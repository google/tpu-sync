// Copyright 2026 Google LLC.
// SPDX-License-Identifier: Apache-2.0
#include "tpu_sync/core/recv_nan_check.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <exception>
#include <initializer_list>
#include <map>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>

#include "absl/cleanup/cleanup.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/core/recv_nan_check_program.h"
#include "tpu_sync/core/recv_nan_check_worker.h"
#include "tpu_sync/kv_cache/kv_cache_manager_base.h"
#include "tpu_sync/kv_cache/pool_layout.h"
#include "tpu_sync/rpc/raiden_service.pb.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/literal.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/proto/compile_options.pb.h"
#include "xla/shape_util.h"

namespace tpu_raiden {
namespace {
struct ReceivedBlock {
  int64_t id;
  std::vector<NanCheckChunk> chunks;
};
struct ReceivedPool {
  size_t index;
  size_t storage;
  std::string tag;
  std::string dtype;
  NanFormat format;
  std::vector<ReceivedBlock> blocks;
  std::vector<raiden::BufferHoldAndAlias> shards;
};
struct ShardInput {
  const ReceivedPool* pool;
  size_t shard;
  const raiden::BufferHoldAndAlias* hold;
  std::vector<NanCheckChunk> chunks;
  int64_t bucket = 1;
};
struct BufferDeleter {
  const PJRT_Api* api;
  void operator()(PJRT_Buffer* buffer) const {
    PJRT_Buffer_Destroy_Args args{};
    args.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
    args.buffer = buffer;
    raiden::PjrtErrorToStatusLocal(api, api->PJRT_Buffer_Destroy(&args))
        .IgnoreError();
  }
};
using Buffer = std::unique_ptr<PJRT_Buffer, BufferDeleter>;
struct Program {
  const PJRT_Api* api;
  PJRT_LoadedExecutable* executable;
  ~Program() {
    PJRT_LoadedExecutable_Destroy_Args args{};
    args.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
    args.executable = executable;
    raiden::PjrtErrorToStatusLocal(api,
                                   api->PJRT_LoadedExecutable_Destroy(&args))
        .IgnoreError();
  }
};

absl::StatusOr<xla::Shape> BufferShape(const PJRT_Api* api, PJRT_Buffer* buffer,
                                       xla::PrimitiveType type,
                                       std::initializer_list<int64_t> dims) {
  PJRT_Buffer_GetMemoryLayout_Args args{};
  args.struct_size = PJRT_Buffer_GetMemoryLayout_Args_STRUCT_SIZE;
  args.buffer = buffer;
  auto status = raiden::PjrtErrorToStatusLocal(
      api, api->PJRT_Buffer_GetMemoryLayout(&args));
  if (!status.ok()) return status;
  if (args.layout.type != PJRT_Buffer_MemoryLayout_Type_Tiled)
    return absl::UnimplementedError(
        "KV diagnostic requires tiled PJRT device layout");
  auto proto = xla::ShapeUtil::MakeShape(type, dims).ToProto();
  auto* layout = proto.mutable_layout();
  layout->clear_minor_to_major();
  const auto& tiled = args.layout.tiled;
  for (size_t i = 0; i < tiled.minor_to_major_size; ++i)
    layout->add_minor_to_major(tiled.minor_to_major[i]);
  size_t offset = 0;
  for (size_t i = 0; i < tiled.num_tiles; ++i) {
    auto* tile = layout->add_tiles();
    for (size_t j = 0; j < tiled.tile_dim_sizes[i]; ++j)
      tile->add_dimensions(tiled.tile_dims[offset++]);
  }
  return xla::Shape::FromProto(proto);
}

absl::StatusOr<std::shared_ptr<Program>> Compile(
    const PJRT_Api* api, PJRT_Client* client,
    std::span<const recv_nan_internal::BatchInput> inputs,
    const recv_nan_internal::ScanBatch& batch,
    const std::vector<xla::Shape>& shapes) {
  auto computation =
      recv_nan_internal::BuildNanCheckComputation(inputs, batch, shapes);
  if (!computation.ok()) return computation.status();
  xla::CompileOptionsProto options;
  options.set_compile_portable_executable(true);
  for (const auto& shape : shapes)
    *options.add_argument_layouts() = shape.ToProto();
  auto* build = options.mutable_executable_build_options();
  build->set_device_ordinal(-1);
  build->set_num_replicas(1);
  build->set_num_partitions(1);
  *build->mutable_debug_options() = xla::DefaultDebugOptionsIgnoringFlags();
  std::string code = computation->proto().SerializeAsString();
  const std::string serialized = options.SerializeAsString();
  PJRT_Program program{};
  program.struct_size = PJRT_Program_STRUCT_SIZE;
  program.code = code.data();
  program.code_size = code.size();
  program.format = "hlo";
  program.format_size = 3;
  PJRT_Client_Compile_Args compile{};
  compile.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
  compile.client = client;
  compile.program = &program;
  compile.compile_options = serialized.data();
  compile.compile_options_size = serialized.size();
  auto status =
      raiden::PjrtErrorToStatusLocal(api, api->PJRT_Client_Compile(&compile));
  if (!status.ok()) return status;
  return std::shared_ptr<Program>(new Program{api, compile.executable});
}

// Device resources and host destinations outlive asynchronous bool readback.
struct Readback {
  ~Readback() {
    // Normal retirement is already ready. An exception during submission must
    // still drain issued device operations before releasing their buffers.
    if (!done.IsReady()) done.Await().IgnoreError();
  }
  std::shared_ptr<Program> program;
  std::vector<int64_t> metadata;
  std::vector<uint8_t> flags;
  std::vector<size_t> flag_inputs;
  std::vector<Buffer> buffers;
  raiden::PjRtCopyFuture done;
  absl::Status status;
};
struct PendingDevice {
  std::vector<ShardInput> inputs;
  std::shared_ptr<Readback> result;
};
}  // namespace

struct RecvNanCheck::Impl {
  // Declared first so device programs and the joined worker retire before
  // the framework's typed buffer handles are released.
  std::shared_ptr<void> inputs_owner;
  std::mutex mu;
  std::map<std::string, std::shared_ptr<Program>> programs;
  std::map<size_t, NanDumpWindow> windows;
  recv_nan_internal::Worker worker;

  explicit Impl(std::shared_ptr<void> owner) : inputs_owner(std::move(owner)) {}
  ~Impl() { worker.Drain(); }
  void Drain() { worker.Drain(); }

  std::shared_ptr<Readback> Submit(const std::vector<ShardInput>& inputs) {
    const auto start = std::chrono::steady_clock::now();
    const auto elapsed_ms = [&] {
      return std::chrono::duration<double, std::milli>(
                 std::chrono::steady_clock::now() - start).count();
    };
    std::vector<recv_nan_internal::BatchInput> signatures;
    signatures.reserve(inputs.size());
    for (const auto& input : inputs)
      signatures.push_back({input.pool->storage, input.shard, input.bucket,
                            input.pool->format});
    const auto batch = recv_nan_internal::BuildNanCheckBatch(signatures);
    const auto* api = inputs[0].hold->c_api();
    auto out = std::make_shared<Readback>();
    std::vector<PJRT_Event*> events;
    absl::Cleanup settle_events = [out, api, &events] {
      out->done = raiden::PjRtCopyFuture::FromEvents(api, std::move(events), {});
    };
    out->status = [&]() -> absl::Status {
      const auto& first = *inputs[0].hold;
      auto handles = raiden::GetCApiDeviceHandles(first.device);
      if (!handles.ok()) return handles.status();
      PJRT_RawBuffer_GetMemorySpace_Args mem{};
      mem.struct_size = PJRT_RawBuffer_GetMemorySpace_Args_STRUCT_SIZE;
      mem.buffer = first.c_raw_buffer;
      auto status = raiden::PjrtErrorToStatusLocal(
          api, first.c_hold->extension->PJRT_RawBuffer_GetMemorySpace(&mem));
      if (!status.ok()) return status;
      auto& buffers = out->buffers;
      std::vector<PJRT_Buffer*> args;
      std::vector<xla::Shape> shapes;
      auto& metadata = out->metadata;
      int64_t rows = 0;
      for (size_t index : batch.raw_inputs) {
        const auto& hold = *inputs[index].hold;
        const auto physical = hold.GetOnDeviceSizeInBytes();
        if (physical == 0 || physical % 4 != 0)
          return absl::InvalidArgumentError(
              "KV diagnostic storage must contain whole raw words");
        const int64_t words = physical / 4;
        // Static PJRT inputs may have any shape with matching device bytes.
        // inputs_owner pins the original typed handle; this borrows it without
        // copying KV, consuming the buffer, or creating a foreign-memory view.
        args.push_back(raiden::BorrowCApiBuffer(hold.buffer));
        auto shape = xla::ShapeUtil::MakeShape(xla::U32, {words});
        shapes.push_back(shape);
      }
      for (const auto& group : batch.groups) {
        for (size_t index : group.inputs) {
          const auto& input = inputs[index];
          for (const auto& chunk : input.chunks) {
            metadata.push_back(chunk.word_begin);
            metadata.push_back(chunk.byte_begin);
            metadata.push_back(chunk.byte_end);
          }
          metadata.resize(
              metadata.size() + (input.bucket - input.chunks.size()) * 3, 0);
          rows += input.bucket;
          out->flag_inputs.push_back(index);
        }
      }
      const double prepare_ms = elapsed_ms();
      const std::array<int64_t, 2> dimensions = {rows, 3};
      PJRT_Client_BufferFromHostBuffer_Args meta{};
      meta.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
      meta.client = handles->client;
      meta.memory = mem.memory_space;
      meta.data = metadata.data();
      meta.type = PJRT_Buffer_Type_S64;
      meta.dims = dimensions.data();
      meta.num_dims = dimensions.size();
      meta.host_buffer_semantics =
          PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;
      status = raiden::PjrtErrorToStatusLocal(
          api, api->PJRT_Client_BufferFromHostBuffer(&meta));
      if (!status.ok()) return status;
      buffers.emplace_back(meta.buffer, BufferDeleter{api});
      args.push_back(meta.buffer);
      if (meta.done_with_host_buffer) events.push_back(meta.done_with_host_buffer);
      auto meta_shape = BufferShape(api, meta.buffer, xla::S64, {rows, 3});
      if (!meta_shape.ok()) return meta_shape.status();
      shapes.push_back(*meta_shape);
      const auto cache_key = recv_nan_internal::NanCheckProgramKey(
          handles->client, signatures, batch, shapes);
      const auto key_hash = std::hash<std::string>{}(cache_key);
      const double metadata_ms = elapsed_ms() - prepare_ms;
      bool cache_hit = false, evicted = false;
      size_t cache_size = 0;
      double compile_ms = 0;
      auto& program = out->program;
      {
        std::lock_guard lock(mu);
        const auto it = programs.find(cache_key);
        if (it != programs.end()) {
          program = it->second;
          cache_hit = true;
        } else {
          const double compile_start_ms = elapsed_ms();
          auto compiled = Compile(api, handles->client, signatures, batch, shapes);
          compile_ms = elapsed_ms() - compile_start_ms;
          if (!compiled.ok()) return compiled.status();
          program = *std::move(compiled);
          if (programs.size() >= 32) {
            programs.erase(programs.begin());
            evicted = true;
          }
          programs.emplace(cache_key, program);
        }
        cache_size = programs.size();
      }
      if (!cache_hit)
        LOG(INFO) << "Receive KV diagnostic compile: key_hash=" << key_hash
                  << " cache_size=" << cache_size << " evicted=" << evicted
                  << " pools=" << inputs.size()
                  << " storages=" << batch.raw_inputs.size()
                  << " scan_groups=" << batch.groups.size()
                  << " metadata_rows=" << rows << " compile_ms=" << compile_ms
                  << " signature=" << cache_key;
      const double execute_start_ms = elapsed_ms();
      std::vector<int64_t> no_donation(args.size());
      std::iota(no_donation.begin(), no_donation.end(), 0);
      PJRT_ExecuteOptions options{};
      options.struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;
      options.non_donatable_input_indices = no_donation.data();
      options.num_non_donatable_input_indices = no_donation.size();
      PJRT_Buffer* output = nullptr;
      PJRT_Buffer* const* input_list = args.data();
      PJRT_Buffer** output_list = &output;
      PJRT_Event* complete = nullptr;
      PJRT_LoadedExecutable_Execute_Args execute{};
      execute.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
      execute.executable = program->executable;
      execute.options = &options;
      execute.argument_lists = &input_list;
      execute.num_devices = 1;
      execute.num_args = args.size();
      execute.output_lists = &output_list;
      execute.device_complete_events = &complete;
      execute.execute_device = handles->device;
      status = raiden::PjrtErrorToStatusLocal(
          api, api->PJRT_LoadedExecutable_Execute(&execute));
      if (complete) events.push_back(complete);
      if (output) buffers.emplace_back(output, BufferDeleter{api});
      if (!status.ok()) return status;
      const double execute_ms = elapsed_ms() - execute_start_ms;
      out->flags.resize(inputs.size());
      const int64_t minor_to_major = 0;
      PJRT_Buffer_MemoryLayout layout{};
      layout.struct_size = PJRT_Buffer_MemoryLayout_STRUCT_SIZE;
      layout.type = PJRT_Buffer_MemoryLayout_Type_Tiled;
      layout.tiled.struct_size = PJRT_Buffer_MemoryLayout_Tiled_STRUCT_SIZE;
      layout.tiled.minor_to_major = &minor_to_major;
      layout.tiled.minor_to_major_size = 1;
      PJRT_Buffer_ToHostBuffer_Args copy{};
      copy.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
      copy.src = output;
      copy.host_layout = &layout;
      copy.dst = out->flags.data();
      copy.dst_size = out->flags.size();
      status = raiden::PjrtErrorToStatusLocal(
          api, api->PJRT_Buffer_ToHostBuffer(&copy));
      if (copy.event) events.push_back(copy.event);
      VLOG(1) << "Receive KV diagnostic submit: key_hash=" << key_hash
              << " cache_hit=" << cache_hit << " cache_size=" << cache_size
              << " prepare_ms=" << prepare_ms << " metadata_ms=" << metadata_ms
              << " compile_ms=" << compile_ms << " execute_ms=" << execute_ms
              << " readback_submit_ms="
              << elapsed_ms() - execute_start_ms - execute_ms
              << " total_ms=" << elapsed_ms();
      return status;
    }();
    return out;
  }
};
struct RecvNanCheck::Task::Impl {
  std::shared_ptr<RecvNanCheck> owner;
  uint64_t uuid;
  std::string req;
  std::vector<ReceivedPool> pools;
};
RecvNanCheck::RecvNanCheck(std::weak_ptr<void> inputs_owner)
    : inputs_owner_(std::move(inputs_owner)) {}
RecvNanCheck::~RecvNanCheck() = default;
void RecvNanCheck::Drain() {
  std::lock_guard lock(init_mu_);
  if (impl_) impl_->Drain();
}
RecvNanCheck::Task::Task(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
RecvNanCheck::Task::~Task() = default;
void RecvNanCheck::Task::Schedule(std::function<void()> submitted) {
  auto self = shared_from_this();
  const auto queued_at = std::chrono::steady_clock::now();
  auto submit = [self, queued_at]() -> recv_nan_internal::Worker::Poll {
    try {
      const auto started_at = std::chrono::steady_clock::now();
      VLOG(1) << "Receive KV diagnostic worker: req_id=" << self->impl_->req
              << " uuid=" << self->impl_->uuid << " queue_ms="
              << std::chrono::duration<double, std::milli>(started_at - queued_at)
                     .count();
      auto poll = self->Submit();
      VLOG(1) << "Receive KV diagnostic submitted: req_id=" << self->impl_->req
              << " uuid=" << self->impl_->uuid << " submit_ms="
              << std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - started_at).count();
      return [self, poll = std::move(poll)] {
        try {
          return poll();
        } catch (const std::exception& e) {
          LOG(WARNING) << "Receive KV NaN/Inf result unavailable: req_id="
                       << self->impl_->req << " uuid=" << self->impl_->uuid
                       << " exception=" << e.what();
          return true;
        }
      };
    } catch (const std::exception& e) {
      LOG(WARNING) << "Receive KV NaN/Inf submission unavailable: req_id="
                   << self->impl_->req << " uuid=" << self->impl_->uuid
                   << " exception=" << e.what();
      return [] { return true; };
    }
  };
  if (!impl_->owner->impl_->worker.Schedule(std::move(submit), submitted)) {
    LOG(WARNING) << "Receive KV NaN/Inf check skipped: req_id=" << impl_->req
                 << " uuid=" << impl_->uuid << " diagnostic queue full or draining";
    submitted();
  }
}

std::shared_ptr<RecvNanCheck::Task> RecvNanCheck::Create(
    kv_cache::KVCacheManagerBase* base,
    const ::tpu_sync::rpc::StartTransferRequest& plan,
    std::shared_ptr<RecvNanCheck>* manager_owner) {
  auto task = std::make_unique<Task::Impl>();
  task->uuid = plan.uuid();
  task->req = plan.req_id();
  for (const auto& group : plan.pool_groups())
    for (int32_t pool_idx : group.pool_indices()) {
      const auto& spec = *base->pool(pool_idx);
      const auto format = NanFormatForPool(spec.tag, spec.dtype_tag);
      if (!format) continue;
      auto segments = kv_cache::ExpandPoolLiveSegments(spec);
      if (!segments.ok()) {
        LOG(WARNING) << "Receive KV NaN/Inf check skipped: req_id="
                     << plan.req_id() << " pool=" << pool_idx
                     << " tag=" << spec.tag << " dtype=" << spec.dtype_tag;
        continue;
      }
      ReceivedPool pool{
          static_cast<size_t>(pool_idx), spec.storage_index, spec.tag,
          spec.dtype_tag, *format, {}, {}};
      std::vector<NanCheckRange> ranges;
      bool aligned = true;
      for (const auto& segment : *segments) {
        ranges.push_back({segment.physical_offset, segment.size});
        aligned &= segment.physical_offset % format->element_bytes == 0 &&
                   segment.size % format->element_bytes == 0;
      }
      for (int i = 0; i < group.dst_device_block_ids_size(); ++i) {
        const int64_t block = group.dst_device_block_ids(i),
                      live = group.dst_expected_extent_bytes(i);
        aligned &= (spec.base_offset_bytes + block * spec.block_stride_bytes) %
                       format->element_bytes ==
                   0;
        aligned &= live % format->element_bytes == 0;
        pool.blocks.push_back(
            {block, BuildNanCheckChunks(spec.base_offset_bytes +
                                            block * spec.block_stride_bytes,
                                        ranges, live)});
      }
      if (!aligned) {
        LOG(WARNING) << "Receive KV NaN/Inf check skipped: req_id="
                     << plan.req_id() << " pool=" << pool_idx
                     << " live ranges split numeric elements";
        continue;
      }
      pool.shards = base->buffer_holds()[spec.storage_index].holds;
      if (std::any_of(pool.shards.begin(), pool.shards.end(),
                      [](const auto& h) {
                        return h.is_common_buffer || !h.shape.is_static();
                      })) {
        LOG(WARNING) << "Receive KV NaN/Inf check unavailable: req_id="
                     << plan.req_id() << " pool=" << pool_idx
                     << " static PJRT C API buffers required";
        continue;
      }
      task->pools.push_back(std::move(pool));
    }
  if (task->pools.empty()) return nullptr;
  if (!*manager_owner) {
    LOG(WARNING) << "Receive KV NaN/Inf check unavailable: req_id="
                 << plan.req_id() << " typed buffer owner not registered";
    return nullptr;
  }
  auto& diagnostic = **manager_owner;
  {
    std::lock_guard lock(diagnostic.init_mu_);
    if (!diagnostic.impl_) {
      auto inputs_owner = diagnostic.inputs_owner_.lock();
      if (!inputs_owner) {
        LOG(WARNING) << "Receive KV NaN/Inf check unavailable: req_id="
                     << plan.req_id() << " typed buffer owner expired";
        return nullptr;
      }
      diagnostic.impl_ = std::make_unique<Impl>(std::move(inputs_owner));
    }
  }
  task->owner = *manager_owner;
  return std::shared_ptr<Task>(new Task(std::move(task)));
}

std::function<bool()> RecvNanCheck::Task::Submit() {
  auto& owner = *impl_->owner->impl_;
  std::map<xla::PjRtDevice*, std::vector<ShardInput>> groups;
  for (const auto& pool : impl_->pools)
    for (size_t shard = 0; shard < pool.shards.size(); ++shard) {
      ShardInput input{&pool, shard, &pool.shards[shard], {}};
      for (const auto& block : pool.blocks)
        input.chunks.insert(input.chunks.end(), block.chunks.begin(),
                            block.chunks.end());
      if (input.chunks.empty()) continue;
      while (input.bucket < static_cast<int64_t>(input.chunks.size()))
        input.bucket *= 2;
      groups[input.hold->device].push_back(std::move(input));
    }
  auto pending = std::make_shared<std::vector<PendingDevice>>();
  pending->reserve(groups.size());
  // Submit every device before publishing receive completion; no host Await.
  for (auto& [device, inputs] : groups) {
    auto result = owner.Submit(inputs);
    pending->push_back({std::move(inputs), std::move(result)});
  }
  return [self = shared_from_this(), pending] {
    for (auto& group : *pending) {
      auto error = group.result->done.PollError();
      if (!error.ok()) {
        // IsReady alone maps a readiness-query error to false forever. Only
        // the exceptional path drains issued operations before retiring them.
        group.result->done.Await().IgnoreError();
        group.result->status.Update(error);
      } else if (!group.result->done.IsReady()) {
        return false;
      }
    }
    auto& owner = *self->impl_->owner->impl_;
    std::map<size_t, std::vector<size_t>> bad_shards;
    std::map<size_t, size_t> checked;
    for (auto& group : *pending) {
      const auto status = group.result->status.ok()
                              ? group.result->done.PollError()
                              : group.result->status;
      if (!status.ok()) {
        LOG(WARNING) << "Receive KV NaN/Inf check unavailable: req_id="
                     << self->impl_->req << " uuid=" << self->impl_->uuid
                     << " status=" << status;
        continue;
      }
      for (size_t i = 0; i < group.inputs.size(); ++i) {
        const auto& input = group.inputs[group.result->flag_inputs[i]];
        if (group.result->flags[i])
          bad_shards[input.pool->index].push_back(input.shard);
        ++checked[input.pool->index];
      }
    }
    for (const auto& pool : self->impl_->pools) {
      if (checked[pool.index] != pool.shards.size()) continue;
      const auto& shards = bad_shards[pool.index];
      if (!owner.windows[pool.index].OnCheck(!shards.empty())) continue;
      std::ostringstream locations;
      for (size_t i = 0; i < shards.size(); ++i) {
        if (i) locations << ',';
        locations << shards[i];
      }
      LOG(ERROR) << "NaN/Inf detected in received KV cache: req_id="
                 << self->impl_->req << " uuid=" << self->impl_->uuid
                 << " pool=" << pool.index << " tag=" << pool.tag
                 << " dtype=" << pool.dtype << " shards=[" << locations.str()
                 << "] source=device_hbm";
    }
    return true;
  };
}
}  // namespace tpu_raiden
