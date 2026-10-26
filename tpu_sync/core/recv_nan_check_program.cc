// Copyright 2026 Google LLC.
// SPDX-License-Identifier: Apache-2.0
#include "tpu_sync/core/recv_nan_check_program.h"

#include <sstream>

#include "tpu_sync/core/recv_nan_check_fp8.h"
#include "xla/hlo/builder/lib/constants.h"
#include "xla/shape_util.h"

namespace tpu_raiden::recv_nan_internal {
namespace {
xla::XlaOp ScalarU32(xla::XlaBuilder* b, uint32_t v) {
  return xla::ConstantR0<uint32_t>(b, v);
}
xla::XlaOp ScalarS64(xla::XlaBuilder* b, int64_t v) {
  return xla::ConstantR0<int64_t>(b, v);
}

}  // namespace

absl::StatusOr<xla::XlaComputation> BuildNanCheckComputation(
    std::span<const BatchInput> inputs, const ScanBatch& batch,
    const std::vector<xla::Shape>& shapes) {
  xla::XlaBuilder b("received_kv_nan");
  xla::XlaBuilder rb("received_kv_nan_or");
  auto a =
      xla::Parameter(&rb, 0, xla::ShapeUtil::MakeShape(xla::PRED, {}), "a");
  auto c =
      xla::Parameter(&rb, 1, xla::ShapeUtil::MakeShape(xla::PRED, {}), "b");
  auto reduce_or = rb.Build(xla::Or(a, c));
  if (!reduce_or.ok()) return reduce_or.status();
  auto meta = xla::Parameter(&b, batch.raw_inputs.size(), shapes.back(),
                             "received_byte_ranges");
  std::vector<xla::XlaOp> flags;
  int64_t row = 0;
  constexpr int64_t words = kNanCheckChunkBytes / 4;
  std::vector<xla::XlaOp> raw_buffers;
  for (size_t i = 0; i < batch.raw_inputs.size(); ++i)
    raw_buffers.push_back(xla::Parameter(&b, i, shapes[i], "raw_kv_words"));
  for (const auto& group : batch.groups) {
    const auto& in = inputs[group.inputs.front()];
    const int64_t count = group.inputs.size();
    const int64_t rows = count * in.bucket;
    auto raw = raw_buffers[group.raw_argument];
    auto column = [&](int col) {
      auto x = xla::Slice(meta, {row, col}, {row + rows, col + 1}, {1, 1});
      return xla::BroadcastInDim(xla::Reshape(x, {rows}), {rows, words}, {0});
    };
    auto word_index = xla::Add(
        column(0),
        xla::Iota(&b, xla::ShapeUtil::MakeShape(xla::S64, {rows, words}), 1));
    xla::GatherDimensionNumbers dims;
    dims.add_collapsed_slice_dims(0);
    dims.add_start_index_map(0);
    dims.set_index_vector_dim(2);  // implicit scalar index for every word
    auto bits = xla::Gather(raw, word_index, dims, {1});
    auto byte = xla::Mul(word_index, ScalarS64(&b, 4));
    auto begin = column(1), end = column(2);
    const auto& f = in.format;
    xla::XlaOp bad;
    if (f.element_bytes == 1) {
      bad =
          recv_nan_internal::Fp8NonFinite(bits, byte, begin, end, f.scan_kind);
    } else {
      // IEEE non-finite values have all exponent bits set, including both
      // infinities and NaNs. Raw words preserve the declared pool's dtype.
      auto is_nonfinite = [&](xla::XlaOp value) {
        return xla::Eq(xla::And(value, ScalarU32(&b, f.exponent_mask)),
                       ScalarU32(&b, f.exponent_mask));
      };
      auto valid = [&](xla::XlaOp start, int64_t bytes) {
        return xla::And(xla::Ge(start, begin),
                        xla::Le(xla::Add(start, ScalarS64(&b, bytes)), end));
      };
      bad = xla::And(is_nonfinite(bits), valid(byte, f.element_bytes));
      if (f.element_bytes == 2) {
        auto high = xla::ShiftRightLogical(bits, ScalarU32(&b, 16));
        bad =
            xla::Or(bad, xla::And(is_nonfinite(high),
                                  valid(xla::Add(byte, ScalarS64(&b, 2)), 2)));
      }
    }
    // Preserve one independent result per logical pool. Only the identical
    // storage/format/bucket graph is shared, never the byte-range predicate.
    auto flag =
        xla::Reduce(xla::Reshape(bad, {count, in.bucket, words}),
                    xla::ConstantR0<bool>(&b, false), *reduce_or, {1, 2});
    flags.push_back(flag);
    row += rows;
  }
  return b.Build(xla::ConcatInDim(&b, flags, 0));
}

std::string NanCheckProgramKey(PJRT_Client* client,
                               std::span<const BatchInput> inputs,
                               const ScanBatch& batch,
                               const std::vector<xla::Shape>& shapes) {
  std::ostringstream key;
  key << client;
  for (size_t i = 0; i < batch.raw_inputs.size(); ++i)
    key << ':' << shapes[i].ToString(true);
  for (const auto& group : batch.groups) {
    const auto& input = inputs[group.inputs.front()];
    key << ";g:" << group.raw_argument << ':' << group.inputs.size() << ':'
        << input.bucket << ':' << input.format.element_bytes << ':'
        << input.format.exponent_mask << ':'
        << static_cast<int>(input.format.scan_kind);
  }
  key << ";meta:" << shapes.back().ToString(true);
  return key.str();
}
}  // namespace tpu_raiden::recv_nan_internal
