// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef THIRD_PARTY_TPU_RAIDEN_CORE_KV_MANAGER_HOLDER_H_
#define THIRD_PARTY_TPU_RAIDEN_CORE_KV_MANAGER_HOLDER_H_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tpu_sync/core/raiden_transfer_endpoint.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/rpc/raiden_service.pb.h"

namespace tpu_raiden {

namespace internal {

template <typename T, typename = void>
struct has_h2d_write : std::false_type {};

template <typename T>
struct has_h2d_write<T, std::void_t<decltype(std::declval<T&>().H2dWrite(
                            std::declval<absl::string_view>(),
                            std::declval<const std::vector<int64_t>&>(),
                            std::declval<const std::vector<int64_t>&>(),
                            std::declval<const std::vector<int64_t>&>(),
                            std::declval<const std::vector<int64_t>&>()))>>
    : std::true_type {};

template <typename T>
inline constexpr bool has_h2d_write_v = has_h2d_write<T>::value;

template <typename T, typename = void>
struct has_h2d_read : std::false_type {};

template <typename T>
struct has_h2d_read<T, std::void_t<decltype(std::declval<T&>().H2dRead(
                           std::declval<absl::string_view>(),
                           std::declval<const std::vector<int64_t>&>(),
                           std::declval<const std::vector<int64_t>&>(),
                           std::declval<const std::vector<int64_t>&>(),
                           std::declval<const std::vector<int64_t>&>()))>>
    : std::true_type {};

template <typename T>
inline constexpr bool has_h2d_read_v = has_h2d_read<T>::value;

template <typename T, typename = void>
struct has_d2h_write : std::false_type {};

template <typename T>
struct has_d2h_write<T, std::void_t<decltype(std::declval<T&>().D2hWrite(
                            std::declval<absl::string_view>(),
                            std::declval<const std::vector<int64_t>&>(),
                            std::declval<const std::vector<int64_t>&>(),
                            std::declval<const std::vector<int64_t>&>(),
                            std::declval<const std::vector<int64_t>&>()))>>
    : std::true_type {};

template <typename T>
inline constexpr bool has_d2h_write_v = has_d2h_write<T>::value;

template <typename T, typename = void>
struct has_d2h_read : std::false_type {};

template <typename T>
struct has_d2h_read<T, std::void_t<decltype(std::declval<T&>().D2hRead(
                           std::declval<absl::string_view>(),
                           std::declval<const std::vector<int64_t>&>(),
                           std::declval<const std::vector<int64_t>&>(),
                           std::declval<const std::vector<int64_t>&>()))>>
    : std::true_type {};

template <typename T>
inline constexpr bool has_d2h_read_v = has_d2h_read<T>::value;

template <typename T, typename = void>
struct has_vector_h2h_write : std::false_type {};

template <typename T>
struct has_vector_h2h_write<
    T, std::void_t<decltype(std::declval<T&>().H2hWrite(
           std::declval<const std::vector<RaidenTransferEndpoint>&>(),
           std::declval<const std::vector<int>&>(),
           std::declval<const std::vector<int>&>()))>> : std::true_type {};

template <typename T>
inline constexpr bool has_vector_h2h_write_v = has_vector_h2h_write<T>::value;

template <typename T, typename = void>
struct has_vector_h2h_read : std::false_type {};

template <typename T>
struct has_vector_h2h_read<
    T, std::void_t<decltype(std::declval<T&>().H2hRead(
           std::declval<const std::vector<RaidenTransferEndpoint>&>(),
           std::declval<const std::vector<int>&>()))>> : std::true_type {};

template <typename T>
inline constexpr bool has_vector_h2h_read_v = has_vector_h2h_read<T>::value;

template <typename T, typename = void>
struct has_vector_h2h_read_explicit : std::false_type {};

template <typename T>
struct has_vector_h2h_read_explicit<
    T, std::void_t<decltype(std::declval<T&>().H2hReadExplicit(
           std::declval<const std::vector<RaidenTransferEndpoint>&>(),
           std::declval<const std::vector<int>&>(),
           std::declval<const std::vector<int>&>()))>> : std::true_type {};

template <typename T>
inline constexpr bool has_vector_h2h_read_explicit_v =
    has_vector_h2h_read_explicit<T>::value;

template <typename T, typename = void>
struct has_peer_h2h_read_explicit : std::false_type {};

template <typename T>
struct has_peer_h2h_read_explicit<
    T, std::void_t<decltype(std::declval<T&>().H2hReadExplicit(
           std::declval<std::string>(),
           std::declval<const std::vector<int>&>(),
           std::declval<const std::vector<int>&>(),
           std::declval<const std::vector<uint8_t*>&>()))>> : std::true_type {
};

template <typename T>
inline constexpr bool has_peer_h2h_read_explicit_v =
    has_peer_h2h_read_explicit<T>::value;

template <typename T, typename = void>
struct has_vector_h2d_read : std::false_type {};

template <typename T>
struct has_vector_h2d_read<
    T, std::void_t<decltype(std::declval<T&>().H2dRead(
           std::declval<const std::vector<RaidenTransferEndpoint>&>(),
           std::declval<const std::vector<int64_t>&>(),
           std::declval<const std::vector<int64_t>&>(),
           std::declval<const std::vector<int64_t>&>(),
           std::declval<const std::vector<int64_t>&>()))>> : std::true_type {};

template <typename T>
inline constexpr bool has_vector_h2d_read_v = has_vector_h2d_read<T>::value;

template <typename T, typename = void>
struct has_pool_reshard_push : std::false_type {};

template <typename T>
struct has_pool_reshard_push<
    T, std::void_t<decltype(std::declval<T&>().PoolReshardPush(
           std::declval<const tpu_sync::rpc::StartTransferRequest&>(),
           std::declval<absl::Span<const int64_t>>(), std::declval<int>()))>>
    : std::true_type {};

template <typename T>
inline constexpr bool has_pool_reshard_push_v =
    has_pool_reshard_push<T>::value;

template <typename T, typename = void>
struct has_pool_reshard_register_recv : std::false_type {};

template <typename T>
struct has_pool_reshard_register_recv<
    T, std::void_t<decltype(std::declval<T&>().PoolReshardRegisterRecv(
           std::declval<const tpu_sync::rpc::StartTransferRequest&>(),
           std::declval<absl::Span<const int64_t>>()))>> : std::true_type {};

template <typename T>
inline constexpr bool has_pool_reshard_register_recv_v =
    has_pool_reshard_register_recv<T>::value;

}  // namespace internal

// Type-erased wrapper for any KV Cache Manager or Transfer Manager
// implementation that provides asynchronous D2H and H2D transfers.
class KVManagerHolder {
 public:
  class Concept {
   public:
    virtual ~Concept() = default;
    virtual absl::StatusOr<raiden::PjRtCopyFuture> D2h(
        const std::vector<int64_t>& src_offsets,
        const std::vector<int64_t>& dst_offsets,
        const std::vector<int64_t>& copy_sizes) = 0;
    virtual absl::StatusOr<raiden::PjRtCopyFuture> H2d(
        const std::vector<int64_t>& src_offsets,
        const std::vector<int64_t>& dst_offsets,
        const std::vector<int64_t>& copy_sizes) = 0;
    virtual absl::StatusOr<raiden::PjRtCopyFuture> H2hRead(
        absl::string_view peer, const std::vector<int64_t>& src_offsets,
        const std::vector<int64_t>& dst_offsets) = 0;
    virtual absl::StatusOr<raiden::PjRtCopyFuture> H2hWrite(
        absl::string_view peer, const std::vector<int64_t>& src_offsets,
        const std::vector<int64_t>& dst_offsets) = 0;
    virtual absl::StatusOr<raiden::PjRtCopyFuture> H2hRead(
        const std::vector<RaidenTransferEndpoint>& remote_descriptors,
        const std::vector<int64_t>& src_offsets,
        const std::vector<int64_t>& dst_offsets) = 0;
    virtual absl::StatusOr<raiden::PjRtCopyFuture> H2hWrite(
        const std::vector<RaidenTransferEndpoint>& remote_descriptors,
        const std::vector<int64_t>& src_offsets,
        const std::vector<int64_t>& dst_offsets) = 0;
    virtual absl::StatusOr<raiden::PjRtCopyFuture> H2dWrite(
        absl::string_view peer, const std::vector<int64_t>& src_host_offsets,
        const std::vector<int64_t>& dst_host_offsets,
        const std::vector<int64_t>& dst_device_offsets,
        const std::vector<int64_t>& copy_sizes) = 0;
    virtual absl::StatusOr<raiden::PjRtCopyFuture> H2dRead(
        absl::string_view peer, const std::vector<int64_t>& src_host_offsets,
        const std::vector<int64_t>& dst_host_offsets,
        const std::vector<int64_t>& dst_device_offsets,
        const std::vector<int64_t>& copy_sizes) = 0;
    virtual absl::StatusOr<raiden::PjRtCopyFuture> H2dRead(
        const std::vector<RaidenTransferEndpoint>& remote_descriptors,
        const std::vector<int64_t>& src_host_offsets,
        const std::vector<int64_t>& dst_host_offsets,
        const std::vector<int64_t>& dst_device_offsets,
        const std::vector<int64_t>& copy_sizes) = 0;
    virtual absl::StatusOr<raiden::PjRtCopyFuture> D2hWrite(
        absl::string_view peer, const std::vector<int64_t>& src_device_offsets,
        const std::vector<int64_t>& src_host_offsets,
        const std::vector<int64_t>& dst_host_offsets,
        const std::vector<int64_t>& copy_sizes) = 0;
    virtual absl::StatusOr<raiden::PjRtCopyFuture> D2hRead(
        absl::string_view peer, const std::vector<int64_t>& src_offsets,
        const std::vector<int64_t>& dst_offsets,
        const std::vector<int64_t>& copy_sizes) = 0;
    // Pool-reshard executor hooks used by the transfer-program worker entry.
    // Fail closed with Unimplemented when the wrapped manager lacks the pool
    // executor, matching the KVCacheManagerBase defaults.
    virtual absl::Status PoolReshardPush(
        const tpu_sync::rpc::StartTransferRequest& request,
        absl::Span<const int64_t> src_block_ids, int parallelism) = 0;
    virtual absl::Status PoolReshardRegisterRecv(
        const tpu_sync::rpc::StartTransferRequest& request,
        absl::Span<const int64_t> chip_block_ids) = 0;
  };

  template <typename T>
  class Model final : public Concept {
   public:
    explicit Model(T* impl) : impl_(impl) {}
    absl::StatusOr<raiden::PjRtCopyFuture> D2h(
        const std::vector<int64_t>& src_offsets,
        const std::vector<int64_t>& dst_offsets,
        const std::vector<int64_t>& copy_sizes) override {
      return impl_->D2h(src_offsets, dst_offsets, copy_sizes);
    }
    absl::StatusOr<raiden::PjRtCopyFuture> H2d(
        const std::vector<int64_t>& src_offsets,
        const std::vector<int64_t>& dst_offsets,
        const std::vector<int64_t>& copy_sizes) override {
      return impl_->H2d(src_offsets, dst_offsets, copy_sizes);
    }
    absl::StatusOr<raiden::PjRtCopyFuture> H2hRead(
        absl::string_view peer, const std::vector<int64_t>& src_offsets,
        const std::vector<int64_t>& dst_offsets) override {
      ABSL_ASSIGN_OR_RETURN(std::vector<int> src_ids,
                            SafeCastOffsets(src_offsets));
      // When the caller named its destination blocks, land the data THERE:
      // plain H2hRead auto-allocates destination blocks from the manager's
      // own accounting, which neither matches the ids the caller reserved
      // and committed to its directory nor respects blocks the controller
      // already handed out.
      if constexpr (internal::has_peer_h2h_read_explicit_v<T>) {
        if (!dst_offsets.empty()) {
          ABSL_ASSIGN_OR_RETURN(std::vector<int> dst_ids,
                                SafeCastOffsets(dst_offsets));
          return impl_->H2hReadExplicit(std::string(peer), src_ids, dst_ids,
                                        /*explicit_dst_ptrs=*/{});
        }
      }
      ABSL_ASSIGN_OR_RETURN(auto res,
                            impl_->H2hRead(std::string(peer), src_ids));
      return res.second;
    }
    absl::StatusOr<raiden::PjRtCopyFuture> H2hWrite(
        absl::string_view peer, const std::vector<int64_t>& src_offsets,
        const std::vector<int64_t>& dst_offsets) override {
      ABSL_ASSIGN_OR_RETURN(std::vector<int> src_ids,
                            SafeCastOffsets(src_offsets));
      ABSL_ASSIGN_OR_RETURN(std::vector<int> dst_ids,
                            SafeCastOffsets(dst_offsets));
      ABSL_ASSIGN_OR_RETURN(
          auto res, impl_->H2hWrite(std::string(peer), src_ids, dst_ids));
      return res.second;
    }
    absl::StatusOr<raiden::PjRtCopyFuture> H2hRead(
        const std::vector<RaidenTransferEndpoint>& remote_descriptors,
        const std::vector<int64_t>& src_offsets,
        const std::vector<int64_t>& dst_offsets) override {
      ABSL_ASSIGN_OR_RETURN(std::vector<int> src_ids,
                            SafeCastOffsets(src_offsets));
      // When the caller named its destination blocks, land the data THERE.
      // Plain H2hRead auto-allocates, which is fine for a fire-and-forget pull
      // but wrong for a store-level read: the store already reserved landing
      // blocks and commits those ids into its directory, so auto-allocated
      // blocks would leave the directory pointing at the wrong memory.
      if constexpr (internal::has_vector_h2h_read_explicit_v<T>) {
        if (!dst_offsets.empty()) {
          ABSL_ASSIGN_OR_RETURN(std::vector<int> dst_ids,
                                SafeCastOffsets(dst_offsets));
          return impl_->H2hReadExplicit(remote_descriptors, src_ids, dst_ids);
        }
      } else if constexpr (internal::has_peer_h2h_read_explicit_v<T>) {
        // No descriptor-shaped explicit read; the peer-string one lands the
        // blocks just as precisely.
        if (!dst_offsets.empty() && !remote_descriptors.empty()) {
          ABSL_ASSIGN_OR_RETURN(std::vector<int> dst_ids,
                                SafeCastOffsets(dst_offsets));
          return impl_->H2hReadExplicit(remote_descriptors[0].endpoint,
                                        src_ids, dst_ids,
                                        /*explicit_dst_ptrs=*/{});
        }
      }
      if constexpr (internal::has_vector_h2h_read_v<T>) {
        ABSL_ASSIGN_OR_RETURN(auto res,
                              impl_->H2hRead(remote_descriptors, src_ids));
        return res.second;
      } else {
        std::string peer =
            remote_descriptors.empty() ? "" : remote_descriptors[0].endpoint;
        ABSL_ASSIGN_OR_RETURN(auto res, impl_->H2hRead(peer, src_ids));
        return res.second;
      }
    }
    absl::StatusOr<raiden::PjRtCopyFuture> H2hWrite(
        const std::vector<RaidenTransferEndpoint>& remote_descriptors,
        const std::vector<int64_t>& src_offsets,
        const std::vector<int64_t>& dst_offsets) override {
      ABSL_ASSIGN_OR_RETURN(std::vector<int> src_ids,
                            SafeCastOffsets(src_offsets));
      ABSL_ASSIGN_OR_RETURN(std::vector<int> dst_ids,
                            SafeCastOffsets(dst_offsets));
      if constexpr (internal::has_vector_h2h_write_v<T>) {
        ABSL_ASSIGN_OR_RETURN(
            auto res, impl_->H2hWrite(remote_descriptors, src_ids, dst_ids));
        return res.second;
      } else {
        std::string peer =
            remote_descriptors.empty() ? "" : remote_descriptors[0].endpoint;
        ABSL_ASSIGN_OR_RETURN(auto res,
                              impl_->H2hWrite(peer, src_ids, dst_ids));
        return res.second;
      }
    }
    absl::StatusOr<raiden::PjRtCopyFuture> H2dWrite(
        absl::string_view peer, const std::vector<int64_t>& src_host_offsets,
        const std::vector<int64_t>& dst_host_offsets,
        const std::vector<int64_t>& dst_device_offsets,
        const std::vector<int64_t>& copy_sizes) override {
      if constexpr (internal::has_h2d_write_v<T>) {
        return impl_->H2dWrite(peer, src_host_offsets, dst_host_offsets,
                               dst_device_offsets, copy_sizes);
      } else {
        return absl::UnimplementedError(
            "H2dWrite is not implemented by the underlying transfer manager.");
      }
    }
    absl::StatusOr<raiden::PjRtCopyFuture> H2dRead(
        absl::string_view peer, const std::vector<int64_t>& src_host_offsets,
        const std::vector<int64_t>& dst_host_offsets,
        const std::vector<int64_t>& dst_device_offsets,
        const std::vector<int64_t>& copy_sizes) override {
      if constexpr (internal::has_h2d_read_v<T>) {
        return impl_->H2dRead(peer, src_host_offsets, dst_host_offsets,
                              dst_device_offsets, copy_sizes);
      } else {
        return absl::UnimplementedError(
            "H2dRead is not implemented by the underlying transfer manager.");
      }
    }
    absl::StatusOr<raiden::PjRtCopyFuture> H2dRead(
        const std::vector<RaidenTransferEndpoint>& remote_descriptors,
        const std::vector<int64_t>& src_host_offsets,
        const std::vector<int64_t>& dst_host_offsets,
        const std::vector<int64_t>& dst_device_offsets,
        const std::vector<int64_t>& copy_sizes) override {
      if constexpr (internal::has_vector_h2d_read_v<T>) {
        return impl_->H2dRead(remote_descriptors, src_host_offsets,
                              dst_host_offsets, dst_device_offsets, copy_sizes);
      } else {
        // Fall back to the single-peer overload (which itself handles impls
        // without any H2dRead), mirroring the vector H2hRead/H2hWrite paths.
        std::string peer =
            remote_descriptors.empty() ? "" : remote_descriptors[0].endpoint;
        return this->H2dRead(peer, src_host_offsets, dst_host_offsets,
                             dst_device_offsets, copy_sizes);
      }
    }
    absl::StatusOr<raiden::PjRtCopyFuture> D2hWrite(
        absl::string_view peer, const std::vector<int64_t>& src_device_offsets,
        const std::vector<int64_t>& src_host_offsets,
        const std::vector<int64_t>& dst_host_offsets,
        const std::vector<int64_t>& copy_sizes) override {
      if constexpr (internal::has_d2h_write_v<T>) {
        return impl_->D2hWrite(peer, src_device_offsets, src_host_offsets,
                               dst_host_offsets, copy_sizes);
      } else {
        return absl::UnimplementedError(
            "D2hWrite is not implemented by the underlying transfer manager.");
      }
    }
    absl::StatusOr<raiden::PjRtCopyFuture> D2hRead(
        absl::string_view peer, const std::vector<int64_t>& src_offsets,
        const std::vector<int64_t>& dst_offsets,
        const std::vector<int64_t>& copy_sizes) override {
      if constexpr (internal::has_d2h_read_v<T>) {
        return impl_->D2hRead(peer, src_offsets, dst_offsets, copy_sizes);
      } else {
        return absl::UnimplementedError(
            "D2hRead is not implemented by the underlying transfer manager.");
      }
    }
    absl::Status PoolReshardPush(
        const tpu_sync::rpc::StartTransferRequest& request,
        absl::Span<const int64_t> src_block_ids, int parallelism) override {
      if constexpr (internal::has_pool_reshard_push_v<T>) {
        return impl_->PoolReshardPush(request, src_block_ids, parallelism);
      } else {
        return absl::UnimplementedError(
            "PoolReshardPush is not implemented by the underlying transfer "
            "manager.");
      }
    }
    absl::Status PoolReshardRegisterRecv(
        const tpu_sync::rpc::StartTransferRequest& request,
        absl::Span<const int64_t> chip_block_ids) override {
      if constexpr (internal::has_pool_reshard_register_recv_v<T>) {
        return impl_->PoolReshardRegisterRecv(request, chip_block_ids);
      } else {
        return absl::UnimplementedError(
            "PoolReshardRegisterRecv is not implemented by the underlying "
            "transfer manager.");
      }
    }

   private:
    absl::StatusOr<std::vector<int>> SafeCastOffsets(
        const std::vector<int64_t>& offsets) {
      std::vector<int> ids;
      ids.reserve(offsets.size());
      for (int64_t offset : offsets) {
        if (offset < std::numeric_limits<int>::min() ||
            offset > std::numeric_limits<int>::max()) {
          return absl::InvalidArgumentError(
              absl::StrCat("Offset ", offset, " overflows int"));
        }
        ids.push_back(static_cast<int>(offset));
      }
      return ids;
    }
    T* impl_;
  };

  KVManagerHolder() : self_(nullptr) {}

  KVManagerHolder(std::nullptr_t) : self_(nullptr) {}  // NOLINT

  template <typename T>
  KVManagerHolder(T* impl)  // NOLINT
      : self_(impl ? std::make_unique<Model<T>>(impl) : nullptr) {}

  absl::StatusOr<raiden::PjRtCopyFuture> D2h(
      const std::vector<int64_t>& src_offsets,
      const std::vector<int64_t>& dst_offsets,
      const std::vector<int64_t>& copy_sizes) const {
    if (!self_) {
      return absl::InternalError("KVManagerHolder is null");
    }
    return self_->D2h(src_offsets, dst_offsets, copy_sizes);
  }

  absl::StatusOr<raiden::PjRtCopyFuture> H2d(
      const std::vector<int64_t>& src_offsets,
      const std::vector<int64_t>& dst_offsets,
      const std::vector<int64_t>& copy_sizes) const {
    if (!self_) {
      return absl::InternalError("KVManagerHolder is null");
    }
    return self_->H2d(src_offsets, dst_offsets, copy_sizes);
  }

  absl::StatusOr<raiden::PjRtCopyFuture> H2hRead(
      absl::string_view peer, const std::vector<int64_t>& src_offsets,
      const std::vector<int64_t>& dst_offsets) const {
    if (!self_) {
      return absl::InternalError("KVManagerHolder is null");
    }
    return self_->H2hRead(peer, src_offsets, dst_offsets);
  }

  absl::StatusOr<raiden::PjRtCopyFuture> H2hWrite(
      absl::string_view peer, const std::vector<int64_t>& src_offsets,
      const std::vector<int64_t>& dst_offsets) const {
    if (!self_) {
      return absl::InternalError("KVManagerHolder is null");
    }
    return self_->H2hWrite(peer, src_offsets, dst_offsets);
  }

  absl::StatusOr<raiden::PjRtCopyFuture> H2hRead(
      const std::vector<RaidenTransferEndpoint>& remote_descriptors,
      const std::vector<int64_t>& src_offsets,
      const std::vector<int64_t>& dst_offsets) const {
    if (!self_) {
      return absl::InternalError("KVManagerHolder is null");
    }
    return self_->H2hRead(remote_descriptors, src_offsets, dst_offsets);
  }

  absl::StatusOr<raiden::PjRtCopyFuture> H2hWrite(
      const std::vector<RaidenTransferEndpoint>& remote_descriptors,
      const std::vector<int64_t>& src_offsets,
      const std::vector<int64_t>& dst_offsets) const {
    if (!self_) {
      return absl::InternalError("KVManagerHolder is null");
    }
    return self_->H2hWrite(remote_descriptors, src_offsets, dst_offsets);
  }

  absl::StatusOr<raiden::PjRtCopyFuture> H2dWrite(
      absl::string_view peer, const std::vector<int64_t>& src_host_offsets,
      const std::vector<int64_t>& dst_host_offsets,
      const std::vector<int64_t>& dst_device_offsets,
      const std::vector<int64_t>& copy_sizes) const {
    if (!self_) {
      return absl::InternalError("KVManagerHolder is null");
    }
    return self_->H2dWrite(peer, src_host_offsets, dst_host_offsets,
                           dst_device_offsets, copy_sizes);
  }

  absl::StatusOr<raiden::PjRtCopyFuture> H2dRead(
      absl::string_view peer, const std::vector<int64_t>& src_host_offsets,
      const std::vector<int64_t>& dst_host_offsets,
      const std::vector<int64_t>& dst_device_offsets,
      const std::vector<int64_t>& copy_sizes) const {
    if (!self_) {
      return absl::InternalError("KVManagerHolder is null");
    }
    return self_->H2dRead(peer, src_host_offsets, dst_host_offsets,
                          dst_device_offsets, copy_sizes);
  }

  absl::StatusOr<raiden::PjRtCopyFuture> H2dRead(
      const std::vector<RaidenTransferEndpoint>& remote_descriptors,
      const std::vector<int64_t>& src_host_offsets,
      const std::vector<int64_t>& dst_host_offsets,
      const std::vector<int64_t>& dst_device_offsets,
      const std::vector<int64_t>& copy_sizes) const {
    if (!self_) {
      return absl::InternalError("KVManagerHolder is null");
    }
    return self_->H2dRead(remote_descriptors, src_host_offsets,
                          dst_host_offsets, dst_device_offsets, copy_sizes);
  }

  absl::StatusOr<raiden::PjRtCopyFuture> D2hWrite(
      absl::string_view peer, const std::vector<int64_t>& src_device_offsets,
      const std::vector<int64_t>& src_host_offsets,
      const std::vector<int64_t>& dst_host_offsets,
      const std::vector<int64_t>& copy_sizes) const {
    if (!self_) {
      return absl::InternalError("KVManagerHolder is null");
    }
    return self_->D2hWrite(peer, src_device_offsets, src_host_offsets,
                           dst_host_offsets, copy_sizes);
  }

  absl::StatusOr<raiden::PjRtCopyFuture> D2hRead(
      absl::string_view peer, const std::vector<int64_t>& src_offsets,
      const std::vector<int64_t>& dst_offsets,
      const std::vector<int64_t>& copy_sizes) const {
    if (!self_) {
      return absl::InternalError("KVManagerHolder is null");
    }
    return self_->D2hRead(peer, src_offsets, dst_offsets, copy_sizes);
  }

  absl::Status PoolReshardPush(
      const tpu_sync::rpc::StartTransferRequest& request,
      absl::Span<const int64_t> src_block_ids, int parallelism) const {
    if (!self_) {
      return absl::InternalError("KVManagerHolder is null");
    }
    return self_->PoolReshardPush(request, src_block_ids, parallelism);
  }

  absl::Status PoolReshardRegisterRecv(
      const tpu_sync::rpc::StartTransferRequest& request,
      absl::Span<const int64_t> chip_block_ids) const {
    if (!self_) {
      return absl::InternalError("KVManagerHolder is null");
    }
    return self_->PoolReshardRegisterRecv(request, chip_block_ids);
  }

  explicit operator bool() const { return self_ != nullptr; }
  bool operator==(std::nullptr_t) const { return self_ == nullptr; }
  bool operator!=(std::nullptr_t) const { return self_ != nullptr; }

 private:
  std::unique_ptr<Concept> self_;
};

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_CORE_KV_MANAGER_HOLDER_H_
