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

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
namespace xla {
class PjRtBuffer;
class PjRtClient;
}  // namespace xla
#include "absl/status/status_macros.h"
#include "xla/pjrt/status_casters.h"
#include "tpu_sync/core/raw_transfer_core.h"
#include "tpu_sync/core/raw_transfer_impl.h"
#include "tpu_sync/frameworks/jax/jax_utils.h"
#ifndef WITHOUT_PYTHON
#include <nanobind/nanobind.h>
#else
#include "tpu_sync/frameworks/jax/mock_nanobind.h"
#endif
#include "tpu_sync/frameworks/jax/raw_transfer_internal.h"

namespace raiden {

using ::xla::PjRtBuffer;

absl::StatusOr<PjRtCopyFuture> transfer_d2h_async_internal(
    const nb::object& src_arr, const nb::object& dst_arr,
    absl::Span<const int64_t> src_offsets,
    absl::Span<const int64_t> dst_offsets, absl::Span<const int64_t> copy_sizes,
    bool unsafe_skip_buffer_lock) {
  std::vector<RaidenBufferHandle> src_buffers =
      jax::ExtractPjRtBuffersFromPyArray(src_arr);

  nb::object dst_addressable_shards = dst_arr.attr("addressable_shards");
  size_t num_dst_shards = nb::len(dst_addressable_shards);
  std::vector<uint8_t*> dst_ptrs;
  std::vector<size_t> dst_sizes;
  dst_ptrs.reserve(num_dst_shards);
  dst_sizes.reserve(num_dst_shards);

  for (size_t i = 0; i < num_dst_shards; ++i) {
    nb::object dst_shard_data = dst_addressable_shards[i].attr("data");
    size_t dst_ptr_val =
        nb::cast<size_t>(dst_shard_data.attr("unsafe_buffer_pointer")());
    dst_ptrs.push_back(reinterpret_cast<uint8_t*>(dst_ptr_val));
    dst_sizes.push_back(
        nb::cast<size_t>(dst_shard_data.attr("on_device_size_in_bytes")()));
  }

  return transfer_d2h_core(src_buffers, dst_ptrs, dst_sizes, src_offsets,
                           dst_offsets, copy_sizes, unsafe_skip_buffer_lock);
}

// Unpack and forward to pure C++ transfer_d2h_async_internal
absl::StatusOr<PjRtCopyFuture> transfer_d2h_async(
    const nb::object& src_arr, const nb::object& dst_arr,
    const nb::list& src_offsets_major_dim,
    const nb::list& dst_offsets_major_dim, const nb::list& copy_sizes_major_dim,
    bool unsafe_skip_buffer_lock) {
  return transfer_d2h_async_internal(
      src_arr, dst_arr, jax::UnpackListToVector(src_offsets_major_dim),
      jax::UnpackListToVector(dst_offsets_major_dim),
      jax::UnpackListToVector(copy_sizes_major_dim), unsafe_skip_buffer_lock);
}

absl::StatusOr<PjRtCopyFuture> transfer_h2d_async_internal(
    const nb::object& src_arr, const nb::object& dst_arr,
    absl::Span<const int64_t> src_offsets,
    absl::Span<const int64_t> dst_offsets, absl::Span<const int64_t> copy_sizes,
    bool unsafe_skip_buffer_lock) {
  std::vector<RaidenBufferHandle> dst_buffers =
      jax::ExtractPjRtBuffersFromPyArray(dst_arr);

  nb::object src_addressable_shards = src_arr.attr("addressable_shards");
  size_t num_src_shards = nb::len(src_addressable_shards);
  std::vector<const uint8_t*> src_ptrs;
  std::vector<size_t> src_sizes;
  src_ptrs.reserve(num_src_shards);
  src_sizes.reserve(num_src_shards);

  for (size_t i = 0; i < num_src_shards; ++i) {
    nb::object src_shard_data = src_addressable_shards[i].attr("data");
    size_t src_ptr_val =
        nb::cast<size_t>(src_shard_data.attr("unsafe_buffer_pointer")());
    src_ptrs.push_back(reinterpret_cast<const uint8_t*>(src_ptr_val));
    src_sizes.push_back(
        nb::cast<size_t>(src_shard_data.attr("on_device_size_in_bytes")()));
  }

  return transfer_h2d_core(dst_buffers, src_ptrs, src_sizes, src_offsets,
                           dst_offsets, copy_sizes, unsafe_skip_buffer_lock);
}

// Unpack and forward to pure C++ transfer_h2d_async_internal
absl::StatusOr<PjRtCopyFuture> transfer_h2d_async(
    const nb::object& src_arr, const nb::object& dst_arr,
    const nb::list& src_offsets_major_dim,
    const nb::list& dst_offsets_major_dim, const nb::list& copy_sizes_major_dim,
    bool unsafe_skip_buffer_lock) {
  return transfer_h2d_async_internal(
      src_arr, dst_arr, jax::UnpackListToVector(src_offsets_major_dim),
      jax::UnpackListToVector(dst_offsets_major_dim),
      jax::UnpackListToVector(copy_sizes_major_dim), unsafe_skip_buffer_lock);
}

// Pure FFI JAX helper to run parallel batch transfers using pure C++ core
inline absl::StatusOr<PjRtCopyFuture> transfer_d2h_batch_async_impl(
    const nb::list& src_arrs, const nb::list& dst_arrs,
    const nb::list& src_offsets_major_dim,
    const nb::list& dst_offsets_major_dim, const nb::list& copy_sizes_major_dim,
    bool unsafe_skip_buffer_lock) {
  if (nb::len(src_arrs) != nb::len(dst_arrs)) {
    throw std::runtime_error("Lengths of src_arrs and dst_arrs must match");
  }
  size_t n = nb::len(src_arrs);
  if (n == 0) return PjRtCopyFuture(std::vector<BufferHolder>{});

  std::vector<PjRtCopyFuture> futures;
  futures.reserve(n);

  std::vector<int64_t> s_offsets =
      jax::UnpackListToVector(src_offsets_major_dim);
  std::vector<int64_t> d_offsets =
      jax::UnpackListToVector(dst_offsets_major_dim);
  std::vector<int64_t> c_sizes = jax::UnpackListToVector(copy_sizes_major_dim);

  for (size_t i = 0; i < n; ++i) {
    ABSL_ASSIGN_OR_RETURN(PjRtCopyFuture f,
                          transfer_d2h_async_internal(
                              src_arrs[i], dst_arrs[i], s_offsets, d_offsets,
                              c_sizes, unsafe_skip_buffer_lock));
    futures.push_back(std::move(f));
  }
  return JoinPjRtCopyFutures(absl::MakeSpan(futures));
}

// Pure FFI JAX helper to run parallel batch transfers using pure C++ core
inline absl::StatusOr<PjRtCopyFuture> transfer_h2d_batch_async_impl(
    const nb::list& src_arrs, const nb::list& dst_arrs,
    const nb::list& src_offsets_major_dim,
    const nb::list& dst_offsets_major_dim, const nb::list& copy_sizes_major_dim,
    bool unsafe_skip_buffer_lock) {
  if (nb::len(src_arrs) != nb::len(dst_arrs)) {
    throw std::runtime_error("Lengths of src_arrs and dst_arrs must match");
  }
  size_t n = nb::len(src_arrs);
  if (n == 0) return PjRtCopyFuture(std::vector<BufferHolder>{});

  std::vector<PjRtCopyFuture> futures;
  futures.reserve(n);

  std::vector<int64_t> s_offsets =
      jax::UnpackListToVector(src_offsets_major_dim);
  std::vector<int64_t> d_offsets =
      jax::UnpackListToVector(dst_offsets_major_dim);
  std::vector<int64_t> c_sizes = jax::UnpackListToVector(copy_sizes_major_dim);

  for (size_t i = 0; i < n; ++i) {
    ABSL_ASSIGN_OR_RETURN(PjRtCopyFuture f,
                          transfer_h2d_async_internal(
                              src_arrs[i], dst_arrs[i], s_offsets, d_offsets,
                              c_sizes, unsafe_skip_buffer_lock));
    futures.push_back(std::move(f));
  }
  return JoinPjRtCopyFutures(absl::MakeSpan(futures));
}

absl::StatusOr<PjRtCopyFuture> transfer_d2h_batch_async(
    const nb::list& src_arrs, const nb::list& dst_arrs,
    const nb::list& src_offsets_major_dim,
    const nb::list& dst_offsets_major_dim, const nb::list& copy_sizes_major_dim,
    bool unsafe_skip_buffer_lock) {
  PjRtCopyFuture acc = xla::ValueOrThrow(transfer_d2h_batch_async_impl(
      src_arrs, dst_arrs, src_offsets_major_dim, dst_offsets_major_dim,
      copy_sizes_major_dim, unsafe_skip_buffer_lock));
  return acc;
}

absl::StatusOr<PjRtCopyFuture> transfer_h2d_batch_async(
    const nb::list& src_arrs, const nb::list& dst_arrs,
    const nb::list& src_offsets_major_dim,
    const nb::list& dst_offsets_major_dim, const nb::list& copy_sizes_major_dim,
    bool unsafe_skip_buffer_lock) {
  PjRtCopyFuture acc = xla::ValueOrThrow(transfer_h2d_batch_async_impl(
      src_arrs, dst_arrs, src_offsets_major_dim, dst_offsets_major_dim,
      copy_sizes_major_dim, unsafe_skip_buffer_lock));
  return acc;
}

void transfer_d2h_batch(const nb::list& src_arrs, const nb::list& dst_arrs,
                        const nb::list& src_offsets_major_dim,
                        const nb::list& dst_offsets_major_dim,
                        const nb::list& copy_sizes_major_dim,
                        bool unsafe_skip_buffer_lock) {
  auto future = xla::ValueOrThrow(transfer_d2h_batch_async(
      src_arrs, dst_arrs, src_offsets_major_dim, dst_offsets_major_dim,
      copy_sizes_major_dim, unsafe_skip_buffer_lock));
  nb::gil_scoped_release release;
  absl::Status status = future.Await();
  if (!status.ok()) {
    throw std::runtime_error(std::string("Async copy failed: ") +
                             std::string(status.message()));
  }
}

void transfer_h2d_batch(const nb::list& src_arrs, const nb::list& dst_arrs,
                        const nb::list& src_offsets_major_dim,
                        const nb::list& dst_offsets_major_dim,
                        const nb::list& copy_sizes_major_dim,
                        bool unsafe_skip_buffer_lock) {
  auto future = xla::ValueOrThrow(transfer_h2d_batch_async(
      src_arrs, dst_arrs, src_offsets_major_dim, dst_offsets_major_dim,
      copy_sizes_major_dim, unsafe_skip_buffer_lock));
  nb::gil_scoped_release release;
  absl::Status status = future.Await();
  if (!status.ok()) {
    throw std::runtime_error(std::string("Async copy failed: ") +
                             std::string(status.message()));
  }
}

void transfer_d2h(const nb::object& src_arr, const nb::object& dst_arr,
                  const nb::list& src_offsets_major_dim,
                  const nb::list& dst_offsets_major_dim,
                  const nb::list& copy_sizes_major_dim,
                  bool unsafe_skip_buffer_lock) {
  auto future = xla::ValueOrThrow(transfer_d2h_async(
      src_arr, dst_arr, src_offsets_major_dim, dst_offsets_major_dim,
      copy_sizes_major_dim, unsafe_skip_buffer_lock));
  nb::gil_scoped_release release;
  absl::Status status = future.Await();
  if (!status.ok()) {
    throw std::runtime_error(std::string("Async copy failed: ") +
                             std::string(status.message()));
  }
}

void transfer_h2d(const nb::object& src_arr, const nb::object& dst_arr,
                  const nb::list& src_offsets_major_dim,
                  const nb::list& dst_offsets_major_dim,
                  const nb::list& copy_sizes_major_dim,
                  bool unsafe_skip_buffer_lock) {
  auto future = xla::ValueOrThrow(transfer_h2d_async(
      src_arr, dst_arr, src_offsets_major_dim, dst_offsets_major_dim,
      copy_sizes_major_dim, unsafe_skip_buffer_lock));
  nb::gil_scoped_release release;
  absl::Status status = future.Await();
  if (!status.ok()) {
    throw std::runtime_error(std::string("Async copy failed: ") +
                             std::string(status.message()));
  }
}

PjRtCopyFuture transfer_d2h_batch_async_naive(
    const nb::list& src_arrs, const nb::list& dst_arrs,
    const nb::list& src_offsets_major_dim,
    const nb::list& dst_offsets_major_dim, const nb::list& copy_sizes_major_dim,
    bool unsafe_skip_buffer_lock) {
  if (nb::len(src_arrs) != nb::len(dst_arrs)) {
    throw std::runtime_error("Lengths of src_arrs and dst_arrs must match");
  }
  size_t n = nb::len(src_arrs);
  if (n == 0) return PjRtCopyFuture(std::vector<BufferHolder>{});

  std::vector<PjRtCopyFuture> futures;
  futures.reserve(n);

  for (size_t i = 0; i < n; ++i) {
    futures.push_back(xla::ValueOrThrow(transfer_d2h_async(
        src_arrs[i], dst_arrs[i], src_offsets_major_dim, dst_offsets_major_dim,
        copy_sizes_major_dim, unsafe_skip_buffer_lock)));
  }
  return JoinPjRtCopyFutures(absl::MakeSpan(futures));
}

PjRtCopyFuture transfer_h2d_batch_async_naive(
    const nb::list& src_arrs, const nb::list& dst_arrs,
    const nb::list& src_offsets_major_dim,
    const nb::list& dst_offsets_major_dim, const nb::list& copy_sizes_major_dim,
    bool unsafe_skip_buffer_lock) {
  if (nb::len(src_arrs) != nb::len(dst_arrs)) {
    throw std::runtime_error("Lengths of src_arrs and dst_arrs must match");
  }
  size_t n = nb::len(src_arrs);
  if (n == 0) return PjRtCopyFuture(std::vector<BufferHolder>{});

  std::vector<PjRtCopyFuture> futures;
  futures.reserve(n);

  for (size_t i = 0; i < n; ++i) {
    futures.push_back(xla::ValueOrThrow(transfer_h2d_async(
        src_arrs[i], dst_arrs[i], src_offsets_major_dim, dst_offsets_major_dim,
        copy_sizes_major_dim, unsafe_skip_buffer_lock)));
  }
  return JoinPjRtCopyFutures(absl::MakeSpan(futures));
}

void await_all(const nb::object& future_obj) {
  if (nb::isinstance<PjRtCopyFuture>(future_obj)) {
    PjRtCopyFuture& future = nb::cast<PjRtCopyFuture&>(future_obj);
    nb::gil_scoped_release release;
    absl::Status status = future.Await();
    if (!status.ok()) {
      throw std::runtime_error(std::string("Async copy failed: ") +
                               std::string(status.message()));
    }
  } else if (nb::isinstance<nb::list>(future_obj)) {
    nb::list futures = nb::cast<nb::list>(future_obj);
    for (size_t i = 0; i < futures.size(); ++i) {
      PjRtCopyFuture& future = nb::cast<PjRtCopyFuture&>(futures[i]);
      nb::gil_scoped_release release;
      absl::Status status = future.Await();
      if (!status.ok()) {
        throw std::runtime_error(std::string("Async copy failed: ") +
                                 std::string(status.message()));
      }
    }
  }
}

bool is_ready(const nb::object& future_obj) {
  if (nb::isinstance<PjRtCopyFuture>(future_obj)) {
    return nb::cast<const PjRtCopyFuture&>(future_obj).IsReady();
  } else if (nb::isinstance<nb::list>(future_obj)) {
    nb::list futures = nb::cast<nb::list>(future_obj);
    for (size_t i = 0; i < futures.size(); ++i) {
      if (!nb::cast<const PjRtCopyFuture&>(futures[i]).IsReady()) {
        return false;
      }
    }
    return true;
  }
  return true;
}

#ifndef WITHOUT_PYTHON
NB_MODULE(_raw_transfer, m) {
  nb::class_<PjRtCopyFuture>(m, "PjRtCopyFuture")
      .def("Await",
           [](PjRtCopyFuture& future) {
             nb::gil_scoped_release release;
             absl::Status status = future.Await();
             if (!status.ok()) {
               throw std::runtime_error(std::string("Async copy failed: ") +
                                        std::string(status.message()));
             }
           })
      .def("IsReady", &PjRtCopyFuture::IsReady);
  m.def("await_all", &await_all, nb::arg("futures"));
  m.def("is_ready", &is_ready, nb::arg("futures"));

  m.def("transfer_d2h_async", xla::ValueOrThrowWrapper(transfer_d2h_async),
        nb::arg("src_arr"), nb::arg("dst_arr"), nb::kw_only(),
        nb::arg("src_offsets_major_dim") = nb::list(),
        nb::arg("dst_offsets_major_dim") = nb::list(),
        nb::arg("copy_sizes_major_dim") = nb::list(),
        nb::arg("unsafe_skip_buffer_lock") = false);
  m.def("transfer_h2d_async", xla::ValueOrThrowWrapper(transfer_h2d_async),
        nb::arg("src_arr"), nb::arg("dst_arr"), nb::kw_only(),
        nb::arg("src_offsets_major_dim") = nb::list(),
        nb::arg("dst_offsets_major_dim") = nb::list(),
        nb::arg("copy_sizes_major_dim") = nb::list(),
        nb::arg("unsafe_skip_buffer_lock") = false);
  m.def("transfer_d2h", &transfer_d2h, nb::arg("src_arr"), nb::arg("dst_arr"),
        nb::kw_only(), nb::arg("src_offsets_major_dim") = nb::list(),
        nb::arg("dst_offsets_major_dim") = nb::list(),
        nb::arg("copy_sizes_major_dim") = nb::list(),
        nb::arg("unsafe_skip_buffer_lock") = false);
  m.def("transfer_h2d", &transfer_h2d, nb::arg("src_arr"), nb::arg("dst_arr"),
        nb::kw_only(), nb::arg("src_offsets_major_dim") = nb::list(),
        nb::arg("dst_offsets_major_dim") = nb::list(),
        nb::arg("copy_sizes_major_dim") = nb::list(),
        nb::arg("unsafe_skip_buffer_lock") = false);

  m.def("transfer_d2h_batch_async_naive", &transfer_d2h_batch_async_naive,
        nb::arg("src_arrs"), nb::arg("dst_arrs"), nb::kw_only(),
        nb::arg("src_offsets_major_dim") = nb::list(),
        nb::arg("dst_offsets_major_dim") = nb::list(),
        nb::arg("copy_sizes_major_dim") = nb::list(),
        nb::arg("unsafe_skip_buffer_lock") = false);
  m.def("transfer_h2d_batch_async_naive", &transfer_h2d_batch_async_naive,
        nb::arg("src_arrs"), nb::arg("dst_arrs"), nb::kw_only(),
        nb::arg("src_offsets_major_dim") = nb::list(),
        nb::arg("dst_offsets_major_dim") = nb::list(),
        nb::arg("copy_sizes_major_dim") = nb::list(),
        nb::arg("unsafe_skip_buffer_lock") = false);

  m.def("transfer_d2h_batch_async",
        xla::ValueOrThrowWrapper(transfer_d2h_batch_async), nb::arg("src_arrs"),
        nb::arg("dst_arrs"), nb::kw_only(),
        nb::arg("src_offsets_major_dim") = nb::list(),
        nb::arg("dst_offsets_major_dim") = nb::list(),
        nb::arg("copy_sizes_major_dim") = nb::list(),
        nb::arg("unsafe_skip_buffer_lock") = false);
  m.def("transfer_h2d_batch_async",
        xla::ValueOrThrowWrapper(transfer_h2d_batch_async), nb::arg("src_arrs"),
        nb::arg("dst_arrs"), nb::kw_only(),
        nb::arg("src_offsets_major_dim") = nb::list(),
        nb::arg("dst_offsets_major_dim") = nb::list(),
        nb::arg("copy_sizes_major_dim") = nb::list(),
        nb::arg("unsafe_skip_buffer_lock") = false);
  m.def("transfer_d2h_batch", &transfer_d2h_batch, nb::arg("src_arrs"),
        nb::arg("dst_arrs"), nb::kw_only(),
        nb::arg("src_offsets_major_dim") = nb::list(),
        nb::arg("dst_offsets_major_dim") = nb::list(),
        nb::arg("copy_sizes_major_dim") = nb::list(),
        nb::arg("unsafe_skip_buffer_lock") = false);
  m.def("transfer_h2d_batch", &transfer_h2d_batch, nb::arg("src_arrs"),
        nb::arg("dst_arrs"), nb::kw_only(),
        nb::arg("src_offsets_major_dim") = nb::list(),
        nb::arg("dst_offsets_major_dim") = nb::list(),
        nb::arg("copy_sizes_major_dim") = nb::list(),
        nb::arg("unsafe_skip_buffer_lock") = false);
}
#endif

}  // namespace raiden
