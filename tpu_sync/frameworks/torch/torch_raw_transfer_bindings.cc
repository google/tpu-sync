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

#include "tpu_sync/frameworks/torch/torch_raw_transfer_bindings.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "absl/status/status.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/optional.h"
#include "nanobind/stl/shared_ptr.h"
#include "nanobind/stl/string.h"
#include "nanobind/stl/vector.h"
#include "tpu_sync/frameworks/torch/torch_nanobind_utils.h"
#include "tpu_sync/frameworks/torch/torch_raw_transfer.h"

namespace nb = nanobind;

namespace raiden {
namespace {

using TensorList = std::vector<at::Tensor>;

void AwaitAll(nb::object futures) {
  if (nb::isinstance<PjRtCopyFuture>(futures)) {
    PjRtCopyFuture& future = nb::cast<PjRtCopyFuture&>(futures);
    nb::gil_scoped_release release;
    absl::Status status = future.Await();
    if (!status.ok()) {
      throw std::runtime_error(std::string("Async copy failed: ") +
                               std::string(status.message()));
    }
    return;
  }
  for (nb::handle item : futures) {
    PjRtCopyFuture& future = nb::cast<PjRtCopyFuture&>(item);
    nb::gil_scoped_release release;
    absl::Status status = future.Await();
    if (!status.ok()) {
      throw std::runtime_error(std::string("Async copy failed: ") +
                               std::string(status.message()));
    }
  }
}

bool IsReady(nb::object futures) {
  if (nb::isinstance<PjRtCopyFuture>(futures)) {
    return nb::cast<PjRtCopyFuture&>(futures).IsReady();
  }
  for (nb::handle item : futures) {
    if (!nb::cast<PjRtCopyFuture&>(item).IsReady()) {
      return false;
    }
  }
  return true;
}

}  // namespace

void BindTorchRawTransfer(nb::module_& m) {
  nb::class_<RawHostBuffer>(m, "RawHostBuffer")
      .def(nb::init<int64_t>(), nb::arg("size_bytes"))
      .def_prop_ro("size_bytes", &RawHostBuffer::SizeBytes)
      .def_prop_ro("data_ptr", &RawHostBuffer::DataPtr)
      .def_prop_ro("is_pjrt_backed", &RawHostBuffer::IsPjRtBacked);

  auto future_cls =
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
          .def("wait",
               [](PjRtCopyFuture& future) {
                 nb::gil_scoped_release release;
                 absl::Status status = future.Await();
                 if (!status.ok()) {
                   throw std::runtime_error(std::string("Async copy failed: ") +
                                            std::string(status.message()));
                 }
               })
          .def("IsReady", &PjRtCopyFuture::IsReady)
          .def("is_ready", &PjRtCopyFuture::IsReady);
  m.attr("PjRtCopyFuture") = future_cls;

  nb::class_<PreparedTorchRawTransfer>(m, "PreparedTorchRawTransfer")
      .def(nb::new_([](const at::Tensor& tpu_tensor,
                       std::shared_ptr<RawHostBuffer> host_buffer,
                       bool unsafe_skip_buffer_lock) {
             return std::make_shared<PreparedTorchRawTransfer>(
                 tpu_tensor, host_buffer, unsafe_skip_buffer_lock);
           }),
           nb::arg("tpu_tensor"), nb::arg("host_buffer"),
           nb::arg("unsafe_skip_buffer_lock") = true)
      .def_prop_ro("physical_size_bytes",
                   &PreparedTorchRawTransfer::PhysicalSizeBytes)
      .def_prop_ro("host_buffer", &PreparedTorchRawTransfer::HostBuffer)
      .def("d2h_async", &PreparedTorchRawTransfer::D2HAsync,
           nb::call_guard<nb::gil_scoped_release>())
      .def("h2d_async", &PreparedTorchRawTransfer::H2DAsync,
           nb::call_guard<nb::gil_scoped_release>())
      .def("d2h", &PreparedTorchRawTransfer::D2H,
           nb::call_guard<nb::gil_scoped_release>())
      .def("h2d", &PreparedTorchRawTransfer::H2D,
           nb::call_guard<nb::gil_scoped_release>());

  m.def("await_all", &AwaitAll, nb::arg("futures"));
  m.def("is_ready", &IsReady, nb::arg("futures"));

  m.def("transfer_d2h_async", &TransferD2HAsync, nb::arg("src_arr"),
        nb::arg("dst_arr"), nb::kw_only(),
        nb::arg("src_offsets_major_dim") = std::vector<int64_t>{},
        nb::arg("dst_offsets_major_dim") = std::vector<int64_t>{},
        nb::arg("copy_sizes_major_dim") = std::vector<int64_t>{},
        nb::arg("unsafe_skip_buffer_lock") = false,
        nb::call_guard<nb::gil_scoped_release>());
  m.def("transfer_h2d_async", &TransferH2DAsync, nb::arg("src_arr"),
        nb::arg("dst_arr"), nb::kw_only(),
        nb::arg("src_offsets_major_dim") = std::vector<int64_t>{},
        nb::arg("dst_offsets_major_dim") = std::vector<int64_t>{},
        nb::arg("copy_sizes_major_dim") = std::vector<int64_t>{},
        nb::arg("unsafe_skip_buffer_lock") = false,
        nb::call_guard<nb::gil_scoped_release>());
  m.def(
      "transfer_d2h",
      [](const at::Tensor& src_arr, const at::Tensor& dst_arr,
         const std::vector<int64_t>& src_offsets_major_dim,
         const std::vector<int64_t>& dst_offsets_major_dim,
         const std::vector<int64_t>& copy_sizes_major_dim,
         bool unsafe_skip_buffer_lock) {
        auto future = TransferD2HAsync(src_arr, dst_arr, src_offsets_major_dim,
                                       dst_offsets_major_dim,
                                       copy_sizes_major_dim,
                                       unsafe_skip_buffer_lock);
        absl::Status status = future.Await();
        if (!status.ok()) {
          throw std::runtime_error(std::string("Async copy failed: ") +
                                   std::string(status.message()));
        }
      },
      nb::arg("src_arr"), nb::arg("dst_arr"), nb::kw_only(),
      nb::arg("src_offsets_major_dim") = std::vector<int64_t>{},
      nb::arg("dst_offsets_major_dim") = std::vector<int64_t>{},
      nb::arg("copy_sizes_major_dim") = std::vector<int64_t>{},
      nb::arg("unsafe_skip_buffer_lock") = false,
      nb::call_guard<nb::gil_scoped_release>());
  m.def(
      "transfer_h2d",
      [](const at::Tensor& src_arr, const at::Tensor& dst_arr,
         const std::vector<int64_t>& src_offsets_major_dim,
         const std::vector<int64_t>& dst_offsets_major_dim,
         const std::vector<int64_t>& copy_sizes_major_dim,
         bool unsafe_skip_buffer_lock) {
        auto future = TransferH2DAsync(src_arr, dst_arr, src_offsets_major_dim,
                                       dst_offsets_major_dim,
                                       copy_sizes_major_dim,
                                       unsafe_skip_buffer_lock);
        absl::Status status = future.Await();
        if (!status.ok()) {
          throw std::runtime_error(std::string("Async copy failed: ") +
                                   std::string(status.message()));
        }
      },
      nb::arg("src_arr"), nb::arg("dst_arr"), nb::kw_only(),
      nb::arg("src_offsets_major_dim") = std::vector<int64_t>{},
      nb::arg("dst_offsets_major_dim") = std::vector<int64_t>{},
      nb::arg("copy_sizes_major_dim") = std::vector<int64_t>{},
      nb::arg("unsafe_skip_buffer_lock") = false,
      nb::call_guard<nb::gil_scoped_release>());

  m.def("transfer_d2h_batch_async", &TransferD2HBatchAsync, nb::arg("src_arrs"),
        nb::arg("dst_arrs"), nb::kw_only(),
        nb::arg("src_offsets_major_dim") = std::vector<int64_t>{},
        nb::arg("dst_offsets_major_dim") = std::vector<int64_t>{},
        nb::arg("copy_sizes_major_dim") = std::vector<int64_t>{},
        nb::arg("unsafe_skip_buffer_lock") = false,
        nb::call_guard<nb::gil_scoped_release>());
  m.def("transfer_h2d_batch_async", &TransferH2DBatchAsync, nb::arg("src_arrs"),
        nb::arg("dst_arrs"), nb::kw_only(),
        nb::arg("src_offsets_major_dim") = std::vector<int64_t>{},
        nb::arg("dst_offsets_major_dim") = std::vector<int64_t>{},
        nb::arg("copy_sizes_major_dim") = std::vector<int64_t>{},
        nb::arg("unsafe_skip_buffer_lock") = false,
        nb::call_guard<nb::gil_scoped_release>());
  m.def(
      "transfer_d2h_batch",
      [](const TensorList& src_arrs, const TensorList& dst_arrs,
         const std::vector<int64_t>& src_offsets_major_dim,
         const std::vector<int64_t>& dst_offsets_major_dim,
         const std::vector<int64_t>& copy_sizes_major_dim,
         bool unsafe_skip_buffer_lock) {
        auto future = TransferD2HBatchAsync(
            src_arrs, dst_arrs, src_offsets_major_dim, dst_offsets_major_dim,
            copy_sizes_major_dim, unsafe_skip_buffer_lock);
        absl::Status status = future.Await();
        if (!status.ok()) {
          throw std::runtime_error(std::string("Async copy failed: ") +
                                   std::string(status.message()));
        }
      },
      nb::arg("src_arrs"), nb::arg("dst_arrs"), nb::kw_only(),
      nb::arg("src_offsets_major_dim") = std::vector<int64_t>{},
      nb::arg("dst_offsets_major_dim") = std::vector<int64_t>{},
      nb::arg("copy_sizes_major_dim") = std::vector<int64_t>{},
      nb::arg("unsafe_skip_buffer_lock") = false,
      nb::call_guard<nb::gil_scoped_release>());
  m.def(
      "transfer_h2d_batch",
      [](const TensorList& src_arrs, const TensorList& dst_arrs,
         const std::vector<int64_t>& src_offsets_major_dim,
         const std::vector<int64_t>& dst_offsets_major_dim,
         const std::vector<int64_t>& copy_sizes_major_dim,
         bool unsafe_skip_buffer_lock) {
        auto future = TransferH2DBatchAsync(
            src_arrs, dst_arrs, src_offsets_major_dim, dst_offsets_major_dim,
            copy_sizes_major_dim, unsafe_skip_buffer_lock);
        absl::Status status = future.Await();
        if (!status.ok()) {
          throw std::runtime_error(std::string("Async copy failed: ") +
                                   std::string(status.message()));
        }
      },
      nb::arg("src_arrs"), nb::arg("dst_arrs"), nb::kw_only(),
      nb::arg("src_offsets_major_dim") = std::vector<int64_t>{},
      nb::arg("dst_offsets_major_dim") = std::vector<int64_t>{},
      nb::arg("copy_sizes_major_dim") = std::vector<int64_t>{},
      nb::arg("unsafe_skip_buffer_lock") = false,
      nb::call_guard<nb::gil_scoped_release>());
}

}  // namespace raiden
