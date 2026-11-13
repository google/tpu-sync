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

#include "tpu_sync/solutions/mooncake/torch/mooncake_provider_bindings.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ATen/core/TensorBody.h"
#include "nanobind/nanobind.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "csrc/api/tensor_buffer.h"
#include "tpu_sync/frameworks/torch/torch_nanobind_utils.h"
#include "tpu_sync/frameworks/torch/torch_tpu_utils.h"
#include "tpu_sync/solutions/mooncake/mooncake_provider.h"

namespace nb = nanobind;

namespace raiden::mooncake {
namespace {

using ::tpu_raiden::torch::FoldedViews;
using ::tpu_raiden::torch::UnpackedTensor;
using ::tpu_raiden::torch::UnpackTorchTensor;

// Exports the buffer that `unpacked` resolved as one provider handle that
// reports `device_index`.
absl::StatusOr<BufferHandle> Export(UnpackedTensor unpacked, int device_index) {
  // `unpacked.buffer` is valid only while this reference lives, so the handle
  // keeps it.
  auto buffer_ref = std::make_shared<torch_tpu::TensorBufferHandle>(
      std::move(unpacked.ref).value());
  // Copies through the handle do not wait for TorchTPU's pending executions.
  ABSL_RETURN_IF_ERROR(buffer_ref->Synchronize());
  std::vector<ExportRequest> requests;
  requests.push_back({.buffer = std::move(unpacked.buffer),
                      .device_index = device_index,
                      .keep_alive = std::move(buffer_ref)});
  ABSL_ASSIGN_OR_RETURN(std::vector<BufferHandle> handles,
                        ExportBuffers(std::move(requests)));
  return handles.front();
}

// Exports the device buffer that backs `tensor` now as one provider handle,
// which the caller owns. Throws std::invalid_argument (Python ValueError) for
// a tensor that cannot be exported, and std::runtime_error (RuntimeError) if
// TorchTPU fails to produce its buffer.
BufferHandle ExportTensor(const at::Tensor& tensor) {
  // Mooncake copies byte ranges of the whole buffer, so a tensor that folds
  // the leading dimensions of its buffer is fine as long as the buffer keeps
  // its bytes where a buffer of the tensor's own shape would. No usage hold: a
  // handle may live as long as the process (mooncake_provider.h).
  absl::StatusOr<BufferHandle> handle =
      Export(UnpackTorchTensor(tensor, /*unsafe_skip_buffer_lock=*/true,
                               FoldedViews::kAcceptIfLayoutMatches),
             tensor.device().index());
  if (handle.ok()) return *handle;
  const std::string message = absl::StrCat(
      "cannot export the [", absl::StrJoin(tensor.sizes(), ", "),
      "] tensor on ", tensor.device().str(), ": ", handle.status().message());
  if (absl::IsInvalidArgument(handle.status())) {
    throw std::invalid_argument(message);
  }
  throw std::runtime_error(message);
}

}  // namespace

void BindMooncakeProvider(nb::module_& m) {
  // Waits for TorchTPU's pending work, so it releases the GIL, as the
  // raw-transfer bindings do.
  m.def("export_tensor", &ExportTensor, nb::arg("tensor"),
        nb::call_guard<nb::gil_scoped_release>(),
        "Exports the device buffer that backs the TPU tensor now as one "
        "Mooncake provider handle, which the caller owns.");
  m.def("live_buffer_count", &LiveBufferCount,
        "Number of exported handles not yet released.");
  m.def("live_dma_mapping_count", &LiveDmaMappingCount,
        "Number of host ranges mapped by the table's dma_map and not yet "
        "unmapped.");
  m.def("mark_process_exiting", &MarkProcessExiting,
        "Declares that interpreter exit has begun: from now on dma_map fails "
        "and dma_unmap skips PJRT, whose client torch_tpu's atexit handler "
        "may destroy. mooncake_provider registers it with atexit.");
}

}  // namespace raiden::mooncake
