// Copyright 2026 Google LLC.
// SPDX-License-Identifier: Apache-2.0
#ifndef TPU_SYNC_CORE_RECV_NAN_CHECK_PROGRAM_H_
#define TPU_SYNC_CORE_RECV_NAN_CHECK_PROGRAM_H_

#include <span>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "tpu_sync/core/recv_nan_check.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/shape.h"

struct PJRT_Client;

namespace tpu_raiden::recv_nan_internal {
// shapes contains the deduplicated U32 storage arguments followed by the
// S64 [rows, 3] received-range metadata. Both production and host tests use
// the same graph and cache signature; range contents are runtime arguments.
absl::StatusOr<xla::XlaComputation> BuildNanCheckComputation(
    std::span<const BatchInput> inputs, const ScanBatch& batch,
    const std::vector<xla::Shape>& shapes);
std::string NanCheckProgramKey(PJRT_Client* client,
                               std::span<const BatchInput> inputs,
                               const ScanBatch& batch,
                               const std::vector<xla::Shape>& shapes);
}  // namespace tpu_raiden::recv_nan_internal

#endif  // TPU_SYNC_CORE_RECV_NAN_CHECK_PROGRAM_H_
