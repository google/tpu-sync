// Copyright 2026 Google LLC.
// SPDX-License-Identifier: Apache-2.0
#ifndef TPU_SYNC_CORE_RECV_NAN_CHECK_FP8_H_
#define TPU_SYNC_CORE_RECV_NAN_CHECK_FP8_H_

#include "tpu_sync/core/recv_nan_check.h"

namespace xla {
class XlaOp;
}

namespace tpu_raiden::recv_nan_internal {
// One predicate per gathered U32. byte, begin and end have the same shape as
// bits; byte starts at floor(begin / 4) * 4 and increases in whole words.
// The result excludes individual bytes outside the chunk's [begin, end).
xla::XlaOp Fp8NonFinite(xla::XlaOp bits, xla::XlaOp byte, xla::XlaOp begin,
                        xla::XlaOp end, NanScanKind kind);
}  // namespace tpu_raiden::recv_nan_internal

#endif  // TPU_SYNC_CORE_RECV_NAN_CHECK_FP8_H_
