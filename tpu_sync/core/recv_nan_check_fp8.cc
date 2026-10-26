// Copyright 2026 Google LLC.
// SPDX-License-Identifier: Apache-2.0
#include "tpu_sync/core/recv_nan_check_fp8.h"

#include <cassert>

#include "xla/hlo/builder/lib/constants.h"
#include "xla/hlo/builder/xla_builder.h"

namespace tpu_raiden::recv_nan_internal {
xla::XlaOp Fp8NonFinite(xla::XlaOp bits, xla::XlaOp byte, xla::XlaOp begin,
                        xla::XlaOp end, NanScanKind kind) {
  auto* b = bits.builder();
  const auto u32 = [&](uint32_t value) {
    return xla::ConstantR0<uint32_t>(b, value);
  };
  const auto s64 = [&](int64_t value) {
    return xla::ConstantR0<int64_t>(b, value);
  };
  constexpr uint32_t kLow = 0x7f7f7f7fU, kHigh = 0x80808080U;
  xla::XlaOp marks;
  // Every byte sum is at most 128 (E4M3FN/E5M2) or 254 (FNUZ), so
  // carries cannot cross byte boundaries. Each bad byte sets its high bit.
  if (kind == NanScanKind::kFp8E4M3Fn) {
    marks = xla::And(xla::Add(xla::And(bits, u32(kLow)), u32(0x01010101U)),
                     u32(kHigh));
  } else if (kind == NanScanKind::kFp8E5M2) {
    marks =
        xla::And(xla::Add(xla::And(bits, u32(0x7c7c7c7cU)), u32(0x04040404U)),
                 u32(kHigh));
  } else {
    assert(kind == NanScanKind::kFp8Fnuz && "FP8 scanner requires FP8 format");
    marks = xla::And(xla::And(bits, xla::Not(xla::Add(xla::And(bits, u32(kLow)),
                                                      u32(kLow)))),
                     u32(kHigh));
  }

  // Only the first/last word needs byte-lane clipping. Keep the same U32
  // shape as the gather instead of expanding four arrays of byte values.
  auto high = xla::FullLike(bits, kHigh);
  auto first_shift = xla::Mul(
      xla::ConvertElementType(xla::And(begin, s64(3)), xla::U32), u32(8));
  auto end_lane = xla::ConvertElementType(xla::And(end, s64(3)), xla::U32);
  auto last_shift =
      xla::Mul(xla::And(xla::Sub(u32(4), end_lane), u32(3)), u32(8));
  auto first = xla::Select(xla::Eq(byte, xla::And(begin, s64(-4))),
                           xla::ShiftLeft(high, first_shift), high);
  auto last =
      xla::Select(xla::Eq(byte, xla::And(xla::Sub(end, s64(1)), s64(-4))),
                  xla::ShiftRightLogical(high, last_shift), high);
  auto live_marks = xla::And(marks, xla::And(first, last));
  // This also excludes zero-filled metadata rows and words past a short tail.
  return xla::And(xla::Lt(byte, end), xla::Ne(live_marks, u32(0)));
}
}  // namespace tpu_raiden::recv_nan_internal
