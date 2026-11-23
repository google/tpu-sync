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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_LOGICAL_SHAPE_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_LOGICAL_SHAPE_H_

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/xla_data.pb.h"
#include "tpu_sync/rpc/weight_transfer_worker_service.pb.h"

namespace tpu_raiden::weight_sync {

// The logical shape of an array whose row-major bytes are transferred: an XLA
// array element type with whole-byte storage and non-negative dimensions, whose
// logical byte count fits in int64_t and size_t. FromProto() and ParseAttr()
// validate, so every LogicalShape is valid, and
// xla::ShapeUtil::MakeValidatedShape() accepts it.
class LogicalShape final {
 public:
  // Returns the shape that `proto` describes, or InvalidArgumentError naming
  // `field` and the invalid field: an `element_type` that is not an XLA array
  // type with whole-byte storage, a negative dimension, or a byte count that
  // overflows int64_t or size_t, including before the first zero dimension
  // (which XLA rejects).
  static absl::StatusOr<LogicalShape> FromProto(
      const tpu_sync::rpc::LogicalShapeProto& proto, absl::string_view field);

  // Parses a semicolon-separated `logical_shapes` FFI attribute (e.g.
  // "bf16[1024,4096];f32[4096]"), validating each entry as FromProto() does.
  // Returns the shapes in attribute order, or InvalidArgumentError naming the
  // offending entry.
  static absl::StatusOr<std::vector<LogicalShape>> ParseAttr(
      absl::string_view logical_shapes_attr);

  // Returns the proto form, which FromProto() accepts.
  tpu_sync::rpc::LogicalShapeProto ToProto() const;

  xla::PrimitiveType element_type() const { return element_type_; }
  absl::Span<const int64_t> dims() const ABSL_ATTRIBUTE_LIFETIME_BOUND {
    return dims_;
  }
  // Logical byte count: element count times element byte width (0 if any
  // dimension is 0).
  int64_t byte_size() const { return byte_size_; }

  // Returns true iff `layout_fingerprint` (as produced by
  // `ShapeUtil::HumanStringWithLayout`) is this shape's row-major untiled
  // layout, optionally in a memory space (`:S(<n>)`): the only layouts whose
  // raw device bytes are the logical bytes.
  bool IsRowMajorUntiledFingerprint(absl::string_view layout_fingerprint) const;

  friend bool operator==(const LogicalShape&, const LogicalShape&) = default;

  // Formats as "<type>[<dims>]", e.g. "f32[8,128]" or "bf16[]".
  template <typename Sink>
  friend void AbslStringify(Sink& sink, const LogicalShape& shape) {
    sink.Append(shape.Format());
  }

 private:
  LogicalShape(xla::PrimitiveType element_type, std::vector<int64_t> dims,
               int64_t byte_size)
      : element_type_(element_type),
        dims_(std::move(dims)),
        byte_size_(byte_size) {}

  std::string Format() const;

  xla::PrimitiveType element_type_;
  std::vector<int64_t> dims_;
  int64_t byte_size_;
};

}  // namespace tpu_raiden::weight_sync

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_WEIGHT_SYNC_LOGICAL_SHAPE_H_
