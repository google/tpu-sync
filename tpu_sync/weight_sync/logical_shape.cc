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

#include "tpu_sync/weight_sync/logical_shape.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"
#include "tpu_sync/rpc/weight_transfer_worker_service.pb.h"

namespace tpu_raiden::weight_sync {
namespace {

constexpr int kBitsPerByte = 8;

// For a row-major untiled layout in a non-default memory space, ":S(<n>)}" is
// the only layout attribute (preceded by "{" for scalars).
constexpr absl::string_view kMemorySpacePrefix = ":S(";
constexpr absl::string_view kMemorySpaceSuffix = ")}";

// Formats an array shape as "<type>[<dims>]", with dimensions as given.
std::string FormatShape(xla::PrimitiveType type,
                        absl::Span<const int64_t> dims) {
  return absl::StrCat(xla::primitive_util::LowercasePrimitiveTypeName(type),
                      "[", absl::StrJoin(dims, ","), "]");
}

}  // namespace

absl::StatusOr<LogicalShape> LogicalShape::FromProto(
    const tpu_sync::rpc::LogicalShapeProto& proto, absl::string_view field) {
  if (!xla::PrimitiveType_IsValid(proto.element_type())) {
    return absl::InvalidArgumentError(absl::StrCat(
        field, " has invalid element_type=", proto.element_type()));
  }
  const xla::PrimitiveType type =
      static_cast<xla::PrimitiveType>(proto.element_type());
  if (!xla::primitive_util::IsArrayType(type)) {
    return absl::InvalidArgumentError(absl::StrCat(
        field, " has non-array element_type=", xla::PrimitiveType_Name(type)));
  }
  if (xla::primitive_util::IsSubByteNonPredType(type) ||
      xla::primitive_util::IsF6Type(type) ||
      xla::primitive_util::StorageBitWidth(type) < kBitsPerByte ||
      xla::primitive_util::StorageBitWidth(type) % kBitsPerByte != 0) {
    return absl::InvalidArgumentError(
        absl::StrCat(field, " has unsupported sub-byte element_type=",
                     xla::primitive_util::LowercasePrimitiveTypeName(type)));
  }
  const int64_t elem_bytes =
      xla::primitive_util::StorageBitWidth(type) / kBitsPerByte;

  int64_t elem_count = 1;
  bool has_zero_dim = false;
  for (int i = 0; i < proto.dims_size(); ++i) {
    const int64_t dim = proto.dims(i);
    if (dim < 0) {
      return absl::InvalidArgumentError(
          absl::StrCat(field, " has negative dims[", i, "]=", dim, " in ",
                       FormatShape(type, proto.dims())));
    }
    if (dim == 0) {
      // Like xla::ShapeUtil::MakeValidatedShape(), reject a byte count that
      // overflows before the first zero dimension.
      if (!has_zero_dim &&
          elem_count > std::numeric_limits<int64_t>::max() / elem_bytes) {
        return absl::InvalidArgumentError(
            absl::StrCat(field, " byte size overflows before zero dims[", i,
                         "] for ", FormatShape(type, proto.dims())));
      }
      has_zero_dim = true;
      elem_count = 0;
      continue;
    }
    if (!has_zero_dim) {
      if (elem_count > std::numeric_limits<int64_t>::max() / dim) {
        return absl::InvalidArgumentError(
            absl::StrCat(field, " element count overflows int64_t for ",
                         FormatShape(type, proto.dims())));
      }
      elem_count *= dim;
    }
  }
  if (elem_count > std::numeric_limits<int64_t>::max() / elem_bytes ||
      static_cast<uint64_t>(elem_count) >
          std::numeric_limits<size_t>::max() /
              static_cast<uint64_t>(elem_bytes)) {
    return absl::InvalidArgumentError(absl::StrCat(
        field, " byte size overflows for ", FormatShape(type, proto.dims())));
  }
  return LogicalShape(
      type, std::vector<int64_t>(proto.dims().begin(), proto.dims().end()),
      elem_count * elem_bytes);
}

absl::StatusOr<std::vector<LogicalShape>> LogicalShape::ParseAttr(
    absl::string_view logical_shapes_attr) {
  if (logical_shapes_attr.empty()) {
    return absl::InvalidArgumentError("logical_shapes attribute is empty");
  }
  std::vector<LogicalShape> out;
  size_t idx = 0;
  for (const absl::string_view token :
       absl::StrSplit(logical_shapes_attr, ';')) {
    const std::string ctx =
        absl::StrCat("logical_shapes[", idx, "] ('", token, "')");
    if (token.empty()) {
      return absl::InvalidArgumentError(absl::StrCat(ctx, " is empty"));
    }
    const size_t lbracket = token.find('[');
    if (lbracket == absl::string_view::npos || token.back() != ']') {
      return absl::InvalidArgumentError(
          absl::StrCat(ctx, " must have format '<dtype>[<dims>]'"));
    }
    const absl::string_view dtype_str = token.substr(0, lbracket);
    const absl::string_view dims_str =
        token.substr(lbracket + 1, token.size() - lbracket - 2);
    ABSL_ASSIGN_OR_RETURN(
        const xla::PrimitiveType dtype,
        xla::primitive_util::StringToPrimitiveType(dtype_str),
        absl::InvalidArgumentError(
            absl::StrCat(ctx, " has unknown XLA dtype '", dtype_str, "'")));
    tpu_sync::rpc::LogicalShapeProto proto;
    proto.set_element_type(static_cast<int32_t>(dtype));
    if (!dims_str.empty()) {
      for (const absl::string_view dim_str : absl::StrSplit(dims_str, ',')) {
        int64_t dim = 0;
        if (dim_str.empty() || !absl::SimpleAtoi(dim_str, &dim)) {
          return absl::InvalidArgumentError(
              absl::StrCat(ctx, " has invalid dimension '", dim_str, "'"));
        }
        proto.add_dims(dim);
      }
    }
    ABSL_ASSIGN_OR_RETURN(LogicalShape shape, FromProto(proto, ctx));
    out.push_back(std::move(shape));
    ++idx;
  }
  return out;
}

tpu_sync::rpc::LogicalShapeProto LogicalShape::ToProto() const {
  tpu_sync::rpc::LogicalShapeProto proto;
  proto.set_element_type(static_cast<int32_t>(element_type_));
  proto.mutable_dims()->Add(dims_.begin(), dims_.end());
  return proto;
}

bool LogicalShape::IsRowMajorUntiledFingerprint(
    absl::string_view layout_fingerprint) const {
  if (layout_fingerprint.empty()) {
    return false;
  }
  // Never fails: FromProto() rejects every shape that MakeValidatedShape()
  // rejects.
  ABSL_ASSIGN_OR_RETURN(
      const xla::Shape canonical_shape,
      xla::ShapeUtil::MakeValidatedShape(element_type_, dims_), false);
  const std::string canonical =
      xla::ShapeUtil::HumanStringWithLayout(canonical_shape);
  if (layout_fingerprint == canonical) {
    return true;
  }
  std::string memspace_prefix;
  if (xla::ShapeUtil::IsScalar(canonical_shape)) {
    // ShapeUtil::HumanStringWithLayout omits "{}" for default-layout scalars
    // ("f32[]"), but emits "{:S(<n>)}" when a non-zero memory space is set.
    memspace_prefix = absl::StrCat(canonical, "{", kMemorySpacePrefix);
  } else {
    if (canonical.empty() || canonical.back() != '}') {
      return false;
    }
    memspace_prefix = absl::StrCat(canonical.substr(0, canonical.size() - 1),
                                   kMemorySpacePrefix);
  }
  if (!absl::StartsWith(layout_fingerprint, memspace_prefix) ||
      !absl::EndsWith(layout_fingerprint, kMemorySpaceSuffix)) {
    return false;
  }
  const absl::string_view digits = layout_fingerprint.substr(
      memspace_prefix.size(), layout_fingerprint.size() -
                                  memspace_prefix.size() -
                                  kMemorySpaceSuffix.size());
  if (digits.empty()) return false;
  for (const char c : digits) {
    if (!absl::ascii_isdigit(static_cast<unsigned char>(c))) {
      return false;
    }
  }
  return true;
}

std::string LogicalShape::Format() const {
  return FormatShape(element_type_, dims_);
}

}  // namespace tpu_raiden::weight_sync
