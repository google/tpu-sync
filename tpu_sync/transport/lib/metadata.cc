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

#include "tpu_sync/transport/lib/metadata.h"

#include <algorithm>
#include <cstdint>
#include <cstring>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "tpu_sync/transport/lib/adapter/transport_adapter.h"
#include "tpu_sync/transport/lib/metadata_generated.h"

namespace tpu_raiden::transport::lib {

namespace {

static_assert(sizeof(flatbuf::PostMetadataV1) == kMetadataSize);
static_assert(sizeof(flatbuf::RequestMetadataV1) == kMetadataSize);

// LINT.IfChange(post_metadata_v1)
MetadataBytes SerializePostMetadataV1(const PostMetadata& post) {
  const flatbuf::PostMetadataV1 m(
      kMetadataVersionV1, /*reserved0=*/0, /*reserved1=*/0, /*reserved2=*/0,
      post.uuid, post.remote_id, post.local_id, post.shard_mask,
      static_cast<uint8_t>(post.op), /*reserved3=*/0, /*reserved4=*/0);
  MetadataBytes bytes;
  std::memcpy(bytes.data(), &m, sizeof(m));
  return bytes;
}

absl::Status ValidatePostMetadataV1(const flatbuf::PostMetadataV1& m) {
  if (m.reserved0() != 0 || m.reserved1() != 0 || m.reserved2() != 0 ||
      m.reserved3() != 0 || m.reserved4() != 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Non-zero reserved field in post metadata: reserved0=", m.reserved0(),
        ", reserved1=", m.reserved1(), ", reserved2=", m.reserved2(),
        ", reserved3=", m.reserved3(), ", reserved4=", m.reserved4()));
  }
  return absl::OkStatus();
}

void DeserializePostMetadataV1(const flatbuf::PostMetadataV1& m,
                               PostMetadata& post) {
  post.uuid = m.uuid();
  post.remote_id = m.remote_id();
  post.local_id = m.local_id();
  post.shard_mask = m.shard_mask();
  post.op = static_cast<TransferOp>(m.op());
  post.version = m.version();
}
// LINT.ThenChange(metadata.fbs:post_metadata)

// LINT.IfChange(request_metadata_v1)
MetadataBytes SerializeRequestMetadataV1(const RequestMetadata& req) {
  const flatbuf::RequestMetadataV1 m(req.shard, req.block_id, req.src_block_id,
                                     req.chunk_idx, /*reserved0=*/0, req.len,
                                     /*reserved1=*/0, /*reserved2=*/0);
  MetadataBytes bytes;
  std::memcpy(bytes.data(), &m, sizeof(m));
  return bytes;
}

absl::Status ValidateRequestMetadataV1(const flatbuf::RequestMetadataV1& m) {
  if (m.reserved0() != 0 || m.reserved1() != 0 || m.reserved2() != 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Non-zero reserved field in request metadata: reserved0=",
                     m.reserved0(), ", reserved1=", m.reserved1(),
                     ", reserved2=", m.reserved2()));
  }
  // A real request has len >= 1, so it is never mistaken for an empty blob.
  if (m.len() == 0) {
    return absl::InvalidArgumentError(
        "Zero len in request metadata; an empty blob means no "
        "metadata and must not be decoded");
  }
  return absl::OkStatus();
}

void DeserializeRequestMetadataV1(const flatbuf::RequestMetadataV1& m,
                                  RequestMetadata& req) {
  req.shard = m.shard();
  req.block_id = m.block_id();
  req.src_block_id = m.src_block_id();
  req.chunk_idx = m.chunk_idx();
  req.len = m.len();
}
// LINT.ThenChange(metadata.fbs:request_metadata)

}  // namespace

bool IsEmptyMetadata(const MetadataBytes& bytes) {
  return std::all_of(bytes.begin(), bytes.end(),
                     [](uint8_t b) { return b == 0; });
}

MetadataBytes EncodePostMetadata(const PostMetadata& post) {
  const uint8_t ver = post.version;
  switch (ver) {
    case kMetadataVersionV1:
      return SerializePostMetadataV1(post);
    default:
      LOG(FATAL) << "Unsupported post metadata version: "
                 << static_cast<int>(ver);
  }
}

MetadataBytes EncodeRequestMetadata(const PostMetadata& post,
                                    const RequestMetadata& req) {
  CHECK_NE(req.len, 0u) << "Request metadata must carry a non-zero len";
  const uint8_t ver = post.version;
  switch (ver) {
    case kMetadataVersionV1:
      return SerializeRequestMetadataV1(req);
    default:
      LOG(FATAL) << "Unsupported request metadata version: "
                 << static_cast<int>(ver);
  }
}

absl::StatusOr<PostMetadata> DecodePostMetadata(const MetadataBytes& bytes) {
  const uint8_t ver = bytes[0];  // `version` is byte 0 in every layout.
  switch (ver) {
    case kMetadataVersionV1: {
      flatbuf::PostMetadataV1 m;
      std::memcpy(&m, bytes.data(), sizeof(m));
      ABSL_RETURN_IF_ERROR(ValidatePostMetadataV1(m));
      PostMetadata post = {};
      DeserializePostMetadataV1(m, post);
      return post;
    }
    default:
      return absl::FailedPreconditionError(absl::StrCat(
          "Unsupported post metadata version: ", static_cast<int>(ver)));
  }
}

absl::StatusOr<RequestMetadata> DecodeRequestMetadata(
    const PostMetadata& post, const MetadataBytes& bytes) {
  const uint8_t ver = post.version;
  switch (ver) {
    case kMetadataVersionV1: {
      flatbuf::RequestMetadataV1 m;
      std::memcpy(&m, bytes.data(), sizeof(m));
      ABSL_RETURN_IF_ERROR(ValidateRequestMetadataV1(m));
      RequestMetadata req = {};
      DeserializeRequestMetadataV1(m, req);
      return req;
    }
    default:
      return absl::FailedPreconditionError(absl::StrCat(
          "Unsupported request metadata version: ", static_cast<int>(ver)));
  }
}

}  // namespace tpu_raiden::transport::lib
