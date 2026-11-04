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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TRANSPORT_LIB_METADATA_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TRANSPORT_LIB_METADATA_H_

#include <cstddef>

#include "absl/status/statusor.h"
#include "peregrine/src/api/transport_types.h"
#include "tpu_sync/transport/lib/adapter/transport_adapter.h"

namespace tpu_raiden::transport::lib {

// Fixed-size metadata blob attached to each Post and request.
inline constexpr size_t kMetadataSize = ::peregrine::kMetadataSize;
using MetadataBytes = ::peregrine::Metadata;

// Returns true if the sender attached no metadata (all-zero blob). Check this
// before decoding.
bool IsEmptyMetadata(const MetadataBytes& bytes);

// Encodes Post metadata at `post.version`.
MetadataBytes EncodePostMetadata(const PostMetadata& post);

// Decodes Post metadata. `op` is not validated; callers decide which ops they
// accept.
absl::StatusOr<PostMetadata> DecodePostMetadata(const MetadataBytes& bytes);

// Encodes request metadata at its Post's version.
MetadataBytes EncodeRequestMetadata(const PostMetadata& post,
                                    const RequestMetadata& req);

// Decodes request metadata at its Post's version.
absl::StatusOr<RequestMetadata> DecodeRequestMetadata(
    const PostMetadata& post, const MetadataBytes& bytes);

}  // namespace tpu_raiden::transport::lib

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TRANSPORT_LIB_METADATA_H_
