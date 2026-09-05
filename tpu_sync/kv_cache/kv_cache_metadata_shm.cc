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

#include "tpu_sync/kv_cache/kv_cache_metadata_shm.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tpu_sync/common/raiden_id.h"
#include "tpu_sync/core/host_memory_allocator.h"
#include "tpu_sync/kv_cache/kv_cache_metadata.h"

namespace tpu_raiden {
namespace kv_cache {
namespace {

std::string NormalizeShmKey(absl::string_view shm_key) {
  std::string key(shm_key);
  if (key.empty() || key[0] != '/') {
    key.insert(0, "/");
  }
  return key;
}

size_t PageAlignedTableSize(int num_blocks) {
  return (KVCacheMetadata::RequiredSizeBytes(num_blocks) + 4095) & ~size_t{
             4095};
}

// The charset SharedMemoryHostMemoryAllocator::ValidateShmNameParts enforces
// on user configuration; internal identities are rewritten to it instead --
// anything else (above all '/', fatal to shm_open) becomes '_'.
std::string SanitizeForShmName(absl::string_view part) {
  std::string sanitized(part);
  for (char& c : sanitized) {
    if (!absl::ascii_isalnum(c) && c != '_' && c != '-' && c != '.') {
      c = '_';
    }
  }
  return sanitized;
}

}  // namespace

std::string MetadataShmKey(const RaidenId& raiden_id) {
  const char* shm_key = std::getenv("RAIDEN_SHM_KEY");
  if (shm_key == nullptr || std::strlen(shm_key) == 0) {
    return "";
  }
  // No per-device suffix — the table spans the store, not one device — so
  // the RaidenId is the only thing telling colocated stores' tables apart.
  std::string key = absl::StrCat(shm_key, "_metadata");
  const char* server_name = std::getenv("RAIDEN_SHM_SERVER_NAME");
  if (server_name != nullptr && std::strlen(server_name) > 0) {
    absl::StrAppend(&key, "_", server_name);
  }
  if (!raiden_id.empty()) {
    absl::StrAppend(&key, "_",
                    SanitizeForShmName(absl::StrCat(
                        raiden_id.job_name, "_", raiden_id.job_replica_id, "_",
                        raiden_id.data_name, "_", raiden_id.data_replica_idx)));
  }
  return key;
}

absl::StatusOr<std::unique_ptr<KVCacheMetadataShmRegion>>
KVCacheMetadataShmRegion::AttachOrFormat(absl::string_view shm_key,
                                         int num_blocks,
                                         absl::string_view model_uid) {
  const std::string key = NormalizeShmKey(shm_key);
  const size_t size = PageAlignedTableSize(num_blocks);

  // The attach-or-create decision runs under the segment's creation lock
  // (the same discipline SharedMemoryHostMemoryAllocator applies to the KV
  // pool segments), so a concurrent opener cannot unlink a half-created
  // table as incompatible.
  ABSL_ASSIGN_OR_RETURN(ScopedShmLock lock, ScopedShmLock::Acquire(key));

  // Warm path: attach to a segment left behind by a previous incarnation and
  // validate the table it carries. Any incompatibility falls through to the
  // cold path below, which re-creates the segment.
  int fd = shm_open(key.c_str(), O_RDWR, 0666);
  if (fd >= 0) {
    struct stat st;
    void* mapped = MAP_FAILED;
    if (fstat(fd, &st) == 0 && static_cast<size_t>(st.st_size) >= size) {
      mapped = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    }
    if (mapped != MAP_FAILED) {
      auto span = absl::MakeSpan(static_cast<uint8_t*>(mapped), size);
      auto metadata_or = KVCacheMetadata::Attach(span, num_blocks, model_uid);
      if (metadata_or.ok()) {
        VLOG(1) << "Attached to the surviving KV metadata table " << key;
        return std::unique_ptr<KVCacheMetadataShmRegion>(
            new KVCacheMetadataShmRegion(fd, mapped, size, *metadata_or,
                                         /*warm=*/true, num_blocks,
                                         std::string(model_uid)));
      }
      LOG(WARNING) << "Surviving KV metadata table " << key
                   << " failed validation, re-creating: "
                   << metadata_or.status().message();
      munmap(mapped, size);
    } else {
      LOG(WARNING) << "Surviving KV metadata segment " << key
                   << " is unusable, re-creating";
    }
    close(fd);
    shm_unlink(key.c_str());
  }

  // Cold path: create the segment and format an empty table into it. Under
  // the creation lock an EEXIST can only be an uncooperative writer: fail
  // loudly rather than overwrite its table.
  fd = shm_open(key.c_str(), O_CREAT | O_RDWR | O_EXCL, 0666);
  if (fd < 0) {
    return absl::InternalError(
        absl::StrCat("shm_open failed to create KV metadata segment ", key,
                     ": ", std::strerror(errno)));
  }
  if (ftruncate(fd, size) != 0) {
    absl::Status status = absl::InternalError(
        absl::StrCat("ftruncate failed on KV metadata segment ", key, ": ",
                     std::strerror(errno)));
    close(fd);
    shm_unlink(key.c_str());
    return status;
  }
  void* mapped = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (mapped == MAP_FAILED) {
    absl::Status status = absl::ResourceExhaustedError(
        absl::StrCat("mmap failed on KV metadata segment ", key, ": ",
                     std::strerror(errno)));
    close(fd);
    shm_unlink(key.c_str());
    return status;
  }
  auto span = absl::MakeSpan(static_cast<uint8_t*>(mapped), size);
  auto metadata_or = KVCacheMetadata::Format(span, num_blocks, model_uid);
  if (!metadata_or.ok()) {
    munmap(mapped, size);
    close(fd);
    shm_unlink(key.c_str());
    return metadata_or.status();
  }
  VLOG(1) << "Formatted a fresh KV metadata table " << key;
  return std::unique_ptr<KVCacheMetadataShmRegion>(new KVCacheMetadataShmRegion(
      fd, mapped, size, *metadata_or, /*warm=*/false, num_blocks,
      std::string(model_uid)));
}

KVCacheMetadataShmRegion::KVCacheMetadataShmRegion(
    int fd, void* mapped, size_t mapped_size, KVCacheMetadata metadata,
    bool warm, int num_blocks, std::string model_uid)
    : fd_(fd),
      mapped_(mapped),
      mapped_size_(mapped_size),
      metadata_(std::move(metadata)),
      warm_(warm),
      num_blocks_(num_blocks),
      model_uid_(std::move(model_uid)) {}

KVCacheMetadataShmRegion::~KVCacheMetadataShmRegion() {
  munmap(mapped_, mapped_size_);
  close(fd_);
}

absl::Status KVCacheMetadataShmRegion::Reformat() {
  auto span = absl::MakeSpan(static_cast<uint8_t*>(mapped_), mapped_size_);
  auto metadata_or = KVCacheMetadata::Format(span, num_blocks_, model_uid_);
  if (!metadata_or.ok()) {
    return metadata_or.status();
  }
  metadata_ = *metadata_or;
  warm_ = false;
  return absl::OkStatus();
}

}  // namespace kv_cache
}  // namespace tpu_raiden
