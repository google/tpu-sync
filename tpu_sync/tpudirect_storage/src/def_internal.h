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

#ifndef TDSUL_DEF_INTERNAL_H_
#define TDSUL_DEF_INTERNAL_H_

#include <cstddef>
#include <string>
#include <vector>

#include "tdsul/def.h"

#ifndef IOV_MAX
#define IOV_MAX 1024
#endif

struct tds_config {
  int num_worker_threads = 4;
  bool enable_io_uring = false;
  bool enable_request_chunking = false;
  size_t chunk_size_bytes = 0;
  bool enable_p2p = false;
  int worker_core_start = -1;
  int worker_core_end = -1;
};

struct tds_storage_opts {
  size_t stripe_size = 0;
};

struct tds_storage_handle {
  std::string uri;
  int fd = -1;
  bool owns_fd = false;
  size_t stripe_size = 0;
};

struct tds_buffer_handle {
  int memory_type = TDS_MEM_HOST;
  void* vaddr = nullptr;
  int dma_buf_fd = -1;
  size_t offset = 0;
  size_t size = 0;
  bool is_mapped = false;
};

struct tds_io_future {
  tds_io_status_t status;
};

struct tds_batch {
  std::vector<tds_storage_io_t> storage_ios;
  std::vector<tds_buffer_io_t> buffer_ios;
  std::vector<OpType> ops;
};

namespace tdsul {
enum class RequestType { SINGLE, BATCHED, CHUNKED };
enum class RequestState {
  // Blocked in the IO queue waiting for its turn due to FIFO ordering or batch
  // barriers.
  PENDING,
  // Scheduled to the worker threads (actively processing or waiting in thread
  // pool).
  SCHEDULED,
  // Processing is complete and the request is awaiting cleanup/polling by the
  // user.
  FINISHED
};

struct WorkRequest {
  WorkRequest(RequestType type, tds_io_queue_t* queue, tds_future_t* future)
      : type_(type),
        queue_(queue),
        future_(future),
        state_(RequestState::PENDING) {}
  virtual ~WorkRequest() = default;

  RequestType type_;
  tds_io_queue_t* queue_;
  tds_future_t* future_;
  RequestState state_;
  virtual size_t GetTransferSize() const = 0;
};

struct SingleWorkRequest : public WorkRequest {
  SingleWorkRequest(tds_storage_io_t storage, tds_buffer_io_t buffer, OpType op,
                    tds_io_queue_t* queue, tds_future_t* future)
      : WorkRequest(RequestType::SINGLE, queue, future),
        storage_io_desc(storage),
        buffer_io_desc(buffer),
        op_type(op) {}

  size_t GetTransferSize() const override {
    // TODO: add support for readv and writev.
    if (buffer_io_desc.handle) {
      return buffer_io_desc.registered.size;
    }
    return buffer_io_desc.raw_ptr.size;
  }

  tds_storage_io_t storage_io_desc;
  tds_buffer_io_t buffer_io_desc;
  OpType op_type;
  // TODO: we need a backpointer to the father work request in case this is
  // a chunked work request. Or we use another struct for the chunked
  // subrequest.
};
// TODO: define ChunkedWorkRequest, define BatchWorkRequest.
}  // namespace tdsul

#endif  // TDSUL_DEF_INTERNAL_H_
