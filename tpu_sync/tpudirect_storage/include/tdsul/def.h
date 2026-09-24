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

#ifndef TDSUL_DEF_H_
#define TDSUL_DEF_H_

#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================== */
/*                                ENUMERATIONS                                */
/* ========================================================================== */

typedef enum tds_result {
  TDS_SUCCESS = 0,
  TDS_PENDING = 1,
  TDS_PARTIAL_FINISH = 2,
  TDS_ERROR_INITIALIZATION = 3,
  TDS_ERROR_NO_MEMORY = 4,
  TDS_ERROR_INVALID_PARAMETER = 5,
  TDS_ERROR_UNALIGNED_BUFFER = 6,
  TDS_ERROR_P2P_UNSUPPORTED = 7,
  TDS_ERROR_IN_USE = 8,
  TDS_ERROR_IO = 9,
  TDS_ERROR_UNSUPPORTED = 10
} tds_result_t;

typedef enum { TDS_MEM_HOST = 0, TDS_MEM_DEVICE = 1 } MemoryType;

typedef enum { TDS_OP_READ = 0, TDS_OP_WRITE = 1 } OpType;

/* ========================================================================== */
/*                               OPAQUE HANDLES                               */
/* ========================================================================== */

typedef struct tds_config tds_config_t;
typedef struct tds_storage_opts tds_storage_opts_t;
typedef struct tds_storage_handle tds_storage_handle_t;
typedef struct tds_buffer_handle tds_buffer_handle_t;
typedef struct tds_io_queue tds_io_queue_t;
typedef struct tds_io_future tds_future_t;
typedef struct tds_batch tds_batch_t;

/* ========================================================================== */
/*                           TRANSPARENT DESCRIPTORS                          */
/* ========================================================================== */

typedef struct tds_storage_descr {
  const char* uri;
  tds_storage_opts_t* options;
} tds_storage_descr_t;

struct tds_file_io_args {
  off_t
      file_offset;  // Used by POSIX file and borrowed file descriptor backends
  size_t size;      // optional, using the buffer-side size as the ground truth.
};

typedef struct tds_storage_io {
  tds_storage_handle_t* handle;
  union {
    struct tds_file_io_args file;
    uint64_t reserved[7];  // 56 bytes reserved inside union (aligns struct to
                           // 64-byte cache line)
  };
} tds_storage_io_t;

struct tds_raw_ptr_io_args {
  void* vptr;
  size_t size;
};

struct tds_registered_io_args {
  size_t offset;
  size_t size;
};

typedef struct tds_buffer_io {
  tds_buffer_handle_t*
      handle;  // Registered handle, or NULL for on-the-fly resolution via vaddr
  union {
    struct tds_raw_ptr_io_args
        raw_ptr;  // Virtual address pointer (used when handle == NULL)
    struct tds_registered_io_args
        registered;        // Slice offset in bytes (used when handle != NULL)
    uint64_t reserved[7];  // 56 bytes reserved inside union (aligns struct to
                           // 64-byte cache line)
  };
} tds_buffer_io_t;

typedef struct tds_io_status {
  tds_result_t status;    // e.g., TDS_SUCCESS, TDS_ERROR_IO, TDS_PARTIAL_FINISH
  int error_num;          // POSIX errno if an error occurred
  size_t finished_bytes;  // Bytes successfully transferred
} tds_io_status_t;

/* TODO: Add GcsIoArgs and GCS storage backend support in the future */

/* ========================================================================== */
/*                       CREATION HELPER INLINES                              */
/* ========================================================================== */

static inline tds_storage_io_t tds_create_file_io(tds_storage_handle_t* handle,
                                                  off_t file_offset,
                                                  size_t size) {
  tds_storage_io_t io;
  memset(&io, 0, sizeof(io));
  io.handle = handle;
  io.file.file_offset = file_offset;
  io.file.size = size;
  return io;
}

static inline tds_buffer_io_t tds_create_registered_buffer_io(
    tds_buffer_handle_t* handle, size_t offset, size_t size) {
  tds_buffer_io_t io;
  memset(&io, 0, sizeof(io));
  io.handle = handle;
  io.registered.offset = offset;
  io.registered.size = size;
  return io;
}

static inline tds_buffer_io_t tds_create_raw_buffer_io(void* vptr,
                                                       size_t size) {
  tds_buffer_io_t io;
  memset(&io, 0, sizeof(io));
  io.handle = NULL;
  io.raw_ptr.vptr = vptr;
  io.raw_ptr.size = size;
  return io;
}

#ifdef __cplusplus
}
#endif

#endif  // TDSUL_DEF_H_
