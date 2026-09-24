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

#ifndef TDSUL_H_
#define TDSUL_H_

#include <stddef.h>
#include <sys/types.h>

#include "tdsul/def.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================== */
/* 4.1 Initialization & Lifecycle Management                                 */
/* ========================================================================== */

tds_config_t* tds_config_create(void);
void tds_config_destroy(tds_config_t* config);
tds_result_t tds_config_set_int(tds_config_t* config, const char* key,
                                int value);
tds_result_t tds_config_set_string(tds_config_t* config, const char* key,
                                   const char* value);

tds_storage_opts_t* tds_storage_opts_create(void);
void tds_storage_opts_destroy(tds_storage_opts_t* opts);
tds_result_t tds_storage_opts_set_int(tds_storage_opts_t* opts, const char* key,
                                      int value);
tds_result_t tds_storage_opts_set_string(tds_storage_opts_t* opts,
                                         const char* key, const char* value);

tds_result_t tds_init(const tds_config_t* config);
tds_result_t tds_shutdown(void);

/* ========================================================================== */
/* 4.2 Registration Management                                               */
/* ========================================================================== */

tds_result_t tds_buffer_register_vaddr(void* vaddr, int memory_type,
                                       size_t size,
                                       tds_buffer_handle_t** buf_handle);
tds_result_t tds_buffer_register_dmabuf(int dma_buf_fd, size_t offset,
                                        int memory_type, size_t size,
                                        tds_buffer_handle_t** buf_handle);
tds_result_t tds_buffer_deregister(tds_buffer_handle_t* buf_handle);

tds_result_t tds_storage_handle_register(const tds_storage_descr_t* descr,
                                         tds_storage_handle_t** storage_handle);
tds_result_t tds_storage_handle_deregister(
    tds_storage_handle_t* storage_handle);

/* ========================================================================== */
/* 4.3 Synchronous I/O                                                       */
/* ========================================================================== */

ssize_t tds_read(const tds_storage_io_t* storage_io,
                 const tds_buffer_io_t* buffer_io);
ssize_t tds_write(const tds_storage_io_t* storage_io,
                  const tds_buffer_io_t* buffer_io);
/**
 * @brief Synchronous vectored read/write operations.
 *
 * @note Unlike scalar I/O (tds_read/tds_write) which defaults a 0-length buffer
 * to the size of the file mapping, vectored I/O treats a 0-length segment as a
 * valid empty buffer and does NOT default it to the file size. This matches
 * standard POSIX behavior to prevent unexpected buffer overflows when
 * iterating.
 */
ssize_t tds_readv(const tds_storage_io_t* storage_io,
                  const tds_buffer_io_t* buffer_io_arr, int num_buffers);
ssize_t tds_writev(const tds_storage_io_t* storage_io,
                   const tds_buffer_io_t* buffer_io_arr, int num_buffers);

/* ========================================================================== */
/* 4.4 Asynchronous I/O & Queue Management                                   */
/* ========================================================================== */

tds_result_t tds_queue_create(tds_io_queue_t** queue);
tds_result_t tds_queue_destroy(tds_io_queue_t* queue);
tds_result_t tds_queue_synchronize(tds_io_queue_t* queue);

tds_result_t tds_read_async(const tds_storage_io_t* storage_io,
                            const tds_buffer_io_t* buffer_io,
                            tds_io_queue_t* queue, tds_future_t** future);
tds_result_t tds_write_async(const tds_storage_io_t* storage_io,
                             const tds_buffer_io_t* buffer_io,
                             tds_io_queue_t* queue, tds_future_t** future);
tds_result_t tds_readv_async(const tds_storage_io_t* storage_io,
                             const tds_buffer_io_t* buffer_io_arr,
                             int num_buffers, tds_io_queue_t* queue,
                             tds_future_t** future);
tds_result_t tds_writev_async(const tds_storage_io_t* storage_io,
                              const tds_buffer_io_t* buffer_io_arr,
                              int num_buffers, tds_io_queue_t* queue,
                              tds_future_t** future);

void tds_future_wait(tds_future_t* future);
tds_io_status_t tds_future_get_status(tds_future_t* future, int index);
void tds_future_destroy(tds_future_t* future);

static inline tds_io_status_t tds_batch_future_get_status(tds_future_t* future,
                                                          int index) {
  return tds_future_get_status(future, index);
}

/* ========================================================================== */
/* 4.5 Batched I/O operations                                                */
/* ========================================================================== */

tds_batch_t* tds_batch_create(void);
tds_result_t tds_batch_add(tds_batch_t* batch,
                           const tds_storage_io_t* storage_io,
                           const tds_buffer_io_t* buffer_io, OpType opt,
                           int* index);
tds_result_t tds_batch_addv(tds_batch_t* batch,
                            const tds_storage_io_t* storage_io,
                            const tds_buffer_io_t* buffer_io_arr,
                            int num_buffers, OpType opt, int* index);
int tds_batch_get_count(const tds_batch_t* batch);
void tds_batch_destroy(tds_batch_t* batch);

tds_result_t tds_batch_execute(tds_batch_t* batch, ssize_t* ret_arr);
tds_result_t tds_batch_submit(tds_batch_t* batch, tds_io_queue_t* queue,
                              tds_future_t** future);

#ifdef __cplusplus
}
#endif

#endif  // TDSUL_H_
