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

#include "tdsul/tdsul.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

#include <cerrno>
#include <cstddef>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/strings/match.h"
#include "def_internal.h"
#include "syscall_internal.h"
#include "tdsul/def.h"
#include "util/IoConductor.h"
#include "util/IoQueue.h"
#include "util/ThreadPoolConductor.h"

namespace tdsul {
static Syscall default_syscall;
Syscall* g_syscall = &default_syscall;
static bool g_p2p_enabled = false;
std::unique_ptr<tdsul::IoConductor> g_io_conductor;
std::mutex g_init_mutex;
bool tds_initialized = false;
bool g_use_io_uring = false;

#ifdef TDSUL_ENABLE_IO_URING
static bool check_io_uring_support() {
#ifdef __NR_io_uring_setup
  long ret = syscall(__NR_io_uring_setup, 1, nullptr);
  if (ret < 0) {
    if (errno == ENOSYS || errno == EPERM || errno == EACCES) {
      return false;
    }
    // EFAULT means the syscall exists and tried to read our nullptr
    return true;
  }
  close(ret);
  return true;
#else
  return false;
#endif
}
#endif

static bool ValidateIoArgs(const char* op_name, int fd, const void* buf,
                           off_t file_offset, off_t buf_offset,
                           std::size_t io_size, std::size_t buf_size) {
  if (fd < 0) {
    LOG(ERROR) << op_name << " failed: invalid file descriptor: " << fd;
    errno = EINVAL;
    return false;
  }
  if (!buf) {
    LOG(ERROR) << op_name << " failed: buffer pointer is null";
    errno = EINVAL;
    return false;
  }
  if (file_offset < 0) {
    LOG(ERROR) << op_name << " failed: negative file offset: " << file_offset;
    errno = EINVAL;
    return false;
  }
  if (buf_offset < 0) {
    LOG(ERROR) << op_name << " failed: negative buffer offset: " << buf_offset;
    errno = EINVAL;
    return false;
  }
  if (static_cast<std::size_t>(buf_offset) + io_size > buf_size) {
    LOG(ERROR) << op_name << " failed: I/O range [" << buf_offset << ", "
               << buf_offset + io_size << ") exceeds registered buffer size ("
               << buf_size << ")";
    errno = EINVAL;
    return false;
  }
  return true;
}

static bool GetBufferInfo(const tds_buffer_io_t* bio, void** vaddr,
                          size_t* offset, size_t* io_size,
                          size_t* full_buffer_size) {
  if (bio->handle) {
    if (!bio->handle->vaddr) {
      LOG(ERROR) << "Invalid buffer handle: vaddr is null";
      return false;
    }
    if (bio->registered.offset > bio->handle->size ||
        bio->registered.size > bio->handle->size - bio->registered.offset) {
      LOG(ERROR) << "Buffer I/O out of bounds: offset="
                 << bio->registered.offset << ", size=" << bio->registered.size
                 << ", buffer_size=" << bio->handle->size;
      return false;
    }
    *vaddr = bio->handle->vaddr;
    *offset = bio->registered.offset;
    *io_size = bio->registered.size;
    if (full_buffer_size) {
      *full_buffer_size = bio->handle->size;
    }
  } else {
    if (!bio->raw_ptr.vptr) {
      LOG(ERROR) << "Invalid raw buffer: vptr is null";
      return false;
    }
    *vaddr = bio->raw_ptr.vptr;
    *offset = 0;
    *io_size = bio->raw_ptr.size;
    if (full_buffer_size) {
      *full_buffer_size = bio->raw_ptr.size;
    }
  }
  return true;
}

static ssize_t ReadFileToRawBuffer(const tds_storage_io_t* storage_io,
                                   const tds_buffer_io_t* buffer_io) {
  int fd = storage_io->handle->fd;
  off_t file_offset = storage_io->file.file_offset;

  void* raw_buf = nullptr;
  size_t buf_offset = 0;
  size_t io_size = 0;
  size_t full_buffer_size = 0;
  if (!GetBufferInfo(buffer_io, &raw_buf, &buf_offset, &io_size,
                     &full_buffer_size)) {
    errno = EINVAL;
    return -1;
  }

  if (io_size == 0) {
    return 0;  // No-op, return success with 0 bytes read
  }

  if (!ValidateIoArgs("Read", fd, raw_buf, file_offset, buf_offset, io_size,
                      full_buffer_size)) {
    return -1;
  }

  char* buf_ptr = static_cast<char*>(raw_buf) + buf_offset;

  std::size_t total_to_read = io_size;

  ssize_t bytes_read;
  do {
    bytes_read = g_syscall->pread(fd, buf_ptr, total_to_read, file_offset);
  } while (bytes_read < 0 && errno == EINTR);

  if (bytes_read < 0) {
    const int saved_errno = errno;
    LOG(ERROR) << "pread failed: " << std::strerror(saved_errno);
    errno = saved_errno;
    return -1;
  }

  return bytes_read;
}

static ssize_t WriteFileFromRawBuffer(const tds_storage_io_t* storage_io,
                                      const tds_buffer_io_t* buffer_io) {
  int fd = storage_io->handle->fd;
  off_t file_offset = storage_io->file.file_offset;

  void* raw_buf = nullptr;
  size_t buf_offset = 0;
  size_t io_size = 0;
  size_t full_buffer_size = 0;
  if (!GetBufferInfo(buffer_io, &raw_buf, &buf_offset, &io_size,
                     &full_buffer_size)) {
    errno = EINVAL;
    return -1;
  }

  if (io_size == 0) {
    return 0;  // No-op, return success with 0 bytes written
  }

  if (!ValidateIoArgs("Write", fd, raw_buf, file_offset, buf_offset, io_size,
                      full_buffer_size)) {
    return -1;
  }

  char* buf_ptr = static_cast<char*>(raw_buf) + buf_offset;

  std::size_t total_to_write = io_size;

  ssize_t bytes_written;
  do {
    bytes_written = g_syscall->pwrite(fd, buf_ptr, total_to_write, file_offset);
  } while (bytes_written < 0 && errno == EINTR);

  if (bytes_written < 0) {
    const int saved_errno = errno;
    LOG(ERROR) << "pwrite failed: " << std::strerror(saved_errno);
    errno = saved_errno;
    return -1;
  }

  return bytes_written;
}

static ssize_t ReadvFileToRawBuffers(const tds_storage_io_t* storage_io,
                                     const tds_buffer_io_t* buffer_iov,
                                     int iovcnt) {
  if (iovcnt > IOV_MAX) {
    LOG(ERROR) << "Readv failed: iovcnt (" << iovcnt << ") exceeds IOV_MAX ("
               << IOV_MAX << ")";
    return -1;
  }

  int fd = storage_io->handle->fd;
  off_t file_offset = storage_io->file.file_offset;

  std::vector<struct iovec> iov(iovcnt);
  std::size_t total_to_read = 0;

  for (int i = 0; i < iovcnt; ++i) {
    const tds_buffer_io_t& bio = buffer_iov[i];
    void* raw_buf = nullptr;
    std::size_t io_size = 0;
    std::size_t buf_offset = 0;
    std::size_t full_buffer_size = 0;
    if (!GetBufferInfo(&bio, &raw_buf, &buf_offset, &io_size,
                       &full_buffer_size)) {
      errno = EINVAL;
      return -1;
    }

    if (!ValidateIoArgs("Readv", fd, raw_buf, file_offset, buf_offset, io_size,
                        full_buffer_size)) {
      return -1;
    }

    char* buf_ptr = static_cast<char*>(raw_buf) + buf_offset;
    iov[i].iov_base = buf_ptr;
    iov[i].iov_len = io_size;

    if (io_size > SSIZE_MAX - total_to_read) {
      LOG(ERROR) << "Readv failed: total size exceeds SSIZE_MAX";
      errno = EINVAL;
      return -1;
    }
    total_to_read += io_size;
  }

  if (total_to_read == 0) {
    return 0;
  }

  ssize_t bytes_read;
  do {
    bytes_read = g_syscall->preadv(fd, iov.data(), iov.size(), file_offset);
  } while (bytes_read < 0 && errno == EINTR);

  if (bytes_read < 0) {
    const int saved_errno = errno;
    LOG(ERROR) << "preadv failed: " << std::strerror(saved_errno);
    errno = saved_errno;
    return -1;
  }

  return bytes_read;
}

static ssize_t WritevFileFromRawBuffers(const tds_storage_io_t* storage_io,
                                        const tds_buffer_io_t* buffer_iov,
                                        int iovcnt) {
  if (iovcnt > IOV_MAX) {
    LOG(ERROR) << "Writev failed: iovcnt (" << iovcnt << ") exceeds IOV_MAX ("
               << IOV_MAX << ")";
    return -1;
  }

  int fd = storage_io->handle->fd;
  off_t file_offset = storage_io->file.file_offset;

  std::vector<struct iovec> iov(iovcnt);
  std::size_t total_to_write = 0;

  for (int i = 0; i < iovcnt; ++i) {
    const tds_buffer_io_t& bio = buffer_iov[i];
    void* raw_buf = nullptr;
    std::size_t io_size = 0;
    std::size_t buf_offset = 0;
    std::size_t full_buffer_size = 0;
    if (!GetBufferInfo(&bio, &raw_buf, &buf_offset, &io_size,
                       &full_buffer_size)) {
      errno = EINVAL;
      return -1;
    }

    if (!ValidateIoArgs("Writev", fd, raw_buf, file_offset, buf_offset, io_size,
                        full_buffer_size)) {
      return -1;
    }

    char* buf_ptr = static_cast<char*>(raw_buf) + buf_offset;
    iov[i].iov_base = buf_ptr;
    iov[i].iov_len = io_size;

    if (io_size > SSIZE_MAX - total_to_write) {
      LOG(ERROR) << "Writev failed: total size exceeds SSIZE_MAX";
      errno = EINVAL;
      return -1;
    }
    total_to_write += io_size;
  }

  if (total_to_write == 0) {
    return 0;
  }

  ssize_t bytes_written;
  do {
    bytes_written = g_syscall->pwritev(fd, iov.data(), iov.size(), file_offset);
  } while (bytes_written < 0 && errno == EINTR);

  if (bytes_written < 0) {
    const int saved_errno = errno;
    LOG(ERROR) << "pwritev failed: " << std::strerror(saved_errno);
    errno = saved_errno;
    return -1;
  }

  return bytes_written;
}
}  // namespace tdsul

/* ========================================================================== */
/* 4.1 Initialization & Lifecycle Management                                 */
/* ========================================================================== */

tds_config_t* tds_config_create(void) {
  return new (std::nothrow) tds_config_t();
}

void tds_config_destroy(tds_config_t* config) { delete config; }

tds_result_t tds_config_set_int(tds_config_t* config, const char* key,
                                int value) {
  if (!config || !key) return TDS_ERROR_INVALID_PARAMETER;
  if (std::strcmp(key, "num_worker_threads") == 0) {
    config->num_worker_threads = value;
    return TDS_SUCCESS;
  }
  if (std::strcmp(key, "enable_io_uring") == 0) {
    config->enable_io_uring = (value != 0);
    return TDS_SUCCESS;
  }
  if (std::strcmp(key, "worker_core_start") == 0) {
    config->worker_core_start = value;
    return TDS_SUCCESS;
  }
  if (std::strcmp(key, "worker_core_end") == 0) {
    config->worker_core_end = value;
    return TDS_SUCCESS;
  }
  // TODO: Check if the hardware is physically present and the kernel driver is
  // loaded to avoid relying solely on the Python layer.
  if (std::strcmp(key, "enable_p2p") == 0) {
    config->enable_p2p = (value != 0);
    return TDS_SUCCESS;
  }
  return TDS_ERROR_INVALID_PARAMETER;
}

tds_result_t tds_config_set_string(tds_config_t* config, const char* key,
                                   const char* value) {
  if (!config || !key || !value) return TDS_ERROR_INVALID_PARAMETER;
  return TDS_SUCCESS;
}

tds_storage_opts_t* tds_storage_opts_create(void) {
  return new (std::nothrow) tds_storage_opts_t();
}

void tds_storage_opts_destroy(tds_storage_opts_t* opts) { delete opts; }

tds_result_t tds_storage_opts_set_int(tds_storage_opts_t* opts, const char* key,
                                      int value) {
  if (!opts || !key) return TDS_ERROR_INVALID_PARAMETER;
  if (std::strcmp(key, "stripe_size") == 0) {
    opts->stripe_size = static_cast<size_t>(value);
    return TDS_SUCCESS;
  }
  return TDS_ERROR_INVALID_PARAMETER;
}

tds_result_t tds_storage_opts_set_string(tds_storage_opts_t* opts,
                                         const char* key, const char* value) {
  if (!opts || !key || !value) return TDS_ERROR_INVALID_PARAMETER;
  return TDS_SUCCESS;
}

tds_result_t tds_init(const tds_config_t* config) {
  std::lock_guard<std::mutex> lock(tdsul::g_init_mutex);
  if (tdsul::tds_initialized) {
    LOG(WARNING) << "TDS is already initialized.";
    return TDS_SUCCESS;
  }

  int num_threads = 4;
  int core_start = -1;
  int core_end = -1;
  tdsul::g_use_io_uring = false;
  tdsul::g_p2p_enabled = false;
  bool enable_io_uring = false;

  if (config) {
    tdsul::g_p2p_enabled = config->enable_p2p;
    if (config->num_worker_threads > 0) {
      num_threads = config->num_worker_threads;
    }
    core_start = config->worker_core_start;
    core_end = config->worker_core_end;
    enable_io_uring = config->enable_io_uring;
  }

#ifdef TDSUL_ENABLE_IO_URING
  if (enable_io_uring) {
    if (tdsul::check_io_uring_support()) {
      tdsul::g_use_io_uring = true;
      LOG(INFO) << "io_uring is supported and enabled by user config.";
    } else {
      LOG(WARNING) << "io_uring is requested but not supported by the system. "
                      "Falling back to worker threads.";
    }
  } else {
    LOG(INFO) << "io_uring is not enabled. Using worker threads.";
  }

  if (tdsul::g_use_io_uring) {
    tdsul::g_io_conductor = std::make_unique<tdsul::IoUringConductor>();
  } else {
    tdsul::g_io_conductor = std::make_unique<tdsul::ThreadPoolConductor>(
        num_threads, core_start, core_end);
  }
#else
  if (enable_io_uring) {
    LOG(WARNING) << "io_uring requested but disabled at compile time! "
                    "Falling back to worker threads.";
  } else {
    LOG(INFO) << "io_uring is not enabled. Using worker threads.";
  }

  tdsul::g_io_conductor = std::make_unique<tdsul::ThreadPoolConductor>(
      num_threads, core_start, core_end);
#endif

  tdsul::g_io_conductor->Start();
  tdsul::tds_initialized = true;
  return TDS_SUCCESS;
}

tds_result_t tds_shutdown(void) {
  std::lock_guard<std::mutex> lock(tdsul::g_init_mutex);
  if (tdsul::g_io_conductor) {
    tdsul::g_io_conductor->Stop();
    tdsul::g_io_conductor.reset();
  }
  tdsul::g_use_io_uring = false;
  tdsul::tds_initialized = false;
  return TDS_SUCCESS;
}

/* ========================================================================== */
/* 4.2 Registration Management                                               */
/* ========================================================================== */

tds_result_t tds_buffer_register_vaddr(void* vaddr, int memory_type,
                                       size_t size,
                                       tds_buffer_handle_t** buf_handle) {
  if (!buf_handle) {
    LOG(ERROR) << "tds_buffer_register_vaddr failed: buf_handle output pointer "
                  "is null";
    return TDS_ERROR_INVALID_PARAMETER;
  }

  if (vaddr == nullptr) {
    LOG(ERROR) << "tds_buffer_register_vaddr failed: vaddr is null";
    return TDS_ERROR_INVALID_PARAMETER;
  }

  if (memory_type == TDS_MEM_DEVICE && !tdsul::g_p2p_enabled) {
    LOG(ERROR) << "tds_buffer_register_vaddr failed: P2P device memory "
                  "registration is not supported in this environment.";
    *buf_handle = nullptr;
    return TDS_ERROR_P2P_UNSUPPORTED;
  }

  auto* handle = new tds_buffer_handle_t();
  handle->memory_type = memory_type;
  handle->vaddr = vaddr;
  handle->size = size;
  handle->is_mapped = false;

  *buf_handle = handle;
  LOG(INFO) << "Successfully registered raw DRAM host buffer at " << vaddr
            << " (size: " << size << ")";
  return TDS_SUCCESS;
}

tds_result_t tds_buffer_register_dmabuf(int dma_buf_fd, size_t offset,
                                        int memory_type, size_t size,
                                        tds_buffer_handle_t** buf_handle) {
  if (!buf_handle) {
    LOG(ERROR) << "tds_buffer_register_dmabuf failed: buf_handle output "
                  "pointer is null";
    return TDS_ERROR_INVALID_PARAMETER;
  }

  if (dma_buf_fd < 0) {
    LOG(ERROR) << "tds_buffer_register_dmabuf failed: dma_buf_fd is invalid: "
               << dma_buf_fd;
    return TDS_ERROR_INVALID_PARAMETER;
  }

  if (memory_type == TDS_MEM_DEVICE && !tdsul::g_p2p_enabled) {
    LOG(ERROR) << "tds_buffer_register_dmabuf failed: P2P device memory "
                  "registration is not supported in this environment.";
    *buf_handle = nullptr;
    return TDS_ERROR_P2P_UNSUPPORTED;
  }

  void* mapped_addr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED,
                           dma_buf_fd, static_cast<off_t>(offset));
  if (mapped_addr == MAP_FAILED) {
    LOG(ERROR)
        << "tds_buffer_register_dmabuf failed: mmap failed for dma_buf_fd: "
        << dma_buf_fd << " at offset: " << offset << " (size: " << size
        << "): " << std::strerror(errno);
    return TDS_ERROR_IO;
  }

  auto* handle = new tds_buffer_handle_t();
  handle->memory_type = TDS_MEM_HOST;
  handle->vaddr = mapped_addr;
  handle->dma_buf_fd = dma_buf_fd;
  handle->offset = offset;
  handle->size = size;
  handle->is_mapped = true;

  *buf_handle = handle;
  LOG(INFO) << "Successfully registered mapped DRAM dmabuf at " << mapped_addr
            << " (size: " << size << ", dma_buf_fd: " << dma_buf_fd << ")";
  return TDS_SUCCESS;
}

tds_result_t tds_buffer_deregister(tds_buffer_handle_t* buf_handle) {
  if (!buf_handle) {
    LOG(ERROR) << "tds_buffer_deregister failed: buf_handle is null";
    return TDS_ERROR_INVALID_PARAMETER;
  }

  if (buf_handle->is_mapped && buf_handle->vaddr != nullptr) {
    if (munmap(buf_handle->vaddr, buf_handle->size) != 0) {
      LOG(ERROR) << "munmap failed for address " << buf_handle->vaddr << ": "
                 << std::strerror(errno);
    }
  }

  delete buf_handle;
  LOG(INFO) << "Successfully deregistered buffer handle";
  return TDS_SUCCESS;
}

tds_result_t tds_storage_handle_register(
    const tds_storage_descr_t* descr, tds_storage_handle_t** storage_handle) {
  if (!descr || !descr->uri || !storage_handle)
    return TDS_ERROR_INVALID_PARAMETER;

  std::string uri(descr->uri);
  int fd = -1;
  bool owns_fd = false;

  if (uri.rfind("fd://", 0) == 0) {
    try {
      fd = std::stoi(uri.substr(5));
    } catch (...) {
      LOG(ERROR) << "Invalid file descriptor in URI: " << uri;
      return TDS_ERROR_INVALID_PARAMETER;
    }
    owns_fd = false;
  } else {
    std::string path;
    if (uri.rfind("file://", 0) == 0) {
      path = uri.substr(7);
    } else if (absl::StrContains(uri, "://")) {
      LOG(WARNING) << "Storage URI scheme is not supported: " << uri;
      return TDS_ERROR_UNSUPPORTED;
    } else {
      path = uri;
    }

    fd = ::open(path.c_str(), O_RDWR | O_CREAT, 0644);
    if (fd < 0) {
      fd = ::open(path.c_str(), O_RDONLY);
    }
    if (fd < 0) {
      LOG(ERROR) << "Failed to open file path '" << path
                 << "': " << std::strerror(errno);
      return TDS_ERROR_IO;
    }
    owns_fd = true;
  }

  auto* handle = new tds_storage_handle_t();
  handle->uri = uri;
  handle->fd = fd;
  handle->owns_fd = owns_fd;
  if (descr->options) {
    handle->stripe_size = descr->options->stripe_size;
  }

  *storage_handle = handle;
  LOG(INFO) << "Successfully registered storage handle for URI: " << uri
            << " (fd: " << fd << ", owns_fd: " << owns_fd << ")";
  return TDS_SUCCESS;
}

tds_result_t tds_storage_handle_deregister(
    tds_storage_handle_t* storage_handle) {
  if (!storage_handle) return TDS_ERROR_INVALID_PARAMETER;
  if (storage_handle->owns_fd && storage_handle->fd >= 0) {
    if (::close(storage_handle->fd) != 0) {
      LOG(ERROR) << "Failed to close storage handle fd " << storage_handle->fd
                 << ": " << std::strerror(errno);
    }
  }
  delete storage_handle;
  LOG(INFO) << "Successfully deregistered storage handle";
  return TDS_SUCCESS;
}

/* ========================================================================== */
/* 4.3 Synchronous I/O                                                       */
/* ========================================================================== */

ssize_t tds_read(const tds_storage_io_t* storage_io,
                 const tds_buffer_io_t* buffer_io) {
  if (!storage_io || !storage_io->handle || !buffer_io) return -1;
  return tdsul::ReadFileToRawBuffer(storage_io, buffer_io);
}

ssize_t tds_write(const tds_storage_io_t* storage_io,
                  const tds_buffer_io_t* buffer_io) {
  if (!storage_io || !storage_io->handle || !buffer_io) return -1;
  return tdsul::WriteFileFromRawBuffer(storage_io, buffer_io);
}

ssize_t tds_readv(const tds_storage_io_t* storage_io,
                  const tds_buffer_io_t* buffer_io_arr, int num_buffers) {
  if (!storage_io || !storage_io->handle || !buffer_io_arr || num_buffers <= 0)
    return -1;
  return tdsul::ReadvFileToRawBuffers(storage_io, buffer_io_arr, num_buffers);
}

ssize_t tds_writev(const tds_storage_io_t* storage_io,
                   const tds_buffer_io_t* buffer_io_arr, int num_buffers) {
  if (!storage_io || !storage_io->handle || !buffer_io_arr || num_buffers <= 0)
    return -1;
  return tdsul::WritevFileFromRawBuffers(storage_io, buffer_io_arr,
                                         num_buffers);
}

/* ========================================================================== */
/* 4.4 Asynchronous I/O & Queue Management                                   */
/* ========================================================================== */

tds_result_t tds_queue_create(tds_io_queue_t** queue) {
  if (!queue) return TDS_ERROR_INVALID_PARAMETER;
  *queue = nullptr;
  return TDS_ERROR_UNSUPPORTED;
}

tds_result_t tds_queue_destroy(tds_io_queue_t* queue) {
  if (!queue) return TDS_ERROR_INVALID_PARAMETER;
  delete reinterpret_cast<tdsul::IoQueue*>(queue);
  return TDS_SUCCESS;
}

tds_result_t tds_queue_synchronize(tds_io_queue_t* queue) {
  if (!queue) return TDS_ERROR_INVALID_PARAMETER;
  return TDS_ERROR_UNSUPPORTED;
}

tds_result_t tds_read_async(const tds_storage_io_t* storage_io,
                            const tds_buffer_io_t* buffer_io,
                            tds_io_queue_t* queue, tds_future_t** future) {
  if (future) *future = nullptr;
  if (!storage_io || !buffer_io) return TDS_ERROR_INVALID_PARAMETER;

  auto* new_future = new tds_future_t();
  new_future->status.status = TDS_PENDING;

  auto* req = new tdsul::SingleWorkRequest(*storage_io, *buffer_io, TDS_OP_READ,
                                           queue, new_future);

  if (queue) {
    auto* io_queue = reinterpret_cast<tdsul::IoQueue*>(queue);
    io_queue->Push(req);
  } else {
    if (tdsul::g_io_conductor) {
      tdsul::g_io_conductor->Dispatch(req);
    }
  }

  if (future) *future = new_future;
  return TDS_SUCCESS;
}

tds_result_t tds_write_async(const tds_storage_io_t* storage_io,
                             const tds_buffer_io_t* buffer_io,
                             tds_io_queue_t* queue, tds_future_t** future) {
  if (future) *future = nullptr;
  if (!storage_io || !buffer_io) return TDS_ERROR_INVALID_PARAMETER;

  auto* new_future = new tds_future_t();
  new_future->status.status = TDS_PENDING;

  auto* req = new tdsul::SingleWorkRequest(*storage_io, *buffer_io,
                                           TDS_OP_WRITE, queue, new_future);

  if (queue) {
    auto* io_queue = reinterpret_cast<tdsul::IoQueue*>(queue);
    io_queue->Push(req);
  } else {
    if (tdsul::g_io_conductor) {
      tdsul::g_io_conductor->Dispatch(req);
    }
  }

  if (future) *future = new_future;
  return TDS_SUCCESS;
}

tds_result_t tds_readv_async(const tds_storage_io_t* storage_io,
                             const tds_buffer_io_t* buffer_io_arr,
                             int num_buffers, tds_io_queue_t* queue,
                             tds_future_t** future) {
  if (future) *future = nullptr;
  if (!queue || !storage_io || !buffer_io_arr || num_buffers <= 0)
    return TDS_ERROR_INVALID_PARAMETER;
  return TDS_ERROR_UNSUPPORTED;
}

tds_result_t tds_writev_async(const tds_storage_io_t* storage_io,
                              const tds_buffer_io_t* buffer_io_arr,
                              int num_buffers, tds_io_queue_t* queue,
                              tds_future_t** future) {
  if (future) *future = nullptr;
  if (!queue || !storage_io || !buffer_io_arr || num_buffers <= 0)
    return TDS_ERROR_INVALID_PARAMETER;
  return TDS_ERROR_UNSUPPORTED;
}

void tds_future_wait(tds_future_t* future) { (void)future; }

tds_io_status_t tds_future_get_status(tds_future_t* future, int index) {
  (void)index;
  if (!future) {
    return {TDS_ERROR_INVALID_PARAMETER, EINVAL, 0};
  }
  return {TDS_ERROR_UNSUPPORTED, 0, 0};
}

void tds_future_destroy(tds_future_t* future) { delete future; }

/* ========================================================================== */
/* 4.5 Batched I/O operations                                                */
/* ========================================================================== */

tds_batch_t* tds_batch_create(void) { return nullptr; }

tds_result_t tds_batch_add(tds_batch_t* batch,
                           const tds_storage_io_t* storage_io,
                           const tds_buffer_io_t* buffer_io, OpType opt,
                           int* index) {
  (void)batch;
  (void)storage_io;
  (void)buffer_io;
  (void)opt;
  if (index) *index = -1;
  return TDS_ERROR_UNSUPPORTED;
}

tds_result_t tds_batch_addv(tds_batch_t* batch,
                            const tds_storage_io_t* storage_io,
                            const tds_buffer_io_t* buffer_io_arr,
                            int num_buffers, OpType opt, int* index) {
  (void)batch;
  (void)storage_io;
  (void)buffer_io_arr;
  (void)num_buffers;
  (void)opt;
  if (index) *index = -1;
  return TDS_ERROR_UNSUPPORTED;
}

int tds_batch_get_count(const tds_batch_t* batch) {
  (void)batch;
  return 0;
}

void tds_batch_destroy(tds_batch_t* batch) { (void)batch; }

tds_result_t tds_batch_execute(tds_batch_t* batch, ssize_t* ret_arr) {
  (void)batch;
  (void)ret_arr;
  return TDS_ERROR_UNSUPPORTED;
}

tds_result_t tds_batch_submit(tds_batch_t* batch, tds_io_queue_t* queue,
                              tds_future_t** future) {
  (void)batch;
  (void)queue;
  if (future) *future = nullptr;
  return TDS_ERROR_UNSUPPORTED;
}
