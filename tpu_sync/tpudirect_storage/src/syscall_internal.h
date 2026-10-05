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

#ifndef TDSUL_SRC_SYSCALL_INTERNAL_H_
#define TDSUL_SRC_SYSCALL_INTERNAL_H_

#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>

#include <cstddef>

namespace tdsul {

class Syscall {
 public:
  virtual ~Syscall() = default;
  virtual ssize_t pread(int fd, void* buf, size_t count, off_t offset) {
    return ::pread(fd, buf, count, offset);
  }
  virtual ssize_t pwrite(int fd, const void* buf, size_t count, off_t offset) {
    return ::pwrite(fd, buf, count, offset);
  }
  virtual ssize_t preadv(int fd, const struct iovec* iov, int iovcnt,
                         off_t offset) {
    return ::preadv(fd, iov, iovcnt, offset);
  }
  virtual ssize_t pwritev(int fd, const struct iovec* iov, int iovcnt,
                          off_t offset) {
    return ::pwritev(fd, iov, iovcnt, offset);
  }
};

extern Syscall* g_syscall;

}  // namespace tdsul

#endif  // TDSUL_SRC_SYSCALL_INTERNAL_H_
