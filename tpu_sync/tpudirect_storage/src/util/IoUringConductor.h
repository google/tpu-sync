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

#ifndef TDSUL_SRC_UTIL_IO_URING_CONDUCTOR_H_
#define TDSUL_SRC_UTIL_IO_URING_CONDUCTOR_H_

#include "util/IoConductor.h"

#ifdef TDSUL_ENABLE_IO_URING

namespace tdsul {

class IoUringConductor : public IoConductor {
 public:
  IoUringConductor();
  ~IoUringConductor() override;

  void Start() override;
  void Stop() override;
  void Dispatch(WorkRequest* request) override;
  IoConductorType GetType() const override { return IoConductorType::IO_URING; }
};

}  // namespace tdsul

#endif  // TDSUL_ENABLE_IO_URING

#endif  // TDSUL_SRC_UTIL_IO_URING_CONDUCTOR_H_
