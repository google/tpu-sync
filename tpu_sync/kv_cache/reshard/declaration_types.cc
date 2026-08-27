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

#include "tpu_sync/kv_cache/reshard/declaration_types.h"

#include <cstdint>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tpu_sync/common/raiden_id.h"

namespace tpu_raiden {
namespace kv_cache {
namespace reshard {

std::string PythonRepr(const RaidenId& id) {
  return absl::StrCat("RaidenId(job_name='", id.job_name, "', job_replica_id='",
                      id.job_replica_id, "', data_name='", id.data_name,
                      "', data_replica_idx=", id.data_replica_idx, ")");
}

std::string PythonReprIntList(const std::vector<int64_t>& values) {
  return absl::StrCat("[", absl::StrJoin(values, ", "), "]");
}

RaidenId RaidenIdFromProto(const tpu_sync::rpc::RaidenIdProto& proto) {
  RaidenId id;
  id.job_name = proto.job_name();
  id.job_replica_id = proto.job_replica_id();
  id.data_name = proto.data_name();
  id.data_replica_idx = proto.data_replica_idx();
  return id;
}

tpu_sync::rpc::RaidenIdProto RaidenIdToProto(const RaidenId& id) {
  tpu_sync::rpc::RaidenIdProto proto;
  proto.set_job_name(id.job_name);
  proto.set_job_replica_id(id.job_replica_id);
  proto.set_data_name(id.data_name);
  proto.set_data_replica_idx(id.data_replica_idx);
  return proto;
}

}  // namespace reshard
}  // namespace kv_cache
}  // namespace tpu_raiden
