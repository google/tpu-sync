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

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_FAULT_INJECTION_HOOKS_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_FAULT_INJECTION_HOOKS_H_

#include <string_view>

namespace tpu_raiden {
namespace hooks {

// Hook labels follow `<module>.<channel>.<action>`, where module is the
// component that owns the site (e.g. kv_cache_manager, grpc_control_plane) and
// channel is the path it runs on (e.g. recv, send, pull, api, h2d, d2h).

// Block transport, receive path.
inline constexpr std::string_view kBlockTransportRecvIdsAlloc =
    "block_transport.recv.ids_alloc";
inline constexpr std::string_view kBlockTransportRecvProgress =
    "block_transport.recv.progress";
inline constexpr std::string_view kBlockTransportRecvLayerDone =
    "block_transport.recv.layer_done";
inline constexpr std::string_view kBlockTransportRecvBeforeAck =
    "block_transport.recv.before_ack";

// Transfer receive session.
inline constexpr std::string_view kTransferRecvSessionH2dDispatch =
    "transfer_recv_session.h2d.dispatch";
inline constexpr std::string_view kTransferRecvSessionH2dComplete =
    "transfer_recv_session.h2d.complete";
inline constexpr std::string_view kTransferRecvSessionPullRequest =
    "transfer_recv_session.pull.request";
inline constexpr std::string_view kTransferRecvSessionPullReply =
    "transfer_recv_session.pull.reply";

// KV cache manager, scheduler API.
inline constexpr std::string_view kKvCacheManagerApiStartRead =
    "kv_cache_manager.api.start_read";
inline constexpr std::string_view kKvCacheManagerApiStartReadSubmitPull =
    "kv_cache_manager.api.start_read.submit_pull";
inline constexpr std::string_view kKvCacheManagerApiCompleteRead =
    "kv_cache_manager.api.complete_read";

// KV cache manager, pull stream handler.
inline constexpr std::string_view kKvCacheManagerPullRegisterWait =
    "kv_cache_manager.pull.register_wait";
inline constexpr std::string_view kKvCacheManagerPullSpawn =
    "kv_cache_manager.pull.spawn";

// gRPC control plane.
inline constexpr std::string_view kGrpcControlPlanePullReply =
    "grpc_control_plane.pull.reply";

// TCP control plane.
inline constexpr std::string_view kTcpControlPlanePullReply =
    "tcp_control_plane.pull.reply";
inline constexpr std::string_view kTcpControlPlaneAccept =
    "tcp_control_plane.accept";

// Staging block allocator.
inline constexpr std::string_view kStagingAllocatorAcquire =
    "staging_allocator.acquire";

// Transfer send session.
inline constexpr std::string_view kTransferSendSessionD2hDispatch =
    "transfer_send_session.d2h.dispatch";
inline constexpr std::string_view kTransferSendSessionD2hComplete =
    "transfer_send_session.d2h.complete";
// Raiden manager base.
inline constexpr std::string_view kRaidenManagerBaseH2hWrite =
    "raiden_manager_base.h2h.write";

// Socket transport, send path.
inline constexpr std::string_view kSocketTransportSendConnect =
    "socket_transport.send.connect";
inline constexpr std::string_view kSocketTransportSendProgress =
    "socket_transport.send.progress";

// Raw buffer transport listener.
inline constexpr std::string_view kRawBufferTransportAccept =
    "raw_buffer_transport.accept";

// Hooks that execute on PJRT callbacks, under mutexes, or at non-blocking
// boundaries where sleeping is disallowed; FaultInjector::Install rejects
// kDelay rules targeting any hook in this list.
inline constexpr std::string_view kFailOnly[] = {
    kBlockTransportRecvIdsAlloc,     kTransferRecvSessionH2dDispatch,
    kTransferRecvSessionH2dComplete, kTransferRecvSessionPullReply,
    kKvCacheManagerApiStartRead,     kKvCacheManagerApiStartReadSubmitPull,
    kKvCacheManagerApiCompleteRead,  kKvCacheManagerPullSpawn,
    kStagingAllocatorAcquire,        kTransferSendSessionD2hDispatch,
    kTransferSendSessionD2hComplete, kRawBufferTransportAccept,
    kTcpControlPlaneAccept,
};

}  // namespace hooks
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_FAULT_INJECTION_HOOKS_H_
