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

#ifndef THIRD_PARTY_TPU_RAIDEN_CORE_TPU_UTILS_H_
#define THIRD_PARTY_TPU_RAIDEN_CORE_TPU_UTILS_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "xla/pjrt/pjrt_client.h"

struct ifaddrs;

namespace tpu_raiden {

struct TpuPciDevice {
  std::string bdf;        // BDF address, e.g. "0000:16:00.0"
  std::string device_id;  // Device ID, e.g. "0x0075"
  int numa_node = -1;     // NUMA node, e.g. 0
};

// Memory policy modes matching Linux kernel set_mempolicy syscall:
inline constexpr int kMpolDefault = 0;
inline constexpr int kMpolPreferred = 1;
inline constexpr int kMpolBind = 2;

// Sets the memory policy for the current thread using set_mempolicy syscall.
// If the environment variable RAIDEN_NUMA_POLICY is set, it overrides the
// requested mode when node >= 0 (e.g. "preferred", "bind", "default").
// Returns 0 on success, or a negative error code on failure.
int64_t SetThreadMempolicy(int mode, int node = -1);

// Returns the CPU core IDs belonging to a given NUMA node.
// Parses /sys/devices/system/node/node<N>/cpulist.
std::vector<int> GetNumaNodeCpuCores(int numa_node);

// Pins the current thread to the given CPU cores.
// Returns 0 on success, or a negative error code on failure.
int PinCurrentThreadToCores(const std::vector<int>& cores);

// Pins the current thread to the given NUMA node and sets its memory
// policy. By default uses kMpolBind (subject to RAIDEN_NUMA_POLICY override).
// Returns 0 on success, or a negative error code on failure.
int PinCurrentThreadToNumaNode(int node, int mode = kMpolBind);

// Scans the PCI bus and returns all detected TPU PCI devices, sorted by BDF.
const std::vector<TpuPciDevice>& GetTpuPciDevices();

// Returns the NUMA node for a given PjRtDevice.
// Maps the device's local_hardware_id to the sorted PCI devices.
// Returns -1 if the node cannot be determined.
#if defined(__GNUC__) || defined(__clang__)
__attribute__((visibility("default")))
#endif
int GetPjRtDeviceNumaNode(const xla::PjRtDevice* device);

// Prints the detected TPU hardware topology to std::cout.
void PrintTpuHardwareTopology();

enum class NicClassification {
  kUnknown,
  kControlPlane,
  kDataPlane,
};

struct HostNicAddress {
  std::string interface_name;  // e.g. "eth0"
  std::string ip_address;      // e.g. "10.128.0.10"
  int numa_node = -1;          // e.g. 0 from sysfs
  NicClassification classification = NicClassification::kUnknown;
};

// Returns non-loopback network interfaces on this host enriched with NUMA
// node and classification.
std::vector<HostNicAddress> GetLocalHostNicAddresses(
    absl::string_view sysfs_root = "/sys");

namespace internal {
std::vector<HostNicAddress> GetLocalHostNicAddressesInternal(
    struct ifaddrs* ifaddr, absl::string_view sysfs_root);
}  // namespace internal

// Returns non-loopback IPv4 addresses discovered on this host.
// Falls back to {"127.0.0.1"} if none exist.
std::vector<std::string> GetLocalHostIpAddresses();

// Returns the NUMA node for a given network interface name.
// Returns -1 if the node cannot be resolved.
int GetInterfaceNumaNode(const char* ifname,
                         absl::string_view sysfs_root = "/sys");

// Uses getsockname to look up the local IPv4 address and returns its
// corresponding interface's host NIC address. Returns std::nullopt on failure
// or if not found.
std::optional<HostNicAddress> GetSocketLocalNic(int fd);

// Pins the thread to the local NUMA node if pin_thread is true.
// Returns 0 on success or -1 on error.
int ApplySocketAffinityAndBinding(int fd, bool pin_thread = true);

}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_CORE_TPU_UTILS_H_
