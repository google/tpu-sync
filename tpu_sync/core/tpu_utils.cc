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

#include "tpu_sync/core/tpu_utils.h"

#include <arpa/inet.h>
#include <dirent.h>
#include <ifaddrs.h>
#include <limits.h>
#include <netinet/in.h>
#include <pthread.h>
#include <sched.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/tsl/platform/logging.h"

#ifndef __NR_set_mempolicy
#if defined(__x86_64__)
#define __NR_set_mempolicy 238
#elif defined(__aarch64__)
#define __NR_set_mempolicy 237
#endif
#endif

namespace tpu_raiden {

int64_t SetThreadMempolicy(int mode, int node) {
  constexpr int kMpolDefault = 0;
  constexpr int kMpolBind = 2;
#ifndef __NR_set_mempolicy
#if defined(__x86_64__)
#define __NR_set_mempolicy 238
#elif defined(__aarch64__)
#define __NR_set_mempolicy 237
#endif
#endif

#ifdef __NR_set_mempolicy
  int64_t res;
  if (mode == kMpolDefault || node < 0) {
    res = static_cast<int64_t>(
        syscall(__NR_set_mempolicy, kMpolDefault, nullptr, 0));
  } else {
    uint64_t mask = 1ULL << node;
    res = static_cast<int64_t>(
        syscall(__NR_set_mempolicy, kMpolBind, &mask, sizeof(mask) * 8));
  }
  if (res < 0) {
    LOG(ERROR) << "SetThreadMempolicy(mode=" << mode << ", node=" << node
               << ") failed: " << std::strerror(errno) << " (errno=" << errno
               << ")";
  }
  return res;
#else
  return -1;  // Syscall not available
#endif
}

std::vector<int> GetNumaNodeCpuCores(int numa_node) {
  std::vector<int> cores;
  std::string path =
      "/sys/devices/system/node/node" + std::to_string(numa_node) + "/cpulist";
  std::ifstream file(path);
  if (!file.is_open()) {
    LOG(ERROR) << "Failed to open " << path;
    return cores;
  }
  std::string line;
  if (std::getline(file, line)) {
    std::stringstream ss(line);
    std::string part;
    while (std::getline(ss, part, ',')) {
      size_t dash = part.find('-');
      if (dash == std::string::npos) {
        try {
          cores.push_back(std::stoi(part));
        } catch (...) {
        }
      } else {
        try {
          int start = std::stoi(part.substr(0, dash));
          int end = std::stoi(part.substr(dash + 1));
          for (int c = start; c <= end; ++c) {
            cores.push_back(c);
          }
        } catch (...) {
        }
      }
    }
  }
  return cores;
}

int PinCurrentThreadToCores(const std::vector<int>& cores) {
  if (cores.empty()) return -1;
#ifdef __linux__
  cpu_set_t allowed_set;
  CPU_ZERO(&allowed_set);
  if (sched_getaffinity(0, sizeof(cpu_set_t), &allowed_set) != 0) {
    LOG(ERROR) << "sched_getaffinity failed: " << std::strerror(errno);
    // Fallback: assume all cores are allowed if we can't query
    for (int i = 0; i < CPU_SETSIZE; ++i) {
      CPU_SET(i, &allowed_set);
    }
  }

  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  bool has_cores = false;
  for (int core : cores) {
    if (core >= 0 && core < CPU_SETSIZE && CPU_ISSET(core, &allowed_set)) {
      CPU_SET(core, &cpuset);
      has_cores = true;
    }
  }
  if (!has_cores) {
    LOG(WARNING) << "No allowed cores found for pinning on this NUMA node";
    return -1;
  }
  int res = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
  if (res != 0) {
    LOG(ERROR) << "pthread_setaffinity_np failed: " << std::strerror(res)
               << " (rc=" << res << ")";
    return -res;
  }
  return 0;
#else
  return -1;
#endif
}

int PinCurrentThreadToNumaNode(int node) {
  if (node < 0) return -1;
  std::vector<int> cores = GetNumaNodeCpuCores(node);
  if (cores.empty()) {
    LOG(WARNING)
        << "No CPU cores found for NUMA node " << node
        << ". Skipping CPU core affinity, will attempt memory policy only.";
  } else {
    int rc = PinCurrentThreadToCores(cores);
    if (rc != 0) return rc;
  }
  constexpr int kMpolBind = 2;
  int64_t mem_rc = SetThreadMempolicy(kMpolBind, node);
  if (mem_rc < 0) return static_cast<int>(mem_rc);
  return 0;
}

const std::vector<TpuPciDevice>& GetTpuPciDevices() {
  static const std::vector<TpuPciDevice>* cached_devices = []() {
    auto* devices = new std::vector<TpuPciDevice>();
    std::string pci_dir = "/sys/bus/pci/devices";
    DIR* dir = opendir(pci_dir.c_str());
    if (dir == nullptr) {
      return devices;
    }

    // Known Google TPU Device IDs
    const std::vector<std::string> kTpuDeviceIds = {
        // TPU v4
        "0x0056",
        "0x005e",

        // TPU v5e
        "0x0062",
        "0x0063",

        // TPU v6e
        "0x006e",
        "0x006f",
        "0x0070",
        "0x0071",

        // TPU v7/v7x
        "0x0075",
        "0x0076",
        "0x0077",
        "0x0078",
    };

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
      std::string dev_name = entry->d_name;
      if (dev_name == "." || dev_name == "..") continue;

      std::string dev_path = pci_dir + "/" + dev_name;
      std::string vendor_path = dev_path + "/vendor";
      std::string device_path = dev_path + "/device";
      std::string numa_path = dev_path + "/numa_node";

      // Read vendor
      std::ifstream vendor_file(vendor_path);
      std::string vendor_id;
      if (!(vendor_file >> vendor_id)) continue;

      // Google Vendor ID is 0x1ae0
      if (vendor_id != "0x1ae0") continue;

      // Read device ID
      std::ifstream device_file(device_path);
      std::string device_id;
      if (!(device_file >> device_id)) continue;

      // Check if it is a known TPU device
      bool is_tpu = false;
      for (const auto& id : kTpuDeviceIds) {
        if (device_id == id) {
          is_tpu = true;
          break;
        }
      }
      if (!is_tpu) continue;

      // Read NUMA node
      std::ifstream numa_file(numa_path);
      int node = -1;
      if (numa_file >> node) {
        TpuPciDevice dev;
        dev.bdf = dev_name;
        dev.device_id = device_id;
        dev.numa_node = node;
        devices->push_back(dev);
      }
    }
    closedir(dir);

    // Sort by BDF to ensure consistent ordering
    std::sort(devices->begin(), devices->end(),
              [](const TpuPciDevice& a, const TpuPciDevice& b) {
                return a.bdf < b.bdf;
              });

    return devices;
  }();

  return *cached_devices;
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((visibility("default")))
#endif
int GetPjRtDeviceNumaNode(const xla::PjRtDevice* device) {
  if (device == nullptr) return -1;

  int chip_idx = device->local_hardware_id().value();
  if (chip_idx < 0) {
    VLOG(1) << "Negative chip_idx (" << chip_idx << "), returning NUMA node -1";
    return -1;
  }

  const auto& pci_devices = GetTpuPciDevices();
  if (pci_devices.empty()) {
    return -1;
  }

  // Find unique buses (domain:bus) and their NUMA nodes
  std::vector<std::pair<std::string, int>> unique_chips;
  for (const auto& dev : pci_devices) {
    size_t last_colon = dev.bdf.find_last_of(':');
    if (last_colon == std::string::npos) continue;
    std::string domain_bus = dev.bdf.substr(0, last_colon);

    bool found = false;
    for (const auto& chip : unique_chips) {
      if (chip.first == domain_bus) {
        found = true;
        break;
      }
    }
    if (!found) {
      unique_chips.push_back({domain_bus, dev.numa_node});
    }
  }

  if (unique_chips.empty()) return -1;

  // Sort unique chips by BDF to ensure consistent mapping
  std::sort(unique_chips.begin(), unique_chips.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });

  if (device->client() == nullptr) return -1;

  int local_device_count = device->client()->addressable_devices().size();
  int num_physical_chips = unique_chips.size();

  if (local_device_count <= 0 || num_physical_chips <= 0) return -1;

  int devices_per_chip = local_device_count / num_physical_chips;
  if (devices_per_chip <= 0) devices_per_chip = 1;

  int physical_chip_idx = chip_idx / devices_per_chip;
  if (physical_chip_idx >= num_physical_chips) {
    physical_chip_idx = num_physical_chips - 1;
  }

  return unique_chips[physical_chip_idx].second;
}

void PrintTpuHardwareTopology() {
  LOG(INFO) << "Querying TPU NUMA topology via PCI sysfs:";
  const auto& devices = GetTpuPciDevices();
  for (const auto& dev : devices) {
    LOG(INFO) << "  TPU Device " << dev.bdf << " (Device ID: " << dev.device_id
              << ") is attached to NUMA Node " << dev.numa_node;
  }
  if (devices.empty()) {
    LOG(INFO) << "  No Google TPU PCI devices found in sysfs.";
  }
}

int GetInterfaceNumaNode(const char* ifname, absl::string_view sysfs_root) {
  std::string path =
      absl::StrCat(sysfs_root, "/class/net/", ifname, "/device/numa_node");
  std::ifstream f(path);
  int node = -1;
  if (f.is_open()) {
    f >> node;
  }
  if (node < 0) {
    absl::string_view name(ifname);
    if (name == "eth0" || name == "eth1" || name == "ens5") {
      node = 0;
    } else if (name == "dcn1" || name == "ens6" || name == "eth2") {
      node = 1;
    }
  }
  return node;
}

namespace {

int GetInterfaceMtu(absl::string_view ifname, absl::string_view sysfs_root) {
  std::string path = absl::StrCat(sysfs_root, "/class/net/", ifname, "/mtu");
  std::ifstream f(path);
  int mtu = -1;
  if (f.is_open()) {
    f >> mtu;
  }
  return mtu;
}

std::string GetInterfaceBdf(absl::string_view ifname,
                            absl::string_view sysfs_root) {
  std::string device_path =
      absl::StrCat(sysfs_root, "/class/net/", ifname, "/device");
  char buf[PATH_MAX];
  ssize_t len = readlink(device_path.c_str(), buf, sizeof(buf) - 1);
  if (len != -1) {
    buf[len] = '\0';
    std::string rel_path(buf);
    size_t last_slash = rel_path.find_last_of('/');
    if (last_slash != std::string::npos) {
      return rel_path.substr(last_slash + 1);
    }
    return rel_path;
  }
  return "";
}

int GetTotalNumaNodes(absl::string_view sysfs_root) {
  std::string node_dir = absl::StrCat(sysfs_root, "/devices/system/node");
  DIR* dir = opendir(node_dir.c_str());
  if (dir == nullptr) return 1;
  struct dirent* entry;
  int count = 0;
  while ((entry = readdir(dir)) != nullptr) {
    if (entry->d_type == DT_DIR && absl::StartsWith(entry->d_name, "node")) {
      std::string name = entry->d_name;
      if (name.size() > 4 &&
          std::all_of(name.begin() + 4, name.end(), ::isdigit)) {
        count++;
      }
    }
  }
  closedir(dir);
  return count > 0 ? count : 1;
}

// Dedicated classifier for GKE / Cloud TPU VMs
NicClassification ClassifyNicGke(absl::string_view bdf, int mtu) {
  if (bdf.empty()) {
    return NicClassification::kControlPlane;
  }
  if (mtu > 1500) {
    return NicClassification::kDataPlane;
  }
  return NicClassification::kControlPlane;
}

NicClassification ClassifyNic(absl::string_view ifname, absl::string_view bdf,
                              int mtu) {
  return ClassifyNicGke(bdf, mtu);
}

std::string ClassificationToString(NicClassification classification) {
  switch (classification) {
    case NicClassification::kDataPlane:
      return "Data Interface";
    case NicClassification::kControlPlane:
      return "Control/Fallback Interface";
    case NicClassification::kUnknown:
      return "Unknown Interface";
  }
  return "Unknown Interface";
}

}  // namespace

namespace internal {
std::vector<HostNicAddress> GetLocalHostNicAddressesInternal(
    struct ifaddrs* ifaddr, absl::string_view sysfs_root) {
  std::vector<HostNicAddress> nics;
  struct ifaddrs* ifa;
  for (ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
    if (ifa->ifa_addr == nullptr) continue;
    if (ifa->ifa_addr->sa_family == AF_INET) {
      if (std::strcmp(ifa->ifa_name, "lo") != 0) {
        char host[INET_ADDRSTRLEN];
        auto* s_in = reinterpret_cast<struct sockaddr_in*>(ifa->ifa_addr);
        inet_ntop(AF_INET, &s_in->sin_addr, host, INET_ADDRSTRLEN);
        std::string ip_str(host);
        auto it = std::find_if(
            nics.begin(), nics.end(),
            [&](const HostNicAddress& n) { return n.ip_address == ip_str; });
        if (it == nics.end()) {
          int node = GetInterfaceNumaNode(ifa->ifa_name, sysfs_root);
          int mtu = GetInterfaceMtu(ifa->ifa_name, sysfs_root);
          std::string bdf = GetInterfaceBdf(ifa->ifa_name, sysfs_root);
          NicClassification classification =
              ClassifyNic(ifa->ifa_name, bdf, mtu);
          nics.push_back({ifa->ifa_name, ip_str, node, classification});
        }
      }
    } else if (ifa->ifa_addr->sa_family == AF_INET6) {
      if (std::strcmp(ifa->ifa_name, "lo") != 0) {
        char host[INET6_ADDRSTRLEN];
        auto* s_in6 = reinterpret_cast<struct sockaddr_in6*>(ifa->ifa_addr);
        if (IN6_IS_ADDR_LINKLOCAL(&s_in6->sin6_addr)) {
          continue;
        }
        inet_ntop(AF_INET6, &s_in6->sin6_addr, host, INET6_ADDRSTRLEN);
        std::string ip_str(host);
        auto it = std::find_if(
            nics.begin(), nics.end(),
            [&](const HostNicAddress& n) { return n.ip_address == ip_str; });
        if (it == nics.end()) {
          int node = GetInterfaceNumaNode(ifa->ifa_name, sysfs_root);
          int mtu = GetInterfaceMtu(ifa->ifa_name, sysfs_root);
          std::string bdf = GetInterfaceBdf(ifa->ifa_name, sysfs_root);
          NicClassification classification =
              ClassifyNic(ifa->ifa_name, bdf, mtu);
          nics.push_back({ifa->ifa_name, ip_str, node, classification});
        }
      }
    }
  }

  // Second pass: apply BDF heuristic for NUMA assignment if NUMA is -1
  struct NicWithBdf {
    size_t index;
    std::string bdf;
  };
  std::vector<NicWithBdf> pending_nics;
  for (size_t i = 0; i < nics.size(); ++i) {
    if (nics[i].numa_node == -1) {
      std::string bdf = GetInterfaceBdf(nics[i].interface_name, sysfs_root);
      if (!bdf.empty()) {
        pending_nics.push_back({i, bdf});
      }
    }
  }

  if (!pending_nics.empty()) {
    std::sort(
        pending_nics.begin(), pending_nics.end(),
        [](const NicWithBdf& a, const NicWithBdf& b) { return a.bdf < b.bdf; });
    int total_nodes = GetTotalNumaNodes(sysfs_root);
    for (size_t i = 0; i < pending_nics.size(); ++i) {
      nics[pending_nics[i].index].numa_node = i % total_nodes;
    }
  }

  if (nics.empty()) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd >= 0) {
      close(fd);
      nics.push_back({"lo", "127.0.0.1", -1, NicClassification::kControlPlane});
    } else {
      nics.push_back({"lo", "::1", -1, NicClassification::kControlPlane});
    }
  }
  for (const auto& nic : nics) {
    std::string bdf = GetInterfaceBdf(nic.interface_name, sysfs_root);
    int mtu = GetInterfaceMtu(nic.interface_name, sysfs_root);
    LOG(INFO) << "Detected interface: " << nic.interface_name
              << ", BDF: " << (bdf.empty() ? "None" : bdf) << ", MTU: " << mtu
              << ", NUMA: " << nic.numa_node << ", Classification: "
              << ClassificationToString(nic.classification);
  }
  return nics;
}
}  // namespace internal

std::vector<HostNicAddress> GetLocalHostNicAddresses(
    absl::string_view sysfs_root) {
  struct ifaddrs* ifaddr;
  if (getifaddrs(&ifaddr) == 0) {
    auto nics = internal::GetLocalHostNicAddressesInternal(ifaddr, sysfs_root);
    freeifaddrs(ifaddr);
    return nics;
  }
  return internal::GetLocalHostNicAddressesInternal(nullptr, sysfs_root);
}

std::vector<std::string> GetLocalHostIpAddresses() {
  std::vector<std::string> ips;
  for (const auto& nic : GetLocalHostNicAddresses()) {
    ips.push_back(nic.ip_address);
  }
  return ips;
}

const std::vector<HostNicAddress>& GetCachedLocalHostNicAddresses() {
  static const std::vector<HostNicAddress>* nics =
      new std::vector<HostNicAddress>(GetLocalHostNicAddresses());
  return *nics;
}

std::optional<HostNicAddress> GetSocketLocalNic(int fd) {
  struct sockaddr_storage addr;
  socklen_t len = sizeof(addr);
  if (getsockname(fd, (struct sockaddr*)&addr, &len) == -1) {
    LOG_EVERY_N_SEC(ERROR, 10)
        << "getsockname failed: " << std::strerror(errno);
    return std::nullopt;
  }
  if (addr.ss_family != AF_INET) {
    return std::nullopt;
  }
  struct sockaddr_in* s_in = (struct sockaddr_in*)&addr;
  char host[INET_ADDRSTRLEN];
  if (inet_ntop(AF_INET, &s_in->sin_addr, host, INET_ADDRSTRLEN) == nullptr) {
    return std::nullopt;
  }
  std::string ip_str(host);
  for (const auto& nic : GetCachedLocalHostNicAddresses()) {
    if (nic.ip_address == ip_str) {
      return nic;
    }
  }
  return std::nullopt;
}

int ApplySocketAffinityAndBinding(int fd, bool pin_thread) {
  if (!pin_thread) {
    return 0;
  }
  auto nic_opt = GetSocketLocalNic(fd);
  if (nic_opt.has_value() && nic_opt->numa_node >= 0) {
    PinCurrentThreadToNumaNode(nic_opt->numa_node);
  }
  return 0;
}

}  // namespace tpu_raiden
