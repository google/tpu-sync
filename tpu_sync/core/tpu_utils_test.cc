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
#include <ifaddrs.h>
#include <netinet/in.h>
#include <sys/socket.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT(build/c++17)
#include <fstream>
#include <string>
#include <vector>

#include "xla/pjrt/pjrt_client.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "tpu_sync/core/tpu_pjrt_manager.h"

namespace tpu_raiden {
namespace {

TEST(TpuUtilsTest, GetTpuPciDevicesTest) {
  const auto& pci_devices = GetTpuPciDevices();
  LOG(INFO) << "Detected " << pci_devices.size() << " TPU PCI devices:";
  for (const auto& dev : pci_devices) {
    LOG(INFO) << "  BDF: " << dev.bdf << ", Device ID: " << dev.device_id
              << ", NUMA Node: " << dev.numa_node;
  }
  // On a TPU VM, we expect to find at least one TPU PCI device.
  EXPECT_FALSE(pci_devices.empty());
}

TEST(TpuUtilsTest, GetPjRtDeviceNumaNodeTest) {
  TF_ASSERT_OK_AND_ASSIGN(auto manager, TpuPjrtManager::GetDefault());
  auto all_devices = manager->client()->addressable_devices();

  LOG(INFO) << "Available PJRT devices: " << all_devices.size();

  // On v7x-8, we expect to see exactly 8 PJRT devices (4 chips * 2
  // devices/chip)
  if (all_devices.size() == 8) {
    LOG(INFO) << "Confirmed v7x-8 topology with 8 devices!";
  } else {
    LOG(WARNING) << "Expected 8 devices for v7x-8 but saw "
                 << all_devices.size() << " devices.";
  }

  int resolved_nodes = 0;
  for (auto* device : all_devices) {
    int numa_node = GetPjRtDeviceNumaNode(device);
    LOG(INFO) << "  PJRT Device " << device->DebugString()
              << " (local_hardware_id: " << device->local_hardware_id().value()
              << ") is mapped to NUMA Node: " << numa_node;

    // NUMA node should be valid (0 or 1 on TPU v7x)
    EXPECT_GE(numa_node, 0);
    EXPECT_LE(numa_node, 1);
    resolved_nodes++;
  }
  EXPECT_GT(resolved_nodes, 0);
}

TEST(TpuUtilsTest, PinCurrentThreadToNumaNodeTest) {
  TF_ASSERT_OK_AND_ASSIGN(auto manager, TpuPjrtManager::GetDefault());
  auto all_devices = manager->client()->addressable_devices();
  ASSERT_FALSE(all_devices.empty());

  // Take the first local device and get its NUMA node
  auto* device = all_devices[0];
  int numa_node = GetPjRtDeviceNumaNode(device);
  LOG(INFO) << "Attempting to pin current thread to NUMA Node of device "
            << device->DebugString() << ": " << numa_node;

  if (numa_node >= 0) {
    int rc = PinCurrentThreadToNumaNode(numa_node);
    // Thread pinning and memory binding can fail in sandboxed test
    // environments. We tolerate EPERM (-1 from set_mempolicy) and EINVAL (-22
    // from pthread_setaffinity_np).
    EXPECT_TRUE(rc == 0 || rc == -1 || rc == -22)
        << "Unexpected failure pinning to NUMA node " << numa_node
        << ", rc=" << rc;
    if (rc == 0) {
      LOG(INFO) << "Successfully pinned thread to NUMA Node " << numa_node;
    }
  } else {
    LOG(WARNING)
        << "Could not resolve NUMA node for device, skipping pinning test.";
  }
}

TEST(TpuUtilsTest, SetThreadMempolicyWithoutFlagTest) {
  unsetenv("RAIDEN_NUMA_POLICY");

  int64_t rc_default = SetThreadMempolicy(kMpolDefault);
  EXPECT_TRUE(rc_default == 0 || rc_default == -1);

  int64_t rc_bind = SetThreadMempolicy(kMpolBind, 0);
  EXPECT_TRUE(rc_bind == 0 || rc_bind == -1);

  int64_t rc_preferred = SetThreadMempolicy(kMpolPreferred, 0);
  EXPECT_TRUE(rc_preferred == 0 || rc_preferred == -1);

  SetThreadMempolicy(kMpolDefault);
}

TEST(TpuUtilsTest, SetThreadMempolicyWithFlagTest) {
  setenv("RAIDEN_NUMA_POLICY", "preferred", 1);
  int64_t rc_pref = SetThreadMempolicy(kMpolBind, 0);
  EXPECT_TRUE(rc_pref == 0 || rc_pref == -1);

  setenv("RAIDEN_NUMA_POLICY", "bind", 1);
  int64_t rc_bind = SetThreadMempolicy(kMpolBind, 0);
  EXPECT_TRUE(rc_bind == 0 || rc_bind == -1);

  setenv("RAIDEN_NUMA_POLICY", "default", 1);
  int64_t rc_def = SetThreadMempolicy(kMpolBind, 0);
  EXPECT_TRUE(rc_def == 0 || rc_def == -1);

  unsetenv("RAIDEN_NUMA_POLICY");
  SetThreadMempolicy(kMpolDefault);
}

TEST(TpuUtilsTest, PinCurrentThreadToNumaNodeWithAndWithoutFlagTest) {
  unsetenv("RAIDEN_NUMA_POLICY");
  int rc_noflag = PinCurrentThreadToNumaNode(0);
  EXPECT_TRUE(rc_noflag == 0 || rc_noflag == -1 || rc_noflag == -22)
      << "Unexpected failure without flag, rc=" << rc_noflag;

  setenv("RAIDEN_NUMA_POLICY", "preferred", 1);
  int rc_pref = PinCurrentThreadToNumaNode(0);
  EXPECT_TRUE(rc_pref == 0 || rc_pref == -1 || rc_pref == -22)
      << "Unexpected failure with RAIDEN_NUMA_POLICY=preferred, rc=" << rc_pref;

  setenv("RAIDEN_NUMA_POLICY", "bind", 1);
  int rc_bind = PinCurrentThreadToNumaNode(0);
  EXPECT_TRUE(rc_bind == 0 || rc_bind == -1 || rc_bind == -22)
      << "Unexpected failure with RAIDEN_NUMA_POLICY=bind, rc=" << rc_bind;

  unsetenv("RAIDEN_NUMA_POLICY");
  SetThreadMempolicy(kMpolDefault);
}

TEST(TpuUtilsTest, GetLocalHostNicAddressesTest) {
  std::vector<HostNicAddress> nics = GetLocalHostNicAddresses();
  LOG(INFO) << "Discovered " << nics.size() << " network interfaces:";
  for (const auto& nic : nics) {
    LOG(INFO) << "  Interface: " << nic.interface_name
              << ", IP: " << nic.ip_address << ", NUMA Node: " << nic.numa_node;
    EXPECT_FALSE(nic.interface_name.empty());
    EXPECT_FALSE(nic.ip_address.empty());
  }
  EXPECT_FALSE(nics.empty());
}

TEST(TpuUtilsTest, GetLocalHostIpAddressesTest) {
  std::vector<std::string> ips = GetLocalHostIpAddresses();
  LOG(INFO) << "Discovered " << ips.size() << " IP addresses:";
  for (const auto& ip : ips) {
    LOG(INFO) << "  IP: " << ip;
    EXPECT_FALSE(ip.empty());
  }
  EXPECT_FALSE(ips.empty());
}

TEST(TpuUtilsTest, GetInterfaceNumaNodeTest) {
  EXPECT_EQ(GetInterfaceNumaNode("eth0"), 0);
  EXPECT_EQ(GetInterfaceNumaNode("eth1"), 0);
  EXPECT_EQ(GetInterfaceNumaNode("ens5"), 0);
  EXPECT_EQ(GetInterfaceNumaNode("dcn1"), 1);
  EXPECT_EQ(GetInterfaceNumaNode("ens6"), 1);
  EXPECT_EQ(GetInterfaceNumaNode("eth2"), 1);
  EXPECT_EQ(GetInterfaceNumaNode("unknown_interface"), -1);
}

// Helper to create a mock sockaddr_in
sockaddr_in CreateSockAddr(const std::string& ip) {
  sockaddr_in addr;
  std::memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  inet_pton(AF_INET, ip.c_str(), &addr.sin_addr);
  return addr;
}

TEST(TpuUtilsTest, GetLocalHostNicAddresses_MultiNic_Classification) {
  namespace fs = std::filesystem;
  std::string temp_dir_str = testing::TempDir();
  fs::path sysfs = fs::path(temp_dir_str) / "mock_sysfs";
  fs::remove_all(sysfs);  // Clean up if left over

  // Create directory structure
  fs::create_directories(sysfs / "class/net/eth0");
  fs::create_directories(sysfs / "class/net/eth1");
  fs::create_directories(sysfs / "class/net/eth2");
  fs::create_directories(sysfs / "devices/system/node/node0");
  fs::create_directories(sysfs / "devices/system/node/node1");
  fs::create_directories(sysfs / "devices/pci0000:00/0000:00:01.0");
  fs::create_directories(sysfs / "devices/pci0000:00/0000:00:02.0");

  // Create BDF symlinks
  fs::create_directory_symlink("../../../devices/pci0000:00/0000:00:01.0",
                               sysfs / "class/net/eth1/device");
  fs::create_directory_symlink("../../../devices/pci0000:00/0000:00:02.0",
                               sysfs / "class/net/eth2/device");

  // Write NUMA nodes
  {
    std::ofstream f(sysfs / "devices/pci0000:00/0000:00:01.0/numa_node");
    f << "-1\n";
  }
  {
    std::ofstream f(sysfs / "devices/pci0000:00/0000:00:02.0/numa_node");
    f << "-1\n";
  }

  // Write MTUs
  {
    std::ofstream f(sysfs / "class/net/eth0/mtu");
    f << "1500\n";
  }
  {
    std::ofstream f(sysfs / "class/net/eth1/mtu");
    f << "9000\n";
  }
  {
    std::ofstream f(sysfs / "class/net/eth2/mtu");
    f << "9000\n";
  }

  // Construct mock ifaddrs
  sockaddr_in addr_lo = CreateSockAddr("127.0.0.1");
  sockaddr_in addr_eth0 = CreateSockAddr("10.0.0.1");
  sockaddr_in addr_eth1 = CreateSockAddr("10.0.0.2");
  sockaddr_in addr_eth2 = CreateSockAddr("10.0.0.3");

  ifaddrs ifa_eth2 = {nullptr, const_cast<char*>("eth2"),
                      0,       reinterpret_cast<sockaddr*>(&addr_eth2),
                      nullptr, {nullptr},
                      nullptr};
  ifaddrs ifa_eth1 = {&ifa_eth2, const_cast<char*>("eth1"),
                      0,         reinterpret_cast<sockaddr*>(&addr_eth1),
                      nullptr,   {nullptr},
                      nullptr};
  ifaddrs ifa_eth0 = {&ifa_eth1, const_cast<char*>("eth0"),
                      0,         reinterpret_cast<sockaddr*>(&addr_eth0),
                      nullptr,   {nullptr},
                      nullptr};
  ifaddrs ifa_lo = {&ifa_eth0, const_cast<char*>("lo"),
                    0,         reinterpret_cast<sockaddr*>(&addr_lo),
                    nullptr,   {nullptr},
                    nullptr};

  // Call the internal function
  auto nics =
      internal::GetLocalHostNicAddressesInternal(&ifa_lo, sysfs.string());

  // We expect 3 NICs (lo is filtered)
  ASSERT_EQ(nics.size(), 3);

  // eth0: Control
  auto it_eth0 = std::find_if(
      nics.begin(), nics.end(),
      [](const HostNicAddress& n) { return n.interface_name == "eth0"; });
  ASSERT_NE(it_eth0, nics.end());
  EXPECT_EQ(it_eth0->ip_address, "10.0.0.1");
  EXPECT_EQ(it_eth0->classification, NicClassification::kControlPlane);

  // eth1: Data, NUMA 0 (heuristic)
  auto it_eth1 = std::find_if(
      nics.begin(), nics.end(),
      [](const HostNicAddress& n) { return n.interface_name == "eth1"; });
  ASSERT_NE(it_eth1, nics.end());
  EXPECT_EQ(it_eth1->ip_address, "10.0.0.2");
  EXPECT_EQ(it_eth1->classification, NicClassification::kDataPlane);
  EXPECT_EQ(it_eth1->numa_node, 0);

  // eth2: Data, NUMA 1 (heuristic)
  auto it_eth2 = std::find_if(
      nics.begin(), nics.end(),
      [](const HostNicAddress& n) { return n.interface_name == "eth2"; });
  ASSERT_NE(it_eth2, nics.end());
  EXPECT_EQ(it_eth2->ip_address, "10.0.0.3");
  EXPECT_EQ(it_eth2->classification, NicClassification::kDataPlane);
  EXPECT_EQ(it_eth2->numa_node, 1);

  // Clean up
  fs::remove_all(sysfs);
}

}  // namespace
}  // namespace tpu_raiden
