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

#include "absl/base/log_severity.h"
#include "absl/log/scoped_mock_log.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
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

  // The resolved node must be one the PCI scan actually reported for a TPU
  // device. This is stronger and less host-specific than a hardcoded 0..1
  // bound: it also catches a bus index that lands on the wrong entry.
  std::vector<int> tpu_numa_nodes;
  for (const auto& dev : GetTpuPciDevices()) {
    if (std::find(tpu_numa_nodes.begin(), tpu_numa_nodes.end(),
                  dev.numa_node) == tpu_numa_nodes.end()) {
      tpu_numa_nodes.push_back(dev.numa_node);
    }
  }
  ASSERT_FALSE(tpu_numa_nodes.empty());

  int resolved_nodes = 0;
  for (auto* device : all_devices) {
    int numa_node = GetPjRtDeviceNumaNode(device);
    LOG(INFO) << "  PJRT Device " << device->DebugString()
              << " (local_hardware_id: " << device->local_hardware_id().value()
              << ") is mapped to NUMA Node: " << numa_node;

    EXPECT_GE(numa_node, 0);
    EXPECT_NE(std::find(tpu_numa_nodes.begin(), tpu_numa_nodes.end(),
                        numa_node),
              tpu_numa_nodes.end())
        << "Resolved NUMA node " << numa_node
        << " is not one of the nodes reported by any TPU PCI device";
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

TEST(TpuUtilsTest, GetLocalHostNicAddresses_AuthoritativeAllowlistAndDocker) {
  namespace fs = std::filesystem;
  std::string temp_dir_str = testing::TempDir();
  fs::path sysfs = fs::path(temp_dir_str) / "mock_sysfs_allowlist";
  fs::remove_all(sysfs);

  // Setup sysfs
  fs::create_directories(sysfs / "class/net/eth0");
  fs::create_directories(sysfs / "class/net/eth1");
  fs::create_directories(sysfs / "class/net/eth2");
  fs::create_directories(sysfs / "class/net/docker0");
  fs::create_directories(sysfs / "devices/system/node/node0");
  fs::create_directories(sysfs / "devices/system/node/node1");
  fs::create_directories(sysfs / "devices/pci0000:00/0000:00:01.0");
  fs::create_directories(sysfs / "devices/pci0000:00/0000:00:02.0");

  fs::create_directory_symlink("../../../devices/pci0000:00/0000:00:01.0",
                               sysfs / "class/net/eth1/device");
  fs::create_directory_symlink("../../../devices/pci0000:00/0000:00:02.0",
                               sysfs / "class/net/eth2/device");

  // Write MTUs
  { std::ofstream(sysfs / "class/net/eth0/mtu") << "1500\n"; }
  { std::ofstream(sysfs / "class/net/eth1/mtu") << "1460\n"; }
  { std::ofstream(sysfs / "class/net/eth2/mtu") << "1460\n"; }
  { std::ofstream(sysfs / "class/net/docker0/mtu") << "1500\n"; }

  // ifaddrs: docker0, eth2, eth1, eth0, lo
  sockaddr_in addr_docker0 = CreateSockAddr("172.17.0.1");
  sockaddr_in addr_eth2 = CreateSockAddr("10.0.0.3");
  sockaddr_in addr_eth1 = CreateSockAddr("10.0.0.2");
  sockaddr_in addr_eth0 = CreateSockAddr("10.0.0.1");
  sockaddr_in addr_lo = CreateSockAddr("127.0.0.1");

  ifaddrs ifa_docker0 = {nullptr, const_cast<char*>("docker0"),
                         0,       reinterpret_cast<sockaddr*>(&addr_docker0),
                         nullptr, {nullptr},
                         nullptr};
  ifaddrs ifa_eth2 = {&ifa_docker0, const_cast<char*>("eth2"),
                      0,            reinterpret_cast<sockaddr*>(&addr_eth2),
                      nullptr,      {nullptr},
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

  // Test 1: Without TPU_RAIDEN_DATA_NICS, auto-discovery relies solely on the
  // BDF filter and the MTU > 1500 rule. docker0 has no BDF, and eth1/eth2 sit
  // at MTU 1460, so nothing qualifies as a data rail.
  unsetenv("TPU_RAIDEN_DATA_NICS");
  auto nics1 =
      internal::GetLocalHostNicAddressesInternal(&ifa_lo, sysfs.string());
  auto it_dock = std::find_if(
      nics1.begin(), nics1.end(),
      [](const HostNicAddress& n) { return n.interface_name == "docker0"; });
  ASSERT_NE(it_dock, nics1.end());
  // No PCI BDF -> control plane.
  EXPECT_EQ(it_dock->classification, NicClassification::kControlPlane);

  auto it1_eth1 = std::find_if(
      nics1.begin(), nics1.end(),
      [](const HostNicAddress& n) { return n.interface_name == "eth1"; });
  ASSERT_NE(it1_eth1, nics1.end());
  // Has a BDF but MTU 1460 -> control plane. This NIC used to be misclassified
  // as data plane purely because it holds no IPv4 default route.
  EXPECT_EQ(it1_eth1->classification, NicClassification::kControlPlane);

  auto it1_eth2 = std::find_if(
      nics1.begin(), nics1.end(),
      [](const HostNicAddress& n) { return n.interface_name == "eth2"; });
  ASSERT_NE(it1_eth2, nics1.end());
  EXPECT_EQ(it1_eth2->classification, NicClassification::kControlPlane);

  // Test 2: With TPU_RAIDEN_DATA_NICS="eth1", ONLY eth1 is data plane.
  // eth2 must NOT fall through to data plane!
  setenv("TPU_RAIDEN_DATA_NICS", "eth1", 1);
  auto nics2 =
      internal::GetLocalHostNicAddressesInternal(&ifa_lo, sysfs.string());
  unsetenv("TPU_RAIDEN_DATA_NICS");

  auto it2_eth1 = std::find_if(
      nics2.begin(), nics2.end(),
      [](const HostNicAddress& n) { return n.interface_name == "eth1"; });
  ASSERT_NE(it2_eth1, nics2.end());
  // The allowlist overrides the MTU rule that would otherwise reject MTU 1460.
  EXPECT_EQ(it2_eth1->classification, NicClassification::kDataPlane);

  auto it2_eth2 = std::find_if(
      nics2.begin(), nics2.end(),
      [](const HostNicAddress& n) { return n.interface_name == "eth2"; });
  ASSERT_NE(it2_eth2, nics2.end());
  // CRITICAL: eth2 was NOT in allowlist -> must be control plane!
  EXPECT_EQ(it2_eth2->classification, NicClassification::kControlPlane);

  // Test 3: Allowlist with leading/trailing whitespace and empty tokens.
  setenv("TPU_RAIDEN_DATA_NICS", " eth1 , , eth2 ", 1);
  auto nics3 =
      internal::GetLocalHostNicAddressesInternal(&ifa_lo, sysfs.string());
  unsetenv("TPU_RAIDEN_DATA_NICS");

  auto it3_eth1 = std::find_if(
      nics3.begin(), nics3.end(),
      [](const HostNicAddress& n) { return n.interface_name == "eth1"; });
  ASSERT_NE(it3_eth1, nics3.end());
  EXPECT_EQ(it3_eth1->classification, NicClassification::kDataPlane);

  auto it3_eth2 = std::find_if(
      nics3.begin(), nics3.end(),
      [](const HostNicAddress& n) { return n.interface_name == "eth2"; });
  ASSERT_NE(it3_eth2, nics3.end());
  EXPECT_EQ(it3_eth2->classification, NicClassification::kDataPlane);

  auto it3_eth0 = std::find_if(
      nics3.begin(), nics3.end(),
      [](const HostNicAddress& n) { return n.interface_name == "eth0"; });
  ASSERT_NE(it3_eth0, nics3.end());
  EXPECT_EQ(it3_eth0->classification, NicClassification::kControlPlane);

  // Test 4: Whitespace-only TPU_RAIDEN_DATA_NICS falls back to heuristic
  // discovery. With the default-route rule gone, the MTU-1460 eth1/eth2 and
  // the BDF-less docker0 are all control plane.
  setenv("TPU_RAIDEN_DATA_NICS", "   ", 1);
  auto nics4 =
      internal::GetLocalHostNicAddressesInternal(&ifa_lo, sysfs.string());
  unsetenv("TPU_RAIDEN_DATA_NICS");

  auto it4_eth1 = std::find_if(
      nics4.begin(), nics4.end(),
      [](const HostNicAddress& n) { return n.interface_name == "eth1"; });
  ASSERT_NE(it4_eth1, nics4.end());
  EXPECT_EQ(it4_eth1->classification, NicClassification::kControlPlane);

  auto it4_eth2 = std::find_if(
      nics4.begin(), nics4.end(),
      [](const HostNicAddress& n) { return n.interface_name == "eth2"; });
  ASSERT_NE(it4_eth2, nics4.end());
  EXPECT_EQ(it4_eth2->classification, NicClassification::kControlPlane);

  auto it4_dock = std::find_if(
      nics4.begin(), nics4.end(),
      [](const HostNicAddress& n) { return n.interface_name == "docker0"; });
  ASSERT_NE(it4_dock, nics4.end());
  EXPECT_EQ(it4_dock->classification, NicClassification::kControlPlane);

  // Test 5: The allowlist is authoritative even over the BDF filter, so an
  // explicitly listed interface with no PCI BDF is still a data rail.
  setenv("TPU_RAIDEN_DATA_NICS", "docker0", 1);
  auto nics5 =
      internal::GetLocalHostNicAddressesInternal(&ifa_lo, sysfs.string());
  unsetenv("TPU_RAIDEN_DATA_NICS");

  auto it5_dock = std::find_if(
      nics5.begin(), nics5.end(),
      [](const HostNicAddress& n) { return n.interface_name == "docker0"; });
  ASSERT_NE(it5_dock, nics5.end());
  EXPECT_EQ(it5_dock->classification, NicClassification::kDataPlane);

  auto it5_eth1 = std::find_if(
      nics5.begin(), nics5.end(),
      [](const HostNicAddress& n) { return n.interface_name == "eth1"; });
  ASSERT_NE(it5_eth1, nics5.end());
  EXPECT_EQ(it5_eth1->classification, NicClassification::kControlPlane);

  fs::remove_all(sysfs);
}

TEST(TpuUtilsTest, GetLocalHostNicAddresses_AllowlistBraceSyntax) {
  namespace fs = std::filesystem;
  fs::path sysfs = fs::path(testing::TempDir()) / "mock_sysfs_brace_syntax";
  fs::remove_all(sysfs);

  // eth0: no BDF. eth1/eth2: PCI BDFs with sysfs numa_node=-1 so the allowlist
  // position (or an explicit :N suffix) decides their NUMA node.
  fs::create_directories(sysfs / "class/net/eth0");
  fs::create_directories(sysfs / "class/net/eth1");
  fs::create_directories(sysfs / "class/net/eth2");
  fs::create_directories(sysfs / "devices/system/node/node0");
  fs::create_directories(sysfs / "devices/system/node/node1");
  fs::create_directories(sysfs / "devices/pci0000:00/0000:00:01.0");
  fs::create_directories(sysfs / "devices/pci0000:00/0000:00:02.0");
  fs::create_directory_symlink("../../../devices/pci0000:00/0000:00:01.0",
                               sysfs / "class/net/eth1/device");
  fs::create_directory_symlink("../../../devices/pci0000:00/0000:00:02.0",
                               sysfs / "class/net/eth2/device");
  {
    std::ofstream(sysfs / "devices/pci0000:00/0000:00:01.0/numa_node")
        << "-1\n";
  }
  {
    std::ofstream(sysfs / "devices/pci0000:00/0000:00:02.0/numa_node")
        << "-1\n";
  }
  {
    std::ofstream(sysfs / "class/net/eth0/mtu") << "1500\n";
  }
  {
    std::ofstream(sysfs / "class/net/eth1/mtu") << "1460\n";
  }
  {
    std::ofstream(sysfs / "class/net/eth2/mtu") << "1460\n";
  }

  sockaddr_in addr_eth2 = CreateSockAddr("10.0.0.3");
  sockaddr_in addr_eth1 = CreateSockAddr("10.0.0.2");
  sockaddr_in addr_eth0 = CreateSockAddr("10.0.0.1");
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

  auto discover = [&](const char* allowlist) {
    setenv("TPU_RAIDEN_DATA_NICS", allowlist, 1);
    auto nics =
        internal::GetLocalHostNicAddressesInternal(&ifa_eth0, sysfs.string());
    unsetenv("TPU_RAIDEN_DATA_NICS");
    return nics;
  };
  auto find_nic = [](const std::vector<HostNicAddress>& nics,
                     absl::string_view name) {
    auto it = std::find_if(
        nics.begin(), nics.end(),
        [&](const HostNicAddress& n) { return n.interface_name == name; });
    EXPECT_NE(it, nics.end()) << "missing interface " << name;
    return it;
  };

  // 1. Curly braces around the whole list are stripped; the entries inside
  // behave exactly like a plain comma-separated allowlist.
  {
    auto nics = discover("{eth1,eth2}");
    ASSERT_EQ(nics.size(), 3);
    EXPECT_EQ(find_nic(nics, "eth0")->classification,
              NicClassification::kControlPlane);
    EXPECT_EQ(find_nic(nics, "eth1")->classification,
              NicClassification::kDataPlane);
    EXPECT_EQ(find_nic(nics, "eth2")->classification,
              NicClassification::kDataPlane);
  }

  // 2. Square brackets with interior whitespace are accepted the same way.
  {
    auto nics = discover("[ eth1 , eth2 ]");
    ASSERT_EQ(nics.size(), 3);
    EXPECT_EQ(find_nic(nics, "eth0")->classification,
              NicClassification::kControlPlane);
    EXPECT_EQ(find_nic(nics, "eth1")->classification,
              NicClassification::kDataPlane);
    EXPECT_EQ(find_nic(nics, "eth2")->classification,
              NicClassification::kDataPlane);
  }

  // 3. The explicit name:numa suffix still works inside braces and overrides
  // the positional index (which would give eth1 -> 0, eth2 -> 1).
  {
    auto nics = discover("{eth1:1,eth2:0}");
    ASSERT_EQ(nics.size(), 3);
    auto it_eth1 = find_nic(nics, "eth1");
    EXPECT_EQ(it_eth1->classification, NicClassification::kDataPlane);
    EXPECT_EQ(it_eth1->numa_node, 1);
    auto it_eth2 = find_nic(nics, "eth2");
    EXPECT_EQ(it_eth2->classification, NicClassification::kDataPlane);
    EXPECT_EQ(it_eth2->numa_node, 0);
  }

  // 4. An unbalanced brace is NOT guessed at: the token "{eth1" is matched
  // literally, matches nothing, and the authoritative allowlist demotes every
  // interface to control plane.
  {
    auto nics = discover("{eth1");
    ASSERT_EQ(nics.size(), 3);
    EXPECT_EQ(find_nic(nics, "eth0")->classification,
              NicClassification::kControlPlane);
    EXPECT_EQ(find_nic(nics, "eth1")->classification,
              NicClassification::kControlPlane);
    EXPECT_EQ(find_nic(nics, "eth2")->classification,
              NicClassification::kControlPlane);
  }

  // 5. Each allowlist entry that matches no discovered interface is named in
  // a warning; a fully matched allowlist emits no such warning.
  {
    absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
    EXPECT_CALL(
        log,
        Log(absl::LogSeverity::kWarning, testing::_,
            testing::AllOf(testing::HasSubstr("\"eth9\""),
                           testing::HasSubstr("matches no local interface"),
                           testing::HasSubstr("discovered: [eth0,eth1,eth2]"))))
        .Times(1);
    log.StartCapturingLogs();
    auto nics = discover("{eth0,eth9}");
    EXPECT_EQ(find_nic(nics, "eth0")->classification,
              NicClassification::kDataPlane);
  }
  {
    absl::ScopedMockLog log(absl::MockLogDefault::kIgnoreUnexpected);
    EXPECT_CALL(log, Log(absl::LogSeverity::kWarning, testing::_,
                         testing::HasSubstr("matches no local interface")))
        .Times(0);
    log.StartCapturingLogs();
    auto nics = discover("{eth1,eth2}");
    EXPECT_EQ(find_nic(nics, "eth1")->classification,
              NicClassification::kDataPlane);
  }

  fs::remove_all(sysfs);
}

TEST(TpuUtilsTest, ContainerNoSysfs_AllowlistPositionalNumaFallback) {
  namespace fs = std::filesystem;
  std::string missing_sysfs =
      (fs::path(testing::TempDir()) / "nonexistent_sysfs").string();
  fs::remove_all(missing_sysfs);

  sockaddr_in addr_ens5 = CreateSockAddr("10.128.0.10");
  sockaddr_in addr_ens6 = CreateSockAddr("10.10.0.10");
  sockaddr_in addr_enp192s4 = CreateSockAddr("10.190.0.10");
  sockaddr_in addr_lo = CreateSockAddr("127.0.0.1");

  ifaddrs ifa_enp192s4 = {nullptr, const_cast<char*>("enp192s4"),
                          0,       reinterpret_cast<sockaddr*>(&addr_enp192s4),
                          nullptr, {nullptr},
                          nullptr};
  ifaddrs ifa_ens6 = {&ifa_enp192s4,
                      const_cast<char*>("ens6"),
                      0,
                      reinterpret_cast<sockaddr*>(&addr_ens6),
                      nullptr,
                      {nullptr},
                      nullptr};
  ifaddrs ifa_ens5 = {&ifa_ens6, const_cast<char*>("ens5"),
                      0,         reinterpret_cast<sockaddr*>(&addr_ens5),
                      nullptr,   {nullptr},
                      nullptr};
  ifaddrs ifa_lo = {&ifa_ens5, const_cast<char*>("lo"),
                    0,         reinterpret_cast<sockaddr*>(&addr_lo),
                    nullptr,   {nullptr},
                    nullptr};

  // 1. With TPU_RAIDEN_DATA_NICS="ens6,enp192s4" and no sysfs, positional
  // allowlist index assigns ens6 -> NUMA 0 and enp192s4 -> NUMA 1.
  setenv("TPU_RAIDEN_DATA_NICS", "ens6,enp192s4", 1);
  auto nics =
      internal::GetLocalHostNicAddressesInternal(&ifa_lo, missing_sysfs);
  unsetenv("TPU_RAIDEN_DATA_NICS");

  ASSERT_EQ(nics.size(), 3);

  auto it_ens5 = std::find_if(
      nics.begin(), nics.end(),
      [](const HostNicAddress& n) { return n.interface_name == "ens5"; });
  ASSERT_NE(it_ens5, nics.end());
  EXPECT_EQ(it_ens5->classification, NicClassification::kControlPlane);
  EXPECT_EQ(it_ens5->numa_node, 0);

  auto it_ens6 = std::find_if(
      nics.begin(), nics.end(),
      [](const HostNicAddress& n) { return n.interface_name == "ens6"; });
  ASSERT_NE(it_ens6, nics.end());
  EXPECT_EQ(it_ens6->classification, NicClassification::kDataPlane);
  EXPECT_EQ(it_ens6->numa_node, 0);

  auto it_enp = std::find_if(
      nics.begin(), nics.end(),
      [](const HostNicAddress& n) { return n.interface_name == "enp192s4"; });
  ASSERT_NE(it_enp, nics.end());
  EXPECT_EQ(it_enp->classification, NicClassification::kDataPlane);
  EXPECT_EQ(it_enp->numa_node, 1);

  // 2. When TPU_RAIDEN_DATA_NICS is unset, GetInterfaceNumaNode legacy
  // fallback remains intact (ens5 -> 0, ens6 -> 1, enp192s4 -> -1).
  auto nics_unset =
      internal::GetLocalHostNicAddressesInternal(&ifa_lo, missing_sysfs);
  auto it_unset_ens5 = std::find_if(
      nics_unset.begin(), nics_unset.end(),
      [](const HostNicAddress& n) { return n.interface_name == "ens5"; });
  ASSERT_NE(it_unset_ens5, nics_unset.end());
  EXPECT_EQ(it_unset_ens5->numa_node, 0);

  auto it_unset_ens6 = std::find_if(
      nics_unset.begin(), nics_unset.end(),
      [](const HostNicAddress& n) { return n.interface_name == "ens6"; });
  ASSERT_NE(it_unset_ens6, nics_unset.end());
  EXPECT_EQ(it_unset_ens6->numa_node, 1);

  auto it_unset_enp = std::find_if(
      nics_unset.begin(), nics_unset.end(),
      [](const HostNicAddress& n) { return n.interface_name == "enp192s4"; });
  ASSERT_NE(it_unset_enp, nics_unset.end());
  EXPECT_EQ(it_unset_enp->numa_node, -1);
}

TEST(TpuUtilsTest, ContainerNoSysfs_AllowlistExplicitNumaOverride) {
  namespace fs = std::filesystem;
  std::string missing_sysfs =
      (fs::path(testing::TempDir()) / "nonexistent_sysfs_explicit").string();
  fs::remove_all(missing_sysfs);

  sockaddr_in addr_ens6 = CreateSockAddr("10.10.0.10");
  sockaddr_in addr_enp192s4 = CreateSockAddr("10.190.0.10");

  ifaddrs ifa_enp192s4 = {nullptr, const_cast<char*>("enp192s4"),
                          0,       reinterpret_cast<sockaddr*>(&addr_enp192s4),
                          nullptr, {nullptr},
                          nullptr};
  ifaddrs ifa_ens6 = {&ifa_enp192s4,
                      const_cast<char*>("ens6"),
                      0,
                      reinterpret_cast<sockaddr*>(&addr_ens6),
                      nullptr,
                      {nullptr},
                      nullptr};

  setenv("TPU_RAIDEN_DATA_NICS", " ens6 : 1 , enp192s4 : 0 ", 1);
  auto nics =
      internal::GetLocalHostNicAddressesInternal(&ifa_ens6, missing_sysfs);
  unsetenv("TPU_RAIDEN_DATA_NICS");

  ASSERT_EQ(nics.size(), 2);

  auto it_ens6 = std::find_if(
      nics.begin(), nics.end(),
      [](const HostNicAddress& n) { return n.interface_name == "ens6"; });
  ASSERT_NE(it_ens6, nics.end());
  EXPECT_EQ(it_ens6->classification, NicClassification::kDataPlane);
  EXPECT_EQ(it_ens6->numa_node, 1);

  auto it_enp = std::find_if(
      nics.begin(), nics.end(),
      [](const HostNicAddress& n) { return n.interface_name == "enp192s4"; });
  ASSERT_NE(it_enp, nics.end());
  EXPECT_EQ(it_enp->classification, NicClassification::kDataPlane);
  EXPECT_EQ(it_enp->numa_node, 0);

  // Verify TPU_RAIDEN_SYSFS_ROOT actually overrides the default "/sys" path
  // in GetLocalHostNicAddresses().
  auto base_nics = GetLocalHostNicAddresses();
  ASSERT_FALSE(base_nics.empty());
  if (base_nics[0].interface_name != "lo") {
    fs::path custom_sysfs =
        fs::path(testing::TempDir()) / "mock_sysfs_env_override";
    fs::remove_all(custom_sysfs);
    fs::path dev_dir =
        custom_sysfs / "class/net" / base_nics[0].interface_name / "device";
    fs::create_directories(dev_dir);
    {
      std::ofstream f(dev_dir / "numa_node");
      f << "7\n";
    }
    setenv("TPU_RAIDEN_SYSFS_ROOT", custom_sysfs.c_str(), 1);
    auto overridden_nics = GetLocalHostNicAddresses();
    unsetenv("TPU_RAIDEN_SYSFS_ROOT");
    auto it_first = std::find_if(overridden_nics.begin(), overridden_nics.end(),
                                 [&](const HostNicAddress& n) {
                                   return n.interface_name ==
                                          base_nics[0].interface_name;
                                 });
    ASSERT_NE(it_first, overridden_nics.end());
    EXPECT_EQ(it_first->numa_node, 7);
    fs::remove_all(custom_sysfs);
  }
}

TEST(TpuUtilsTest,
     SysfsPresent_AllowlistPreservesSysfsNumaAndSupportsOverrides) {
  namespace fs = std::filesystem;
  fs::path sysfs =
      fs::path(testing::TempDir()) / "mock_sysfs_precedence_and_bdf";
  fs::remove_all(sysfs);

  fs::create_directories(sysfs / "class/net/ens6");
  fs::create_directories(sysfs / "class/net/enp192s4");
  fs::create_directories(sysfs / "devices/pci0000:00/0000:00:06.0");
  fs::create_directories(sysfs / "devices/pci0000:c0/0000:c0:04.0");

  fs::create_directory_symlink("../../../devices/pci0000:00/0000:00:06.0",
                               sysfs / "class/net/ens6/device");
  fs::create_directory_symlink("../../../devices/pci0000:c0/0000:c0:04.0",
                               sysfs / "class/net/enp192s4/device");

  // Populate sysfs with ens6 -> NUMA 1 and enp192s4 -> NUMA 0 (opposite of
  // their 0-based allowlist order).
  {
    std::ofstream f(sysfs / "devices/pci0000:00/0000:00:06.0/numa_node");
    f << "1\n";
  }
  {
    std::ofstream f(sysfs / "devices/pci0000:c0/0000:c0:04.0/numa_node");
    f << "0\n";
  }

  sockaddr_in addr_ens6 = CreateSockAddr("10.10.0.10");
  sockaddr_in addr_enp192s4 = CreateSockAddr("10.190.0.10");

  ifaddrs ifa_enp192s4 = {nullptr, const_cast<char*>("enp192s4"),
                          0,       reinterpret_cast<sockaddr*>(&addr_enp192s4),
                          nullptr, {nullptr},
                          nullptr};
  ifaddrs ifa_ens6 = {&ifa_enp192s4,
                      const_cast<char*>("ens6"),
                      0,
                      reinterpret_cast<sockaddr*>(&addr_ens6),
                      nullptr,
                      {nullptr},
                      nullptr};

  // 1. When sysfs device/numa_node >= 0 and no :N suffix is given, sysfs NUMA
  // takes precedence over positional allowlist index.
  setenv("TPU_RAIDEN_DATA_NICS", "ens6,enp192s4", 1);
  auto nics_sysfs =
      internal::GetLocalHostNicAddressesInternal(&ifa_ens6, sysfs.string());
  unsetenv("TPU_RAIDEN_DATA_NICS");

  ASSERT_EQ(nics_sysfs.size(), 2);
  auto it1_ens6 = std::find_if(
      nics_sysfs.begin(), nics_sysfs.end(),
      [](const HostNicAddress& n) { return n.interface_name == "ens6"; });
  ASSERT_NE(it1_ens6, nics_sysfs.end());
  EXPECT_EQ(it1_ens6->classification, NicClassification::kDataPlane);
  EXPECT_EQ(it1_ens6->numa_node, 1);

  auto it1_enp = std::find_if(
      nics_sysfs.begin(), nics_sysfs.end(),
      [](const HostNicAddress& n) { return n.interface_name == "enp192s4"; });
  ASSERT_NE(it1_enp, nics_sysfs.end());
  EXPECT_EQ(it1_enp->classification, NicClassification::kDataPlane);
  EXPECT_EQ(it1_enp->numa_node, 0);

  // 2. Explicit :N suffix overrides valid sysfs device/numa_node, including
  // when matched by PCI BDF.
  setenv("TPU_RAIDEN_DATA_NICS", "0000:00:06.0:0,0000:c0:04.0:1", 1);
  auto nics_bdf_override =
      internal::GetLocalHostNicAddressesInternal(&ifa_ens6, sysfs.string());
  unsetenv("TPU_RAIDEN_DATA_NICS");

  ASSERT_EQ(nics_bdf_override.size(), 2);
  auto it2_ens6 = std::find_if(
      nics_bdf_override.begin(), nics_bdf_override.end(),
      [](const HostNicAddress& n) { return n.interface_name == "ens6"; });
  ASSERT_NE(it2_ens6, nics_bdf_override.end());
  EXPECT_EQ(it2_ens6->classification, NicClassification::kDataPlane);
  EXPECT_EQ(it2_ens6->numa_node, 0);

  auto it2_enp = std::find_if(
      nics_bdf_override.begin(), nics_bdf_override.end(),
      [](const HostNicAddress& n) { return n.interface_name == "enp192s4"; });
  ASSERT_NE(it2_enp, nics_bdf_override.end());
  EXPECT_EQ(it2_enp->classification, NicClassification::kDataPlane);
  EXPECT_EQ(it2_enp->numa_node, 1);

  fs::remove_all(sysfs);
}

TEST(TpuUtilsTest, SelectPrimaryHostIpTest) {
  // Empty span returns default fallback ("unknown") or custom fallback.
  EXPECT_EQ(SelectPrimaryHostIp({}), "unknown");
  EXPECT_EQ(SelectPrimaryHostIp({}, "127.0.0.1"), "127.0.0.1");

  // All NICs have empty ip_address -> returns fallback.
  std::vector<HostNicAddress> empty_ip_nics = {
      {"eth0", "", 0, NicClassification::kDataPlane},
      {"eth1", "", 0, NicClassification::kControlPlane},
      {"eth2", "", 0, NicClassification::kUnknown},
  };
  EXPECT_EQ(SelectPrimaryHostIp(empty_ip_nics), "unknown");
  EXPECT_EQ(SelectPrimaryHostIp(empty_ip_nics, "fallback_ip"), "fallback_ip");

  // Prefers first kDataPlane NIC with non-empty IP over earlier
  // control/unknown or empty-IP data-plane NICs.
  std::vector<HostNicAddress> mixed_nics = {
      {"eth0", "10.0.0.1", 0, NicClassification::kControlPlane},
      {"eth1", "", 0, NicClassification::kDataPlane},
      {"eth2", "192.168.1.10", 0, NicClassification::kDataPlane},
      {"eth3", "192.168.1.11", 1, NicClassification::kDataPlane},
  };
  EXPECT_EQ(SelectPrimaryHostIp(mixed_nics), "192.168.1.10");

  // Falls back to first kControlPlane NIC with non-empty IP when no
  // data-plane NIC has a non-empty IP.
  std::vector<HostNicAddress> control_and_unknown_nics = {
      {"eth0", "172.16.0.1", -1, NicClassification::kUnknown},
      {"eth1", "", 0, NicClassification::kControlPlane},
      {"eth2", "10.0.0.2", 0, NicClassification::kControlPlane},
      {"eth3", "10.0.0.3", 0, NicClassification::kControlPlane},
  };
  EXPECT_EQ(SelectPrimaryHostIp(control_and_unknown_nics), "10.0.0.2");

  // Falls back to first NIC with non-empty IP (e.g. kUnknown) when neither
  // kDataPlane nor kControlPlane has a non-empty IP.
  std::vector<HostNicAddress> unknown_only_nics = {
      {"eth0", "", -1, NicClassification::kUnknown},
      {"eth1", "172.16.0.5", -1, NicClassification::kUnknown},
      {"eth2", "172.16.0.6", -1, NicClassification::kUnknown},
  };
  EXPECT_EQ(SelectPrimaryHostIp(unknown_only_nics), "172.16.0.5");
}

TEST(TpuUtilsTest, GetCachedLocalHostNicAddressesAndPrimaryIpTest) {
  const auto& cached_nics_1 = GetCachedLocalHostNicAddresses();
  const auto& cached_nics_2 = GetCachedLocalHostNicAddresses();
  EXPECT_EQ(&cached_nics_1, &cached_nics_2);
  EXPECT_FALSE(cached_nics_1.empty());

  const std::string& primary_ip_1 = GetLocalHostPrimaryIp();
  const std::string& primary_ip_2 = GetLocalHostPrimaryIp();
  EXPECT_EQ(&primary_ip_1, &primary_ip_2);
  EXPECT_FALSE(primary_ip_1.empty());
  EXPECT_EQ(primary_ip_1, SelectPrimaryHostIp(cached_nics_1));
}

}  // namespace
}  // namespace tpu_raiden
