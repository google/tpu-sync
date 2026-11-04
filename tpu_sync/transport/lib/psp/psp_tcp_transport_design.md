# TPU Raiden: PSP-TCP Transport Architecture & Design

This document details the PSP-TCP encryption architecture, socket
lifecycle, and out-of-band key exchange flow in the TPU Raiden transport
layer.

---

## 1. Overview

PSP (PSP Security Protocol) provides hardware/kernel-level cryptographic
encryption for TCP data transfers between TPU nodes.
Because raw TCP connections cannot exchange cryptographic keys in-band
without mutual authentication and coordination, TPU Raiden uses an
out-of-band control plane via gRPC (`PeregrineControlService`) to perform
the key exchange before establishing encrypted data streams.

---

## 2. Architecture & Connection Flow

```
+==========================================================================+
|                TPU RAIDEN: PSP-TCP TRANSPORT ARCHITECTURE                |
+==========================================================================+

  NODE A (Client / Initiator)                 NODE B (Server / Receiver)
 +----------------------------+              +----------------------------+
 |       BlockTransport       |              |       BlockTransport       |
 |   (SyncPush / SyncPull)    |              | -peregrine_control_service |
 +-------------+--------------+              +-------------+--------------+
               |                                           | Exposes Service
               v                                           v
 +----------------------------+              +----------------------------+
 |     RawBufferTransport     |              |    gRPC Server Builder     |
 |   BorrowConnection(peer)   |              | builder.RegisterService(   |
 +-------------+--------------+              |   transport.               |
               |                             |   peregrine_control_       |
               | (1) Channel =               |     service())             |
               |     raw_delegate_->         +-------------+--------------+
               |       GetPeregrineChannel                 |
               |         (peer)                            | Host gRPC
               v                                           | (Control Plane)
 +----------------------------+                            |
 |          ConnPool          |                            |
 |  Borrow(peer, ip, Channel) |                            |
 +-------------+--------------+                            |
               |                                           |
               v                                           |
 +----------------------------+                            |
 |       tcp_psp_helper       |                            |
 |   TcpPspConnect(sock_fd)   |                            |
 +-------------+--------------+                            |
               |                                           |
   (2) AcquireRxSpiAndKey                                  |
       (sock_fd) -> (client_                               |
       spi, client_key)                                    |
               |                                           |
               |                                           |
 ===============[ STEP 1: OUT-OF-BAND KEY EXCHANGE OVER gRPC ]==============
               |                                           |
               | (3) gRPC: ExchangePspKey(req)             |
               |     Request: (client_spi, client_key)     v
               +---------------------------------> +---------------+
               |                                   |  Peregrine    |
               |                                   |  Control      |
               |                                   |  ServiceImpl  |
               |                                   +-------+-------+
               |                                           |
               |                                   (4) [psp_mu_]
               |                                       RegisterPspPeer
               |                                           |
               |                                           v
               |                                   +---------------+
               |                                   | tcp_psp_      |
               |                                   | helper        |
               |                                   | RegisterPsp-  |
               |                                   | PeerKey       |
               |                                   +-------+-------+
               |                                           |
               |                                   (5) AddSecureListener
               |                                       (server_fd_,
               |                                        client_tx_key)
               |                                       -> (server_spi,
               |                                           server_key)
               |     Response: (server_spi, server_key)    |
               |<------------------------------------------+
               |
   (6) SetTxSpiAndKey(sock_fd,
       server_tx_key)
               |
               |
 ===============[ STEP 2: PSP-TCP CONNECTION & VERIFICATION ]===============
               |                                           |
   (7) connect(sock_fd, addr)                              |
               |                                           |
               | -------- TCP SYN (Client Initial SPI) --->|
               |                                           |
               | <------- TCP SYN-ACK (Server Initial SPI)-+
               |                                           |
   (8) Verify Client Rx SPI:                       (9) accept(server_fd_)
       GetInitialRxSpi == rx_spi                       -> client_fd
               |                                           |
               |                                   (10) Verify Server PSP:
               |                                        PspEnabled(fd)
               |                                           |
 ===============[ STEP 3: HARDWARE/KERNEL ENCRYPTED DATA TRANSFER ]=========
               |                                           |
               |<=========================================>|
               |    Hardware/Kernel PSP-Encrypted Payload  |
               |          (Raw Chunks / Blocks)            |
               |                                           |
```

---

## 3. Protocol Steps

### Step 1: Out-of-Band Key Exchange (gRPC Control Plane)
1. When `BorrowConnection(peer)` is called on Node A, `RawBufferTransport`
   resolves the control-plane gRPC channel for `peer` via
   `RawBufferTransportDelegate::GetPeregrineChannel(peer)`.
2. `ConnPool` passes the channel to
   `tcp_psp_helper::TcpPspConnect(sock_fd, addr, addrlen, channel)`.
3. `tcp_psp_helper` calls `AcquireRxSpiAndKey(sock_fd)` on the client
   socket to generate a fresh `(client_spi, client_key)` pair.
4. An `ExchangePspKey` RPC is dispatched to Node B with
   `(client_spi, client_key)`.
5. On Node B, `PeregrineControlServiceImpl` delegates to
   `RawBufferTransport::RegisterPspPeer`, calling
   `RegisterPeerPspToken(server_fd, client_spi, client_key)`.
6. `AddSecureListener` registers the client key on `server_fd` and returns
   allocated `(server_spi, server_key)` in the gRPC response.
7. Node A applies `SetTxSpiAndKey(sock_fd, server_tx_key)` to configure
   the outbound encryption parameters.

### Step 2: Data Plane Connection & Verification
8. Node A executes `connect(sock_fd, addr, addrlen)`. The TCP handshake
   carries the initial PSP SPIs.
9. Node A verifies `GetInitialRxSpi(sock_fd) == client_spi` to ensure
   encryption was negotiated properly.
10. Node B accepts the socket and runs `PspEnabled(client_fd)` to confirm
    hardware encryption is active before data handling.

### Step 3: Encrypted Data Transfer
- All subsequent TCP chunks (data payloads) flowing across the connection
  are encrypted directly by hardware/kernel PSP offload.

---

## 4. Build Configurations & Open-Source Compatibility

- **Flag Control**: Controlled via
  `ABSL_FLAG(bool, require_psp_tcp, false, ...)`.
  When set to `true`, PSP key exchange and encryption verification are
  enforced on all TCP transport connections.
- **Open-Source (OSS) Builds**: Copybara strips internal LOAS PSP
  dependencies on export. If `require_psp_tcp` is enabled in an environment
  without PSP support, the helper functions return `UnimplementedError`.
