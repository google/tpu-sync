#!/usr/bin/env python3
# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fake h2h_benchmark_runner for driver tests (stdlib only, Python 3.9).

Speaks the C++ runner's control-plane handshake and prints the same stdout
lines in the same order (readiness marker, contact line, results block,
H2H_ITER_MS lines, integrity verdict) so the driver's markers/regexes work
unchanged, but synthesizes latencies instead of moving bytes. Every interface
name resolves to 127.0.0.1 so sender and receiver can run on one machine.
Flags are exactly the C++ ABSL_FLAG list; anything else is rejected.

Env knobs (all off by default):
  FAKE_H2H_SPEED_FACTOR        scales synthetic throughput (default 1.0)
  FAKE_H2H_SEED                extra jitter seed (default 0)
  FAKE_H2H_OUTLIER_FRAC        fraction of iterations made 40% slower
  FAKE_H2H_FAIL_INTEGRITY      "1": receiver prints the FAILED verdict
  FAKE_H2H_ACCEPT_TIMEOUT_S    receiver gives up waiting for a sender (300)
  FAKE_H2H_SENDER_DIE_AT_ITER  n (0-based): sender exits 1 when timed
                               iteration n would start -- "Iteration n failed
                               on active manager 0: injected" on stderr and no
                               results block, like the C++ exit(1) from a
                               worker thread
  FAKE_H2H_SENDER_HANG_S       sender sleeps this long after printing the
                               results, before closing the control socket
  FAKE_H2H_SENDER_DIE_ON_PORT  SENDER_DIE_AT_ITER applies only to the process
                               whose --peer_control_port equals this
  FAKE_H2H_RECEIVER_DIE_BEFORE_READY    "1": receiver exits 1 before the
                               readiness marker (a bind failure)
  FAKE_H2H_RECEIVER_DIE_AFTER_HANDSHAKE "1": receiver exits 1 right after the
                               handshake, printing no verdict
  FAKE_H2H_RECEIVER_DIE_ON_PORT the RECEIVER_DIE_* knobs apply only to the
                               process whose --peer_control_port equals this

Jitter is deterministic given (block_size, parallelism, peer_control_port,
FAKE_H2H_SEED), with relative spread 0.02 / parallelism so higher parallelism
is measurably tighter (the driver's ranking tests key on that).
"""

import argparse
import math
import os
import random
import socket
import statistics
import sys
import time

_LAYERS = 32
_SHARDS = 1
_MAX_BYTES_PER_IFACE = 16 * 1024 * 1024 * 1024
_HANDSHAKE_SIG = 'RAIDEN_SENDER_ACTIVE:'
_LOOPBACK = '127.0.0.1'


def _out(msg):
  print(msg, flush=True)


def _err(msg):
  print(msg, file=sys.stderr, flush=True)


class _StrictParser(argparse.ArgumentParser):
  """Rejects anything absl would reject: message on stderr, exit 1."""

  def error(self, message):
    _err('FATAL Flags parsing error: %s' % message)
    sys.exit(1)


def _parse_args(argv):
  # Exactly the ABSL_FLAG list (names, types, defaults) of
  # h2h_benchmark_runner.cc, so a flag the C++ would reject fails here too.
  p = _StrictParser(add_help=False, allow_abbrev=False)
  p.add_argument('--role', default='')
  p.add_argument('--control_ip', default=_LOOPBACK)
  p.add_argument('--control_interface', default='')
  p.add_argument('--data_ip', default=_LOOPBACK)
  p.add_argument('--data_interface', default='')
  p.add_argument('--peer_control_ip', default=_LOOPBACK)
  p.add_argument('--peer_control_port', type=int, default=9999)
  p.add_argument('--block_size', type=int, default=1024 * 1024)
  p.add_argument('--parallelism', type=int, default=1)
  p.add_argument('--numa_node', type=int, default=-1)
  p.add_argument('--num_blocks', type=int, default=64)
  p.add_argument('--iterations', type=int, default=50)
  return p.parse_args(argv)


def _env_int(name):
  v = os.environ.get(name, '').strip()
  return int(v) if v else None


def _env_float(name, default):
  v = os.environ.get(name, '').strip()
  return float(v) if v else default


def _port_selected(port, port_var):
  """True unless `port_var` names a different --peer_control_port."""
  only = _env_int(port_var)
  return only is None or only == port


def _read_line(sock):
  """Byte-at-a-time line read, like the C++ read_line (no trailing '\\n')."""
  buf = bytearray()
  while True:
    c = sock.recv(1)
    if not c:
      break
    if c == b'\n':
      break
    buf += c
  return buf.decode('utf-8', 'replace')


def _send_all(sock, text):
  try:
    sock.sendall(text.encode('utf-8'))
    return True
  except OSError as e:
    _err('send failed: %s' % e)
    return False


def _connect(host, port, deadline_s=20.0):
  """Connect with retries; None after deadline_s (real runner: 30 x 2 s)."""
  start = time.monotonic()
  attempt = 0
  while True:
    try:
      return socket.create_connection((host, port), timeout=5.0)
    except OSError:
      pass
    attempt += 1
    if time.monotonic() - start > deadline_s:
      return None
    if attempt % 20 == 0:
      _out('Connection or resolution to control plane failed, retrying in 2 '
           'seconds...')
    time.sleep(0.1)


def _synth_latencies_ms(total_bytes, parallelism, iterations, block_size,
                        peer_control_port):
  speed = _env_float('FAKE_H2H_SPEED_FACTOR', 1.0)
  outlier_frac = _env_float('FAKE_H2H_OUTLIER_FRAC', 0.0)
  extra_seed = _env_int('FAKE_H2H_SEED') or 0
  base_gbs = 10.0 * (1.0 + 0.15 * math.log2(max(1, parallelism))) * speed
  base_ms = (total_bytes / 1e9) / base_gbs * 1000.0
  rel_sigma = 0.02 / max(1, parallelism)
  seed = (block_size * 1000003 + parallelism * 7919 +
          peer_control_port * 31 + extra_seed) & 0xFFFFFFFF
  rng = random.Random(seed)
  # One normal draw per quantile stratum (jittered inside it): the sample MAD
  # then tracks rel_sigma closely for any seed, so the spread ordering between
  # parallelisms holds run after run while the values still vary per port.
  nd = statistics.NormalDist()
  lat = []
  for i in range(iterations):
    q = min(max((i + rng.random()) / iterations, 1e-6), 1.0 - 1e-6)
    ms = base_ms * (1.0 + rel_sigma * nd.inv_cdf(q))
    lat.append(max(ms, 0.05 * base_ms))
  n_out = int(round(outlier_frac * iterations))
  for idx in rng.sample(range(iterations), min(n_out, iterations)):
    lat[idx] *= 1.4
  return lat


def _print_common(args, control_ip, data_ifaces):
  if args.control_interface:
    _out('Resolved control IP from interface %s: %s' %
         (args.control_interface, control_ip))
  if args.data_interface:
    for name, ip in data_ifaces:
      _out('Using specified data interface override: %s (%s)' % (name, ip))
  _out('Discovered %d data interfaces:' % len(data_ifaces))
  for name, ip in data_ifaces:
    _out('  - %s: %s' % (name, ip))
  if args.numa_node >= 0:
    _out('Pinning main thread to NUMA node %d' % args.numa_node)


def _run_receiver(args, control_ip, data_ifaces):
  port = args.peer_control_port
  data_port = port + 1 if port < 65535 else port - 1
  endpoints = []
  for name, ip in data_ifaces:
    _out('Spawning Receiver Manager on %s (%s)...' % (name, ip))
    endpoints.append('%s:%s:%d' % (name, ip, data_port))
  endpoints_str = ','.join(endpoints)

  server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
  try:
    if (os.environ.get('FAKE_H2H_RECEIVER_DIE_BEFORE_READY') == '1' and
        _port_selected(port, 'FAKE_H2H_RECEIVER_DIE_ON_PORT')):
      raise OSError('injected (FAKE_H2H_RECEIVER_DIE_BEFORE_READY)')
    server.bind((control_ip, port))
    server.listen(1)
  except OSError as e:
    _err('bind failed: %s' % e)
    _err('Failed to start control server')
    server.close()
    return 1
  server.settimeout(_env_float('FAKE_H2H_ACCEPT_TIMEOUT_S', 300.0))

  active_indices = []
  while True:
    # Readiness marker: bound + listening before this line, like the C++.
    _out('Waiting for sender connection on control plane (%s:%d)...' %
         (control_ip, port))
    try:
      conn, _ = server.accept()
    except socket.timeout:
      _err('accept failed: no sender within FAKE_H2H_ACCEPT_TIMEOUT_S')
      server.close()
      return 1
    conn.settimeout(600.0)
    _out('Connection established on control plane. Verifying...')
    if not _send_all(conn, endpoints_str + '\n'):
      _err('Failed to send endpoints, closing connection.')
      conn.close()
      continue
    msg = _read_line(conn)
    _out('Received handshake message: %s' % msg)
    if not msg.startswith(_HANDSHAKE_SIG):
      _err('Warning: Invalid handshake signature from client, closing '
           'connection (suspected scanner).')
      conn.close()
      continue
    active_indices = [int(t) for t in msg[len(_HANDSHAKE_SIG):].split(',')
                      if t.strip()]
    _out('Handshake verified successfully. Sender is active.')
    break

  if (os.environ.get('FAKE_H2H_RECEIVER_DIE_AFTER_HANDSHAKE') == '1' and
      _port_selected(port, 'FAKE_H2H_RECEIVER_DIE_ON_PORT')):
    # A crash here leaves the driver with neither verdict line.
    _err('injected receiver crash (FAKE_H2H_RECEIVER_DIE_AFTER_HANDSHAKE)')
    return 1

  _out('Waiting for sender to complete benchmark...')
  try:
    conn.recv(1)  # blocks until the sender closes
  except OSError as e:
    _err('recv failed: %s' % e)
  _out('Sender disconnected. Benchmark finished.')

  _out('Verifying data integrity across active managers...')
  for m in active_indices:
    name = data_ifaces[m][0] if m < len(data_ifaces) else '?'
    _out('Verifying manager %d (interface: %s)...' % (m, name))
  if os.environ.get('FAKE_H2H_FAIL_INTEGRITY') == '1':
    _err('Data mismatch at manager 0, layer 0, block 0, byte 0. Expected 1, '
         'got 0')
    _out('Data integrity verification FAILED!')
  else:
    _out('Data integrity verification PASSED across active interfaces!')
  conn.close()
  server.close()
  return 0


def _run_sender(args, data_ifaces, num_blocks):
  port = args.peer_control_port
  _out('Connecting to receiver control plane at %s:%d...' %
       (args.peer_control_ip, port))
  sock = _connect(args.peer_control_ip, port)
  if sock is None:
    _err('Failed to connect to receiver control plane')
    return 1
  sock.settimeout(120.0)
  _out('Connected to receiver control plane.')

  peer_data = _read_line(sock)
  _out('Received peer data endpoints: %s' % peer_data)
  remotes = []
  for tok in peer_data.split(','):
    first, last = tok.find(':'), tok.rfind(':')
    if first != -1 and last != -1 and first < last:
      remotes.append((tok[:first], tok[first + 1:last], int(tok[last + 1:])))
  local_name, local_ip = data_ifaces[0]
  for name, ip, dport in remotes:
    _out('Mapping remote %s (%s:%d) to local %s (%s)' %
         (name, ip, dport, local_name, local_ip))

  _out('Running parallel warmup across %d interfaces...' % len(remotes))
  active = list(range(len(remotes)))
  if not active:
    _err('Error: All managers failed warmup! Cannot run benchmark.')
    sock.close()
    return 1
  _out('Successfully warmed up %d / %d interfaces. Proceeding with benchmark.'
       % (len(active), len(remotes)))
  if not _send_all(sock, _HANDSHAKE_SIG + ','.join(str(i) for i in active)
                   + '\n'):
    _err('Failed to send active indices to receiver')
    sock.close()
    return 1

  n = args.iterations
  _out('Running parallel benchmark (%d iterations) across %d active '
       'interfaces...' % (n, len(active)))
  die_at = _env_int('FAKE_H2H_SENDER_DIE_AT_ITER')
  if (die_at is not None and 0 <= die_at < n and
      _port_selected(port, 'FAKE_H2H_SENDER_DIE_ON_PORT')):
    # The C++ worker thread calls exit(1) mid-loop: no results block follows
    # and the control socket is closed by process exit.
    _err('Iteration %d failed on active manager 0: injected' % die_at)
    return 1
  total_bytes = len(active) * _LAYERS * _SHARDS * num_blocks * args.block_size
  lat = sorted(_synth_latencies_ms(total_bytes, args.parallelism, n,
                                   args.block_size, port))
  mean_ms = sum(lat) / n
  p50, p90, p99 = lat[int(n * 0.50)], lat[int(n * 0.90)], lat[int(n * 0.99)]
  gbs = (total_bytes / 1e9) / (mean_ms / 1000.0)

  sys.stdout.write('\n### Collective Benchmark Results (Sender) ###\n')
  sys.stdout.write('Interfaces:  %d (active: %d)\n' % (len(remotes),
                                                       len(active)))
  sys.stdout.write('Block Size:  %d bytes\n' % args.block_size)
  sys.stdout.write('Parallelism: %d (streams per NIC)\n' % args.parallelism)
  sys.stdout.write('p50:   %8.3f ms\n' % p50)
  sys.stdout.write('p90:   %8.3f ms\n' % p90)
  sys.stdout.write('p99:   %8.3f ms\n' % p99)
  sys.stdout.write('Mean:  %8.3f ms\n' % mean_ms)
  sys.stdout.write('Throughput: %8.3f GB/s (collective)\n' % gbs)
  for ms in lat:
    sys.stdout.write('H2H_ITER_MS %.6f\n' % ms)
  sys.stdout.flush()
  hang_s = _env_float('FAKE_H2H_SENDER_HANG_S', 0.0)
  if hang_s > 0:
    time.sleep(hang_s)
  _out('Closing control connection...')
  sock.close()
  return 0


def main():
  args = _parse_args(sys.argv[1:])
  if args.role not in ('sender', 'receiver'):
    _err("Error: --role must be either 'sender' or 'receiver'")
    return 1
  if args.role == 'sender' and args.peer_control_ip in ('', 'auto'):
    _err('Error: --peer_control_ip must be provided for sender.')
    return 1
  if args.iterations < 1:
    _err('Error: --iterations must be >= 1')
    return 1

  num_blocks = args.num_blocks
  bytes_per_block = _LAYERS * _SHARDS * args.block_size
  if num_blocks * bytes_per_block > _MAX_BYTES_PER_IFACE:
    num_blocks = max(1, _MAX_BYTES_PER_IFACE // bytes_per_block)
    _out('Auto-scaling num_blocks to %d to constrain memory bandwidth payload '
         'per iteration to ~16 GiB.' % num_blocks)

  # One machine: an interface name resolves to loopback; an explicit
  # --control_ip is honoured like the C++ does.
  control_ip = _LOOPBACK if args.control_interface else args.control_ip
  if args.data_interface:
    names = [t for t in args.data_interface.replace(',', ' ').split() if t]
    data_ifaces = [(t, _LOOPBACK) for t in names]
  else:
    data_ifaces = [('lo', args.data_ip)]
  _print_common(args, control_ip, data_ifaces)

  if args.role == 'receiver':
    return _run_receiver(args, control_ip, data_ifaces)
  return _run_sender(args, data_ifaces, num_blocks)


if __name__ == '__main__':
  try:
    sys.stdout.reconfigure(line_buffering=True)
  except AttributeError:
    pass
  sys.exit(main())
