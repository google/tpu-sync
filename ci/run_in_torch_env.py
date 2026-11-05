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

"""Runs a Python script in an open-source PyTorch environment.

  python3 ci/run_in_torch_env.py <script.py> [args...]

<script.py> is relative to the repository root. Run this in a virtualenv: it
pip-installs into the Python that runs it.

In the open-source build, torch and torch_tpu reach Bazel only as headers, so
raiden's torch extension can't come from `bazel run`. It has to be compiled
against the installed torch and torch_tpu wheels, and against torch_tpu's own
XLA revision, which only `build.sh torch` sets up. So this script:

  1. installs torch and torch_tpu,
  2. builds the torch extension into the source tree with `build.sh torch`,
  3. installs torch_tpu's runtime requirements, with the libtpu it's tested
     against,
  4. links the extension to torch_tpu's glue library for that torch,
  5. configures libtpu for a multi-process slice in a GKE container and logs
     what the TPU runtime will see, and
  6. replaces itself with `python <script.py> [args...]`.

The torch workloads in tpu_sync/benchmarks/benchmark_registry*.pbtxt run their
scripts through it.
"""

from collections.abc import Sequence
import importlib
import importlib.metadata
import os
import pathlib
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

# The torch release to build and run against. torch_tpu must ship glue for it;
# this is the primary RAIDEN_TORCH_ABIS entry in ci/build_wheel_impl.sh.
_TORCH_VERSION = '2.11.0'
_TORCH_INDEX_URL = 'https://download.pytorch.org/whl/cpu'
# The registry ci/build_wheel_impl.sh installs torch_tpu from.
_TORCH_TPU_INDEX_URL = (
    'https://us-python.pkg.dev/ml-oss-artifacts-transient/'
    'torch-tpu-virtual-registry/simple/'
)
# The glue library build.sh links the in-tree extension against. Only a
# torch_tpu source build has a library by this name.
_UNVERSIONED_GLUE = 'libpywrap_torch_tpu_common.so'
# The libtpu torch_tpu is tested against (its requirements lock files). Its
# wheel only asks for libtpu>=0.0.41, so without a pin requirements.txt's
# newest jax would pull in a libtpu torch_tpu was never tested with.
_LIBTPU_VERSION = '0.0.50'
# The TPU agent socket the GKE TPU device plugin mounts. libtpu talks to it
# before it builds a multi-process slice; torch_tpu's own CI on the same runners
# points VBAR_CONTROL_SERVICE_URL at it.
_VBAR_SOCKET = pathlib.Path('/var/run/tpu-plugin/vbar.sock')


def _log(message: str) -> None:
  print(f'=== {message} ===', flush=True)


def _run(*args: str, **kwargs) -> None:
  subprocess.run(args, check=True, **kwargs)


def _pip_install(*args: str) -> None:
  _run(sys.executable, '-m', 'pip', 'install', '--quiet', *args)


def _authorize_torch_tpu_index() -> None:
  """Lets pip read the torch_tpu registry as the runner's Google identity.

  Like ci/build_wheel_impl.sh, this writes the identity's access token to
  ~/.netrc, and fails with a readable error if the identity can't read the
  registry, instead of leaving pip to report "no matching distribution".
  """
  importlib.invalidate_caches()  # google-auth was installed after startup.
  # pylint: disable=g-import-not-at-top
  import google.auth
  import google.auth.transport.requests
  # pylint: enable=g-import-not-at-top

  credentials, _ = google.auth.default(
      scopes=['https://www.googleapis.com/auth/cloud-platform']
  )
  credentials.refresh(google.auth.transport.requests.Request())
  token = credentials.token

  request = urllib.request.Request(
      _TORCH_TPU_INDEX_URL + 'torch-tpu/',
      headers={'Authorization': f'Bearer {token}'},
  )
  try:
    with urllib.request.urlopen(request) as response:
      status = response.status
  except urllib.error.HTTPError as error:
    status = error.code
  if status != 200:
    identity = getattr(credentials, 'service_account_email', None)
    sys.exit(
        f'{_TORCH_TPU_INDEX_URL} answered HTTP {status} to'
        f' {identity or "the ambient identity"}; it needs Artifact Registry'
        ' read access to that repository.'
    )

  host = urllib.parse.urlparse(_TORCH_TPU_INDEX_URL).netloc
  netrc = pathlib.Path.home() / '.netrc'
  with netrc.open('a') as f:
    f.write(f'machine {host} login oauth2accesstoken password {token}\n')
  netrc.chmod(0o600)


def _install_build_inputs() -> None:
  """Installs what build.sh compiles against: torch and torch_tpu."""
  _log(f'Installing torch {_TORCH_VERSION}')
  _pip_install(f'torch=={_TORCH_VERSION}', '--index-url', _TORCH_INDEX_URL)

  # google-auth reads the torch_tpu registry. patchelf must be on PATH when
  # build.sh runs.
  _pip_install('google-auth', 'requests', 'patchelf')

  version = (_REPO_ROOT / 'torch_tpu.version').read_text().strip()
  _log(f'Installing torch_tpu {version}')
  _authorize_torch_tpu_index()
  # --no-deps keeps pip from swapping the CPU torch wheel for another build.
  # _install_runtime_requirements() installs torch_tpu's other requirements.
  _pip_install(
      '--no-deps', f'torch_tpu=={version}', '--index-url', _TORCH_TPU_INDEX_URL
  )


def _install_runtime_requirements() -> None:
  """Installs what the script needs at runtime that build.sh doesn't.

  That's torch_tpu's own requirements, read from its metadata, minus torch,
  which is already pinned: `import torch` loads the torch_tpu backend, which
  imports them. libtpu is pinned to _LIBTPU_VERSION, plus tensorflow-cpu, which
  bap_metrics (how BAP benchmarks report metrics) writes TensorBoard events
  with.
  """
  importlib.invalidate_caches()  # torch_tpu was installed after startup.
  requirements = [
      requirement
      for requirement in importlib.metadata.requires('torch_tpu') or []
      if 'extra ==' not in requirement
      and re.match(r'[\w.-]+', requirement).group().lower() != 'torch'
  ]
  _log(f'Installing runtime requirements: {", ".join(requirements)}')
  _pip_install(*requirements, f'libtpu=={_LIBTPU_VERSION}', 'tensorflow-cpu')


def _build_extension() -> None:
  """Builds the torch extension into tpu_sync/frameworks/torch/."""
  _log('Building the torch extension with build.sh torch')
  env = dict(os.environ)
  # With torch_tpu installed, importing torch autoloads its backend. The build
  # only needs the wheels' headers, as in ci/build_wheel_impl.sh.
  env['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
  # --skip-server-binaries: skip standalone C++ service binaries unused by torch
  # benchmarks.
  # --skip-pip-install: skip requirements.txt (which installs JAX and newer libtpu);
  # _install_runtime_requirements() installs torch_tpu's runtime requirements.
  # --config=ci: the remote cache the bazel workloads also build with.
  _run(
      './build.sh',
      'torch',
      '--skip-server-binaries',
      '--skip-pip-install',
      '--config=ci',
      cwd=_REPO_ROOT,
      env=env,
  )


def _link_torch_tpu_glue() -> None:
  """Points the extension at torch_tpu's glue library for _TORCH_VERSION.

  The torch_tpu wheel ships one glue library per torch release, and
  torch_tpu_common_loader loads the one matching the installed torch, so the
  extension has to name that one instead of _UNVERSIONED_GLUE.
  ci/build_wheel_impl.sh does the same for the wheel's extension.
  """
  extension = str(_REPO_ROOT / 'tpu_sync/frameworks/torch/_tpu_raiden_torch.so')
  glue = f'libpywrap_{_TORCH_VERSION.replace(".", "_")}_common.so'
  needed = subprocess.run(
      ['patchelf', '--print-needed', extension],
      check=True,
      capture_output=True,
      text=True,
  ).stdout.split()
  if _UNVERSIONED_GLUE in needed:
    _run('patchelf', '--remove-needed', _UNVERSIONED_GLUE, extension)
  if glue not in needed:
    _run('patchelf', '--add-needed', glue, extension)


def _prepare_tpu_environment() -> None:
  """Configures libtpu for a multi-process slice, then logs what it will see.

  libtpu builds the slice of a multi-process job through a SliceBuilder gRPC
  service in each process, which the Cloud build of libtpu secures with ALTS.
  In a GKE container without host networking the ALTS handshake never
  completes, so the processes wait 15 minutes for each other and time out.
  torch_tpu releases from late September 2026 on default libtpu to insecure
  SliceBuilder gRPC themselves; the release in torch_tpu.version was cut
  before that, so this sets the same flags, unless LIBTPU_INIT_ARGS already
  chooses.

  The log is there because a multi-process torch job that can't form its TPU
  slice only reports a timeout, 15 minutes later; what the runner provided is
  the first thing to check then.
  """
  if 'VBAR_CONTROL_SERVICE_URL' not in os.environ and _VBAR_SOCKET.exists():
    os.environ['VBAR_CONTROL_SERVICE_URL'] = f'unix://{_VBAR_SOCKET}'
  init_args = os.environ.get('LIBTPU_INIT_ARGS', '').split()
  if not any('slicebuilder_use_insecure_grpc' in arg for arg in init_args):
    # --undefok: libtpu builds without the flag would otherwise reject it.
    init_args += [
        '--slicebuilder_use_insecure_grpc=true',
        '--undefok=slicebuilder_use_insecure_grpc',
    ]
    os.environ['LIBTPU_INIT_ARGS'] = ' '.join(init_args)

  _log('TPU environment')
  for name, value in sorted(os.environ.items()):
    if re.match(r'(LIBTPU_|TPU_|TORCH_TPU_|VBAR_|(CHIPS_PER_)?HOST_BOUNDS$)',
                name):
      print(f'{name}={value}')
  for directory in (_VBAR_SOCKET.parent, pathlib.Path('/dev/vfio')):
    try:
      print(f'{directory}: {sorted(p.name for p in directory.iterdir())}')
    except OSError as error:
      print(f'{directory}: {error.strerror}')
  shm = os.statvfs('/dev/shm')
  print(f'/dev/shm: {shm.f_blocks * shm.f_frsize / 2**30:.1f} GiB')
  print(f'libtpu {importlib.metadata.version("libtpu")}', flush=True)


def main(argv: Sequence[str]) -> None:
  if len(argv) < 2:
    sys.exit(f'usage: {argv[0]} <script.py> [args...]')
  script = _REPO_ROOT / argv[1]
  if not script.is_file():
    sys.exit(f'{script} is not a file')

  _install_build_inputs()
  _build_extension()
  _install_runtime_requirements()
  _link_torch_tpu_glue()
  _prepare_tpu_environment()

  os.environ['PYTHONPATH'] = os.pathsep.join(
      filter(None, [str(_REPO_ROOT), os.environ.get('PYTHONPATH')])
  )
  _log(f'Running {argv[1]}')
  os.execv(sys.executable, [sys.executable, str(script), *argv[2:]])


if __name__ == '__main__':
  main(sys.argv)
