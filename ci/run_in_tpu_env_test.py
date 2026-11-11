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

import sys
import unittest
from unittest import mock

from ci import run_in_tpu_env


class PipInstallTest(unittest.TestCase):

  def test_pip_install_uses_uv_when_available(self) -> None:
    with (
        mock.patch.object(
            run_in_tpu_env.shutil, 'which', return_value='/usr/bin/uv'
        ) as mock_which,
        mock.patch.object(run_in_tpu_env, '_run') as mock_run,
    ):
      run_in_tpu_env._pip_install('--no-deps', '/tmp/tpu_sync.whl')

    mock_which.assert_called_once_with('uv')
    mock_run.assert_called_once_with(
        'uv',
        'pip',
        'install',
        '--quiet',
        '--python',
        sys.executable,
        '--no-deps',
        '/tmp/tpu_sync.whl',
    )

  def test_pip_install_falls_back_to_python_m_pip_without_uv(self) -> None:
    with (
        mock.patch.object(
            run_in_tpu_env.shutil, 'which', return_value=None
        ) as mock_which,
        mock.patch.object(run_in_tpu_env, '_run') as mock_run,
    ):
      run_in_tpu_env._pip_install(
          '--requirement',
          'requirements.txt',
          '--extra-index-url',
          'https://example.com/simple',
      )

    mock_which.assert_called_once_with('uv')
    mock_run.assert_called_once_with(
        sys.executable,
        '-m',
        'pip',
        'install',
        '--quiet',
        '--requirement',
        'requirements.txt',
        '--extra-index-url',
        'https://example.com/simple',
    )


if __name__ == '__main__':
  unittest.main()
