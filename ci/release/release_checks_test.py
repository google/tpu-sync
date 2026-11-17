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

"""Unit tests for release_checks.py."""

import io
import json
import os
import pathlib
import subprocess
import tempfile
import unittest
from unittest import mock
import urllib.error
import zipfile

from ci.release import release_checks

_FIRST_SHA = "1" * 40
_MAIN_SHA = "2" * 40
_FEATURE_SHA = "3" * 40


class _FakeGit:
  """Hermetic in-memory stand-in for release_checks._run_git."""

  def __init__(self):
    self.refs = {
        _FIRST_SHA: _FIRST_SHA,
        _MAIN_SHA: _MAIN_SHA,
        _FEATURE_SHA: _FEATURE_SHA,
        "main": _MAIN_SHA,
        "origin/main": _MAIN_SHA,
        "golden-20261116": _FIRST_SHA,
        "feature": _FEATURE_SHA,
    }
    self.main_ancestors = {_FIRST_SHA, _MAIN_SHA}
    self.pyproject_by_sha = {
        _FIRST_SHA: '[project]\nname = "tpu_raiden"\nversion = "0.0.1"\n',
        _MAIN_SHA: '[project]\nname = "tpu_raiden"\nversion = "0.0.2"\n',
        _FEATURE_SHA: '[project]\nname = "tpu_raiden"\nversion = "0.0.1"\n',
    }

  def __call__(self, repo_dir, args, *, check=True):
    del repo_dir
    if args[:2] == ["rev-parse", "--verify"] and len(args) == 3:
      target = args[2].removesuffix("^{commit}")
      if target in self.refs:
        return subprocess.CompletedProcess(
            args, 0, stdout=self.refs[target] + "\n", stderr=""
        )
      if check:
        raise release_checks.Error(
            f"git rev-parse failed: unknown ref {target}"
        )
      return subprocess.CompletedProcess(args, 128, stdout="", stderr="unknown")

    if args[:2] == ["merge-base", "--is-ancestor"] and len(args) == 4:
      sha, branch = args[2], args[3]
      rc = 0 if (branch == "origin/main" and sha in self.main_ancestors) else 1
      if check and rc != 0:
        raise release_checks.Error("not an ancestor")
      return subprocess.CompletedProcess(args, rc, stdout="", stderr="")

    if args[:1] == ["show"] and len(args) == 2:
      sha, _, filename = args[1].partition(":")
      if filename == "pyproject.toml" and sha in self.pyproject_by_sha:
        return subprocess.CompletedProcess(
            args, 0, stdout=self.pyproject_by_sha[sha], stderr=""
        )
      if check:
        raise release_checks.Error(f"git show failed for {args[1]}")
      return subprocess.CompletedProcess(args, 128, stdout="", stderr="missing")

    raise AssertionError(f"unexpected git args: {args}")


def _make_wheel(dist_dir, filename, *, name, version):
  path = pathlib.Path(dist_dir) / filename
  dist_info = f"{name.replace('-', '_')}-{version}.dist-info/METADATA"
  metadata = (
      "Metadata-Version: 2.1\n"
      f"Name: {name}\n"
      f"Version: {version}\n"
      "\n"
      "Wheel description.\n"
  )
  with zipfile.ZipFile(path, "w") as zf:
    zf.writestr(dist_info, metadata)
  return path


class PyprojectParserTest(unittest.TestCase):

  def test_parses_project_version_ignoring_other_sections(self):
    content = (
        "# comment\n"
        "[tool.other]\n"
        'version = "9.9.9"\n'
        "\n"
        "[project]\n"
        'name = "tpu_raiden"\n'
        'version = "0.0.2"\n'
    )
    self.assertEqual(release_checks.parse_pyproject_version(content), "0.0.2")

  def test_missing_version_raises(self):
    with self.assertRaises(release_checks.Error):
      release_checks.parse_pyproject_version('[project]\nname = "tpu_raiden"\n')

  def test_rewrite_pyproject_version_preserves_other_sections(self):
    content = (
        "[tool.other]\n"
        'version = "9.9.9"\n'
        "[project]\n"
        'name = "tpu_raiden"\n'
        'version = "0.0.1"\n'
        'license = "Apache-2.0"\n'
    )
    updated, old = release_checks.rewrite_pyproject_version(content, "0.0.2")
    self.assertEqual(old, "0.0.1")
    self.assertEqual(release_checks.parse_pyproject_version(updated), "0.0.2")
    self.assertIn('[tool.other]\nversion = "9.9.9"\n', updated)
    self.assertIn('license = "Apache-2.0"\n', updated)


class ValidationTest(unittest.TestCase):

  def test_parse_stable_version_accepts_plain_and_v_prefix(self):
    self.assertEqual(release_checks.parse_stable_version("0.0.2"), "0.0.2")
    self.assertEqual(release_checks.parse_stable_version(" v1.23.4 "), "1.23.4")

  def test_parse_stable_version_rejects_prereleases_and_garbage(self):
    for bad in ("0.0.1rc20261023185136", "0.0.2.dev1", "0.0", "1.2.3.4", "abc"):
      with self.subTest(bad=bad):
        with self.assertRaises(release_checks.Error):
          release_checks.parse_stable_version(bad)

  def test_parse_frameworks_preserves_canonical_order_and_rejects_unknown(self):
    self.assertEqual(release_checks.parse_frameworks(""), ["jax", "torch"])
    self.assertEqual(
        release_checks.parse_frameworks("torch jax"), ["jax", "torch"]
    )
    self.assertEqual(release_checks.parse_frameworks("torch"), ["torch"])
    with self.assertRaises(release_checks.Error):
      release_checks.parse_frameworks("jax tf")


class RunGitTest(unittest.TestCase):

  def test_run_git_raises_on_nonzero_when_checked(self):
    failed = subprocess.CompletedProcess(
        ["git"], 128, stdout="", stderr="fatal: bad revision"
    )
    with mock.patch.object(subprocess, "run", return_value=failed):
      with self.assertRaisesRegex(release_checks.Error, "bad revision"):
        release_checks._run_git(".", ["rev-parse", "bad"])
      res = release_checks._run_git(".", ["rev-parse", "bad"], check=False)
      self.assertEqual(res.returncode, 128)


class ResolveReleaseTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.tmp = tempfile.TemporaryDirectory()
    self.addCleanup(self.tmp.cleanup)
    self.repo = pathlib.Path(self.tmp.name)
    (self.repo / "lkg.version").write_text(_FIRST_SHA + "\n", encoding="utf-8")
    self.fake_git = _FakeGit()
    patcher = mock.patch.object(
        release_checks, "_run_git", side_effect=self.fake_git
    )
    patcher.start()
    self.addCleanup(patcher.stop)

  def test_tag_push_matching_pyproject(self):
    out = release_checks.resolve_release(
        repo_dir=self.repo,
        github_ref="refs/tags/v0.0.2",
        github_ref_name="v0.0.2",
        github_sha=_MAIN_SHA,
    )
    self.assertEqual(out["checkout_sha"], _MAIN_SHA)
    self.assertEqual(out["version"], "0.0.2")
    self.assertEqual(out["wheel_version"], "0.0.2")
    self.assertEqual(out["wheel_version_override"], "0.0.2")
    self.assertEqual(out["wheel_version_extras"], "")
    self.assertEqual(out["is_stable_release"], "true")
    self.assertEqual(out["will_publish"], "true")
    self.assertEqual(out["tag_name"], "v0.0.2")
    self.assertEqual(json.loads(out["frameworks"]), ["jax", "torch"])

  def test_tag_push_overrides_historical_pyproject_version(self):
    out = release_checks.resolve_release(
        repo_dir=self.repo,
        github_ref="refs/tags/v0.0.2",
        github_ref_name="v0.0.2",
        github_sha=_FIRST_SHA,
    )
    self.assertEqual(out["checkout_sha"], _FIRST_SHA)
    self.assertEqual(out["wheel_version"], "0.0.2")
    self.assertEqual(out["wheel_version_override"], "0.0.2")
    self.assertEqual(out["wheel_version_extras"], "")

  def test_tag_push_rejects_commit_not_on_main(self):
    with self.assertRaisesRegex(release_checks.Error, "origin/main"):
      release_checks.resolve_release(
          repo_dir=self.repo,
          github_ref="refs/tags/v0.0.2",
          github_ref_name="v0.0.2",
          github_sha=_FEATURE_SHA,
      )

  def test_default_workflow_dispatch_preserves_dev_behavior(self):
    out = release_checks.resolve_release(
        repo_dir=self.repo,
        github_ref="refs/heads/main",
        github_ref_name="main",
        github_sha=_MAIN_SHA,
        publish_input="no",
        frameworks_input="jax torch",
        timestamp="20261117120000",
    )
    self.assertEqual(out["checkout_sha"], _MAIN_SHA)
    self.assertEqual(out["version"], "0.0.2")
    self.assertEqual(out["wheel_version"], "0.0.2.dev20261117120000")
    self.assertEqual(out["wheel_version_override"], "")
    self.assertEqual(out["wheel_version_extras"], ".dev20261117120000")
    self.assertEqual(out["is_stable_release"], "false")
    self.assertEqual(out["will_publish"], "false")
    self.assertEqual(out["tag_name"], "")

  def test_workflow_dispatch_stable_release_from_lkg(self):
    out = release_checks.resolve_release(
        repo_dir=self.repo,
        github_ref="refs/heads/main",
        github_ref_name="main",
        github_sha=_MAIN_SHA,
        ref_input="lkg",
        release_version_input="0.0.2",
        publish_input="yes",
        frameworks_input="torch",
    )
    self.assertEqual(out["checkout_sha"], _FIRST_SHA)
    self.assertEqual(out["version"], "0.0.2")
    self.assertEqual(out["wheel_version"], "0.0.2")
    self.assertEqual(out["wheel_version_override"], "0.0.2")
    self.assertEqual(out["wheel_version_extras"], "")
    self.assertEqual(out["is_stable_release"], "true")
    self.assertEqual(out["will_publish"], "true")
    self.assertEqual(out["tag_name"], "v0.0.2")
    self.assertEqual(json.loads(out["frameworks"]), ["torch"])
    self.assertEqual(out["frameworks_space"], "torch")

  def test_workflow_dispatch_stable_release_from_golden_tag(self):
    out = release_checks.resolve_release(
        repo_dir=self.repo,
        github_ref="refs/heads/main",
        github_ref_name="main",
        github_sha=_MAIN_SHA,
        ref_input="golden-20261116",
        release_version_input="v0.0.2",
        publish_input="no",
    )
    self.assertEqual(out["checkout_sha"], _FIRST_SHA)
    self.assertEqual(out["wheel_version"], "0.0.2")
    self.assertEqual(out["is_stable_release"], "true")
    self.assertEqual(out["will_publish"], "false")

  def test_workflow_dispatch_stable_release_rejects_non_main_ref(self):
    with self.assertRaisesRegex(release_checks.Error, "origin/main"):
      release_checks.resolve_release(
          repo_dir=self.repo,
          github_ref="refs/heads/main",
          github_ref_name="main",
          github_sha=_MAIN_SHA,
          ref_input=_FEATURE_SHA,
          release_version_input="0.0.2",
          publish_input="yes",
      )

  def test_workflow_dispatch_dev_dry_run_allows_feature_branch(self):
    out = release_checks.resolve_release(
        repo_dir=self.repo,
        github_ref="refs/heads/feature",
        github_ref_name="feature",
        github_sha=_FEATURE_SHA,
        publish_input="no",
        timestamp="20261117120000",
    )
    self.assertEqual(out["checkout_sha"], _FEATURE_SHA)
    self.assertEqual(out["wheel_version"], "0.0.1.dev20261117120000")
    self.assertEqual(out["is_stable_release"], "false")

  def test_workflow_dispatch_rejects_bad_ref_and_bad_publish(self):
    with self.assertRaises(release_checks.Error):
      release_checks.resolve_release(
          repo_dir=self.repo,
          github_ref="refs/heads/main",
          github_ref_name="main",
          github_sha=_MAIN_SHA,
          ref_input="nonexistent-ref",
      )
    with self.assertRaises(release_checks.Error):
      release_checks.resolve_release(
          repo_dir=self.repo,
          github_ref="refs/heads/main",
          github_ref_name="main",
          github_sha=_MAIN_SHA,
          publish_input="maybe",
      )


class CheckPypiTest(unittest.TestCase):

  def _http_error(
      self, code, url="https://pypi.org/pypi/tpu-sync-jax/0.0.2/json"
  ):
    return urllib.error.HTTPError(
        url, code, f"HTTP {code}", {}, io.BytesIO(b"")
    )

  def test_succeeds_when_all_projects_return_404(self):
    with mock.patch.object(
        urllib.request, "urlopen", side_effect=self._http_error(404)
    ) as open_mock:
      result = release_checks.check_pypi_preflight(
          "0.0.2", ["jax", "torch"], sleep_fn=lambda _: None
      )
    self.assertEqual(
        result["checked_projects"], ["tpu-sync-jax", "tpu-sync-torch"]
    )
    self.assertEqual(open_mock.call_count, 2)

  def test_fails_immediately_when_version_exists_and_reports_conflicts(self):
    resp = mock.MagicMock()
    resp.status = 200
    resp.__enter__.return_value = resp
    with mock.patch.object(
        urllib.request, "urlopen", return_value=resp
    ) as open_mock:
      with self.assertRaisesRegex(
          release_checks.Error,
          r"(?s)tpu-sync-jax==0\.0\.2.*tpu-sync-torch==0\.0\.2",
      ):
        release_checks.check_pypi_preflight(
            "0.0.2", ["jax", "torch"], retries=3, sleep_fn=lambda _: None
        )
    # 200 is definitive, never retried.
    self.assertEqual(open_mock.call_count, 2)

  def test_retries_transient_503_then_succeeds_on_404(self):
    sleeps = []
    side_effects = [
        self._http_error(503),
        urllib.error.URLError("timeout"),
        self._http_error(404),
    ]
    with mock.patch.object(
        urllib.request, "urlopen", side_effect=side_effects
    ) as open_mock:
      release_checks.check_pypi_preflight(
          "0.0.2", ["jax"], retries=3, retry_delay=1.5, sleep_fn=sleeps.append
      )
    self.assertEqual(open_mock.call_count, 3)
    self.assertEqual(sleeps, [1.5, 3.0])

  def test_fails_after_exhausting_retries(self):
    with mock.patch.object(
        urllib.request, "urlopen", side_effect=self._http_error(502)
    ):
      with self.assertRaisesRegex(release_checks.Error, "HTTP 502"):
        release_checks.check_pypi_preflight(
            "0.0.2", ["torch"], retries=2, sleep_fn=lambda _: None
        )


class ApplyVersionOverrideTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.tmp = tempfile.TemporaryDirectory()
    self.addCleanup(self.tmp.cleanup)
    self.repo = pathlib.Path(self.tmp.name)
    (self.repo / "bazel").mkdir()
    (self.repo / "ci").mkdir()
    (self.repo / "pyproject.toml").write_text(
        '[project]\nname = "tpu_raiden"\nversion = "0.0.1"\n', encoding="utf-8"
    )

  def test_leaves_pyproject_untouched_when_native_override_supported(self):
    (self.repo / "bazel" / "wheel_version.bzl").write_text(
        'environ = ["WHEEL_VERSION_OVERRIDE", "WHEEL_VERSION_EXTRAS"]\n',
        encoding="utf-8",
    )
    (self.repo / "ci" / "build_wheel_impl.sh").write_text(
        "--repo_env=WHEEL_VERSION_OVERRIDE=foo\n", encoding="utf-8"
    )
    result = release_checks.apply_version_override(self.repo, "0.0.2")
    self.assertEqual(result["mode"], "native")
    self.assertEqual(
        release_checks.parse_pyproject_version(
            (self.repo / "pyproject.toml").read_text(encoding="utf-8")
        ),
        "0.0.1",
    )

  def test_patches_ephemeral_pyproject_on_historical_commit(self):
    (self.repo / "bazel" / "wheel_version.bzl").write_text(
        'environ = ["WHEEL_VERSION_EXTRAS"]\n', encoding="utf-8"
    )
    (self.repo / "ci" / "build_wheel_impl.sh").write_text(
        "--repo_env=WHEEL_VERSION_EXTRAS=foo\n", encoding="utf-8"
    )
    result = release_checks.apply_version_override(self.repo, "0.0.2")
    self.assertEqual(result["mode"], "patched_pyproject")
    self.assertEqual(result["old_version"], "0.0.1")
    self.assertEqual(
        release_checks.parse_pyproject_version(
            (self.repo / "pyproject.toml").read_text(encoding="utf-8")
        ),
        "0.0.2",
    )


class CheckWheelsTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.tmp = tempfile.TemporaryDirectory()
    self.addCleanup(self.tmp.cleanup)
    self.dist = pathlib.Path(self.tmp.name)

  def test_verifies_metadata_and_invokes_uv_with_scrubbed_env(self):
    _make_wheel(
        self.dist,
        "tpu_sync_jax-0.0.2-0112-cp312-cp312-manylinux_2_31_x86_64.whl",
        name="tpu_sync_jax",
        version="0.0.2",
    )
    captured = {}

    def fake_run(cmd, **kwargs):
      captured["cmd"] = cmd
      captured["kwargs"] = kwargs
      return subprocess.CompletedProcess(
          cmd, 0, stdout="tpu-sync-jax==0.0.2\njax==0.11.2\n", stderr=""
      )

    with mock.patch.dict(
        os.environ,
        {
            "PIP_EXTRA_INDEX_URL": "https://private.example/simple",
            "UV_EXTRA_INDEX_URL": "https://private.example/simple",
            "PATH": "/usr/bin",
        },
        clear=True,
    ):
      result = release_checks.check_wheel_resolution(
          self.dist, "jax", "0.0.2", run_fn=fake_run
      )

    self.assertEqual(result["resolution"], "passed")
    self.assertEqual(captured["kwargs"]["input"], "tpu-sync-jax==0.0.2\n")
    self.assertIn("--python-version", captured["cmd"])
    self.assertIn("3.12", captured["cmd"])
    self.assertIn("--python-platform", captured["cmd"])
    self.assertIn("x86_64-manylinux_2_31", captured["cmd"])
    self.assertIn("https://pypi.org/simple", captured["cmd"])
    self.assertNotIn("PIP_EXTRA_INDEX_URL", captured["kwargs"]["env"])
    self.assertNotIn("UV_EXTRA_INDEX_URL", captured["kwargs"]["env"])

  def test_fails_on_resolution_error_when_strict(self):
    _make_wheel(
        self.dist,
        "tpu_sync_torch-0.0.2-cp312-cp312-manylinux_2_31_x86_64.whl",
        name="tpu_sync_torch",
        version="0.0.2",
    )

    def fake_fail(cmd, **kwargs):
      del kwargs
      return subprocess.CompletedProcess(
          cmd, 1, stdout="", stderr="No solution found when resolving..."
      )

    with self.assertRaisesRegex(release_checks.Error, "No solution found"):
      release_checks.check_wheel_resolution(
          self.dist,
          "torch",
          "0.0.2",
          allow_resolution_failure=False,
          run_fn=fake_fail,
      )

  def test_warns_instead_of_failing_when_allow_resolution_failure(self):
    _make_wheel(
        self.dist,
        "tpu_sync_jax-0.0.2.dev20261117000000-0120-cp312-cp312-manylinux_2_31_x86_64.whl",
        name="tpu_sync_jax",
        version="0.0.2.dev20261117000000",
    )

    def fake_fail(cmd, **kwargs):
      del kwargs
      return subprocess.CompletedProcess(
          cmd, 1, stdout="", stderr="jax==0.99.0 was not found"
      )

    result = release_checks.check_wheel_resolution(
        self.dist,
        "jax",
        "0.0.2.dev20261117000000",
        allow_resolution_failure=True,
        run_fn=fake_fail,
    )
    self.assertEqual(result["resolution"], "warned")

  def test_rejects_metadata_version_mismatch_even_on_dry_run(self):
    _make_wheel(
        self.dist,
        "tpu_sync_torch-0.0.2-cp312-cp312-manylinux_2_31_x86_64.whl",
        name="tpu_sync_torch",
        version="0.0.1",
    )
    with self.assertRaisesRegex(release_checks.Error, "METADATA Version"):
      release_checks.check_wheel_resolution(
          self.dist,
          "torch",
          "0.0.2",
          allow_resolution_failure=True,
          run_fn=mock.MagicMock(),
      )

  def test_rejects_missing_or_duplicate_wheels(self):
    with self.assertRaisesRegex(release_checks.Error, "expected exactly 1"):
      release_checks.find_built_wheel(self.dist, "torch", "0.0.2")

    _make_wheel(
        self.dist,
        "tpu_sync_torch-0.0.2-cp312-cp312-manylinux_2_31_x86_64.whl",
        name="tpu_sync_torch",
        version="0.0.2",
    )
    _make_wheel(
        self.dist,
        "tpu_sync_torch-0.0.2-1-cp312-cp312-manylinux_2_31_x86_64.whl",
        name="tpu_sync_torch",
        version="0.0.2",
    )
    with self.assertRaisesRegex(release_checks.Error, "expected exactly 1"):
      release_checks.find_built_wheel(self.dist, "torch", "0.0.2")


class EmitAndCliTest(unittest.TestCase):

  def test_emit_writes_github_output(self):
    with tempfile.TemporaryDirectory() as tmp:
      out_file = pathlib.Path(tmp) / "github_output"
      with mock.patch.dict(os.environ, {"GITHUB_OUTPUT": str(out_file)}):
        release_checks.emit({"version": "0.0.2", "will_publish": "true"})
      self.assertEqual(
          out_file.read_text(encoding="utf-8"),
          "version=0.0.2\nwill_publish=true\n",
      )

  def test_main_returns_nonzero_on_error(self):
    self.assertEqual(
        release_checks.main(
            ["apply-version-override", "--repo-dir", ".", "--version", "bad"]
        ),
        1,
    )


if __name__ == "__main__":
  unittest.main()
