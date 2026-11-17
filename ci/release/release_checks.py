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

"""Release verification and preflight checks for .github/workflows/release_wheels.yml.

Drives the release workflow's verification, preflight, version override, and
post-build wheel validation steps:

  resolve                Resolve checkout SHA, base version, version override,
                         suffix, frameworks, and publish/release flags from the
                         GitHub event context and optional dispatch inputs.
  check-pypi             Fail fast if any target PyPI project/version already
                         exists (requires HTTP 404 from PyPI JSON API).
  apply-version-override Apply a base version override to an ephemeral checkout
                         when building a historical commit that predates native
                         WHEEL_VERSION_OVERRIDE support.
  check-wheels           Verify built wheel metadata matches the expected
                         distribution and version, then run `uv pip compile`
                         against public PyPI for Python 3.12 /
                         x86_64-manylinux_2_31 before staging to the exit gate.

Only the standard library is used, so the script runs on a stock python3.
"""

import argparse
import datetime
import json
import os
import pathlib
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile

KNOWN_FRAMEWORKS = ("jax", "torch")
DEFAULT_PYPI_BASE_URL = "https://pypi.org/pypi"
DEFAULT_PYPI_SIMPLE_URL = "https://pypi.org/simple"
DEFAULT_PYTHON_VERSION = "3.12"
DEFAULT_PYTHON_PLATFORM = "x86_64-manylinux_2_31"
LKG_FILE = "lkg.version"

_STABLE_VERSION_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_RETRYABLE_HTTP_STATUS = frozenset({429, 500, 502, 503, 504})
_INDEX_ENV_VARS = (
    "PIP_INDEX_URL",
    "PIP_EXTRA_INDEX_URL",
    "PIP_FIND_LINKS",
    "UV_INDEX_URL",
    "UV_EXTRA_INDEX_URL",
    "UV_DEFAULT_INDEX",
    "UV_INDEX",
    "UV_FIND_LINKS",
)
_UTC = datetime.timezone.utc


class Error(Exception):
  """A release check failure reported directly to the caller."""


def _log(message):
  print(message, file=sys.stderr, flush=True)


def emit(outputs):
  """Prints |outputs| as JSON and appends them to $GITHUB_OUTPUT if set."""
  print(json.dumps(outputs, indent=2, sort_keys=True))
  github_output = os.environ.get("GITHUB_OUTPUT")
  if github_output:
    with open(github_output, "a", encoding="utf-8") as f:
      for key, value in outputs.items():
        if "\n" in str(value):
          raise Error(f"GITHUB_OUTPUT value for {key!r} contains a newline")
        f.write(f"{key}={value}\n")


def normalize_package_name(name):
  """PEP 503 distribution name normalization."""
  return re.sub(r"[-_.]+", "-", name).lower()


def pypi_project_name(framework):
  """Returns the PyPI project name for a framework ('jax' or 'torch')."""
  if framework not in KNOWN_FRAMEWORKS:
    raise Error(
        f"unknown framework {framework!r}; choose from {list(KNOWN_FRAMEWORKS)}"
    )
  return f"tpu-sync-{framework}"


def wheel_dist_name(framework):
  """Returns the wheel filename distribution prefix for a framework."""
  if framework not in KNOWN_FRAMEWORKS:
    raise Error(
        f"unknown framework {framework!r}; choose from {list(KNOWN_FRAMEWORKS)}"
    )
  return f"tpu_sync_{framework}"


def parse_pyproject_version(content):
  """Parses [project].version from pyproject.toml text.

  Matches bazel/wheel_version.bzl's section-aware parser so CI and Bazel agree.

  Args:
    content: Contents of pyproject.toml.

  Returns:
    The base version string from [project].version.

  Raises:
    Error: [project].version is missing or empty.
  """
  in_project_section = False
  for raw_line in content.splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#"):
      continue
    if line.startswith("[") and line.endswith("]"):
      in_project_section = line == "[project]"
      continue
    if in_project_section and line.startswith("version"):
      parts = line.split("=", 1)
      if len(parts) == 2 and parts[0].strip() == "version":
        version = parts[1].strip().strip('"').strip("'")
        if version:
          return version
  raise Error("version not found in pyproject.toml [project] section")


def rewrite_pyproject_version(content, new_version):
  """Replaces [project].version in pyproject.toml text, preserving formatting.

  Args:
    content: Existing pyproject.toml text.
    new_version: Replacement base version string.

  Returns:
    Tuple of (updated_content, old_version).

  Raises:
    Error: [project].version was not found.
  """
  lines = content.splitlines(keepends=True)
  in_project_section = False
  for i, raw_line in enumerate(lines):
    stripped = raw_line.strip()
    if not stripped or stripped.startswith("#"):
      continue
    if stripped.startswith("[") and stripped.endswith("]"):
      in_project_section = stripped == "[project]"
      continue
    if in_project_section and stripped.startswith("version"):
      parts = stripped.split("=", 1)
      if len(parts) == 2 and parts[0].strip() == "version":
        old_version = parts[1].strip().strip('"').strip("'")
        newline = "\r\n" if raw_line.endswith("\r\n") else (
            "\n" if raw_line.endswith("\n") else ""
        )
        indent = raw_line[: len(raw_line) - len(raw_line.lstrip())]
        lines[i] = f'{indent}version = "{new_version}"{newline}'
        return "".join(lines), old_version
  raise Error("version not found in pyproject.toml [project] section")


def parse_stable_version(raw):
  """Validates and normalizes a stable X.Y.Z version (accepting optional 'v')."""
  cleaned = raw.strip()
  if cleaned.startswith("v"):
    cleaned = cleaned[1:]
  if not _STABLE_VERSION_RE.match(cleaned):
    raise Error(
        f"release_version {raw!r} is not a stable X.Y.Z version "
        "(pre-release suffixes like rc/dev/a/b are disallowed for releases)"
    )
  return cleaned


def parse_frameworks(frameworks_input):
  """Parses and validates a space-separated framework selection string."""
  wanted = frameworks_input.split() or list(KNOWN_FRAMEWORKS)
  unknown = sorted(set(wanted) - set(KNOWN_FRAMEWORKS))
  if unknown:
    raise Error(
        f"unknown frameworks {unknown}; choose from {list(KNOWN_FRAMEWORKS)}"
    )
  return [f for f in KNOWN_FRAMEWORKS if f in wanted]


# --- Git helpers ------------------------------------------------------------


def _run_git(repo_dir, args, *, check=True):
  result = subprocess.run(
      ["git", "-C", str(repo_dir), *args],
      capture_output=True,
      text=True,
      check=False,
  )
  if check and result.returncode != 0:
    detail = (result.stderr or result.stdout).strip()
    raise Error(f"git {' '.join(args)} failed: {detail}")
  return result


def resolve_commit_sha(repo_dir, ref_input, github_sha):
  """Resolves the target commit SHA from an optional dispatch ref input."""
  repo_path = pathlib.Path(repo_dir)
  ref = ref_input.strip()
  if not ref:
    target = github_sha.strip()
  elif ref.lower() == "lkg":
    lkg_path = repo_path / LKG_FILE
    if not lkg_path.is_file():
      raise Error(f"ref='lkg' requested, but {lkg_path} does not exist")
    target = lkg_path.read_text(encoding="utf-8").strip()
    if not _SHA_RE.match(target):
      raise Error(
          f"{lkg_path} does not contain a 40-char commit SHA: {target!r}"
      )
    _log(f"Resolved ref='lkg' from {LKG_FILE} -> {target}")
  else:
    target = ref

  if not target:
    raise Error("could not determine target commit SHA (empty ref and sha)")
  sha = _run_git(
      repo_path, ["rev-parse", "--verify", f"{target}^{{commit}}"]
  ).stdout.strip()
  if not _SHA_RE.match(sha):
    raise Error(f"git rev-parse returned unexpected commit SHA: {sha!r}")
  return sha


def verify_main_ancestor(repo_dir, sha, *, context):
  """Verifies that |sha| is reachable from origin/main."""
  result = _run_git(
      repo_dir,
      ["merge-base", "--is-ancestor", sha, "origin/main"],
      check=False,
  )
  if result.returncode != 0:
    raise Error(
        f"{context} ({sha}) is not reachable from origin/main; "
        "stable releases are cut from main only."
    )


def read_pyproject_at_commit(repo_dir, sha):
  """Reads [project].version from pyproject.toml at commit |sha|."""
  content = _run_git(repo_dir, ["show", f"{sha}:pyproject.toml"]).stdout
  return parse_pyproject_version(content)


# --- resolve ----------------------------------------------------------------


def resolve_release(
    *,
    repo_dir,
    github_ref,
    github_ref_name,
    github_sha,
    ref_input="",
    release_version_input="",
    publish_input="no",
    frameworks_input="",
    timestamp="",
):
  """Resolves all release workflow parameters from GitHub event context."""
  is_tag = github_ref.startswith("refs/tags/")
  if is_tag:
    if ref_input.strip() or release_version_input.strip():
      raise Error(
          "tag-triggered releases do not accept ref or release_version inputs"
      )
    if not github_ref_name.startswith("v"):
      raise Error(
          f"release tag {github_ref_name!r} must start with 'v' (vX.Y.Z)"
      )
    tag_version = parse_stable_version(github_ref_name)
    checkout_sha = resolve_commit_sha(repo_dir, "", github_sha)
    verify_main_ancestor(
        repo_dir, checkout_sha, context=f"tag {github_ref_name}"
    )
    pyproject_version = read_pyproject_at_commit(repo_dir, checkout_sha)
    if tag_version != pyproject_version:
      _log(
          f"NOTICE: tag {github_ref_name} overrides pyproject.toml base "
          f"version {pyproject_version} at {checkout_sha[:12]} with "
          f"{tag_version}."
      )
    else:
      _log(
          f"Tag {github_ref_name} matches pyproject.toml version "
          f"{pyproject_version} at {checkout_sha[:12]}."
      )
    frameworks = list(KNOWN_FRAMEWORKS)
    return {
        "checkout_sha": checkout_sha,
        "frameworks": json.dumps(frameworks),
        "frameworks_space": " ".join(frameworks),
        "is_stable_release": "true",
        "tag_name": f"v{tag_version}",
        "version": tag_version,
        "wheel_version": tag_version,
        "wheel_version_extras": "",
        "wheel_version_override": tag_version,
        "will_publish": "true",
    }

  publish_clean = (publish_input or "no").strip().lower()
  if publish_clean not in ("yes", "no"):
    raise Error(f"publish input must be 'yes' or 'no', got {publish_input!r}")
  will_publish = publish_clean == "yes"
  frameworks = parse_frameworks(frameworks_input)
  checkout_sha = resolve_commit_sha(repo_dir, ref_input, github_sha)
  pyproject_version = read_pyproject_at_commit(repo_dir, checkout_sha)

  if release_version_input.strip():
    release_version = parse_stable_version(release_version_input)
    verify_main_ancestor(
        repo_dir,
        checkout_sha,
        context=f"release {release_version} target commit",
    )
    if release_version != pyproject_version:
      _log(
          f"NOTICE: release_version {release_version} overrides pyproject.toml "
          f"base version {pyproject_version} at {checkout_sha[:12]}."
      )
    else:
      _log(
          f"release_version {release_version} matches pyproject.toml at "
          f"{checkout_sha[:12]}."
      )
    return {
        "checkout_sha": checkout_sha,
        "frameworks": json.dumps(frameworks),
        "frameworks_space": " ".join(frameworks),
        "is_stable_release": "true",
        "tag_name": f"v{release_version}",
        "version": release_version,
        "wheel_version": release_version,
        "wheel_version_extras": "",
        "wheel_version_override": release_version,
        "will_publish": "true" if will_publish else "false",
    }

  stamp = timestamp.strip() or datetime.datetime.now(_UTC).strftime(
      "%Y%m%d%H%M%S"
  )
  extras = f".dev{stamp}"
  _log(
      f"Dev build from {checkout_sha[:12]}: "
      f"pyproject.toml={pyproject_version}, extras={extras}"
  )
  return {
      "checkout_sha": checkout_sha,
      "frameworks": json.dumps(frameworks),
      "frameworks_space": " ".join(frameworks),
      "is_stable_release": "false",
      "tag_name": "",
      "version": pyproject_version,
      "wheel_version": f"{pyproject_version}{extras}",
      "wheel_version_extras": extras,
      "wheel_version_override": "",
      "will_publish": "true" if will_publish else "false",
  }


# --- check-pypi -------------------------------------------------------------


def _fetch_pypi_status(url, *, timeout=30):
  request = urllib.request.Request(
      url,
      method="GET",
      headers={"User-Agent": "tpu-sync-release-checks"},
  )
  try:
    with urllib.request.urlopen(request, timeout=timeout) as response:
      return response.status
  except urllib.error.HTTPError as e:
    return e.code


def check_pypi_version_available(
    project,
    version,
    *,
    pypi_base_url=DEFAULT_PYPI_BASE_URL,
    retries=3,
    retry_delay=2.0,
    sleep_fn=time.sleep,
):
  """Verifies that |project|==|version| returns HTTP 404 on PyPI."""
  if retries < 1:
    raise Error(f"retries must be >= 1, got {retries}")
  url = f"{pypi_base_url.rstrip('/')}/{project}/{version}/json"
  last_error = None
  for attempt in range(1, retries + 1):
    try:
      status = _fetch_pypi_status(url)
    except (urllib.error.URLError, TimeoutError, OSError) as e:
      last_error = f"{type(e).__name__}: {e}"
      if attempt < retries:
        _log(
            f"Transient error checking {url} (attempt {attempt}/{retries}): "
            f"{last_error}; retrying..."
        )
        sleep_fn(retry_delay * attempt)
        continue
      raise Error(
          f"failed to query PyPI at {url} after {retries} attempts: "
          f"{last_error}"
      ) from e

    if status == 404:
      _log(
          f"OK: {project}=={version} is not published on PyPI "
          f"(HTTP 404: {url})"
      )
      return
    if status == 200:
      raise Error(
          f"{project}=={version} is already published on PyPI ({url}); "
          "PyPI versions are immutable — choose a newer version."
      )
    if status in _RETRYABLE_HTTP_STATUS and attempt < retries:
      last_error = f"HTTP {status}"
      _log(
          f"Transient HTTP {status} from {url} (attempt {attempt}/{retries}); "
          "retrying..."
      )
      sleep_fn(retry_delay * attempt)
      continue
    raise Error(
        f"unexpected HTTP {status} querying PyPI at {url}; "
        "refusing to proceed without a confirmed 404."
    )


def check_pypi_preflight(
    version,
    frameworks,
    *,
    pypi_base_url=DEFAULT_PYPI_BASE_URL,
    retries=3,
    retry_delay=2.0,
    sleep_fn=time.sleep,
):
  """Checks every target framework's PyPI project returns 404 for |version|."""
  errors = []
  checked = []
  for framework in frameworks:
    project = pypi_project_name(framework)
    checked.append(project)
    try:
      check_pypi_version_available(
          project,
          version,
          pypi_base_url=pypi_base_url,
          retries=retries,
          retry_delay=retry_delay,
          sleep_fn=sleep_fn,
      )
    except Error as e:
      errors.append(str(e))
  if errors:
    raise Error("PyPI preflight check failed:\n  - " + "\n  - ".join(errors))
  return {"checked_projects": checked, "version": version}


# --- apply-version-override -------------------------------------------------


def supports_native_version_override(repo_dir):
  """Returns True if the checkout supports WHEEL_VERSION_OVERRIDE natively."""
  repo_path = pathlib.Path(repo_dir)
  bzl = repo_path / "bazel" / "wheel_version.bzl"
  impl = repo_path / "ci" / "build_wheel_impl.sh"
  if not bzl.is_file() or not impl.is_file():
    return False
  return (
      "WHEEL_VERSION_OVERRIDE" in bzl.read_text(encoding="utf-8")
      and "WHEEL_VERSION_OVERRIDE" in impl.read_text(encoding="utf-8")
  )


def apply_version_override(repo_dir, version):
  """Ensures the checkout builds wheels with base version |version|.

  Commits with native WHEEL_VERSION_OVERRIDE support in bazel/wheel_version.bzl
  and ci/build_wheel_impl.sh are left untouched. Historical commits predating
  that support have their ephemeral pyproject.toml [project].version updated
  in place so release_wheels.yml can cut stable releases from them (e.g.
  lkg.version) without editing git history.

  Args:
    repo_dir: Path to the ephemeral checkout directory.
    version: Target stable base version (X.Y.Z).

  Returns:
    Dict describing the action taken ("native" or "patched_pyproject").

  Raises:
    Error: |version| is invalid, or pyproject.toml is missing or malformed.
  """
  normalized = parse_stable_version(version)
  repo_path = pathlib.Path(repo_dir)
  if supports_native_version_override(repo_path):
    _log(
        "Checkout supports WHEEL_VERSION_OVERRIDE natively; "
        "leaving pyproject.toml untouched."
    )
    return {"mode": "native", "version": normalized}

  pyproject_path = repo_path / "pyproject.toml"
  if not pyproject_path.is_file():
    raise Error(f"pyproject.toml not found at {pyproject_path}")
  updated, old_version = rewrite_pyproject_version(
      pyproject_path.read_text(encoding="utf-8"), normalized
  )
  pyproject_path.write_text(updated, encoding="utf-8")
  _log(
      "Checkout predates native WHEEL_VERSION_OVERRIDE support; updated "
      f"ephemeral pyproject.toml [project].version from {old_version} to "
      f"{normalized}."
  )
  return {
      "mode": "patched_pyproject",
      "old_version": old_version,
      "version": normalized,
  }


# --- check-wheels -----------------------------------------------------------


def read_wheel_metadata(wheel_path):
  """Reads Name and Version headers from a wheel's .dist-info/METADATA."""
  path = pathlib.Path(wheel_path)
  if not path.is_file():
    raise Error(f"wheel not found: {path}")
  with zipfile.ZipFile(path) as zf:
    metadata_names = [
        n for n in zf.namelist() if n.endswith(".dist-info/METADATA")
    ]
    if len(metadata_names) != 1:
      raise Error(
          f"{path.name} must contain exactly one .dist-info/METADATA entry, "
          f"found {metadata_names}"
      )
    text = zf.read(metadata_names[0]).decode("utf-8")

  headers = {}
  for line in text.splitlines():
    if not line.strip():
      break
    if ": " in line:
      key, value = line.split(": ", 1)
      headers.setdefault(key.lower(), value.strip())
  if "name" not in headers or "version" not in headers:
    raise Error(
        f"{path.name} METADATA is missing Name or Version headers "
        f"(found keys: {sorted(headers)})"
    )
  return headers["name"], headers["version"]


def find_built_wheel(dist_dir, framework, expected_version):
  """Finds and verifies the single built wheel for |framework| and version."""
  dist_path = pathlib.Path(dist_dir)
  if not dist_path.is_dir():
    raise Error(f"dist directory does not exist: {dist_path}")
  prefix = wheel_dist_name(framework)
  pattern = f"{prefix}-{expected_version}-*.whl"
  matches = sorted(dist_path.glob(pattern))
  if len(matches) != 1:
    present = sorted(p.name for p in dist_path.glob("*.whl"))
    raise Error(
        f"expected exactly 1 wheel matching {pattern} in {dist_path}, "
        f"found {len(matches)}: {[p.name for p in matches]} "
        f"(all wheels in {dist_path}: {present})"
    )

  wheel = matches[0]
  raw_name, actual_version = read_wheel_metadata(wheel)
  expected_pypi = pypi_project_name(framework)
  actual_pypi = normalize_package_name(raw_name)
  if actual_pypi != expected_pypi:
    raise Error(
        f"{wheel.name} METADATA Name is {raw_name!r} (normalized "
        f"{actual_pypi!r}), expected {expected_pypi!r}"
    )
  if actual_version != expected_version:
    raise Error(
        f"{wheel.name} METADATA Version is {actual_version!r}, expected "
        f"{expected_version!r}"
    )
  _log(
      f"OK: {wheel.name} METADATA verified "
      f"(Name={actual_pypi}, Version={actual_version})"
  )
  return wheel


def _clean_index_env(base_env=None):
  env = dict(os.environ if base_env is None else base_env)
  for var in _INDEX_ENV_VARS:
    env.pop(var, None)
  return env


def check_wheel_resolution(
    dist_dir,
    framework,
    expected_version,
    *,
    python_version=DEFAULT_PYTHON_VERSION,
    python_platform=DEFAULT_PYTHON_PLATFORM,
    index_url=DEFAULT_PYPI_SIMPLE_URL,
    allow_resolution_failure=False,
    run_fn=subprocess.run,
):
  """Verifies wheel metadata and resolves dependencies against public PyPI."""
  wheel = find_built_wheel(dist_dir, framework, expected_version)
  project = pypi_project_name(framework)
  requirement = f"{project}=={expected_version}\n"
  cmd = [
      "uv",
      "pip",
      "compile",
      "-",
      "--no-config",
      "--no-cache",
      "--index-url",
      index_url,
      "--find-links",
      str(pathlib.Path(dist_dir).resolve()),
      "--python-version",
      python_version,
      "--python-platform",
      python_platform,
  ]
  _log(
      f"Checking public PyPI resolution for {project}=={expected_version} "
      f"(python={python_version}, platform={python_platform}, "
      f"index={index_url})"
  )
  result = run_fn(
      cmd,
      input=requirement,
      capture_output=True,
      text=True,
      env=_clean_index_env(),
      check=False,
  )
  if result.returncode == 0:
    _log(
        f"OK: {project}=={expected_version} resolved cleanly against "
        f"{index_url}:"
    )
    for line in result.stdout.splitlines():
      stripped = line.strip()
      if stripped and not stripped.startswith("#"):
        _log(f"  {stripped}")
    return {
        "resolution": "passed",
        "wheel": wheel.name,
        "version": expected_version,
    }

  detail = (result.stderr or result.stdout or "").strip()
  message = (
      f"uv pip compile failed for {project}=={expected_version} "
      f"(python={python_version}, platform={python_platform}, "
      f"index={index_url}):\n{detail}"
  )
  if allow_resolution_failure:
    first_line = detail.splitlines()[0] if detail else "unknown error"
    print(
        f"::warning::Dry-run PyPI resolution check failed for "
        f"{project}=={expected_version}: {first_line}",
        flush=True,
    )
    _log(f"WARNING (dry run, non-fatal): {message}")
    return {
        "resolution": "warned",
        "wheel": wheel.name,
        "version": expected_version,
    }
  raise Error(message)


# --- CLI --------------------------------------------------------------------


def build_parser():
  """Constructs the CLI argument parser for release_checks."""
  parser = argparse.ArgumentParser(
      prog="release_checks.py",
      description="Release verification and preflight checks for tpu_sync.",
  )
  sub = parser.add_subparsers(dest="command", required=True)

  p_resolve = sub.add_parser("resolve", help="Resolve release parameters.")
  p_resolve.add_argument("--repo-dir", default=".")
  p_resolve.add_argument("--github-ref", required=True)
  p_resolve.add_argument("--github-ref-name", required=True)
  p_resolve.add_argument("--github-sha", required=True)
  p_resolve.add_argument("--ref-input", default="")
  p_resolve.add_argument("--release-version-input", default="")
  p_resolve.add_argument("--publish-input", default="no")
  p_resolve.add_argument("--frameworks-input", default="")
  p_resolve.add_argument("--timestamp", default="")

  p_pypi = sub.add_parser(
      "check-pypi", help="Verify target version returns 404 on PyPI."
  )
  p_pypi.add_argument("--version", required=True)
  p_pypi.add_argument(
      "--frameworks",
      nargs="+",
      required=True,
      choices=KNOWN_FRAMEWORKS,
  )
  p_pypi.add_argument("--pypi-base-url", default=DEFAULT_PYPI_BASE_URL)
  p_pypi.add_argument("--retries", type=int, default=3)
  p_pypi.add_argument("--retry-delay", type=float, default=2.0)

  p_override = sub.add_parser(
      "apply-version-override",
      help="Apply base version override for pre-override historical commits.",
  )
  p_override.add_argument("--repo-dir", default=".")
  p_override.add_argument("--version", required=True)

  p_wheels = sub.add_parser(
      "check-wheels",
      help="Verify wheel METADATA and resolve dependencies via uv pip compile.",
  )
  p_wheels.add_argument("--dist-dir", required=True)
  p_wheels.add_argument(
      "--framework", required=True, choices=KNOWN_FRAMEWORKS
  )
  p_wheels.add_argument("--expected-version", required=True)
  p_wheels.add_argument("--python-version", default=DEFAULT_PYTHON_VERSION)
  p_wheels.add_argument("--python-platform", default=DEFAULT_PYTHON_PLATFORM)
  p_wheels.add_argument("--index-url", default=DEFAULT_PYPI_SIMPLE_URL)
  p_wheels.add_argument(
      "--allow-resolution-failure",
      action="store_true",
      help="Warn instead of failing on PyPI resolution errors (for dry runs).",
  )
  return parser


def main(argv=None):
  args = build_parser().parse_args(argv)
  try:
    if args.command == "resolve":
      emit(
          resolve_release(
              repo_dir=args.repo_dir,
              github_ref=args.github_ref,
              github_ref_name=args.github_ref_name,
              github_sha=args.github_sha,
              ref_input=args.ref_input,
              release_version_input=args.release_version_input,
              publish_input=args.publish_input,
              frameworks_input=args.frameworks_input,
              timestamp=args.timestamp,
          )
      )
    elif args.command == "check-pypi":
      emit(
          check_pypi_preflight(
              args.version,
              args.frameworks,
              pypi_base_url=args.pypi_base_url,
              retries=args.retries,
              retry_delay=args.retry_delay,
          )
      )
    elif args.command == "apply-version-override":
      emit(apply_version_override(args.repo_dir, args.version))
    elif args.command == "check-wheels":
      emit(
          check_wheel_resolution(
              args.dist_dir,
              args.framework,
              args.expected_version,
              python_version=args.python_version,
              python_platform=args.python_platform,
              index_url=args.index_url,
              allow_resolution_failure=args.allow_resolution_failure,
          )
      )
  except Error as e:
    _log(f"ERROR: {e}")
    return 1
  return 0


if __name__ == "__main__":
  sys.exit(main())
