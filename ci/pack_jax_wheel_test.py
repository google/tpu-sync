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

"""Unit tests for ci/pack_jax_wheel.py and the wheel's ABI-dispatch shims."""

import base64
import csv
import hashlib
import importlib
import importlib.machinery
import importlib.metadata
import io
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock
import zipfile

from ci import pack_jax_wheel


def _make_base_wheel(
    path: pathlib.Path, extra_files: dict[str, bytes] | None = None
) -> None:
  metadata = (
      "Metadata-Version: 2.1\n"
      "Name: tpu_sync_jax\n"
      "Version: 0.1.1\n"
      "Requires-Python: >=3.12,<3.14\n"
      "Requires-Dist: tpu-info==0.7.1\n"
      "Requires-Dist: jax==0.11.2\n"
      "Requires-Dist: jaxlib==0.11.2\n"
      "Requires-Dist: libtpu==0.0.48\n"
      "Requires-Dist: numpy\n"
      "\n"
      "# README body\n"
      "Requires-Dist: jax==0.11.2 inside README stays untouched\n"
  )
  wheel_meta = (
      "Wheel-Version: 1.0\n"
      "Generator: rules_python\n"
      "Root-Is-Purelib: false\n"
      "Tag: cp313-cp313-manylinux_2_31_x86_64\n"
  )
  files = {
      "tpu_sync/__init__.py": b"",
      "tpu_sync/frameworks/jax/_tpu_raiden_jax.so": b"BASE_SO",
      "tpu_sync_jax-0.1.1.dist-info/METADATA": metadata.encode("utf-8"),
      "tpu_sync_jax-0.1.1.dist-info/WHEEL": wheel_meta.encode("utf-8"),
      "tpu_sync_jax-0.1.1.dist-info/RECORD": b"",
  }
  if extra_files:
    files.update(extra_files)
  with zipfile.ZipFile(path, "w") as zf:
    for name, content in files.items():
      zf.writestr(name, content)


class PackJaxWheelTest(unittest.TestCase):

  def test_pack_multi_abi_wheel_rewrites_metadata_shims_pins_and_record(self):
    with tempfile.TemporaryDirectory() as tmp:
      tmp_dir = pathlib.Path(tmp)
      stage = tmp_dir / "stage"
      stage.mkdir()
      base = stage / "tpu_sync_jax-0.1.1-cp313-cp313-manylinux_2_31_x86_64.whl"
      _make_base_wheel(base)
      (stage / "_tpu_raiden_jax_0_12_0.so").write_bytes(b"ELF_0120")
      (stage / "_tpu_raiden_jax_0_11_2.so").write_bytes(b"ELF_0112")

      out_dir = tmp_dir / "dist"
      pins_file = out_dir / "jax_pins.txt"
      rc = pack_jax_wheel.main([
          "--stage-dir",
          str(stage),
          "--output-dir",
          str(out_dir),
          "--pins-file",
          str(pins_file),
      ])
      self.assertEqual(rc, 0)
      self.assertEqual(
          pins_file.read_text(encoding="utf-8"),
          "jax[tpu]>=0.11.2,<=0.12.0\n",
      )

      out = out_dir / base.name
      self.assertTrue(out.is_file())
      with zipfile.ZipFile(out, "r") as zf:
        names = set(zf.namelist())
        self.assertNotIn("tpu_sync/frameworks/jax/_tpu_raiden_jax.so", names)
        self.assertEqual(
            zf.read("tpu_sync/frameworks/jax/_tpu_raiden_jax_0_11_2.so"),
            b"ELF_0112",
        )
        self.assertEqual(
            zf.read("tpu_sync/frameworks/jax/_tpu_raiden_jax_0_12_0.so"),
            b"ELF_0120",
        )
        for shim_path in (
            "tpu_sync/frameworks/jax/_tpu_raiden_jax.py",
            "tpu_sync/frameworks/jax/_weight_synchronizer_ffi.py",
            "tpu_sync/frameworks/jax/_kv_cache_manager_ffi.py",
        ):
          self.assertIn(shim_path, names)

        meta = zf.read("tpu_sync_jax-0.1.1.dist-info/METADATA").decode("utf-8")
        headers, body = meta.split("\n\n", 1)
        self.assertIn("Requires-Dist: jax[tpu]>=0.11.2,<=0.12.0", headers)
        self.assertNotIn("jax==", headers)
        self.assertNotIn("jaxlib==", headers)
        self.assertNotIn("libtpu==", headers)
        self.assertIn("Requires-Dist: tpu-info==0.7.1", headers)
        self.assertIn("Requires-Dist: numpy", headers)
        self.assertIn("inside README stays untouched", body)

        record_text = zf.read("tpu_sync_jax-0.1.1.dist-info/RECORD").decode(
            "utf-8"
        )
        rows = list(csv.reader(io.StringIO(record_text)))
        self.assertEqual({row[0] for row in rows}, names)
        for arcname, digest_str, size_str in rows:
          if arcname.endswith("/RECORD"):
            self.assertEqual(digest_str, "")
            self.assertEqual(size_str, "")
            continue
          payload = zf.read(arcname)
          expected_digest = "sha256=" + base64.urlsafe_b64encode(
              hashlib.sha256(payload).digest()
          ).rstrip(b"=").decode("ascii")
          self.assertEqual(digest_str, expected_digest)
          self.assertEqual(size_str, str(len(payload)))

  def test_pack_single_abi_wheel_uses_exact_jax_tpu_pin(self):
    with tempfile.TemporaryDirectory() as tmp:
      tmp_dir = pathlib.Path(tmp)
      base = (
          tmp_dir / "tpu_sync_jax-0.1.1-cp312-cp312-manylinux_2_31_x86_64.whl"
      )
      _make_base_wheel(base)
      so_120 = tmp_dir / "so_0120.so"
      so_120.write_bytes(b"ELF_0120")

      out = pack_jax_wheel.pack_jax_wheel(
          base, [("0.12.0", so_120)], tmp_dir / "out"
      )
      with zipfile.ZipFile(out, "r") as zf:
        meta = zf.read("tpu_sync_jax-0.1.1.dist-info/METADATA").decode("utf-8")
        headers, _ = meta.split("\n\n", 1)
        self.assertIn("Requires-Dist: jax[tpu]==0.12.0", headers)

  def test_rejects_unconsolidated_extra_extensions(self):
    with tempfile.TemporaryDirectory() as tmp:
      tmp_dir = pathlib.Path(tmp)
      base = (
          tmp_dir / "tpu_sync_jax-0.1.1-cp312-cp312-manylinux_2_31_x86_64.whl"
      )
      _make_base_wheel(
          base,
          extra_files={
              "tpu_sync/frameworks/jax/_weight_synchronizer_ffi.so": b"EXTRA"
          },
      )
      so_112 = tmp_dir / "so_0112.so"
      so_112.write_bytes(b"ELF")

      with self.assertRaisesRegex(RuntimeError, "unconsolidated extension"):
        pack_jax_wheel.pack_jax_wheel(
            base, [("0.11.2", so_112)], tmp_dir / "out"
        )

  def test_discover_stage_dir_rejects_invalid_layouts(self):
    with tempfile.TemporaryDirectory() as tmp:
      stage = pathlib.Path(tmp)
      with self.assertRaisesRegex(ValueError, "Expected exactly 1 base wheel"):
        pack_jax_wheel.discover_stage_dir(stage)

      _make_base_wheel(
          stage / "tpu_sync_jax-0.1.1-cp312-cp312-manylinux_2_31_x86_64.whl"
      )
      with self.assertRaisesRegex(ValueError, "No staged"):
        pack_jax_wheel.discover_stage_dir(stage)

      (stage / "_tpu_raiden_jax_bad.so").write_bytes(b"")
      with self.assertRaisesRegex(ValueError, "Unexpected staged extension"):
        pack_jax_wheel.discover_stage_dir(stage)

  def _extract_wheel(self, tmp_dir: pathlib.Path) -> pathlib.Path:
    stage = tmp_dir / "stage"
    stage.mkdir()
    base = stage / "tpu_sync_jax-0.1.1-cp312-cp312-manylinux_2_31_x86_64.whl"
    _make_base_wheel(base)
    (stage / "_tpu_raiden_jax_0_11_2.so").write_bytes(b"ELF_112")
    (stage / "_tpu_raiden_jax_0_12_0.so").write_bytes(b"ELF_120")
    out = pack_jax_wheel.pack_jax_wheel(
        base, pack_jax_wheel.discover_stage_dir(stage)[1], tmp_dir / "dist"
    )
    site_dir = tmp_dir / "site"
    with zipfile.ZipFile(out, "r") as zf:
      zf.extractall(site_dir)
    return site_dir

  def test_wheel_shims_dispatch_matching_variant_without_importing_jax(self):
    had_jax = "jax" in sys.modules
    had_jaxlib = "jaxlib" in sys.modules

    with tempfile.TemporaryDirectory() as tmp:
      site_dir = self._extract_wheel(pathlib.Path(tmp))
      loaded_paths: list[str] = []
      fake_ws = types.ModuleType("weight_synchronizer_ffi")
      fake_kv = types.ModuleType("kv_cache_manager_ffi")

      class FakeLoader:

        def __init__(self, mod_name: str, path: str):
          self.mod_name = mod_name
          self.path = path

        def create_module(self, spec):
          mod = types.ModuleType(spec.name)
          mod.__file__ = self.path
          mod.weight_synchronizer_ffi = fake_ws
          mod.kv_cache_manager_ffi = fake_kv
          return mod

        def exec_module(self, module):
          loaded_paths.append(module.__file__)

      def fake_version(name: str) -> str:
        if name == "jaxlib":
          return "0.12.0"
        raise importlib.metadata.PackageNotFoundError(name)

      saved_modules = {
          k: sys.modules.pop(k)
          for k in list(sys.modules)
          if k == "tpu_sync" or k.startswith("tpu_sync.")
      }
      sys.path.insert(0, str(site_dir))
      importlib.invalidate_caches()
      try:
        with (
            mock.patch.object(
                importlib.metadata, "version", side_effect=fake_version
            ),
            mock.patch.object(
                importlib.machinery, "ExtensionFileLoader", FakeLoader
            ),
        ):
          ext = importlib.import_module(
              "tpu_sync.frameworks.jax._tpu_raiden_jax"
          )
          expected_so = str(
              (
                  site_dir / "tpu_sync/frameworks/jax/_tpu_raiden_jax_0_12_0.so"
              ).resolve()
          )
          self.assertEqual(ext.__file__, expected_so)
          self.assertEqual(loaded_paths, [expected_so])
          # Re-import returns the cached C++ module without reloading.
          self.assertIs(
              importlib.import_module(
                  "tpu_sync.frameworks.jax._tpu_raiden_jax"
              ),
              ext,
          )
          self.assertEqual(len(loaded_paths), 1)
          # Compatibility FFI shims route to submodules of _tpu_raiden_jax.
          self.assertIs(
              importlib.import_module(
                  "tpu_sync.frameworks.jax._weight_synchronizer_ffi"
              ),
              fake_ws,
          )
          self.assertIs(
              importlib.import_module(
                  "tpu_sync.frameworks.jax._kv_cache_manager_ffi"
              ),
              fake_kv,
          )
      finally:
        sys.path.remove(str(site_dir))
        for k in list(sys.modules):
          if k == "tpu_sync" or k.startswith("tpu_sync."):
            sys.modules.pop(k, None)
        sys.modules.update(saved_modules)

    self.assertEqual("jax" in sys.modules, had_jax)
    self.assertEqual("jaxlib" in sys.modules, had_jaxlib)

  def test_wheel_shim_rejects_missing_or_mismatched_jax_version(self):
    with tempfile.TemporaryDirectory() as tmp:
      site_dir = self._extract_wheel(pathlib.Path(tmp))
      saved_modules = {
          k: sys.modules.pop(k)
          for k in list(sys.modules)
          if k == "tpu_sync" or k.startswith("tpu_sync.")
      }
      sys.path.insert(0, str(site_dir))
      importlib.invalidate_caches()
      try:
        with mock.patch.object(
            importlib.metadata,
            "version",
            side_effect=importlib.metadata.PackageNotFoundError("jaxlib"),
        ):
          with self.assertRaisesRegex(
              ImportError, r"could not determine.*0\.11\.2, 0\.12\.0"
          ):
            importlib.import_module("tpu_sync.frameworks.jax._tpu_raiden_jax")

        for bad_version in ("0.12.1", "0.12.0.dev20261123", "invalid"):
          with self.subTest(version=bad_version):
            for k in list(sys.modules):
              if k == "tpu_sync" or k.startswith("tpu_sync."):
                sys.modules.pop(k, None)
            with mock.patch.object(
                importlib.metadata, "version", return_value=bad_version
            ):
              with self.assertRaisesRegex(
                  ImportError, r"exact version match.*0\.11\.2, 0\.12\.0"
              ):
                importlib.import_module(
                    "tpu_sync.frameworks.jax._tpu_raiden_jax"
                )
      finally:
        sys.path.remove(str(site_dir))
        for k in list(sys.modules):
          if k == "tpu_sync" or k.startswith("tpu_sync."):
            sys.modules.pop(k, None)
        sys.modules.update(saved_modules)


if __name__ == "__main__":
  unittest.main()
