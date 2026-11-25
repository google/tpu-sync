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

"""Unit tests for ci/pack_jax_wheel.py and the wheel's ABI-dispatch shim."""

import importlib
import importlib.machinery
import importlib.metadata
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

from ci import pack_jax_wheel

_METADATA = (
    "Metadata-Version: 2.1\nName: tpu_sync_jax\nVersion: 0.1.1\n"
    "Requires-Python: >=3.12\nRequires-Dist: jax-ai-stack==0.1.0\n"
    "Requires-Dist: libtpu-nightly==0.0.49\nRequires-Dist: jax==0.11.2\n"
    "Requires-Dist: jaxlib==0.11.2\nRequires-Dist: libtpu==0.0.48\n\n# README\n"
    "Requires-Dist: jax==0.11.2 inside README stays untouched\n"
)


def _make_pkg(root: str, versions: list[str]) -> pathlib.Path:
  pkg = pathlib.Path(root)
  jax_dir = pkg / "tpu_sync/frameworks/jax"
  jax_dir.mkdir(parents=True)
  for d in (pkg / "tpu_sync", pkg / "tpu_sync/frameworks", jax_dir):
    (d / "__init__.py").write_text("", "utf-8")
  (jax_dir / "_tpu_raiden_jax.so").write_bytes(b"base")
  for v in versions:
    (jax_dir / f"_tpu_raiden_jax_{v.replace('.', '_')}.so").write_bytes(b"abi")
  dist = pkg / "tpu_sync_jax-0.1.1.dist-info"
  dist.mkdir()
  (dist / "METADATA").write_text(_METADATA, "utf-8")
  return pkg


class PackJaxWheelTest(unittest.TestCase):

  def test_prepare_wheel_and_requires_dist(self):
    fn = pack_jax_wheel.compute_jax_requires_dist
    self.assertEqual(fn(["0.12.0"]), "jax[tpu]==0.12.0")
    self.assertEqual(fn(["0.11.2", "0.12.0"]), "jax[tpu]>=0.11.2,<=0.12.0")
    self.assertEqual(
        fn(["0.10.0", "0.12.0"]), "jax[tpu]>=0.10.0,<=0.12.0,!=0.11.*"
    )
    self.assertEqual(
        fn(["0.11.0", "0.11.2"]), "jax[tpu]>=0.11.0,<=0.11.2,!=0.11.1"
    )
    self.assertEqual(
        fn(["0.11.2", "0.12.1"]), "jax[tpu]>=0.11.2,<=0.12.1,!=0.12.0"
    )
    self.assertEqual(
        fn(["0.11.2", "0.12.0", "0.12.2"]),
        "jax[tpu]>=0.11.2,<=0.12.2,!=0.12.1",
    )
    with tempfile.TemporaryDirectory() as tmp:
      pkg = _make_pkg(tmp, ["0.11.2", "0.12.0"])
      jax_dir = pkg / "tpu_sync/frameworks/jax"
      self.assertEqual(pack_jax_wheel.main([str(pkg), "0.12.0", "0.11.2"]), 0)
      self.assertFalse((jax_dir / "_tpu_raiden_jax.so").exists())
      self.assertTrue((jax_dir / "_tpu_raiden_jax.py").is_file())
      self.assertFalse(list(jax_dir.glob("*_ffi.*")))
      hdr, _, body = (
          (pkg / "tpu_sync_jax-0.1.1.dist-info/METADATA")
          .read_text()
          .partition("\n\n")
      )
      self.assertIn("Requires-Dist: jax[tpu]>=0.11.2,<=0.12.0", hdr)
      self.assertIn("Requires-Dist: jax-ai-stack==0.1.0", hdr)
      self.assertIn("Requires-Dist: libtpu-nightly==0.0.49", hdr)
      for removed in ("jax==0.11.2", "jaxlib==", "libtpu==0.0.48"):
        self.assertNotIn(removed, hdr)
      self.assertIn("Requires-Dist: jax==0.11.2 inside README", body)

  def test_rejects_forbidden_or_mismatched_extensions(self):
    with tempfile.TemporaryDirectory() as tmp:
      pkg = _make_pkg(tmp, ["0.11.2"])
      (pkg / "tpu_sync/frameworks/jax/_weight_synchronizer_ffi.so").write_bytes(
          b"x"
      )
      with self.assertRaisesRegex(RuntimeError, "Forbidden standalone"):
        pack_jax_wheel.prepare_unpacked_jax_wheel(pkg, ["0.11.2"])
    with tempfile.TemporaryDirectory() as tmp:
      pkg = _make_pkg(tmp, ["0.11.2"])
      with self.assertRaisesRegex(RuntimeError, "Variant .so mismatch"):
        pack_jax_wheel.prepare_unpacked_jax_wheel(pkg, ["0.11.2", "0.12.0"])

  def test_shim_dispatches_abi_without_importing_jax(self):
    with tempfile.TemporaryDirectory() as tmp:
      pkg = _make_pkg(tmp, ["0.11.2", "0.12.0"])
      pack_jax_wheel.prepare_unpacked_jax_wheel(pkg, ["0.11.2", "0.12.0"])

      class FakeLoader:

        def __init__(self, name: str, path: str):
          self.name, self.path = name, path

        def create_module(self, spec):
          mod = type(sys)(spec.name)
          mod.loaded_so = pathlib.Path(self.path).name
          return mod

        def exec_module(self, module):
          del module

      target = "tpu_sync.frameworks.jax._tpu_raiden_jax"
      mods = (
          "tpu_sync",
          "tpu_sync.frameworks",
          "tpu_sync.frameworks.jax",
          target,
      )
      saved = {k: sys.modules.pop(k, None) for k in (*mods, "jax", "jaxlib")}
      sys.path.insert(0, str(pkg))
      try:
        with (
            mock.patch.object(
                importlib.metadata, "version", return_value="0.12.0"
            ),
            mock.patch.object(
                importlib.machinery, "ExtensionFileLoader", FakeLoader
            ),
        ):
          ext = importlib.import_module(target)
          self.assertEqual(
              ext.__name__, "tpu_sync.frameworks.jax._tpu_raiden_jax"
          )
          self.assertEqual(
              ext.__spec__.name, "tpu_sync.frameworks.jax._tpu_raiden_jax"
          )
          self.assertEqual(ext.loaded_so, "_tpu_raiden_jax_0_12_0.so")
          self.assertFalse({"jax", "jaxlib"} & sys.modules.keys())
          self.assertIs(importlib.import_module(target), ext)
        for bad in ("0.12.1", "0.12.0.dev20261123", None):
          sys.modules.pop(target, None)
          err = (
              importlib.metadata.PackageNotFoundError("jax")
              if bad is None
              else None
          )
          with mock.patch.object(
              importlib.metadata, "version", side_effect=err, return_value=bad
          ):
            with self.assertRaises(ImportError):
              importlib.import_module(target)
      finally:
        sys.path.remove(str(pkg))
        for k in mods:
          sys.modules.pop(k, None)
        for k, v in saved.items():
          if v is not None:
            sys.modules[k] = v


if __name__ == "__main__":
  unittest.main()
