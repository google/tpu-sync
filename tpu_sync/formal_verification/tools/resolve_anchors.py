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
"""Resolve symbol and code-fragment anchors in `<!-- source-map:begin -->` tables.

Usage (run from `tpu_sync/formal_verification/` or repo root):
    python3 tpu_sync/formal_verification/tools/resolve_anchors.py [--ref REF] [--check | --apply] [SOURCE ...]
    python3 tpu_sync/formal_verification/tools/resolve_anchors.py --diff OLD_REF NEW_REF [--apply] [SOURCE ...]

Each `<!-- source-map:begin --> ... <!-- source-map:end -->` block is a Markdown
table with `File`, `Symbol / anchor` (or `Symbol`), and `Lines` columns:

    | File | Symbol / anchor | Lines |
    |---|---|---|
    | `recv.h` | `TransferReceiveSession::TryBeginRecvOp` | `:117-122` |
    | `recv.h` | `TransferReceiveSession` (`std::vector<int64_t> blocks_received_per_shard_` .. `std::vector<raiden::PjRtCopyFuture> h2d_futures_`) | `:273-296` |
    | `recv.cc` | `TransferReceiveSession::ExecuteLayerH2d` (`future.OnReady`) | `:702-760` |
    | `send_drain_test.cc` | `RecvDrainTest.ExpiredReceiveKeepsStagingUntilH2dEnds` | `:751-778` |

Supported anchor forms in the Symbol column:
  1. `` `Class::Method` `` / `` `Class::Method(param_substr)` `` / `` `Func` ``:
     C++ function/method/constructor/destructor definition.
  2. `` `Suite.TestName` ``: GTest `TEST` / `TEST_F` / `TEST_P` definition (matches `DISABLED_` too).
  3. `` `Class` ``: C++ `class` / `struct` / `enum` or proto `message` / `service` definition.
  4. `` `*` ``: whole file (use with a fragment or `..` range for file-scope items).
  5. `` `Symbol` (`fragment`) ``: unique occurrence of `fragment` inside `Symbol`; if the
     statement opens `{ ... }`, spans to the matching closing `}` (and trailing `);` / `;`),
     otherwise spans the statement/line.
  6. `` `Symbol` (`start_fragment` .. `end_fragment`) ``: line of `start_fragment` through
     line of `end_fragment` inside `Symbol`.
"""

import argparse
import difflib
import os
import re
import subprocess
import sys

ALIASES = {
    "recv.cc": "tpu_sync/kv_cache/transfer_receive_session.cc",
    "recv.h": "tpu_sync/kv_cache/transfer_receive_session.h",
    "send.cc": "tpu_sync/kv_cache/transfer_send_session.cc",
    "send.h": "tpu_sync/kv_cache/transfer_send_session.h",
    "session.h": "tpu_sync/kv_cache/transfer_session.h",
    "mgr.cc": "tpu_sync/kv_cache/kv_cache_manager_with_transfer.cc",
    "mgr.h": "tpu_sync/kv_cache/kv_cache_manager_with_transfer.h",
    "bt.cc": "tpu_sync/transport/block_transport.cc",
    "bt.h": "tpu_sync/transport/block_transport.h",
    "bts.cc": "tpu_sync/transport/block_transport_session.cc",
    "bts.h": "tpu_sync/transport/block_transport_session.h",
    "arena.cc": "tpu_sync/core/staging_arena.cc",
    "arena.h": "tpu_sync/core/staging_arena.h",
    "coord.cc": "tpu_sync/kv_cache/reshard/reshard_coordinator.cc",
    "coord.h": "tpu_sync/kv_cache/reshard/reshard_coordinator.h",
    "base.cc": "tpu_sync/kv_cache/kv_cache_manager_base.cc",
    "base.h": "tpu_sync/kv_cache/kv_cache_manager_base.h",
    "slots.cc": "tpu_sync/core/pool_reshard_send_slots.cc",
    "slots.h": "tpu_sync/core/pool_reshard_send_slots.h",
    "reshard_recv.cc": "tpu_sync/kv_cache/reshard_receive_session.cc",
    "reshard_recv.h": "tpu_sync/kv_cache/reshard_receive_session.h",
    "reshard_send.cc": "tpu_sync/kv_cache/reshard_send_session.cc",
    "reshard_send.h": "tpu_sync/kv_cache/reshard_send_session.h",
    "reg.cc": "tpu_sync/kv_cache/reshard/request_block_registry.cc",
    "reg.h": "tpu_sync/kv_cache/reshard/request_block_registry.h",
    "planner.cc": "tpu_sync/kv_cache/reshard/pool_reshard_planner.cc",
    "planner.h": "tpu_sync/kv_cache/reshard/pool_reshard_planner.h",
    "send_drain_test.cc":
        "tpu_sync/kv_cache/kv_cache_manager_with_transfer_send_drain_test.cc",
    "control_test.cc":
        "tpu_sync/kv_cache/kv_cache_manager_with_transfer_control_test.cc",
    "pool_reshard_test.cc":
        "tpu_sync/kv_cache/kv_cache_manager_with_transfer_pool_reshard_test.cc",
    "reshard_recv_test.cc":
        "tpu_sync/kv_cache/reshard_receive_session_test.cc",
    "reshard_send_test.cc":
        "tpu_sync/kv_cache/reshard_send_session_test.cc",
}

EXTERNAL = {
    "tpu_connector.py",
    "sched.py",
    "scheduler.py",
    "base.py",
    "conn.py",
    "tpu_raiden_connector.py",
    "utils.py",
    "kvcfg.py",
    "mixin.py",
    "kv_transfer.py",
    "vllm/config/kv_transfer.py",
    "vllm/distributed/kv_transfer/kv_connector/v1/base.py",
}

RAW_CITE = re.compile(
    r"(?P<tok>[\w./-]+(?:\.\{h,cc\}|\.(?:cc|h|py|proto))|`?\.(?:cc|h)|`):(?P<nums>\d+(?:\s*[-\u2013]\s*\d+)?)"
)
PROSE_LINE_CITE = re.compile(r"\blines?\s+\d+(?:\s*[-\u2013]\s*\d+)?\b", re.IGNORECASE)
RAW_STR_OPEN = re.compile(r'R"([A-Za-z0-9_{}\[\]#<>%:;.?*+\-/^&|~!=,\']*)\(')

NO_MAP_FILES = {
    "TpuSyncVerify/Common/ListAux.lean",
    "TpuSyncVerify/Common/ModelCheck.lean",
    "TpuSyncVerify/Common/System.lean",
    "docs/conventions.md",
}


def find_repo(start):
    d = os.path.abspath(start)
    while d != "/" and not os.path.exists(os.path.join(d, ".git")):
        d = os.path.dirname(d)
    return d


def split_md_row(line):
    """Split a markdown table row on `|` outside backticks.
    Returns `(cells, pipe_positions)` where `cells[i]` is the raw substring
    between `pipe_positions[i]` and `pipe_positions[i+1]`."""
    pipes = []
    in_bt = False
    for idx, ch in enumerate(line):
        if ch == "`":
            in_bt = not in_bt
        elif ch == "|" and not in_bt:
            pipes.append(idx)
    if len(pipes) < 2:
        return [], []
    cells = [line[pipes[i] + 1:pipes[i + 1]] for i in range(len(pipes) - 1)]
    return cells, pipes


def strip_comments_and_strings(text):
    """Replace comments and string/char literals with spaces, preserving newlines."""
    out = list(text)
    n = len(text)
    i = 0
    while i < n:
        c = text[i]
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] != "\n":
                out[i] = " "
                i += 1
        elif c == "/" and i + 1 < n and text[i + 1] == "*":
            out[i] = " "
            out[i + 1] = " "
            i += 2
            while i < n and not (text[i] == "*" and i + 1 < n and text[i + 1] == "/"):
                if out[i] != "\n":
                    out[i] = " "
                i += 1
            if i < n:
                out[i] = " "
                out[i + 1] = " "
                i += 2
        elif (
            c == "R"
            and i + 1 < n
            and text[i + 1] == '"'
            and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_"))
        ):
            rm = RAW_STR_OPEN.match(text, i)
            if rm:
                delim = ")" + rm.group(1) + '"'
                end_idx = text.find(delim, rm.end())
                end_pos = n if end_idx == -1 else end_idx + len(delim)
                while i < end_pos:
                    if out[i] != "\n":
                        out[i] = " "
                    i += 1
            else:
                i += 1
        elif c == "'" and i > 0 and i + 1 < n and text[i - 1].isalnum() and text[i + 1].isalnum():
            # C++14 digit separator inside numeric literal (e.g. 1'000'000)
            i += 1
        elif c in ('"', "'"):
            quote = c
            out[i] = " "
            i += 1
            while i < n and text[i] != quote:
                if text[i] == "\\" and i + 1 < n:
                    if out[i] != "\n":
                        out[i] = " "
                    i += 1
                if i < n and out[i] != "\n":
                    out[i] = " "
                i += 1
            if i < n:
                out[i] = " "
                i += 1
        else:
            i += 1
    return "".join(out)


class SourceFile:
    def __init__(self, text):
        self.text = text
        self.clean = strip_comments_and_strings(text)
        self.lines = text.splitlines()
        self.clean_lines = self.clean.splitlines()
        self.line_starts = [0]
        for idx, ch in enumerate(text):
            if ch == "\n":
                self.line_starts.append(idx + 1)

    def pos_to_line(self, pos):
        lo, hi = 0, len(self.line_starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.line_starts[mid] <= pos:
                lo = mid
            else:
                hi = mid - 1
        return lo + 1

    def line_to_pos(self, lineno):
        return self.line_starts[lineno - 1]

    def match_brace(self, open_pos):
        depth = 0
        for i in range(open_pos, len(self.clean)):
            c = self.clean[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return i
        return None

    def match_paren(self, open_pos):
        depth = 0
        for i in range(open_pos, len(self.clean)):
            c = self.clean[i]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    return i
        return None


def find_gtest_spans(sf, suite, test):
    pat = re.compile(
        rf"\bTEST(?:_F|_P)?\s*\(\s*{re.escape(suite)}\s*,\s*(?:DISABLED_)?{re.escape(test)}\s*\)"
    )
    spans = []
    for m in pat.finditer(sf.clean):
        brace_pos = sf.clean.find("{", m.end())
        if brace_pos == -1:
            continue
        end_pos = sf.match_brace(brace_pos)
        if end_pos is None:
            continue
        spans.append((sf.pos_to_line(m.start()), sf.pos_to_line(end_pos)))
    return spans


def find_class_spans(sf, cls_name):
    pat = re.compile(rf"\b(?:class|struct|enum(?:\s+class)?|message|service)\s+{re.escape(cls_name)}\b")
    spans = []
    for m in pat.finditer(sf.clean):
        rest = sf.clean[m.end():]
        semi = rest.find(";")
        brace = rest.find("{")
        if brace == -1 or (semi != -1 and semi < brace):
            continue
        brace_pos = m.end() + brace
        end_pos = sf.match_brace(brace_pos)
        if end_pos is None:
            continue
        spans.append((sf.pos_to_line(m.start()), sf.pos_to_line(end_pos)))
    return spans


def _scan_to_body_brace(clean, close_paren_pos):
    i = close_paren_pos + 1
    n = len(clean)
    paren = 0
    while i < n:
        c = clean[i]
        if c == "(":
            paren += 1
        elif c == ")":
            if paren > 0:
                paren -= 1
            else:
                return None
        elif paren == 0:
            if c == "{":
                return i
            if c == ";":
                return None
            if c == ":":
                i += 1
                while i < n:
                    while i < n and clean[i].isspace():
                        i += 1
                    if i < n and clean[i] == "{":
                        return i
                    id_m = re.match(r"[\w:<> ,]+", clean[i:])
                    if not id_m:
                        return None
                    i += id_m.end()
                    while i < n and clean[i].isspace():
                        i += 1
                    if i < n and clean[i] in ("(", "{"):
                        open_c = clean[i]
                        close_c = ")" if open_c == "(" else "}"
                        d = 0
                        while i < n:
                            if clean[i] == open_c:
                                d += 1
                            elif clean[i] == close_c:
                                d -= 1
                                if d == 0:
                                    i += 1
                                    break
                            i += 1
                        while i < n and clean[i].isspace():
                            i += 1
                        if i < n and clean[i] == ",":
                            i += 1
                            continue
                        if i < n and clean[i] == "{":
                            return i
                    return None
        i += 1
    return None


def _adjust_start_line(sf, sym_line):
    if sym_line <= 1:
        return sym_line
    prev = sf.clean_lines[sym_line - 2].strip()
    if not prev or prev.startswith("#"):
        return sym_line
    if prev[-1] in (";", "{", "}", ",", ":"):
        return sym_line
    return sym_line - 1


def find_func_spans(sf, sym, region=None, param_filter=None):
    lo_pos = sf.line_to_pos(region[0]) if region else 0
    hi_pos = (sf.line_to_pos(region[1] + 1)
              if region and region[1] < len(sf.lines) else len(sf.clean))
    pat = re.compile(rf"(?<![\w:~]){re.escape(sym)}\s*\(")
    spans = []
    for m in pat.finditer(sf.clean, lo_pos, hi_pos):
        open_paren = sf.clean.find("(", m.start())
        close_paren = sf.match_paren(open_paren)
        if close_paren is None:
            continue
        if param_filter and param_filter not in sf.text[open_paren:close_paren + 1]:
            continue
        brace_pos = _scan_to_body_brace(sf.clean, close_paren)
        if brace_pos is None or brace_pos >= hi_pos:
            continue
        end_pos = sf.match_brace(brace_pos)
        if end_pos is None:
            continue
        sym_line = sf.pos_to_line(m.start())
        line_prefix = sf.clean_lines[sym_line - 1][:m.start() - sf.line_to_pos(sym_line)].strip()
        start_line = _adjust_start_line(sf, sym_line) if not line_prefix else sym_line
        spans.append((start_line, sf.pos_to_line(end_pos)))
    return spans


def find_py_def_spans(sf, sym):
    pat = re.compile(rf"^(\s*)def\s+{re.escape(sym)}\s*\(")
    spans = []
    n = len(sf.lines)
    for idx, line in enumerate(sf.lines):
        m = pat.match(line)
        if not m:
            continue
        indent = len(m.group(1))
        start_ln = idx + 1
        end_ln = start_ln
        for j in range(idx + 1, n):
            s = sf.lines[j].strip()
            if not s or s.startswith("#"):
                continue
            cur_indent = len(sf.lines[j]) - len(sf.lines[j].lstrip())
            if cur_indent <= indent:
                break
            end_ln = j + 1
        spans.append((start_ln, end_ln))
    return spans


def resolve_symbol(sf, sym):
    if sym == "*":
        return [(1, len(sf.lines))]
    param_filter = None
    if "(" in sym and sym.endswith(")"):
        sym, param_filter = sym[:-1].split("(", 1)
        param_filter = param_filter.strip()
    if "." in sym and "::" not in sym:
        suite, test = sym.split(".", 1)
        return find_gtest_spans(sf, suite, test)
    if "::" not in sym and not param_filter:
        cls_spans = find_class_spans(sf, sym)
        if cls_spans:
            return cls_spans
    spans = find_func_spans(sf, sym, param_filter=param_filter)
    if spans:
        return spans
    if "::" in sym:
        cls_name, meth = sym.rsplit("::", 1)
        for cls_span in find_class_spans(sf, cls_name):
            spans.extend(find_func_spans(sf, meth, region=cls_span, param_filter=param_filter))
        return spans
    py_spans = find_py_def_spans(sf, sym)
    if py_spans:
        return py_spans
    return []


def find_unique_line(sf, lo, hi, frag):
    hits = [
        ln for ln in range(lo, hi + 1)
        if frag in sf.lines[ln - 1]
    ]
    if len(hits) > 1:
        code_hits = [
            ln for ln in hits
            if frag in sf.lines[ln - 1].split("//")[0]
        ]
        if len(code_hits) == 1:
            return code_hits
    return hits


def scan_statement_or_block_end(sf, start_line, sym_hi):
    clean_first = sf.clean_lines[start_line - 1].strip()
    if not clean_first:
        return start_line
    start_pos = sf.line_to_pos(start_line)
    end_limit = (sf.line_to_pos(sym_hi + 1)
                 if sym_hi < len(sf.lines) else len(sf.clean))
    paren = 0
    brace = 0
    saw_brace = False
    i = start_pos
    while i < end_limit:
        c = sf.clean[i]
        if c == "(":
            paren += 1
        elif c == ")":
            if paren > 0:
                paren -= 1
        elif c == "{":
            brace += 1
            saw_brace = True
        elif c == "}":
            if brace > 0:
                brace -= 1
                if brace == 0 and paren == 0 and saw_brace:
                    rest = sf.clean[i + 1:end_limit].lstrip()
                    if rest.startswith("while") and (len(rest) == 5 or not (rest[5].isalnum() or rest[5] == "_")):
                        saw_brace = False
                    else:
                        return sf.pos_to_line(i)
        elif c == ";" and brace == 0 and paren == 0:
            return sf.pos_to_line(i)
        i += 1
    return start_line


def resolve_spec(sf, spec):
    tokens = re.findall(r"`([^`]+)`", spec)
    if not tokens:
        return None, None, "no backticked symbol in spec"
    sym = tokens[0]
    cand_spans = resolve_symbol(sf, sym)
    if not cand_spans:
        return None, None, f"symbol `{sym}` not found"

    if len(tokens) == 1:
        if len(cand_spans) > 1:
            return None, None, f"symbol `{sym}` is ambiguous ({len(cand_spans)} overloads: {cand_spans})"
        return cand_spans[0][0], cand_spans[0][1], None

    if len(tokens) == 3:
        f1, f2 = tokens[1], tokens[2]
        matched = []
        for s_lo, s_hi in cand_spans:
            m1 = find_unique_line(sf, s_lo, s_hi, f1)
            if len(m1) != 1:
                continue
            m2 = find_unique_line(sf, m1[0], s_hi, f2)
            if len(m2) != 1:
                continue
            matched.append((m1[0], m2[0]))
        if not matched:
            return None, None, f"range `{f1}` .. `{f2}` not uniquely found in `{sym}`"
        if len(matched) > 1:
            return None, None, f"range `{f1}` .. `{f2}` matched multiple overloads of `{sym}`: {matched}"
        return matched[0][0], matched[0][1], None

    if len(tokens) == 2:
        frag = tokens[1]
        matched = []
        ambig_lines = None
        for s_lo, s_hi in cand_spans:
            m = find_unique_line(sf, s_lo, s_hi, frag)
            if len(m) == 1:
                end_ln = scan_statement_or_block_end(sf, m[0], s_hi)
                matched.append((m[0], end_ln))
            elif len(m) > 1:
                ambig_lines = m
        if not matched:
            if ambig_lines:
                return None, None, f"fragment `{frag}` matched multiple lines {ambig_lines} in `{sym}`"
            return None, None, f"fragment `{frag}` not found in `{sym}`"
        if len(matched) > 1:
            return None, None, f"fragment `{frag}` matched multiple overloads of `{sym}`: {matched}"
        return matched[0][0], matched[0][1], None

    return None, None, f"unsupported anchor syntax (expected 1, 2, or 3 backticked tokens, got {len(tokens)})"


def fmt_range(lo, hi):
    return f":{lo}" if lo == hi else f":{lo}-{hi}"


class FileLoader:
    def __init__(self, repo, ref="WORKTREE"):
        self.repo = repo
        self.ref = ref
        self.cache = {}

    def resolve_path(self, tok):
        tok = tok.strip("`").strip()
        if tok in ALIASES:
            return ALIASES[tok]
        if tok.startswith("tpu_sync/"):
            return tok
        for sub in ("tpu_sync",):
            for root, dirs, files in os.walk(os.path.join(self.repo, sub)):
                # The corpus itself (and its .lake build tree) holds no C++ sources.
                dirs[:] = [d for d in dirs if d not in ("formal_verification", ".lake")]
                if tok in files:
                    return os.path.relpath(os.path.join(root, tok), self.repo)
        return None

    def load(self, path):
        if path in self.cache:
            return self.cache[path]
        if self.ref == "WORKTREE":
            full = os.path.join(self.repo, path)
            if not os.path.exists(full):
                return None
            with open(full, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            res = subprocess.run(
                ["git", "show", f"{self.ref}:{path}"],
                cwd=self.repo,
                capture_output=True,
                text=True,
            )
            if res.returncode != 0:
                if path.startswith("tpu_sync/formal_verification/"):
                    full = os.path.join(self.repo, path)
                    if os.path.exists(full):
                        with open(full, "r", encoding="utf-8") as f:
                            text = f.read()
                    else:
                        return None
                else:
                    return None
            else:
                text = res.stdout
        sf = SourceFile(text)
        self.cache[path] = sf
        return sf


def process_file(src_path, rel_path, loader, old_loader=None, apply=False):
    with open(src_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines(keepends=True)

    in_map = False
    file_col = 0
    sym_col = 1
    lines_col = 2
    out_lines = []
    stats = {"same": 0, "shift": 0, "modified": 0, "updated": 0, "fail": 0, "stray": 0}
    has_map = False

    for lineno, line in enumerate(lines, 1):
        stripped = line.strip()
        if "<!-- source-map:begin -->" in stripped:
            in_map = True
            has_map = True
            out_lines.append(line)
            continue
        if "<!-- source-map:end -->" in stripped:
            in_map = False
            out_lines.append(line)
            continue

        if not in_map:
            out_lines.append(line)
            continue

        if not stripped.startswith("|") or stripped.startswith("|---") or stripped.startswith("| ---"):
            out_lines.append(line)
            continue

        raw_cells, pipes = split_md_row(line)
        cells = [c.strip() for c in raw_cells]
        if len(cells) < 3:
            out_lines.append(line)
            continue

        if any(c.lower() in ("file", "symbol", "symbol / anchor", "lines") for c in cells):
            for idx, c in enumerate(cells):
                cl = c.lower()
                if cl == "file":
                    file_col = idx
                elif cl in ("symbol", "symbol / anchor"):
                    sym_col = idx
                elif cl == "lines":
                    lines_col = idx
            out_lines.append(line)
            continue

        file_tok = cells[file_col]
        spec = cells[sym_col]
        raw_lines_cell = cells[lines_col].strip()
        cur_lines_cell = raw_lines_cell.strip("`").strip()

        repo_path = loader.resolve_path(file_tok)
        if not repo_path:
            print(f"{rel_path}:{lineno}  FAIL  unknown file {file_tok}")
            stats["fail"] += 1
            out_lines.append(line)
            continue

        sf = loader.load(repo_path)
        if sf is None:
            print(f"{rel_path}:{lineno}  FAIL  cannot load {repo_path} at {loader.ref}")
            stats["fail"] += 1
            out_lines.append(line)
            continue

        lo, hi, err = resolve_spec(sf, spec)
        if err:
            print(f"{rel_path}:{lineno}  FAIL  {file_tok} {spec}: {err}")
            stats["fail"] += 1
            out_lines.append(line)
            continue

        new_range = fmt_range(lo, hi)

        if old_loader is not None:
            old_sf = old_loader.load(repo_path)
            old_lo, old_hi, old_err = (resolve_spec(old_sf, spec) if old_sf else (None, None, "missing file"))
            if old_err:
                print(f"{rel_path}:{lineno}  NEW   {file_tok} {spec}  -> {new_range}")
                stats["modified"] += 1
            else:
                old_range = fmt_range(old_lo, old_hi)
                old_slice = old_sf.lines[old_lo - 1:old_hi]
                new_slice = sf.lines[lo - 1:hi]
                if old_slice == new_slice:
                    if old_range == new_range:
                        stats["same"] += 1
                    else:
                        print(f"{rel_path}:{lineno}  shift     {file_tok} {spec}  {old_range} -> {new_range}")
                        stats["shift"] += 1
                else:
                    print(f"{rel_path}:{lineno}  MODIFIED  {file_tok} {spec}  {old_range} -> {new_range}")
                    for dline in difflib.unified_diff(
                        old_slice, new_slice,
                        fromfile=f"{repo_path}{old_range}@{old_loader.ref}",
                        tofile=f"{repo_path}{new_range}@{loader.ref}",
                        lineterm="",
                    ):
                        print(f"    {dline}")
                    stats["modified"] += 1
        else:
            if raw_lines_cell == f"`{new_range}`":
                stats["same"] += 1
            else:
                print(f"{rel_path}:{lineno}  drift  {file_tok} {spec}: table has `{raw_lines_cell}`, resolved `` `{new_range}` ``")
                stats["updated"] += 1

        if raw_lines_cell != f"`{new_range}`" and apply:
            p_start = pipes[lines_col] + 1
            p_end = pipes[lines_col + 1]
            line = line[:p_start] + f" `{new_range}` " + line[p_end:]
        out_lines.append(line)

    if apply:
        with open(src_path, "w", encoding="utf-8") as f:
            f.writelines(out_lines)

    return stats, has_map


def check_stray_citations(src_path, rel_path):
    with open(src_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    in_map = False
    in_fence = False
    stray = 0
    for lineno, line in enumerate(lines, 1):
        s = line.strip()
        if "<!-- source-map:begin -->" in s:
            in_map = True
            continue
        if "<!-- source-map:end -->" in s:
            in_map = False
            continue
        if in_map:
            continue
        if s.startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence:
            for m in RAW_CITE.finditer(line):
                tok = m.group("tok").strip("`")
                if tok in EXTERNAL or os.path.basename(tok) in EXTERNAL:
                    continue
                print(f"{rel_path}:{lineno}  STRAY raw citation `{m.group(0)}` outside source-map")
                stray += 1
            for pm in PROSE_LINE_CITE.finditer(line):
                print(f"{rel_path}:{lineno}  STRAY prose line citation `{pm.group(0)}` outside source-map")
                stray += 1
    return stray


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", default="WORKTREE", help="Git ref to resolve against (default: WORKTREE)")
    ap.add_argument("--diff", nargs=2, metavar=("OLD_REF", "NEW_REF"),
                    help="Compare resolved spans between OLD_REF and NEW_REF")
    ap.add_argument("--apply", action="store_true", help="Write resolved line numbers into source-map tables")
    ap.add_argument("--check", action="store_true", help="Exit non-zero on any drift, failure, or stray raw citation")
    ap.add_argument("sources", nargs="*", help="Source files to process")
    args = ap.parse_args()

    repo = find_repo(os.getcwd())
    ver_root = os.path.join(repo, "tpu_sync", "formal_verification")

    if args.sources:
        sources = [os.path.abspath(s) for s in args.sources]
    else:
        sources = []
        for sub in ("TpuSyncVerify", "docs"):
            for root, _, files in os.walk(os.path.join(ver_root, sub)):
                for fn in sorted(files):
                    if fn.endswith((".lean", ".md")):
                        sources.append(os.path.join(root, fn))

    if args.diff:
        old_loader = FileLoader(repo, args.diff[0])
        loader = FileLoader(repo, args.diff[1])
    else:
        old_loader = None
        loader = FileLoader(repo, args.ref)

    totals = {"same": 0, "shift": 0, "modified": 0, "updated": 0, "fail": 0, "stray": 0}
    checked_files = 0
    for src in sources:
        rel = os.path.relpath(src, ver_root)
        with open(src, "r", encoding="utf-8") as f:
            has_begin = "<!-- source-map:begin -->" in f.read()
        # A module split into a directory (`X.lean` umbrella + `X/Part.lean`
        # files) keeps its source map in the umbrella; the parts carry no map
        # and are only checked for stray citations.
        is_part = os.path.isfile(os.path.dirname(src) + ".lean")
        if not has_begin and (rel in NO_MAP_FILES or is_part):
            if rel != "docs/conventions.md":
                totals["stray"] += check_stray_citations(src, rel)
            continue
        checked_files += 1
        if not has_begin:
            print(f"{rel}:1  FAIL  missing <!-- source-map:begin --> table")
            totals["fail"] += 1
            totals["stray"] += check_stray_citations(src, rel)
            continue
        st, _ = process_file(src, rel, loader, old_loader=old_loader, apply=args.apply)
        st["stray"] = check_stray_citations(src, rel)
        for k in totals:
            totals[k] += st[k]

    print(
        f"\nSummary ({checked_files} file(s)): "
        f"{totals['same']} same, {totals['shift']} shift, "
        f"{totals['modified']} modified, {totals['updated']} drift, "
        f"{totals['fail']} fail, {totals['stray']} stray"
    )
    if args.check and (
        totals["fail"] > 0
        or totals["stray"] > 0
        or (not args.apply and (totals["updated"] > 0 or totals["modified"] > 0))
    ):
        sys.exit(1)
    if totals["fail"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
