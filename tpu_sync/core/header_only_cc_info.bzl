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

# =============================================================================

"""Rule to create a CcInfo containing only headers from dependencies."""

def _header_only_cc_info_impl(ctx):
    compilation_contexts = [dep[CcInfo].compilation_context for dep in ctx.attr.deps]
    if ctx.attr.defines:
        defines_context = cc_common.create_compilation_context(
            defines = depset(ctx.attr.defines),
        )
        compilation_contexts.append(defines_context)
    return [
        CcInfo(
            compilation_context = cc_common.merge_compilation_contexts(
                compilation_contexts = compilation_contexts,
            ),
            linking_context = cc_common.create_linking_context(
                linker_inputs = depset(),
            ),
        ),
    ]

header_only_cc_info = rule(
    attrs = {
        "deps": attr.label_list(),
        "defines": attr.string_list(),
    },
    implementation = _header_only_cc_info_impl,
)
