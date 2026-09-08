# Project Rules for tpu_raiden Development

## C++ Coding Preferences

* When checking if a string/string_view contains a substring, ALWAYS prefer `absl::StrContains` (from `third_party/absl/strings/match.h`) instead of `std::string::find() != std::string::npos`.

## Google3 Forge Sandbox & Unit Test Data Dependencies

*   Unit tests running under Forge (e.g. `blaze test`) run in a sandboxed network environment and CANNOT access CNS files (`/cns/...`).
*   ALWAYS declare all external assets (such as weight files `.safetensors` or compiled binaries `.bin`) required by a unit test as local data dependencies in the test's `BUILD` rule under `data = [ ... ]`.
*   ALWAYS resolve files in unit tests locally via `::testing::SrcDir()` relative to the sandbox `runfiles` path, and NEVER hardcode `/cns/...` paths in unit tests.

## TPU-Dependent Unit Test Execution

*   NEVER attempt to run physical TPU-dependent unit tests locally on your Cloudtop workstation (e.g. using `blaze run` or executing binaries directly). Cloudtop does not have physical TPUs attached, so these executions will fail.
*   ALWAYS run physical TPU-dependent unit tests via Forge using `blaze test` (e.g., `blaze test //path/to:test`), which correctly allocates remote TPU simulators or hardware resources (like Ghostlite/TPU pools) inside the Borg testing sandbox.

## CNS File Operations

*   ALWAYS use the `fileutil` command directly without any absolute path prefix (e.g. use `fileutil ls` or `fileutil cp`, and NEVER `/google/bin/query/fileutil/fileutil`) when performing CNS/Colossus operations.

## VCS Operations & Binary File Safety

*   NEVER run `hg diff`, `hg status`, or other version control operations directly on the entire workspace when it contains large binary data assets (such as weight files `.safetensors` or compiled binaries `.bin`), to prevent execution hangs.
*   ALWAYS exclude binary data files (e.g. using `--exclude "**.safetensors" --exclude "**.bin"`) when querying repository status or generating diffs.

## Common commands execution

*   Never try to prefix the following commands with anything, you should run them as is: `build_cleaner`, `blaze`.
*   ALWAYS use `-c opt` for any `blaze build` or `blaze test` commands (e.g. `blaze test //path/to:test -c opt`) to ensure compilation optimization and prevent slow test runs.

## Agent Communication & Safety Rules

*   NEVER mail or submit a CL without explicit approval from the user. Always present the proposed changes/diff to the user first and ask for permission before running any `hg mail` or `hg submit` commands.
*   For new features, once the design proposal is approved by `tpu-raiden-design-reviewer`, the main agent MUST explicitly present the final approved proposal to the user and ask for approval before dispatching execution to `tpu-raiden-coder`, unless the user has explicitly authorized auto-execute mode.

## Delegation of Coding & Debugging Tasks

*   ALWAYS delegate all coding tasks (implementing features, refactoring, fixing bugs) to the specialized subagent `tpu-raiden-coder`.
*   ALWAYS delegate all code debugging, problem analysis, and root cause investigation to the specialized subagent `tpu-raiden-debugger`.
*   ALWAYS delegate all design reviews and proposal evaluations to the specialized subagent `tpu-raiden-design-reviewer`.
*   ALWAYS delegate all feature design and research requests to the specialized subagent `tpu-raiden-designer`.
*   ALWAYS delegate all Xmanager-related tasks (launching, monitoring, log fetching) to the specialized subagent `tpu-raiden-xmanager-debugger`.

## Work Log Tracking Rules

*   ALWAYS create and maintain a `worklog.md` file in the project directory (e.g., `tpu_raiden_worklog.md`) to track all project activities and experiment runs.
*   The main agent MUST record an entry in the worklog file every time it invokes a subagent, documenting the activity type, subagent role/id, problem addressed, and status.
