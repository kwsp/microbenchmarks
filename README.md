# C++ project template

Features:

- GitHub Action for reproducible builds.
- VCPKG for package management.
- CMake with CMakePresets for cross-platform builds
  - Configuration presets:
    - `ninja-multi`: A generic Ninja Multi-Config preset.
    - `clang-multi`: A Clang specific Ninja Multi-Config preset.
    - `cl-multi`: (Windows x64) CL specific Ninja Multi-Config preset.
    - `win64`: (Windows x64) Visual Studio 2022 preset.
  - Build presets:
    - Debug, RelWithDebInfo, and Release build presets are provided for all configuration presets above. Just remove the `multi` and add `debug`, `relwithdebinfo`, or `release`. For example, the presets for `clang-multi` are `clang-debug`, `clang-relwithdebinfo`, and `clang-release`.
  - Test presets:
    - Just add `test-` to the build presets names.
- Clang-Tidy and Clang-Format
