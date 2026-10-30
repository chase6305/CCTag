# CCTag library

## Building instructions

For a detailed guide on building the library check the [online documentation](https://cctag.readthedocs.io/).

Required tools:

* CMake >= 3.14 to build the code
* Git
* C/C++ compiler with C++17 and `std::filesystem` support
  * see here: <https://en.cppreference.com/w/cpp/compiler_support>
  * GCC >= 9, Clang with a C++17 filesystem library, or Visual Studio >= 2019

Optional tool:

* CUDA >= 11.0
Note: On Windows, there are compatibility issues to build the GPU part due to conflicts between msvc/nvcc/thrust/eigen/boost.

### Dependencies

Most of the dependencies can be installed from the common repositories (apt, yum etc):

* Eigen3 (libeigen3-dev) >= 3.3.4  (NOTE: in order to have Cuda support on Windows, at least version 3.3.9 is required)
* Boost >= 1.66 ([exception, program-options, ptr-container, serialization, spirit, stacktrace, test]-dev)
* OpenCV >= 3.1
* TBB >= 2021.5.0

Boost.ProgramOptions is needed only for applications, and Boost.Test only for
unit tests. When configuring CCTag directly with the vcpkg toolchain,
`CCTAG_BUILD_APPS` and `CCTAG_BUILD_TESTS` select the corresponding manifest
features before dependency installation. Set both options to `OFF` to install
only the library dependencies. Explicit `VCPKG_MANIFEST_FEATURES` and
`VCPKG_MANIFEST_NO_DEFAULT_FEATURES` settings are respected.

A standalone `vcpkg install` still includes both features by default. Use
`vcpkg install --x-no-default-features` for library dependencies only, adding
`--x-feature=apps` or `--x-feature=tests` when needed. When CCTag is embedded
with `add_subdirectory`, the parent project controls vcpkg feature selection.

The installed `CCTag::CCTag` target propagates the C++17 requirement. Rebuild
consumers when upgrading: filesystem helpers now accept `std::filesystem::path`,
`CCTag::radiusRatiosInit()` and `Color` use `std::array`, and the edge-linking
helper uses `std::unordered_set`. The detection API's owning pointer containers
and serialized XML format are retained.

----------

### Using CCTag as third party

When you install CCTag a file `CCTagConfig.cmake` is installed in `$CCTAG_INSTALL/lib/cmake/CCTag/` that allows you to import the library in your CMake project.
In your `CMakeLists.txt` file you can add the dependency in this way:

```cmake
# Find the package from the CCTagConfig.cmake
# in <prefix>/lib/cmake/CCTag/. Under the namespace CCTag::
# it exposes the target CCTag that allows you to compile
# and link with the library
find_package(CCTag CONFIG REQUIRED)
...
# suppose you want to try it out in a executable
add_executable(cctagtest yourfile.cpp)
# add link to the library
target_link_libraries(cctagtest PUBLIC CCTag::CCTag)
```

Then, in order to build just pass the location of `CCTagConfig.cmake` from the cmake command line:

```bash
cmake .. -DCCTag_DIR=$CCTAG_INSTALL/lib/cmake/CCTag/
```

## Docker Image

A docker image can be built using the Ubuntu based [Dockerfile](Dockerfile),which is based on nvidia/cuda image (<https://hub.docker.com/r/nvidia/cuda/>)

A parameter `CUDA_TAG` can be passed when building the image to select the ubuntu and cuda version.
First build the dependency image, then build CCTag with the same CUDA and Ubuntu versions:

```bash
docker build -f Dockerfile_deps --tag alicevision/cctag-deps:cuda11.8.0-ubuntu20.04 .
docker build --tag cctag .
```

The complete list of available tags can be found on the nvidia [dockerhub page](https://hub.docker.com/r/nvidia/cuda/)
In order to run the image nvidia docker is needed: see the installation instruction here <https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)>
Once installed, the docker can be run, e.g., in interactive mode with

```bash
docker run -it --runtime=nvidia cctag
```
