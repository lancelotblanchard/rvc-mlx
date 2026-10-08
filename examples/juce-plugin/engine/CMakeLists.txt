# rvc_engine: the C++ RVC inference engine used by the plugin (also usable on its own).
cmake_minimum_required(VERSION 3.22)
project(rvc_engine LANGUAGES CXX)

if(NOT TARGET mlx)
  # MLX ships a CMake package with the pip wheel (`python -m mlx --cmake-dir`) and with Homebrew (`brew install mlx`).
  find_package(MLX CONFIG REQUIRED)
endif()

add_library(rvc_engine STATIC
  src/weights.cpp
  src/hubert.cpp
  src/rmvpe.cpp
  src/synthesizer.cpp
  src/dsp.cpp
  src/voice.cpp
  src/engine.cpp)
target_compile_features(rvc_engine PUBLIC cxx_std_17)
target_include_directories(rvc_engine PUBLIC include PRIVATE src)
target_link_libraries(rvc_engine PUBLIC mlx)
set_target_properties(rvc_engine PROPERTIES POSITION_INDEPENDENT_CODE ON)

option(RVC_ENGINE_TESTS "Build the Python-parity test runner" OFF)
if(RVC_ENGINE_TESTS)
  add_executable(rvc_parity_test tests/parity_test.cpp)
  target_include_directories(rvc_parity_test PRIVATE src)
  target_link_libraries(rvc_parity_test PRIVATE rvc_engine)
endif()
