# RVC Morph: an RVC voice-conversion plug-in (AU / VST3 / Standalone, with ARA 2) running on MLX.
#
#   cmake -B build -G Xcode -DARA_SDK_PATH=/path/to/ARA_SDK-releases-2.2.0
#   cmake --build build --config Release
#
# Dependencies are fetched automatically unless you point CMake at local copies:
#   JUCE_DIR / -DRVC_JUCE_SOURCE=<path>   JUCE 7.0.5+ or 8.x
#   -DRVC_MLX_FROM_SOURCE=OFF -DCMAKE_PREFIX_PATH=$(python -m mlx --cmake-dir)   an installed MLX (pip or Homebrew)
cmake_minimum_required(VERSION 3.22)
project(RVCMorph VERSION 0.2.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OSX_DEPLOYMENT_TARGET "14.0" CACHE STRING "MLX needs macOS 14+")
set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "MLX runs on Apple silicon")

set(ARA_SDK_PATH "" CACHE PATH "Celemony ARA SDK (https://github.com/Celemony/ARA_SDK); leave empty to build without ARA")
set(RVC_JUCE_SOURCE "" CACHE PATH "Local JUCE checkout (otherwise fetched)")
option(RVC_MLX_FROM_SOURCE "Build MLX from source (static) instead of using an installed package" ON)

include(FetchContent)

#--- JUCE ------------------------------------------------------------------------------------------------------
if(RVC_JUCE_SOURCE)
  add_subdirectory(${RVC_JUCE_SOURCE} juce)
else()
  find_package(JUCE 7 CONFIG QUIET)
  if(NOT JUCE_FOUND)
    FetchContent_Declare(JUCE GIT_REPOSITORY https://github.com/juce-framework/JUCE.git GIT_TAG 8.0.4 GIT_SHALLOW ON)
    FetchContent_MakeAvailable(JUCE)
  endif()
endif()

#--- MLX -------------------------------------------------------------------------------------------------------
if(RVC_MLX_FROM_SOURCE)
  set(MLX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(MLX_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(MLX_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
  set(MLX_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(mlx GIT_REPOSITORY https://github.com/ml-explore/mlx.git GIT_TAG v0.30.1 GIT_SHALLOW ON)
  FetchContent_MakeAvailable(mlx)
else()
  find_package(MLX CONFIG REQUIRED)
endif()

add_subdirectory(engine)

#--- plug-in ---------------------------------------------------------------------------------------------------
set(RVC_ARA FALSE)
if(ARA_SDK_PATH)
  juce_set_ara_sdk_path(${ARA_SDK_PATH})
  set(RVC_ARA TRUE)
else()
  message(STATUS "ARA_SDK_PATH not set: building without ARA (live mode only)")
endif()

juce_add_plugin(RVCMorph
  PRODUCT_NAME "RVC Morph"
  COMPANY_NAME "rvc-mlx"
  BUNDLE_ID "com.rvc-mlx.morph"
  PLUGIN_MANUFACTURER_CODE Rvcm
  PLUGIN_CODE Rvmo
  FORMATS AU VST3 Standalone
  IS_SYNTH FALSE
  NEEDS_MIDI_INPUT FALSE
  NEEDS_MIDI_OUTPUT FALSE
  IS_MIDI_EFFECT FALSE
  EDITOR_WANTS_KEYBOARD_FOCUS FALSE
  MICROPHONE_PERMISSION_ENABLED TRUE
  MICROPHONE_PERMISSION_TEXT "RVC Morph converts your voice live in standalone mode."
  IS_ARA_EFFECT ${RVC_ARA}
  ARA_FACTORY_ID "\"com.rvc-mlx.morph.arafactory.0.2.0\""
  ARA_DOCUMENT_ARCHIVE_ID "\"com.rvc-mlx.morph.aradocumentarchive.1\""
  COPY_PLUGIN_AFTER_BUILD FALSE)  # done below, after mlx.metallib is in the bundle

juce_generate_juce_header(RVCMorph)

target_sources(RVCMorph PRIVATE
  Source/PluginProcessor.cpp
  Source/PluginEditor.cpp
  Source/EngineHost.cpp
  Source/StreamingConverter.cpp
  Source/ARAVoiceConversion.cpp)

target_compile_definitions(RVCMorph PUBLIC
  JUCE_WEB_BROWSER=0
  JUCE_USE_CURL=0
  JUCE_VST3_CAN_REPLACE_VST2=0
  JUCE_DISPLAY_SPLASH_SCREEN=0)

target_link_libraries(RVCMorph
  PRIVATE
    rvc_engine
    juce::juce_audio_utils
    juce::juce_audio_formats
  PUBLIC
    juce::juce_recommended_config_flags
    juce::juce_recommended_lto_flags
    juce::juce_recommended_warning_flags)

# MLX looks for its Metal kernels (mlx.metallib) next to the binary that contains it: copy it into every bundle,
# then install the AU / VST3 bundles into ~/Library/Audio/Plug-Ins.
if(APPLE)
  if(RVC_MLX_FROM_SOURCE)
    set(RVC_METALLIB "${mlx_BINARY_DIR}/mlx/backend/metal/kernels/mlx.metallib")
  else()
    set(RVC_METALLIB "${MLX_LIBRARY_DIRS}/mlx.metallib")
  endif()
  set(RVC_INSTALL_VST3 "$ENV{HOME}/Library/Audio/Plug-Ins/VST3")
  set(RVC_INSTALL_AU "$ENV{HOME}/Library/Audio/Plug-Ins/Components")
  set(RVC_BUNDLE_VST3 "RVC Morph.vst3")
  set(RVC_BUNDLE_AU "RVC Morph.component")
  foreach(format VST3 AU Standalone)
    if(TARGET RVCMorph_${format})
      add_custom_command(TARGET RVCMorph_${format} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${RVC_METALLIB}" "$<TARGET_FILE_DIR:RVCMorph_${format}>/mlx.metallib"
        COMMENT "Bundling mlx.metallib into ${format}")
      if(NOT format STREQUAL "Standalone")
        add_custom_command(TARGET RVCMorph_${format} POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E make_directory "${RVC_INSTALL_${format}}"
          COMMAND ${CMAKE_COMMAND} -E copy_directory "$<TARGET_BUNDLE_DIR:RVCMorph_${format}>" "${RVC_INSTALL_${format}}/${RVC_BUNDLE_${format}}"
          COMMENT "Installing ${format} into ${RVC_INSTALL_${format}}")
      endif()
    endif()
  endforeach()
endif()

#--- smoke test (headless) -------------------------------------------------------------------------------------
option(RVC_PLUGIN_TESTS "Build the headless plug-in smoke test" OFF)
if(RVC_PLUGIN_TESTS)
  juce_add_console_app(RVCMorphSmokeTest PRODUCT_NAME "RVCMorphSmokeTest")
  juce_generate_juce_header(RVCMorphSmokeTest)
  target_sources(RVCMorphSmokeTest PRIVATE
    tests/PluginSmokeTest.cpp
    Source/PluginProcessor.cpp
    Source/PluginEditor.cpp
    Source/EngineHost.cpp
    Source/StreamingConverter.cpp)
  target_compile_definitions(RVCMorphSmokeTest PRIVATE
    JucePlugin_Name="RVC Morph" JucePlugin_Enable_ARA=0 JUCE_WEB_BROWSER=0 JUCE_USE_CURL=0 JUCE_MODAL_LOOPS_PERMITTED=1)
  target_link_libraries(RVCMorphSmokeTest PRIVATE rvc_engine juce::juce_audio_utils juce::juce_audio_formats
    juce::juce_recommended_config_flags)
endif()
