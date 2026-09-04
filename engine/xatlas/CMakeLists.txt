cmake_minimum_required(VERSION 3.15)
project(xatlas)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# static msvc runtime so the exe ships without vc redist, matching optcuts.
if(MSVC)
  set(CMAKE_POLICY_DEFAULT_CMP0091 NEW)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(xatlas main.cpp xatlas.cpp)

if(MSVC)
  target_compile_options(xatlas PRIVATE $<$<CONFIG:Release>:/O2>)
  # without it a path outside the process codepage arrives as question marks
  target_sources(xatlas PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/utf8.manifest)
else()
  target_compile_options(xatlas PRIVATE $<$<CONFIG:Release>:-O3>)
  # xatlas uses std::thread
  find_package(Threads REQUIRED)
  target_link_libraries(xatlas PRIVATE Threads::Threads)
endif()

# package the exe into a zip. the ci passes -DCPACK_GENERATOR=ZIP and
# -DCPACK_PACKAGE_FILE_NAME; the addon finds the binary in the zip by name.
install(TARGETS xatlas RUNTIME DESTINATION .)
# the engine zip is the only distribution of the binary, so the notice ships here
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/LICENSE DESTINATION . RENAME LICENSE.txt)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../../LICENSE DESTINATION licenses RENAME GPL-3.0.txt)
set(CPACK_PACKAGE_NAME xatlas)
include(CPack)
