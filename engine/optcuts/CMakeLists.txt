cmake_minimum_required(VERSION 3.15)
project(optcuts)

option(UVGAMI_USE_MIMALLOC "Use mimalloc on Windows" ON)
option(UVGAMI_ENABLE_AVX2 "Compile with the AVX2 floor on x86-64" ON)

# some third-party deps (libigl's glad) declare pre-3.5 cmake minimums,
# which cmake 4 rejects
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

# mimalloc's static override needs the whole tree on the static release crt.
# cmp0091 must be new or wjakob/tbb (pre-3.15 min) ignores the runtime-library
# setting and links /md, mixing crts and emitting lnk4098.
if(MSVC AND UVGAMI_USE_MIMALLOC)
  set(CMAKE_POLICY_DEFAULT_CMP0091 NEW)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# needed for glad and stb on windows
if(WIN32)
  set(BUILD_SHARED_LIBS FALSE)
endif()

# read version from VERSION file
if(NOT DEFINED UVGAMI_VERSION_MAJOR)
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" _version_raw)
  string(STRIP "${_version_raw}" _version_stripped)
  string(REPLACE "." ";" _version_list "${_version_stripped}")
  list(GET _version_list 0 UVGAMI_VERSION_MAJOR)
  list(GET _version_list 1 UVGAMI_VERSION_MINOR)
  list(GET _version_list 2 UVGAMI_VERSION_BUILD)
endif()

set(UVGAMI_VERSION
  "${UVGAMI_VERSION_MAJOR}.${UVGAMI_VERSION_MINOR}.${UVGAMI_VERSION_BUILD}")
set(PACKAGE_VERSION
  "${UVGAMI_VERSION_MAJOR}.${UVGAMI_VERSION_MINOR}.${UVGAMI_VERSION_BUILD}")


set(UVGAMI_EXTERNAL "${CMAKE_CURRENT_SOURCE_DIR}/ext")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
add_compile_definitions(_USE_MATH_DEFINES)

# avx2 floor: x86-64 only. resolve the effective target arch, honoring mac
# cross-builds via CMAKE_OSX_ARCHITECTURES so arm64 gets no x86 flags.
# must be global (before deps): mixing avx2 and non-avx2 tus changes
# EIGEN_MAX_ALIGN_BYTES per tu, and the linker folds eigen's inline
# malloc/free instantiations into mismatched pairs (heap corruption).
set(UVGAMI_TARGET_ARCH "${CMAKE_SYSTEM_PROCESSOR}")
if(APPLE AND CMAKE_OSX_ARCHITECTURES)
  set(UVGAMI_TARGET_ARCH "${CMAKE_OSX_ARCHITECTURES}")
endif()
string(TOLOWER "${UVGAMI_TARGET_ARCH}" UVGAMI_TARGET_ARCH)
if(UVGAMI_TARGET_ARCH MATCHES "arm|aarch64")
  set(UVGAMI_IS_X86_64 FALSE)
elseif(UVGAMI_TARGET_ARCH MATCHES "x86_64|amd64|x64|em64t")
  set(UVGAMI_IS_X86_64 TRUE)
else()
  set(UVGAMI_IS_X86_64 FALSE)
endif()
if(UVGAMI_IS_X86_64 AND UVGAMI_ENABLE_AVX2)
  if(MSVC)
    add_compile_options(/arch:AVX2)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    add_compile_options(-mavx2 -mfma)
  endif()
endif()

include(GNUInstallDirs)
if (APPLE)
	list(APPEND UVGAMI_LIBRARY_PROPERTIES INSTALL_NAME_DIR "${CMAKE_INSTALL_FULL_LIBDIR}")
endif()


include(Dependencies)

# libigl. the engine is headless, so no opengl/glfw/png/imgui
set(LIBIGL_RESTRICTED_TRIANGLE ON)
set(LIBIGL_USE_STATIC_LIBRARY ON)
add_subdirectory(ext/libigl EXCLUDE_FROM_ALL)

file(GLOB SRCFILES
  src/uvgami.cpp
  src/Flatten.cpp
  src/Optimizer.cpp
  src/TriMesh.cpp
  src/Scaffold.cpp
  src/Energy/Energy.cpp
  src/Energy/ARAPEnergy.cpp
  src/Energy/SymDirichletEnergy.cpp
  src/Utils/IglUtils.cpp
  src/LinSysSolver/EigenLibSolver.cpp
)

include_directories(
  # uvgami_config.h is generated here
  ${CMAKE_CURRENT_BINARY_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}/_deps/eigen-src
  ${CMAKE_CURRENT_BINARY_DIR}/_deps/triangle-src
  ${CMAKE_CURRENT_SOURCE_DIR}/ext/libigl/include
  src
  src/Energy
  src/Utils
  src/LinSysSolver
  src/include
)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  set(UVGAMI_COMPILE_OPTIONS ${UVGAMI_COMPILE_OPTIONS} -Wall -Wextra -Wconversion -Wsign-conversion -Wunused-parameter)
endif()

add_executable(${PROJECT_NAME} ${SRCFILES})

# argv and the narrow file opens use the process codepage, so a path with a
# character outside it arrives as question marks. needs windows 10 1903
if(MSVC)
  target_sources(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/utf8.manifest)
endif()

target_link_libraries(${PROJECT_NAME} PRIVATE igl::core  igl_restricted::triangle  TBB::tbb ${UVGAMI_COMPILE_OPTIONS})

# link the mimalloc object file so its malloc/free override the crt (windows only)
if(WIN32 AND UVGAMI_USE_MIMALLOC AND TARGET mimalloc-obj)
  target_link_libraries(${PROJECT_NAME} PRIVATE mimalloc-obj)
endif()

install(TARGETS ${PROJECT_NAME}
    DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT Applications
)


configure_file(
 ${CMAKE_CURRENT_SOURCE_DIR}/uvgami_config.h.cmake.in
 ${CMAKE_CURRENT_BINARY_DIR}/uvgami_config.h
 @ONLY)

include (cmake/UvgamiCPack.cmake)
