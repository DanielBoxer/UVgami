cmake_minimum_required(VERSION 3.15)
project(uvgami)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# needed for glad and stb on windows
if(WIN32)
  set(BUILD_SHARED_LIBS FALSE)
endif()

# default is 0 if not provided by CI
if(NOT DEFINED UVGAMI_VERSION_MAJOR)
  set(UVGAMI_VERSION_MAJOR 0)
endif()
if(NOT DEFINED UVGAMI_VERSION_MINOR)
  set(UVGAMI_VERSION_MINOR 0)
endif()
if(NOT DEFINED UVGAMI_VERSION_BUILD)
  set(UVGAMI_VERSION_BUILD 0)
endif()

set(UVGAMI_VERSION
  "${UVGAMI_VERSION_MAJOR}.${UVGAMI_VERSION_MINOR}.${UVGAMI_VERSION_BUILD}")
set(PACKAGE_VERSION
  "${UVGAMI_VERSION_MAJOR}.${UVGAMI_VERSION_MINOR}.${UVGAMI_VERSION_BUILD}")


set(UVGAMI_EXTERNAL "${CMAKE_CURRENT_SOURCE_DIR}/ext")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
add_compile_definitions(_USE_MATH_DEFINES)

include(GNUInstallDirs)
set(UVGAMI_INSTALL_SUBDIR "uvgami-${UVGAMI_VERSION_MAJOR}.${UVGAMI_VERSION_MINOR}")
set(UVGAMI_INSTALL_PACKAGE_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/${UVGAMI_INSTALL_SUBDIR}")
if (APPLE)
	list(APPEND UVGAMI_LIBRARY_PROPERTIES INSTALL_NAME_DIR "${CMAKE_INSTALL_FULL_LIBDIR}")
endif()


include(Dependencies)

# libigl
set(LIBIGL_OPENGL  ON)
set(LIBIGL_PNG  ON)
set(LIBIGL_IMGUI ON)
set(LIBIGL_GLFW ON)
set(LIBIGL_RESTRICTED_TRIANGLE ON)
set(LIBIGL_USE_STATIC_LIBRARY ON)
add_subdirectory(ext/libigl EXCLUDE_FROM_ALL)

file(GLOB SRCFILES
  src/uvgami.cpp
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
  ${CMAKE_CURRENT_BINARY_DIR}/_deps/eigen-src
  ${CMAKE_CURRENT_BINARY_DIR}/_deps/triangle-src
  ${CMAKE_CURRENT_BINARY_DIR}/_deps/stb-src
  ${CMAKE_CURRENT_BINARY_DIR}/_deps/glad-src/include
  ${CMAKE_CURRENT_BINARY_DIR}/_deps/glfw-src/include
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

target_link_libraries(${PROJECT_NAME} PRIVATE igl::glfw  igl::png  igl_restricted::triangle  TBB::tbb ${UVGAMI_COMPILE_OPTIONS})

install(TARGETS ${PROJECT_NAME}
    EXPORT UvgamiTargets
    DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT Applications
)


configure_file(
 ${CMAKE_CURRENT_SOURCE_DIR}/uvgami_config.h.cmake.in
 ${CMAKE_CURRENT_BINARY_DIR}/uvgami_config.h
 @ONLY)

include (cmake/UvgamiCPack.cmake)
install(EXPORT UvgamiTargets DESTINATION ${UVGAMI_INSTALL_PACKAGE_DIR})
