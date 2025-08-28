include(DownloadProject)

# With CMake 3.8 and above, we can hide warnings about git being in a
# detached head by passing an extra GIT_CONFIG option
if(NOT (${CMAKE_VERSION} VERSION_LESS "3.8.0"))
  set(UVGAMI_EXTRA_OPTIONS "GIT_CONFIG advice.detachedHead=false")
else()
  set(UVGAMI_EXTRA_OPTIONS "")
endif()

function(custom_download_project name)
  download_project(
    PROJ         ${name}
    SOURCE_DIR   ${UVGAMI_EXTERNAL}/${name}
    DOWNLOAD_DIR ${UVGAMI_EXTERNAL}/.cache/${name}
    QUIET
    ${UVGAMI_EXTRA_OPTIONS}
    ${ARGN}
  )
endfunction()

################################################################################

# libigl
#function(download_libigl)
#  custom_download_project(libigl
#    GIT_REPOSITORY https://github.com/libigl/libigl.git
#    GIT_TAG        b0d7740e0b7e887a7e93601c4c557ecf762b389b
#  )
#endfunction()

# TBB
function(download_tbb)
  custom_download_project(tbb
    # GIT_REPOSITORY https://github.com/intel/tbb.git
    # GIT_TAG        2018_U5
    GIT_REPOSITORY https://github.com/wjakob/tbb.git
    GIT_TAG        9e219e24fe223b299783200f217e9d27790a87b0
  )
endfunction()

# AMGCL
#function(download_amgcl)
#    custom_download_project(amgcl
#       GIT_REPOSITORY https://github.com/ddemidov/amgcl.git
#       GIT_TAG        b901a26fff09af4a16581429a1c5748d21520b7f
#    )
#endfunction()
