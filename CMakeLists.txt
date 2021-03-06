cmake_minimum_required(VERSION 3.5)
project(drake-superbuild VERSION 0.0.0)

include(CTest)

include(cmake/config.cmake)
include(cmake/externals.cmake)
include(cmake/options.cmake)

drake_setup_superbuild()
drake_setup_platform()
drake_setup_options()

###############################################################################
# BEGIN external projects

# External projects in order of dependencies; 'trivial' ones first
drake_add_external(qt_property_browser CMAKE QT)

# eigen
# N.B. See #5785; do your best not to have to bump this to a newer commit.
drake_add_external(eigen PUBLIC CMAKE
  URL https://bitbucket.org/eigen/eigen/get/3.3.3.tar.gz
  URL_HASH SHA256=94878cbfa27b0d0fbc64c00d4aafa137f678d5315ae62ba4aecddbd4269ae75f
  CMAKE_ARGS
    -DEIGEN_TEST_NO_OPENGL=ON
    -DEIGEN_TEST_NOQT=ON)

# spdlog
drake_add_external(spdlog PUBLIC CMAKE
  DEPENDS fmt)

# bullet
drake_add_external(bullet PUBLIC CMAKE
  CMAKE_ARGS
    -DBUILD_SHARED_LIBS=ON
    -DBUILD_BULLET2_DEMOS=OFF
    -DBUILD_CPU_DEMOS=OFF
    -DBUILD_EXTRAS=OFF
    -DBUILD_OPENGL3_DEMOS=OFF
    -DBUILD_UNIT_TESTS=OFF
    -DCMAKE_DEBUG_POSTFIX=
    -DCMAKE_MINSIZEREL_POSTFIX=
    -DCMAKE_RELWITHDEBINFO_POSTFIX=
    -DINSTALL_LIBS=ON
    -DPKGCONFIG_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}/lib/pkgconfig
    -DUSE_DOUBLE_PRECISION=ON)

# fmt
drake_add_external(fmt PUBLIC CMAKE ALWAYS
  CMAKE_ARGS
    -DFMT_DOC=OFF
    -DFMT_TEST=OFF)

# gflags
drake_add_external(gflags PUBLIC CMAKE
  CMAKE_ARGS
    -DBUILD_SHARED_LIBS=ON
    -DGFLAGS_BUILD_gflags_nothreads_LIB=OFF)

# googletest
drake_add_external(googletest PUBLIC CMAKE
  CMAKE_ARGS
    -DBUILD_SHARED_LIBS=ON
    -DCMAKE_INSTALL_NAME_DIR=${CMAKE_INSTALL_PREFIX}/lib)

# ipopt
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # Work around horrific clang* = clang, */cl* = msvc bug in BuildTools/coin.m4
  get_filename_component(IPOPT_C_COMPILER "${CMAKE_C_COMPILER}" NAME)
  get_filename_component(IPOPT_CXX_COMPILER "${CMAKE_CXX_COMPILER}" NAME)
else()
  set(IPOPT_C_COMPILER "${CMAKE_C_COMPILER}")
  set(IPOPT_CXX_COMPILER "${CMAKE_CXX_COMPILER}")
endif()

drake_add_external(ipopt PUBLIC AUTOTOOLS FORTRAN
  AUTOTOOLS_ENV
    CC=${IPOPT_C_COMPILER}
    CXX=${IPOPT_CXX_COMPILER}
  AUTOTOOLS_CONFIGURE_ARGS
    --disable-shared
    --includedir=${CMAKE_INSTALL_PREFIX}/include/ipopt
    --with-blas=BUILD
    --with-lapack=BUILD
    --with-pic)

# nlopt
drake_add_external(nlopt PUBLIC CMAKE
  CMAKE_ARGS
    -DBUILD_SHARED_LIBS=ON
    -DBUILD_PYTHON=OFF
    -DBUILD_OCTAVE=OFF
    -DBUILD_MATLAB=OFF
    -DBUILD_GUILE=OFF
    -DINSTALL_INCLUDE_DIR=${CMAKE_INSTALL_PREFIX}/include/nlopt
    -DUSE_SWIG=OFF)

# protobuf
drake_add_external(protobuf PUBLIC CMAKE ALWAYS
  SOURCE_SUBDIR cmake
  CMAKE_ARGS
    -DBUILD_SHARED_LIBS=ON
    -Dprotobuf_BUILD_TESTS=OFF
    -Dprotobuf_MODULE_COMPATIBLE=ON)

# pybind11
drake_add_external(pybind11 CMAKE
  CMAKE_ARGS
    -DPYBIND11_PYTHON_VERSION=${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}
    -DPYBIND11_TEST=OFF)

# pythonqt
drake_add_external(pythonqt CMAKE PYTHON QT
  CMAKE_ARGS
    -DPythonQt_Wrap_Qtcore:BOOL=ON
    -DPythonQt_Wrap_Qtgui:BOOL=ON
    -DPythonQt_Wrap_Qtuitools:BOOL=ON
)

# tinyobjloader
drake_add_external(tinyobjloader PUBLIC CMAKE
  CMAKE_ARGS
    -DTINYOBJLOADER_COMPILATION_SHARED=ON
  )

# yaml_cpp
drake_add_external(yaml_cpp PUBLIC CMAKE
 CMAKE_ARGS
   -DBUILD_SHARED_LIBS=ON
   -DYAML_CPP_BUILD_CONTRIB=OFF
   -DYAML_CPP_BUILD_TOOLS=OFF)

# ctk_python_console
drake_add_external(ctk_python_console CMAKE PYTHON QT
  DEPENDS pythonqt)

# lcm
drake_add_external(lcm PUBLIC CMAKE PYTHON
  CMAKE_ARGS
    -DBUILD_SHARED_LIBS=ON
    -DLCM_ENABLE_EXAMPLES=OFF
    -DLCM_ENABLE_LUA=OFF
    -DLCM_ENABLE_TESTS=OFF
    -DLCM_INSTALL_M4MACROS=OFF)

# libbot
drake_add_external(libbot PUBLIC CMAKE PYTHON
  DEPENDS lcm)

# bot_core_lcmtypes
drake_add_external(bot_core_lcmtypes PUBLIC CMAKE PYTHON
  DEPENDS lcm libbot) # Conflicts with libbot; ensure this is built after

# robotlocomotion_lcmtypes
drake_add_external(robotlocomotion_lcmtypes PUBLIC CMAKE PYTHON
  DEPENDS bot_core_lcmtypes lcm)

# Option to skip building drake proper via the superbuild. This allows the
# superbuild to build everything BUT drake, which can still be built separately
# from its build directory. This is used by the dashboards to make separate
# submissions for drake proper and the superbuild without drake. Some users may
# also find it useful, especially to build drake with ninja using fewer than
# the default number of jobs.
option(SKIP_DRAKE_BUILD "Build external projects but not drake itself" OFF)

# drake: For drake, list both compilation AND RUNTIME dependencies. Runtime
# dependencies are needed because the drake project must configure only after
# any dependencies used by MATLAB have been installed.
# TODO: re-add TEST to drake flags when subprojects arrive
drake_add_external(drake LOCAL PUBLIC CMAKE ALWAYS MATLAB PYTHON VTK
  SOURCE_DIR ${PROJECT_SOURCE_DIR}/drake
  BINARY_DIR ${PROJECT_BINARY_DIR}/drake
  CMAKE_ARGS
    -DDISABLE_MATLAB:BOOL=${DISABLE_MATLAB}
    -DDISABLE_PYTHON:BOOL=${DISABLE_PYTHON}
    -DDISABLE_FORTRAN:BOOL=${DISABLE_FORTRAN}
    -DWITH_BOT_CORE_LCMTYPES:BOOL=${WITH_BOT_CORE_LCMTYPES}
    -DWITH_BULLET:BOOL=${WITH_BULLET}
    -DWITH_EIGEN3:BOOL=${WITH_EIGEN}  # TODO(jamiesnape): Fix naming inconsistency
    -DWITH_GFLAGS:BOOL=${WITH_GFLAGS}
    -DWITH_GTEST:BOOL=${WITH_GOOGLETEST}  # TODO(jamiesnape): Fix naming inconsistency
    -DWITH_IPOPT:BOOL=${WITH_IPOPT}
    -DWITH_LCM:BOOL=${WITH_LCM}
    -DWITH_LIBBOT:BOOL=${WITH_LIBBOT}
    -DWITH_NLOPT:BOOL=${WITH_NLOPT}
    -DWITH_PROTOBUF:BOOL=${WITH_PROTOBUF}
    -DWITH_PYBIND11:BOOL=${WITH_PYBIND11}
    -DWITH_PYTHON_3:BOOL=${WITH_PYTHON_3}
    -DWITH_SPDLOG:BOOL=${WITH_SPDLOG}
    -DWITH_TINYOBJLOADER:BOOL=${WITH_TINYOBJLOADER}
    -DWITH_YAML_CPP:BOOL=${WITH_YAML_CPP}
  DEPENDS
    bot_core_lcmtypes
    bullet
    eigen
    fmt
    gflags
    googletest
    ipopt
    lcm
    libbot
    nlopt
    protobuf
    pybind11
    robotlocomotion_lcmtypes
    spdlog
    tinyobjloader
    yaml_cpp
)

# director
drake_add_external(director PUBLIC CMAKE TEST PYTHON QT VTK
  CMAKE_ARGS
    -DENABLE_TESTING=ON
    -DUSE_DRAKE=ON
    -DUSE_LCM=${HAVE_LCM}
    -DUSE_LCMGL=${HAVE_LIBBOT}
    -DUSE_LIBBOT=OFF
  DEPENDS
    bot_core_lcmtypes
    ctk_python_console
    drake
    lcm
    libbot
    pythonqt
    qt_property_browser
)

# END external projects
###############################################################################
