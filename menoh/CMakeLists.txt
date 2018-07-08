cmake_minimum_required(VERSION 2.8.3)
project(menoh)

include(ExternalProject)

set(MKL_URL_ROOT "https://github.com/intel/mkl-dnn/releases/download/v0.14/")
set(MKL_VERSION "2018.0.3.20180406")
set(MKLDNN_URL "https://github.com/intel/mkl-dnn/archive/v0.14.zip")

message(STATUS "CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}")
message(STATUS "CMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}")
message(STATUS "CATKIN_DEVEL_PREFIX=${CATKIN_DEVEL_PREFIX}")

ExternalProject_Add(
    mkl
    PREFIX mkl
    BUILD_IN_SOURCE 1
    URL "${MKL_URL_ROOT}/mklml_lnx_${MKL_VERSION}.tgz"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory . ${CMAKE_INSTALL_PREFIX}/
)

ExternalProject_Add(
    mkldnn
    PREFIX mkldnn
    BUILD_IN_SOURCE 1
    URL ${MKLDNN_URL}
    CONFIGURE_COMMAND ${CMAKE_COMMAND} . -DMKLINC=${CMAKE_INSTALL_PREFIX}/include -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
    BUILD_COMMAND make
    INSTALL_COMMAND make install
)

ExternalProject_Add(
    libmenoh
    PREFIX libmenoh
    BUILD_IN_SOURCE 1
    GIT_REPOSITORY "https://github.com/pfnet-research/menoh.git"
    GIT_TAG v1.0.2
    GIT_SUBMODULES external/onnx
    CONFIGURE_COMMAND ${CMAKE_COMMAND} . -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
    BUILD_COMMAND make
    INSTALL_COMMAND make install
)

add_dependencies(mkldnn mkl)
add_dependencies(libmenoh mkldnn)

set(MENOH_INCLUDE_DIR ${CMAKE_INSTALL_PREFIX}/include)
set(MENOH_LIB_DIR ${CMAKE_INSTALL_PREFIX}/lib)
set(MENOH_LIBRARIES ${MENOH_LIB_DIR}/libmenoh.so)

configure_file(
    menoh-config.cmake.in
    ${PROJECT_BINARY_DIR}/menoh-config.cmake
    @ONLY
)

install(
    FILES ${PROJECT_BINARY_DIR}/menoh-config.cmake
    DESTINATION ${CMAKE_INSTALL_PREFIX}/share/menoh/cmake/
)

install(
    FILES package.xml LICENSE LICENSE.menoh LICENSE.onnx LICENSE.mkldnn LICENSE.mkl
    DESTINATION ${CMAKE_INSTALL_PREFIX}/share/menoh/
)
