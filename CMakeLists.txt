cmake_minimum_required(VERSION 2.8.3)
project(menoh_ros)

find_package(
  catkin REQUIRED
  COMPONENTS pluginlib nodelet cv_bridge roscpp sensor_msgs std_msgs
)

include(ExternalProject)

set(MKL_URL_ROOT "https://github.com/intel/mkl-dnn/releases/download/v0.14/")
set(MKL_VERSION "2018.0.3.20180406")

ExternalProject_Add(
    mkl
    PREFIX mkl
    BUILD_IN_SOURCE 1
    URL "${MKL_URL_ROOT}/mklml_lnx_${MKL_VERSION}.tgz"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND "pwd"
    INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory . ${CATKIN_DEVEL_PREFIX}/
)

ExternalProject_Add(
    mkldnn
    PREFIX mkldnn
    BUILD_IN_SOURCE 1
    URL "https://github.com/intel/mkl-dnn/archive/v0.14.zip"
    CONFIGURE_COMMAND ${CMAKE_COMMAND} . -DMKLINC=${CATKIN_DEVEL_PREFIX}/include -DCMAKE_INSTALL_PREFIX=${CATKIN_DEVEL_PREFIX}
    BUILD_COMMAND make
    INSTALL_COMMAND make install
)

ExternalProject_Add(
    libmenoh
    PREFIX libmenoh
    BUILD_IN_SOURCE 1
    GIT_REPOSITORY "https://github.com/pfnet-research/menoh.git"
    GIT_TAG v1.0.1
    GIT_SUBMODULES external/onnx
    CONFIGURE_COMMAND ${CMAKE_COMMAND} . -DCMAKE_INSTALL_PREFIX=${CATKIN_DEVEL_PREFIX}
    BUILD_COMMAND make
    INSTALL_COMMAND make install
)

add_dependencies(mkldnn mkl)
add_dependencies(libmenoh mkldnn)

find_package(PkgConfig)

pkg_check_modules(menoh REQUIRED menoh)

find_package(Boost REQUIRED COMPONENTS system)


###################################
## catkin specific configuration ##
###################################
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES menoh_ros
  CATKIN_DEPENDS nodelet
  DEPENDS boost
)

###########
## Build ##
###########

add_definitions(-std=c++11)

include_directories(include ${catkin_INCLUDE_DIRS} ${menoh_INCLUDE_DIRS})

add_library(menoh_ros
    src/menoh_ros/nodelet.cpp
    src/menoh_ros/io_nodelets.cpp
)
target_link_libraries(menoh_ros ${catkin_LIBRARIES} ${menoh_LIBRARIES})

#############
## Install ##
#############

## Mark executables and/or libraries for installation
install(TARGETS menoh_ros
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Mark cpp header files for installation
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
)

## Mark other files for installation (e.g. launch and bag files, etc.)
install(FILES launch/vgg16.launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

# Manually install files built by ExternalProject
install(FILES
  nodelet_plugins.xml
  ${CATKIN_DEVEL_PREFIX}/license.txt
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/
)

install(FILES
  ${CATKIN_DEVEL_PREFIX}/lib/libiomp5.so
  ${CATKIN_DEVEL_PREFIX}/lib/libmkldnn.so
  ${CATKIN_DEVEL_PREFIX}/lib/libmklml_gnu.so
  ${CATKIN_DEVEL_PREFIX}/lib/libmenoh.so
  ${CATKIN_DEVEL_PREFIX}/lib/libmkldnn.so.0.14.0
  ${CATKIN_DEVEL_PREFIX}/lib/libmkldnn.so.0
  ${CATKIN_DEVEL_PREFIX}/lib/libmkldnn.so
  ${CATKIN_DEVEL_PREFIX}/lib/libmklml_intel.so
  DESTINATION ${CATKIN_GLOBAL_LIB_DESTINATION}/
)
