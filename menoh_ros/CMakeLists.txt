cmake_minimum_required(VERSION 2.8.3)
project(menoh_ros)

find_package(
  catkin REQUIRED
  COMPONENTS pluginlib nodelet cv_bridge roscpp sensor_msgs std_msgs
)

find_package(menoh CONFIG REQUIRED)

find_package(Boost REQUIRED COMPONENTS system)

###################################
## catkin specific configuration ##
###################################
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES menoh_ros
  CATKIN_DEPENDS nodelet
  DEPENDS Boost
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

##########
## Test ##
##########

# if(CATKIN_ENABLE_TESTNG)
# T.B.D.
# endif()

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
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/
)
