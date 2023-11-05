cmake_minimum_required(VERSION 3.0.2)
project(gym_sawyer)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  rospy
  intera_core_msgs
  gazebo_msgs
)

#############
## Install ##
#############

add_message_files(
   FILES
   RLExperimentInfo.msg
 )



# install(PROGRAMS
#    ./warm_up.py set_gazebo_model.py training.py
#    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

catkin_python_setup()

generate_messages(
   DEPENDENCIES
   std_msgs
 )

catkin_package(
  DEPENDS message_runtime rospy gazebo_msgs std_msgs geometry_msgs controller_manager_msgs intera_core_msgs
)

# catkin_package(
#   CATKIN_DEPENDS
#   rospy
#   intera_core_msgs
#   gazebo_msgs
# )

include_directories(
  ${catkin_INCLUDE_DIRS}
)