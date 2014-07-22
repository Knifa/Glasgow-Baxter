cmake_minimum_required(VERSION 2.8.3)
project(glasgow_baxter)

find_package(catkin
    REQUIRED
    COMPONENTS
        rospy
        std_msgs
        geometry_msgs
        message_generation
)

add_message_files(
    FILES
        DetectedSquare.msg
        DetectedSquares.msg
        TrackedSquare.msg
        TrackedSquares.msg
)

generate_messages(
    DEPENDENCIES
        std_msgs
        geometry_msgs
)

catkin_package(
    CATKIN_DEPENDS
        rospy
        baxter_core_msgs
        baxter_interface
        std_msgs
        geometry_msgs
        message_runtime
        glasgow_baxter_helpers
)

install(
    DIRECTORY scripts/
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
    USE_SOURCE_PERMISSIONS
)