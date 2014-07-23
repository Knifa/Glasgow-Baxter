cmake_minimum_required(VERSION 2.8.3)
project(glasgow_baxter_helpers)

find_package(catkin REQUIRED)

catkin_package(
    LIBRARIES ${PROJECT_NAME}
    CATKIN_DEPENDS
        baxter_interface
        baxter_core_msgs
        rospy
        cv_bridge
        image_geometry
        std_msgs
        geometry_msgs
        sensor_msgs
        tf
)

catkin_python_setup()
catkin_install_python(
    PROGRAMS 
        scripts/avg_camera 
        scripts/rectify_camera
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)