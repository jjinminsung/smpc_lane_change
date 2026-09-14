cmake_minimum_required(VERSION 3.0.2)
project(smpc_lane_change)

add_compile_options(-std=c++17 -Wall -Wextra)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  nav_msgs
  visualization_msgs
  message_generation
  morai_msgs
  waypoint_maker
  waypoint_system
  lidar_ttc_tracker
)

add_message_files(FILES
  TargetVehicle.msg
  TargetVehicleSet.msg
  LaneChangeDecision.msg
  BehaviorPlan.msg
  BehaviorLongitudinalRequest.msg
  BehaviorLongitudinalStatus.msg
  LaneChangeStatus.msg
)

generate_messages(DEPENDENCIES std_msgs)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS roscpp std_msgs nav_msgs visualization_msgs message_runtime morai_msgs waypoint_maker waypoint_system lidar_ttc_tracker
)

include_directories(include ${catkin_INCLUDE_DIRS})

add_executable(target_selector_node src/target_selector_node.cpp)
add_dependencies(target_selector_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(target_selector_node ${catkin_LIBRARIES})

add_executable(lane_change_supervisor_node src/lane_change_supervisor_node.cpp)
add_dependencies(lane_change_supervisor_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(lane_change_supervisor_node ${catkin_LIBRARIES})

add_executable(acc_speed_planner_node src/acc_speed_planner_node.cpp)
add_dependencies(acc_speed_planner_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(acc_speed_planner_node ${catkin_LIBRARIES})

add_executable(smpc_decision_node src/smpc_decision_node.cpp)
add_dependencies(smpc_decision_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(smpc_decision_node ${catkin_LIBRARIES})

add_executable(gap_test_decision_node src/gap_test_decision_node.cpp)
add_dependencies(gap_test_decision_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(gap_test_decision_node ${catkin_LIBRARIES})

install(TARGETS target_selector_node lane_change_supervisor_node acc_speed_planner_node smpc_decision_node gap_test_decision_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(DIRECTORY include/${PROJECT_NAME}/ DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})
install(DIRECTORY launch config rviz meshes DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
