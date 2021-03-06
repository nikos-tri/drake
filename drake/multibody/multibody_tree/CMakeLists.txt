add_library_with_exports(LIB_NAME drakeMultibodyTree SOURCE_FILES
  body.cc
  body_node_impl.cc
  fixed_offset_frame.cc
  mobilizer_impl.cc
  multibody_tree.cc
  multibody_tree_context.cc
  position_kinematics_cache.cc
  revolute_mobilizer.cc
  rigid_body.cc
  uniform_gravity_field_element.cc)

target_link_libraries(drakeMultibodyTree
  drakeCommon
  #TODO(amcastro-tri): Only systems::Context is needed here.
  # Split drakeSystemContext from drakeSystemFramework.
  drakeSystemFramework
  Eigen3::Eigen)
drake_install_libraries(drakeMultibodyTree)
drake_install_headers(
  body.h
  fixed_offset_frame.h
  force_element.h
  multibody_tree.h
  multibody_tree_element.h
  multibody_tree_context.h
  multibody_tree_indexes.h
  position_kinematics_cache.h
  revolute_mobilizer.h
  rigid_body.h
  uniform_gravity_field_element.h)

drake_install_pkg_config_file(drake-multibodytree
  TARGET drakeMultibodyTree
  LIBS -ldrakeMultibodyTree
  REQUIRES
  drake-common
  eigen3)

if(BUILD_TESTING)
  add_subdirectory(test)
endif()

add_subdirectory(math)
