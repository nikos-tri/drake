# -*- python -*-
# This file contains rules for Bazel; see drake/doc/bazel.rst.

load(
    "//tools:drake.bzl",
    "drake_cc_binary",
    "drake_cc_googletest",
    "drake_cc_library",
)
load("//tools:lint.bzl", "add_lint_tests")

package(default_visibility = ["//visibility:public"])

drake_cc_library(
    name = "sensors",
    srcs = [],
    hdrs = [],
    deps = [
        ":accelerometer",
        ":camera_info",
        ":depth_sensor",
        ":rgbd_camera",
        ":rotary_encoders",
    ],
)

drake_cc_library(
    name = "camera_info",
    srcs = [
        "camera_info.cc",
    ],
    hdrs = [
        "camera_info.h",
    ],
    deps = [
        "//drake/systems/framework",
    ],
)

drake_cc_library(
    name = "image",
    srcs = [],
    hdrs = [
        "image.h",
        "pixel_types.h",
    ],
    deps = [
        "//drake/common:essential",
        "//drake/common:reinit_after_move",
    ],
)

drake_cc_library(
    name = "image_to_lcm_image_array_t",
    srcs = [
        "image_to_lcm_image_array_t.cc",
    ],
    hdrs = [
        "image_to_lcm_image_array_t.h",
    ],
    deps = [
        ":image",
        "//drake/common:essential",
        "//drake/systems/framework",
        "@lcmtypes_robotlocomotion",
        "@zlib",
    ],
)

drake_cc_library(
    name = "rotary_encoders",
    srcs = ["rotary_encoders.cc"],
    hdrs = ["rotary_encoders.h"],
    deps = [
        "//drake/common:unused",
        "//drake/systems/framework",
    ],
)

drake_cc_library(
    name = "accelerometer",
    srcs = [
        "accelerometer.cc",
        "accelerometer_output.cc",
    ],
    hdrs = [
        "accelerometer.h",
        "accelerometer_output.h",
    ],
    deps = [
        "//drake/multibody:rigid_body_tree",
        "//drake/multibody/rigid_body_plant",
        "//drake/systems/framework",
    ],
)

drake_cc_library(
    name = "depth_sensor",
    srcs = [
        "depth_sensor.cc",
        "depth_sensor_output.cc",
        "depth_sensor_specification.cc",
    ],
    hdrs = [
        "depth_sensor.h",
        "depth_sensor_output.h",
        "depth_sensor_specification.h",
    ],
    deps = [
        "//drake/multibody:rigid_body_tree",
        "//drake/systems/framework",
        "//drake/systems/rendering:pose_vector",
    ],
)

drake_cc_library(
    name = "depth_sensor_to_lcm_point_cloud_message",
    srcs = [
        "depth_sensor_to_lcm_point_cloud_message.cc",
    ],
    hdrs = [
        "depth_sensor_to_lcm_point_cloud_message.h",
    ],
    deps = [
        ":depth_sensor",
        "//drake/multibody:rigid_body_tree",
        "//drake/systems/rendering:pose_vector",
        "@lcmtypes_bot2_core",
    ],
)

drake_cc_library(
    name = "gyroscope",
    srcs = [
        "gyroscope.cc",
        "gyroscope_output.cc",
    ],
    hdrs = [
        "gyroscope.h",
        "gyroscope_output.h",
    ],
    deps = [
        "//drake/multibody:rigid_body_tree",
        "//drake/systems/framework",
    ],
)

drake_cc_library(
    name = "vtk_util",
    srcs = ["vtk_util.cc"],
    hdrs = ["vtk_util.h"],
    deps = [
        "//drake/common:essential",
        "@eigen",
        "@vtk//:vtkCommonCore",
        "@vtk//:vtkCommonTransforms",
        "@vtk//:vtkFiltersSources",
    ],
)

drake_cc_library(
    name = "rgbd_camera",
    srcs = ["rgbd_camera.cc"],
    hdrs = ["rgbd_camera.h"],
    deps = [
        ":camera_info",
        ":image",
        ":vtk_util",
        "//drake/math:geometric_transform",
        "//drake/multibody:rigid_body_frame",
        "//drake/multibody:rigid_body_tree",
        "//drake/systems/framework",
        "//drake/systems/framework:diagram_builder",
        "//drake/systems/primitives",
        "//drake/systems/rendering:pose_vector",
        "@eigen",
        "@vtk//:vtkCommonCore",
        "@vtk//:vtkCommonDataModel",
        "@vtk//:vtkCommonTransforms",
        "@vtk//:vtkFiltersGeneral",
        "@vtk//:vtkFiltersSources",
        "@vtk//:vtkIOGeometry",
        "@vtk//:vtkIOImage",
        "@vtk//:vtkRenderingCore",
        "@vtk//:vtkRenderingOpenGL2",
    ],
)

drake_cc_binary(
    name = "rgbd_camera_publish_lcm_example",
    srcs = [
        "rgbd_camera_publish_lcm_example.cc",
    ],
    deps = [
        ":image_to_lcm_image_array_t",
        ":rgbd_camera",
        "//drake/common:unused",
        "//drake/lcm",
        "//drake/multibody:rigid_body_tree",
        "//drake/multibody:rigid_body_tree_construction",
        "//drake/multibody/parsers",
        "//drake/multibody/rigid_body_plant",
        "//drake/multibody/rigid_body_plant:drake_visualizer",
        "//drake/systems/analysis:simulator",
        "//drake/systems/lcm",
        "//drake/systems/rendering:pose_stamped_t_pose_vector_translator",
        "@com_github_gflags_gflags//:gflags",
    ],
)

drake_cc_library(
    name = "accelerometer_example_diagram",
    srcs = [
        "test/accelerometer_test/accelerometer_example_diagram.cc",
        "test/accelerometer_test/accelerometer_test_logger.cc",
        "test/accelerometer_test/accelerometer_xdot_hack.cc",
    ],
    hdrs = [
        "test/accelerometer_test/accelerometer_example_diagram.h",
        "test/accelerometer_test/accelerometer_test_logger.h",
        "test/accelerometer_test/accelerometer_xdot_hack.h",
    ],
    data = [
        "//drake/examples/pendulum:models",
    ],
    deps = [
        ":accelerometer",
        "//drake/common:essential",
        "//drake/common:find_resource",
        "//drake/multibody/parsers",
        "//drake/multibody/rigid_body_plant:drake_visualizer",
        "//drake/multibody/rigid_body_plant:rigid_body_plant_that_publishes_xdot",  # noqa
        "//drake/systems/lcm:lcmt_drake_signal_translator",
        "//drake/systems/primitives:constant_vector_source",
        "//drake/systems/primitives:signal_logger",
    ],
)

drake_cc_binary(
    name = "accelerometer_example",
    srcs = [
        "test/accelerometer_test/accelerometer_example.cc",
    ],
    deps = [
        ":accelerometer_example_diagram",
        "//drake/common:text_logging_gflags",
        "//drake/lcm",
        "//drake/systems/analysis:simulator",
        "@com_github_gflags_gflags//:gflags",
    ],
)

# === test/ ===

drake_cc_googletest(
    name = "camera_info_test",
    deps = [
        ":camera_info",
        "//drake/common/test_utilities:eigen_matrix_compare",
    ],
)

drake_cc_googletest(
    name = "image_test",
    deps = [
        ":image",
    ],
)

drake_cc_googletest(
    name = "rotary_encoders_test",
    deps = [
        ":rotary_encoders",
        "//drake/common/test_utilities:eigen_matrix_compare",
        "//drake/common/test_utilities:symbolic_test_util",
        "//drake/systems/framework/test_utilities",
    ],
)

drake_cc_googletest(
    name = "accelerometer_test",
    srcs = [
        "test/accelerometer_test/accelerometer_test.cc",
    ],
    data = [
        "//drake/multibody:models",
    ],
    deps = [
        ":accelerometer_example_diagram",
        "//drake/common:find_resource",
        "//drake/common/test_utilities:eigen_matrix_compare",
        "//drake/lcm:mock",
        "//drake/systems/analysis:simulator",
    ],
)

drake_cc_googletest(
    name = "depth_sensor_test",
    srcs = ["test/depth_sensor_test.cc"],
    deps = [
        ":depth_sensor",
        "//drake/common/test_utilities:eigen_matrix_compare",
    ],
)

drake_cc_googletest(
    name = "depth_sensor_to_lcm_point_cloud_message_test",
    srcs = ["test/depth_sensor_to_lcm_point_cloud_message_test.cc"],
    deps = [
        ":depth_sensor_to_lcm_point_cloud_message",
        "//drake/lcm:mock",
    ],
)

drake_cc_googletest(
    name = "gyroscope_test",
    srcs = ["test/gyroscope_test.cc"],
    data = [
        "//drake/multibody:models",
    ],
    deps = [
        ":gyroscope",
        "//drake/common:find_resource",
        "//drake/common/test_utilities:eigen_matrix_compare",
        "//drake/multibody/parsers",
    ],
)

filegroup(
    name = "test_models",
    testonly = 1,
    srcs = [
        "test/models/box.sdf",
        "test/models/cylinder.sdf",
        "test/models/mesh_box.sdf",
        "test/models/meshes/box.obj",
        "test/models/meshes/box.png",
        "test/models/multiple_visuals.sdf",
        "test/models/nothing.sdf",
        "test/models/sphere.sdf",
        "test/models/three_boxes.sdf",
    ],
    visibility = ["//visibility:private"],
)

# TODO(jamiesnape): Remove the "manual" tag when Xvfb is available on CI.
drake_cc_googletest(
    name = "rgbd_camera_test",
    data = [
        ":test_models",
    ],
    # Mitigate driver-related issues when running under `bazel test` (#7004).
    local = 1,
    tags = ["manual"],
    deps = [
        ":rgbd_camera",
        "//drake/common:find_resource",
        "//drake/common/test_utilities:eigen_matrix_compare",
        "//drake/multibody:rigid_body_tree_construction",
        "//drake/multibody/parsers",
        "//drake/multibody/rigid_body_plant",
    ],
)

sh_test(
    name = "rgbd_camera_publish_lcm_test",
    srcs = ["rgbd_camera_publish_lcm_example"],
    args = [
        # TODO(eric.cousineau): Find skylark rule that can resolve
        # "test/models/sphere.sdf" to the path below.
        "--sdf_dir=drake/systems/sensors/test/models",
        "--duration=0.1",
    ],
    data = [
        ":test_models",
    ],
    tags = ["manual"],
)

drake_cc_googletest(
    name = "vtk_util_test",
    deps = [":vtk_util"],
)

drake_cc_googletest(
    name = "image_to_lcm_image_array_t_test",
    deps = [":image_to_lcm_image_array_t"],
)

add_lint_tests()
