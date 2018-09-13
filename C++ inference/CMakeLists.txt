cmake_minimum_required(VERSION 3.5)
project(tf_detector_example)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp utils.cpp utils.h)
add_executable(tf_detector_example ${SOURCE_FILES})

# OpenCV libs
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(tf_detector_example ${OpenCV_LIBS})


# ==================== PATHS TO SPECIFY! ==================== #
link_directories(/usr/lib /usr/local/lib)

# Eigen lib headers
include_directories("/usr/local/include/eigen3")

# TensorFlow headers
set(TF_INCLUDE_PATH /usr/local/include/google/tensorflow)
include_directories("${TF_INCLUDE_PATH}")
include_directories("${TF_INCLUDE_PATH}/bazel-genfiles/")
include_directories("${TF_INCLUDE_PATH}/tensorflow/")
include_directories("${TF_INCLUDE_PATH}/third-party/")

# Link TensorFlow libs
#target_link_libraries(tf_detector_example "/usr/local/lib/libtensorflow_cc.so")
#target_link_libraries(tf_detector_example "/usr/local/lib/libtensorflow_framework.so")
target_link_libraries(tf_detector_example protobuf-lite tensorflow_cc tensorflow_framework)
