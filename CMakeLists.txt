cmake_minimum_required(VERSION 3.5)
project(cppi3d)
set(CMAKE_CXX_STANDARD 11)
AUX_SOURCE_DIRECTORY(src DIR_SRCS)
SET(TEST_MATH ${DIR_SRCS})
link_directories(/usr/local/lib)
include_directories(
        /home/dell/tensorflow
        /home/dell/tensorflow/tensorflow/contrib/makefile/gen/host_obj
        /home/dell/tensorflow/tensorflow/contrib/makefile/gen/proto
        /home/dell/tensorflow/tensorflow/contrib/makefile/downloads/nsync/public
        /home/dell/tensorflow/tensorflow/contrib/makefile/downloads/eigen
        /home/dell/tensorflow/bazel-out/local_linux-py3-opt/genfiles
        /home/dell/tensorflow/tensorflow/contrib/makefile/downloads/absl
        /home/dell/tensorflow/tensorflow/contrib/makefile/gen/protobuf/include
        /home/dell/tensorflow/bazel-genfiles
        /home/dell/tensorflow/bazel-bin/tensorflow
        /usr/local/include/opencv4
        include
)
add_executable(cppi3d ${TEST_MATH})
target_link_libraries(cppi3d
    /usr/local/lib/libopencv_core.so
    /usr/local/lib/libopencv_videoio.so
    /usr/local/lib/libopencv_highgui.so
    /usr/local/lib/libopencv_imgcodecs.so
    /usr/local/lib/libopencv_imgproc.so
    /home/dell/tensorflow/bazel-bin/tensorflow/libtensorflow_cc.so
    /home/dell/tensorflow/bazel-bin/tensorflow/libtensorflow_framework.so.1)

