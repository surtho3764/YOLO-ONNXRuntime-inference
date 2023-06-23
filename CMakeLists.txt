cmake_minimum_required(VERSION 3.0.0)

project(yolov5)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


message(STATUS "========================================")
message(STATUS "               Onnx Runtime             ")
message(STATUS "========================================")
message(STATUS "ONNXRUNTIME_DIR: ${ONNXRUNTIME_DIR}\n")

set(ONNXRUNTIME_DIR "${CMAKE_CURRENT_LIST_DIR}/onnxruntime/")

include_directories("${ONNXRUNTIME_DIR}/include")

link_directories("${ONNXRUNTIME_DIR}/lib")

message(STATUS "========================================")
message(STATUS "                OpenCV                  ")
message(STATUS "========================================")
find_package(OpenCV REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} )
message(STATUS "OpenCV Library Status:")
message(STATUS "   Version: ${OpenCV_VERSION}")
message(STATUS "   Libraries: ${OpenCV_LIBS}")
message(STATUS "   Include path: ${OpenCV_INCLUDE_DIRS}\n")

include_directories("${CMAKE_CURRENT_LIST_DIR}/include")

add_executable(demo
               src/main.cpp
               src/detector.cpp
               src/helpers.cpp)


target_link_libraries(demo onnxruntime)
target_link_libraries(demo ${OpenCV_LIBS})




