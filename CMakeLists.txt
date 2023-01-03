cmake_minimum_required(VERSION 3.10)

set(Torch_DIR /usr/local/lib/libtorch/)
set(OpenCV_DIR /usr/local/lib)
set(OpenEXR_DIR /usr/local/lib)

# Enable C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

if(MSVC)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17")
endif(MSVC)


project(YoloV5LibTorch)

include_directories(./include/ )

add_subdirectory(./src)
add_subdirectory(./test)

install(TARGETS verification YoloV5 AverageHash
    RUNTIME DESTINATION ${CMAKE_SOURCE_DIR}/bin
    ARCHIVE DESTINATION ${CMAKE_SOURCE_DIR}/lib
)