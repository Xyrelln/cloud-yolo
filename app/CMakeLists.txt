install(FILES ./coco.txt ./yolov5s.cpu.pt ./yolov5s.cuda.pt DESTINATION ${CMAKE_SOURCE_DIR}/bin)

find_package(OpenEXR)
find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

add_executable(verification freezeDetecter.cpp)

target_link_libraries(verification YoloV5 AverageHash)
target_link_libraries(verification ${OpenCV_LIBS} )
target_link_libraries(verification ${TORCH_LIBRARIES})

