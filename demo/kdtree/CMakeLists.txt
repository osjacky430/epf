find_package(OpenCV 4.5.5 EXACT REQUIRED)

add_executable(draw_kdtree draw_kdtree.cpp)
target_include_directories(draw_kdtree PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(draw_kdtree PRIVATE opencv::opencv_core)
