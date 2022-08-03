find_package(Eigen3 3.4.0 EXACT REQUIRED)

add_executable(main main.cpp)
target_include_directories(main PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(main PRIVATE project_option project_warning amclpp Eigen3::Eigen Boost::headers range-v3::range-v3)
