find_package(GTest REQUIRED)

add_executable(kdtree_test kdtree.cpp)
target_include_directories(kdtree_test PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(kdtree_test PUBLIC GTest::gtest_main)

enable_testing()
add_test(NAME "test kdtree" COMMAND kdtree_test)
