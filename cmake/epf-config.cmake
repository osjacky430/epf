include(CMakeFindDependencyMacro)

find_dependency(Boost)
find_dependency(range-v3)
find_dependency(Eigen3)

include("${CMAKE_CURRENT_LIST_DIR}/epf-targets.cmake")
