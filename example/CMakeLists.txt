find_package(Matplot++ QUIET)

add_library(example_config INTERFACE)
target_link_libraries(example_config INTERFACE epf project_option Boost::headers fmt::fmt $<${Matplot++_FOUND}:Matplot++::matplot>)
target_compile_definitions(example_config INTERFACE $<${Matplot++_FOUND}:HAVE_MATPLOTCPP>)

# add_subdirectory(kdtree)
add_subdirectory(robotics)
add_subdirectory(time_series)
