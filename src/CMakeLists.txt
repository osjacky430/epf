add_executable(main main.cpp)
target_link_libraries(main PRIVATE project_option project_warning epf Boost::headers)
