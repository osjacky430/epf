cmake_minimum_required(VERSION 3.21.1 FATAL_ERROR)
project(epf LANGUAGES CXX VERSION 0.0.1)

# For most projects, on non-Windows platforms it is advisable to explicitly set CMAKE_INSTALL_PREFIX to
# a FHS-compliant /opt/... path. This should generally be done only in the top level CMakeLists.txt and
# it should be protected by an appropriate check that the project is in fact the top level of the source
# tree (to support hierarchical project arrangements).
if (NOT WIN32 AND CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(CMAKE_INSTALL_PREFIX "/opt/${PROJECT_NAME}")
endif ()

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" "${CMAKE_BINARY_DIR}")
include(ProjectOption)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# cmake-format: off
configure_project_option(
  WARNINGS
    TARGET project_warning
  LINKER
    TARGET project_option
    LINKER_NAME mold
    LINKER_PATH /usr/local/libexec/mold
  SANITIZER
    TARGET project_option
  IPO
    DISABLE_FOR_CONFIG Debug Coverage
  PROJ_TARGET
    TARGET epf
)
# cmake-format: on

find_package(Boost REQUIRED)
find_package(range-v3 REQUIRED)
find_package(fmt REQUIRED)
find_package(Eigen3 3.4.0 EXACT REQUIRED)

# todo: test install command
add_library(epf::epf ALIAS epf)
target_link_libraries(epf INTERFACE range-v3::range-v3 Eigen3::Eigen)

option(EPF_TEST_ENABLE "Enable epf unit testing" OFF)
if (${EPF_TEST_ENABLE})
  add_subdirectory(test)
endif ()

option(EPF_BUILD_EXAMPLE "Build epf example code" ON)
if (${EPF_BUILD_EXAMPLE})
  add_subdirectory(example)
endif ()

include(CMakePackageConfigHelpers)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/epf-config-version.cmake VERSION ${epf_VERSION} COMPATIBILITY ExactVersion
                                                                                                                           ARCH_INDEPENDENT)

include(GNUInstallDirs)
install(TARGETS epf EXPORT epf-targets INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# this location is both FHS and ROS compatible
set(ConfigPackageLocation share/epf/cmake)
install(EXPORT epf-targets FILE epf-targets.cmake NAMESPACE epf:: DESTINATION ${ConfigPackageLocation})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/epf-config-version.cmake cmake/epf-config.cmake DESTINATION ${ConfigPackageLocation})

# If the project does not define framework targets, and it can use CMake 3.23 or later for its minimum
# version, file sets should be preferred. For all other cases, headers can be installed more generically
# using install(FILES) or install(DIRECTORY) .
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} FILES_MATCHING PATTERN *.hpp)
