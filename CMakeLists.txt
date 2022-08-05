cmake_minimum_required(VERSION 3.21.1 FATAL_ERROR)
project(ExtendableParticleFilter LANGUAGES CXX)

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

add_subdirectory(src)
add_subdirectory(test)

add_subdirectory(demo)
