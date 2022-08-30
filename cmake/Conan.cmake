macro (configure_package_manager)
  if (NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/conan.cmake")
    message(STATUS "Downloading conan.cmake from https://github.com/conan-io/cmake-conan")
    file(DOWNLOAD "https://raw.githubusercontent.com/conan-io/cmake-conan/release/0.17/conan.cmake" "${CMAKE_BINARY_DIR}/conan.cmake")
  endif ()

  include(${CMAKE_CURRENT_BINARY_DIR}/conan.cmake)

  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND NOT ANDROID)
    add_compile_options(-stdlib=libc++)
    add_link_options(-stdlib=libc++)
  endif ()

  conan_cmake_configure(REQUIRES gtest/cci.20210126 range-v3/0.11.0 eigen/3.4.0 fmt/9.1.0 GENERATORS cmake_find_package)

  conan_cmake_autodetect(settings BUILD_TYPE Release)
  conan_cmake_install(PATH_OR_REFERENCE . BUILD missing SETTINGS ${settings})
endmacro ()
