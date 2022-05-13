
cmake_minimum_required(VERSION 2.6)
project(integer_least_squares)

# Global options for the compiler
set(CMAKE_CXX_FLAGS "-std=c++17 -Wall -O3")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake/Modules/)

# There are test cases in all builds
enable_testing()

add_subdirectory(src)


