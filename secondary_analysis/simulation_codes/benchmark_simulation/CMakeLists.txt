cmake_minimum_required(VERSION 3.20)
project(prdes)
find_package(OpenMP)
add_definitions(-std=c++11)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "-fopenmp")
add_executable(prdes benchmark_sim.cpp benchmark_sim.h)
