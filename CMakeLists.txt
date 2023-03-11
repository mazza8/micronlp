cmake_minimum_required(VERSION 3.24)
project(micronlp C)

set(CMAKE_C_STANDARD 11)

include_directories(.)
include_directories(micronlp)

add_executable(micronlp
        micronlp/lm.h
        micronlp/metrics.h
        micronlp/utils.h main.cpp)
