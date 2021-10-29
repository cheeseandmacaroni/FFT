cmake_minimum_required(VERSION 3.5)
project(my_FFT VERSION 1.0.0 DESCRIPTION "Very nice project" LANGUAGES CXX)

file(GLOB CPP_SOURCES "src/*.cpp")
file(GLOB HEADER_FILES "include/*.hpp")

add_executable(${PROJECT_NAME} ${CPP_SOURCES} ${HEADER_FILES})

target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)
