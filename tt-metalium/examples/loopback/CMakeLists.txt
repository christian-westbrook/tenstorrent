cmake_minimum_required(VERSION 3.22...4.2)
project(loopback)

set(TT_METAL_DIR "~/tt-metal" CACHE PATH "Path to tt-metal source")
add_subdirectory(${TT_METAL_DIR} tt-metal)

find_package(MPI REQUIRED)
add_link_options(-lmpi_cxx)

add_executable(loopback)
target_sources(loopback PRIVATE loopback.cpp)
target_link_libraries(loopback PUBLIC TT::Metalium MPI::MPI_CXX)