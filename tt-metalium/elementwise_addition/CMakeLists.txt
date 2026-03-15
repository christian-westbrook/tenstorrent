cmake_minimum_required(VERSION 3.22...4.2)
project(elementwise_addition)

set(TT_METAL_DIR "~/tt-metal" CACHE PATH "Path to tt-metal source")
add_subdirectory(${TT_METAL_DIR} tt-metal)

find_package(MPI REQUIRED)
add_link_options(-lmpi_cxx)

add_executable(elementwise_addition)
target_sources(elementwise_addition PRIVATE elementwise_addition.cpp)
target_link_libraries(elementwise_addition PUBLIC TT::Metalium MPI::MPI_CXX)