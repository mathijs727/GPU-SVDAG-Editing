add_subdirectory("SlabAlloc")

add_library(SlabHash INTERFACE)
target_include_directories(SlabHash INTERFACE "src")
target_link_libraries(SlabHash INTERFACE SlabAlloc)
