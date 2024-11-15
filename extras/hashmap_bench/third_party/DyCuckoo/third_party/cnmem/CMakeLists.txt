add_library(Cnmem "src/cnmem.cu")
target_include_directories(Cnmem PUBLIC "include")
target_link_libraries(Cnmem PRIVATE CUDA::cudart)