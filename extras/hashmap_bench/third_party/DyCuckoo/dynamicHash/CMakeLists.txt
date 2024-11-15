add_library(DyCuckoo INTERFACE)
target_link_libraries(DyCuckoo INTERFACE Cnmem CudaSamples)
target_include_directories(DyCuckoo INTERFACE "include")
target_compile_features(DyCuckoo INTERFACE cxx_std_14 cuda_std_14)
