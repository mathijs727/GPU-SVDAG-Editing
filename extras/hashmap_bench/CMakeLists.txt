add_subdirectory("third_party/SlabHash")
add_subdirectory("third_party/DyCuckoo")

add_executable(HashMapBench "src/main.cu" )
target_compile_features(HashMapBench PUBLIC cxx_std_20 cuda_std_20)
set_target_properties(HashMapBench PROPERTIES CUDA_ARCHITECTURES "86")
target_link_libraries(HashMapBench PRIVATE 
	CLI11::CLI11 TBB::tbb fmt::fmt nlohmann_json::nlohmann_json
	GPUHashDAG_lib
	SlabHash DyCuckoo)
