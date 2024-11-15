enable_testing()
find_package(Catch2 CONFIG REQUIRED)
include(CTest)
include(Catch)

add_executable(GPUHashDAG_tests
    #"array2d.cpp"
    #"utils_tests.cu"
    #"color_utils_tests.cpp"
    #"gpu_hash_dag/gpu_base_dag_test.cpp"
    "gpu_hash_dag/duplicate_hash_table_test.cu"
    #"gpu_hash_dag/gpu_slab_allocator_test.cu" 
    #"gpu_hash_dag/hash_table_test.cpp" 
    #"gpu_hash_dag/hash_table_test_cuda.cu" 
    #"gpu_hash_dag/hash_table_test_cuda.h" 
    #"gpu_hash_dag/gpu_hash_dag_tests.cpp"
    "main.cpp"
)
target_compile_features(GPUHashDAG_tests PUBLIC cxx_std_20 cuda_std_20)
set_target_properties(GPUHashDAG_tests  PROPERTIES
    CUDA_ARCHITECTURES "86" # 86 (Ampere) required for warp reduction functions.
    CUDA_SEPARABLE_COMPILATION ON # Seems like a requirement for CUDA debugging?
    CUDA_RESOLVE_DEVICE_SYMBOLS ON # Required on Windows when using CMake to generate a Visual Studio Solution file.
)
target_compile_definitions(GPUHashDAG_tests PUBLIC "TEST_DATA_FOLDER=\"${CMAKE_CURRENT_LIST_DIR}/data/\"")
target_link_libraries(GPUHashDAG_tests PRIVATE GPUHashDAG_lib Catch2::Catch2)

# Work-around for bug/feature in Visual Studio TestExplorer which hasn't been fixed for years:
# IN TEST EXPLORER SET GROUP-BY TO CLASS/OBJECT (ANYTHING WITHOUT NAMESPACE)
# https://developercommunity.visualstudio.com/t/cmake-weird-test-name-prefix/793209
# https://stackoverflow.com/questions/60520767/specify-namespace-class-in-vs2019-test-explorer-while-using-ctest-cmake
catch_discover_tests(
    GPUHashDAG_tests
	TEST_PREFIX "DAG." # the last dot is required :)
)
