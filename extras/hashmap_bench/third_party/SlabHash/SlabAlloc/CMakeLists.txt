add_library(SlabAlloc
	"src/slab_alloc.cu"
	"src/slab_alloc.cuh"
	"src/slab_alloc_global.cuh"
)
target_include_directories(SlabAlloc PUBLIC "src")