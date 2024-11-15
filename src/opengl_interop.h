#pragma once
#include "array2d.h"
#include "typedefs.h"
#include <GL/glew.h>
#include <cuda_gl_interop.h>

class GLSurface2D {
public:
    static inline GLSurface2D create(GLuint image)
    {
        // https://github.com/NVIDIA/cuda-samples/blob/master/Samples/5_Domain_Specific/postProcessGL/main.cpp
        //
        // Register the texture with CUDA
        GLSurface2D out {};
        cudaGraphicsGLRegisterImage(&out.pCudaGraphicsResource, image, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard);
        return out;
    }

    inline void free()
    {
        cudaGraphicsUnregisterResource(pCudaGraphicsResource);
        pCudaGraphicsResource = nullptr;
    }

    inline void copyFrom(const StaticArray2D<uint32_t>& array2D)
    {
        // https://github.com/NVIDIA/cuda-samples/blob/master/Samples/5_Domain_Specific/postProcessGL/main.cpp
        // https://on-demand.gputechconf.com/gtc/2012/presentations/S0267B-Mixing-Graphics-and-Compute-with-Multiple-GPUs-Part-B.pdf
        //
        // We want to copy cuda_dest_resource data to the texture.
        // Map buffer objects to get CUDA device pointers.
        cudaArray* pCudaArray;
        cudaGraphicsMapResources(1, &pCudaGraphicsResource);
        cudaGraphicsSubResourceGetMappedArray(&pCudaArray, pCudaGraphicsResource, 0, 0);

        const size_t horPitch = array2D.width * sizeof(uint32_t);
        cudaMemcpy2DToArray(pCudaArray, 0, 0, array2D.buffer.data(), horPitch, horPitch, array2D.height, cudaMemcpyDeviceToDevice);

        cudaGraphicsUnmapResources(1, &pCudaGraphicsResource);
    }

private:
    cudaGraphicsResource* pCudaGraphicsResource;
};
