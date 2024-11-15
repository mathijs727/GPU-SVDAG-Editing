/************************************************************/
/*!	\brief CUDA Helpers
 */
/* Copyright (c) 2010, 2011: Markus Billeter
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
/********************************************************************/
/*
 * Modified by Viktor Kämpe to use std::cout instead of fprintf
 */

/* Modified again to not use std::cout and to abort on error. -M
 */

#pragma once

#include "typedefs.h"

#include <cstdio>
#include <cstdlib>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>

#if ENABLE_CHECKS == 0
#define CUDA_CHECKED_CALL
#define CUDA_CHECK_ERROR()
#else
[[maybe_unused]] static void __cudaCheckError(const char* file, unsigned line)
{
    {
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (cudaSuccess != err) {
            std::fprintf(stderr, "ERROR %s:%d: cuda error \"%s\"\n", file, line, cudaGetErrorString(err));
            std::abort();
        }
    }
}
#define CUDA_CHECKED_CALL ::detail::CudaErrorChecker(__LINE__, __FILE__) =
#define CUDA_CHECK_ERROR() __cudaCheckError(__FILE__, __LINE__)

/*EOM*/
#endif

namespace detail {
struct CudaErrorChecker {
    int line;
    const char* file;

    inline CudaErrorChecker(int aLine, const char* aFile) noexcept
        : line(aLine)
        , file(aFile)
    {
    }

    inline cudaError_t operator=(cudaError_t err)
    {
        if (cudaSuccess != err) {
            const std::string error = cudaGetErrorString(err);
            if (error != "driver shutting down") {
                std::fprintf(stderr, "ERROR %s:%d: cuda error \"%s\"\n", file, line, error.c_str());
                std::abort();
            }
        }

        return err;
    }

    inline CUresult operator=(CUresult err)
    {
        if (CUDA_SUCCESS != err) {
            const char* errMsg = nullptr;
            cuGetErrorString(err, &errMsg);
            std::fprintf(stderr, "ERROR [driver] %s:%d: cuda error \"%s\"\n", file, line, errMsg);
            std::abort();
        }

        return err;
    }
};
}