#pragma once
#ifdef FORCEINLINE
#pragma push_macro("FORCEINLINE")
#undef FORCEINLINE
#ifdef __CUDACC__
#include <cooperative_groups.h>
#endif
#pragma pop_macro("FORCEINLINE")
#else
#include <cooperative_groups.h>
#undef FORCEINLINE
#endif