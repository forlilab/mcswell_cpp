/*
 * Workaround for CUDA 11.x with glibc >= 2.38.
 *
 * glibc's <math.h> declares functions using _Float32, _Float64x, _Float128,
 * etc., which nvcc's C++ frontend (cudafe++) does not understand.
 *
 * This header is force-included (--pre-include) before any source file
 * processed by nvcc.  It pre-includes <bits/floatn.h> so the include guard
 * is set, then overrides the capability macros to zero, preventing glibc
 * from emitting the unsupported declarations.
 *
 * Safe to include in non-CUDA translation units (no-op without __CUDACC__).
 */
#ifdef __CUDACC__

#include <bits/floatn.h>

#undef __HAVE_FLOAT16
#undef __HAVE_FLOAT32
#undef __HAVE_FLOAT64
#undef __HAVE_FLOAT32X
#undef __HAVE_FLOAT64X
#undef __HAVE_FLOAT128

#define __HAVE_FLOAT16  0
#define __HAVE_FLOAT32  0
#define __HAVE_FLOAT64  0
#define __HAVE_FLOAT32X 0
#define __HAVE_FLOAT64X 0
#define __HAVE_FLOAT128 0

#undef __HAVE_DISTINCT_FLOAT16
#undef __HAVE_DISTINCT_FLOAT32
#undef __HAVE_DISTINCT_FLOAT64
#undef __HAVE_DISTINCT_FLOAT32X
#undef __HAVE_DISTINCT_FLOAT64X
#undef __HAVE_DISTINCT_FLOAT128

#define __HAVE_DISTINCT_FLOAT16  0
#define __HAVE_DISTINCT_FLOAT32  0
#define __HAVE_DISTINCT_FLOAT64  0
#define __HAVE_DISTINCT_FLOAT32X 0
#define __HAVE_DISTINCT_FLOAT64X 0
#define __HAVE_DISTINCT_FLOAT128 0

#undef __HAVE_FLOATN_NOT_TYPEDEF
#define __HAVE_FLOATN_NOT_TYPEDEF 0

#endif /* __CUDACC__ */
