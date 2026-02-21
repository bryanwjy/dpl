/* Copyright 2023-2025 Bryan Wong */

#pragma once

#include "dpl/configuration/architecture.h"
#include "dpl/configuration/compiler.h"

#if DPL_ARCH_x86

#  ifdef __SSE2__
#    define DPL_SIMD_X86_SSE2 __SSE2__
#  else
// Always enabled on x64
#    define DPL_SIMD_X86_SSE2 DPL_ARCH_x86_64
#  endif

#  ifdef __SSE3__
#    define DPL_SIMD_X86_SSE3 __SSE3__
#  endif

#  ifdef __SSE4_1__
#    define DPL_SIMD_X86_SSE4_1 __SSE4_1__
#  endif

#  ifdef __SSE4_2__
#    define DPL_SIMD_X86_SSE4_2 __SSE4_2__
#  endif

#  ifdef __AVX__
#    ifndef DPL_SIMD_X86_SSE3
#      define DPL_SIMD_X86_SSE3 1
#    endif
#    ifndef DPL_SIMD_X86_SSE4_1
#      define DPL_SIMD_X86_SSE4_1 1
#    endif
#    ifndef DPL_SIMD_X86_SSE4_2
#      define DPL_SIMD_X86_SSE4_2 1
#    endif
#    define DPL_SIMD_X86_AVX __AVX__
#  endif

#  ifdef __FMA__
#    define DPL_SIMD_X86_FMA __FMA__
#  endif

#  ifdef __AVX2__
#    define DPL_SIMD_X86_AVX2 __AVX2__
#    if DPL_COMPILER_MSVC
#      ifndef __FMA__
#        define DPL_SIMD_X86_FMA 1
#      endif
#    endif
#  endif

#  ifdef __AVX512BF16__
#    define DPL_SIMD_X86_AVX512BF16 __AVX512BF16__
#  endif

#  ifdef __AVX512FP16__
#    define DPL_SIMD_X86_AVX512FP16 __AVX512FP16__
#  endif

#  ifdef __AVX512BW__
#    define DPL_SIMD_X86_AVX512BW __AVX512BW__
#  endif

#  ifdef __AVX512CD__
#    define DPL_SIMD_X86_AVX512CD __AVX512CD__
#  endif

#  ifdef __AVX512DQ__
#    define DPL_SIMD_X86_AVX512DQ __AVX512DQ__
#  endif

#  ifdef __AVX512ER__
#    define DPL_SIMD_X86_AVX512ER __AVX512ER__
#  endif

#  ifdef __AVX512PF__
#    define DPL_SIMD_X86_AVX512PF __AVX512PF__
#  endif

#  ifdef __AVX512VL__
#    define DPL_SIMD_X86_AVX512VL __AVX512VL__
#  endif

#  ifdef __AVX512F__
#    define DPL_SIMD_X86_AVX512F __AVX512F__
#  endif

#elif DPL_ARCH_ARM

#  ifdef __ARM_NEON
#    define DPL_SIMD_ARM_NEON 1
#  endif

#  ifdef __ARM_FEATURE_SVE
#    define DPL_SIMD_ARM_SVE 1
#    ifdef __ARM_FEATURE_SVE_BITS
#      define DPL_SIMD_ARM_SVE_BITS __ARM_FEATURE_SVE_BITS
#    endif
#  endif

#  ifdef __ARM_FEATURE_SVE2
#    define DPL_SIMD_ARM_SVE2 1
#  endif
#endif
