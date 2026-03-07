/* Copyright 2023-2025 Bryan Wong */

#pragma once

#include "dpl/configuration/architecture.h"
#include "dpl/configuration/compiler.h" // IWYU pragma: keep

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

#  ifdef __F16C__
#    define DPL_SIMD_X86_F16C __F16C__
#  elif DPL_COMPILER_MSVC && __AVX2__
#    ifndef DPL_ENABLE_MSVC_F16C
#      define DPL_ENABLE_MSVC_F16C 0
#    endif
#    define DPL_SIMD_X86_F16C DPL_ENABLE_MSVC_F16C
#  endif

#  ifdef __AVX512VPOPCNTDQ__
#    define DPL_SIMD_X86_AVX512VPOPCNTDQ __AVX512VPOPCNTDQ__
#  elif DPL_COMPILER_MSVC && defined(__AVX512VL__)
#    ifndef DPL_ENABLE_MSVC_AVX512VPOPCNTDQ
#      define DPL_ENABLE_MSVC_AVX512VPOPCNTDQ 0
#    endif
#    define DPL_SIMD_X86_AVX512VPOPCNTDQ DPL_ENABLE_MSVC_AVX512VPOPCNTDQ
#  endif

#  ifdef __AVX512BITALG__
#    define DPL_SIMD_X86_AVX512BITALG __AVX512BITALG__
#  elif DPL_COMPILER_MSVC && defined(__AVX512VL__)
#    ifndef DPL_ENABLE_MSVC_AVX512BITALG
#      define DPL_ENABLE_MSVC_AVX512BITALG 0
#    endif
#    define DPL_SIMD_X86_AVX512BITALG DPL_ENABLE_MSVC_AVX512BITALG
#  endif

#  ifdef __AVX512BF16__
#    define DPL_SIMD_X86_AVX512BF16 __AVX512BF16__
#  elif DPL_COMPILER_MSVC && defined(__AVX512F__)
#    ifndef DPL_ENABLE_MSVC_AVX512BF16
#      define DPL_ENABLE_MSVC_AVX512BF16 0
#    endif
#    define DPL_SIMD_X86_AVX512BF16 DPL_ENABLE_MSVC_AVX512BF16
#  endif

#  ifdef __AVX512FP16__
#    define DPL_SIMD_X86_AVX512FP16 __AVX512FP16__
#  elif DPL_COMPILER_MSVC && defined(__AVX512F__)
#    ifndef DPL_ENABLE_MSVC_AVX512FP16
#      define DPL_ENABLE_MSVC_AVX512FP16 0
#    endif
#    define DPL_SIMD_X86_AVX512FP16 DPL_ENABLE_MSVC_AVX512FP16
#  endif

#  ifdef __AVX512BW__
#    define DPL_SIMD_X86_AVX512BW __AVX512BW__
#  endif

#  ifdef __AVX512CD__
#    define DPL_SIMD_X86_AVX512CD __AVX512CD__
#  endif

#  ifdef __AVX512F__
#    define DPL_SIMD_X86_AVX512F __AVX512F__
#  endif

#  ifdef __AVX512DQ__
#    define DPL_SIMD_X86_AVX512DQ __AVX512DQ__
#  endif

#  ifdef __AVX512VL__
#    define DPL_SIMD_X86_AVX512VL __AVX512VL__
#  endif

#  ifdef __AVX512ER__
#    define DPL_SIMD_X86_AVX512ER __AVX512ER__
#  endif

#  ifdef __AVX512PF__
#    define DPL_SIMD_X86_AVX512PF __AVX512PF__
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
