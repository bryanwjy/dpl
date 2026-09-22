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
#      define DPL_SIMD_X86_SSE3 __AVX__
#    endif
#    ifndef DPL_SIMD_X86_SSE4_1
#      define DPL_SIMD_X86_SSE4_1 __AVX__
#    endif
#    ifndef DPL_SIMD_X86_SSE4_2
#      define DPL_SIMD_X86_SSE4_2 __AVX__
#    endif
#    define DPL_SIMD_X86_AVX __AVX__
#  endif

#  ifdef __AVX2__
#    define DPL_SIMD_X86_AVX2 __AVX2__
#  endif

#  ifdef __FMA__
#    define DPL_SIMD_X86_FMA __FMA__
#  elif DPL_COMPILER_MSVC && __AVX2__
#    ifndef DPL_DISABLE_MSVC_FMA
#      define DPL_DISABLE_MSVC_FMA 0
#    endif
#    define DPL_SIMD_X86_FMA (!DPL_DISABLE_MSVC_FMA)
#  endif

#  ifdef __F16C__
#    define DPL_SIMD_X86_F16C __F16C__
#  elif DPL_COMPILER_MSVC && __AVX2__
#    ifndef DPL_DISABLE_MSVC_F16C
#      define DPL_DISABLE_MSVC_F16C 0
#    endif
#    define DPL_SIMD_X86_F16C (!DPL_DISABLE_MSVC_F16C)
#  endif

#  ifdef __AVX512F__
#    define DPL_SIMD_X86_AVX512F __AVX512F__
#  endif

#  ifdef __AVX512CD__
#    define DPL_SIMD_X86_AVX512CD __AVX512CD__
#  elif DPL_COMPILER_MSVC && defined(__AVX512F__)
#    ifndef DPL_DISABLE_MSVC_AVX512CD
#      define DPL_DISABLE_MSVC_AVX512CD 0
#    endif
#    define DPL_SIMD_X86_AVX512CD (!DPL_DISABLE_MSVC_AVX512CD)
#  endif
#  ifdef __AVX512DQ__
#    define DPL_SIMD_X86_AVX512DQ __AVX512DQ__
#  elif DPL_COMPILER_MSVC && defined(__AVX512F__)
#    ifndef DPL_DISABLE_MSVC_AVX512DQ
#      define DPL_DISABLE_MSVC_AVX512DQ 0
#    endif
#    define DPL_SIMD_X86_AVX512DQ (!DPL_DISABLE_MSVC_AVX512DQ)
#  endif
#  ifdef __AVX512BW__
#    define DPL_SIMD_X86_AVX512BW __AVX512BW__
#  elif DPL_COMPILER_MSVC && defined(__AVX512F__)
#    ifndef DPL_DISABLE_MSVC_AVX512BW
#      define DPL_DISABLE_MSVC_AVX512BW 0
#    endif
#    define DPL_SIMD_X86_AVX512BW (!DPL_DISABLE_MSVC_AVX512BW)
#  endif
#  ifdef __AVX512VL__
#    define DPL_SIMD_X86_AVX512VL __AVX512VL__
#  elif DPL_COMPILER_MSVC && defined(__AVX512F__)
#    ifndef DPL_DISABLE_MSVC_AVX512VL
#      define DPL_DISABLE_MSVC_AVX512VL 0
#    endif
#    define DPL_SIMD_X86_AVX512VL (!DPL_DISABLE_MSVC_AVX512VL)
#  endif
#  ifdef __AVX512BITALG__
#    define DPL_SIMD_X86_AVX512BITALG __AVX512BITALG__
#  elif DPL_COMPILER_MSVC && defined(__AVX512F__)
#    ifndef DPL_DISABLE_MSVC_AVX512BITALG
#      define DPL_DISABLE_MSVC_AVX512BITALG 0
#    endif
#    define DPL_SIMD_X86_AVX512BITALG (!DPL_DISABLE_MSVC_AVX512BITALG)
#  endif
#  ifdef __AVX512VBMI__
#    define DPL_SIMD_X86_AVX512VBMI __AVX512VBMI__
#  elif DPL_COMPILER_MSVC && defined(__AVX512F__)
#    ifndef DPL_DISABLE_MSVC_AVX512VBMI
#      define DPL_DISABLE_MSVC_AVX512VBMI 0
#    endif
#    define DPL_SIMD_X86_AVX512VBMI (!DPL_DISABLE_MSVC_AVX512VBMI)
#  endif
#  ifdef __AVX512VBMI2__
#    define DPL_SIMD_X86_AVX512VBMI2 __AVX512VBMI2__
#  elif DPL_COMPILER_MSVC && defined(__AVX512F__)
#    ifndef DPL_DISABLE_MSVC_AVX512VBMI2
#      define DPL_DISABLE_MSVC_AVX512VBMI2 0
#    endif
#    define DPL_SIMD_X86_AVX512VBMI2 (!DPL_DISABLE_MSVC_AVX512VBMI2)
#  endif
#  ifdef __AVX512IFMA__
#    define DPL_SIMD_X86_AVX512IFMA __AVX512IFMA__
#  elif DPL_COMPILER_MSVC && defined(__AVX512F__)
#    ifndef DPL_DISABLE_MSVC_AVX512IFMA
#      define DPL_DISABLE_MSVC_AVX512IFMA 0
#    endif
#    define DPL_SIMD_X86_AVX512IFMA (!DPL_DISABLE_MSVC_AVX512IFMA)
#  endif
#  ifdef __AVX512VNNI__
#    define DPL_SIMD_X86_AVX512VNNI __AVX512VNNI__
#  elif DPL_COMPILER_MSVC && defined(__AVX512F__)
#    ifndef DPL_DISABLE_MSVC_AVX512VNNI
#      define DPL_DISABLE_MSVC_AVX512VNNI 0
#    endif
#    define DPL_SIMD_X86_AVX512VNNI (!DPL_DISABLE_MSVC_AVX512VNNI)
#  endif
#  ifdef __AVX512VPOPCNTDQ__
#    define DPL_SIMD_X86_AVX512VPOPCNTDQ __AVX512VPOPCNTDQ__
#  elif DPL_COMPILER_MSVC && defined(__AVX512F__)
#    ifndef DPL_DISABLE_MSVC_AVX512VPOPCNTDQ
#      define DPL_DISABLE_MSVC_AVX512VPOPCNTDQ 0
#    endif
#    define DPL_SIMD_X86_AVX512VPOPCNTDQ (!DPL_DISABLE_MSVC_AVX512VPOPCNTDQ)
#  endif
#  ifdef __AVX512FP16__
#    define DPL_SIMD_X86_AVX512FP16 __AVX512FP16__
#  elif DPL_COMPILER_MSVC && defined(__AVX512F__)
#    ifndef DPL_DISABLE_MSVC_AVX512FP16
#      define DPL_DISABLE_MSVC_AVX512FP16 0
#    endif
#    define DPL_SIMD_X86_AVX512FP16 (!DPL_DISABLE_MSVC_AVX512FP16)
#  endif
#  ifdef __AVX512BF16__
#    define DPL_SIMD_X86_AVX512BF16 __AVX512BF16__
#  elif DPL_COMPILER_MSVC && defined(__AVX512F__)
#    ifndef DPL_DISABLE_MSVC_AVX512BF16
#      define DPL_DISABLE_MSVC_AVX512BF16 0
#    endif
#    define DPL_SIMD_X86_AVX512BF16 (!DPL_DISABLE_MSVC_AVX512BF16)
#  endif
#  ifdef __AVX512VP2INTERSECT__
#    define DPL_SIMD_X86_AVX512VP2INTERSECT __AVX512VP2INTERSECT__
#  elif DPL_COMPILER_MSVC && defined(__AVX512F__)
#    ifndef DPL_DISABLE_MSVC_AVX512VP2INTERSECT
#      define DPL_DISABLE_MSVC_AVX512VP2INTERSECT 0
#    endif
#    define DPL_SIMD_X86_AVX512VP2INTERSECT \
        (!DPL_DISABLE_MSVC_AVX512VP2INTERSECT)
#  endif

#  ifdef __AVX512BMM__
#    define DPL_SIMD_X86_AVX512BMM __AVX512BMM__
#  elif DPL_COMPILER_MSVC && defined(__AVX512F__)
#    ifndef DPL_DISABLE_MSVC_AVX512BMM
#      define DPL_DISABLE_MSVC_AVX512BMM 0
#    endif
#    define DPL_SIMD_X86_AVX512BMM (!DPL_DISABLE_MSVC_AVX512BMM)
#  endif

#  ifdef __AVX10_1__
#    define DPL_SIMD_X86_AVX10_1 __AVX10_1__
#  endif

#  ifdef __AVX10_2__
#    define DPL_SIMD_X86_AVX10_1 __AVX10_2__
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
