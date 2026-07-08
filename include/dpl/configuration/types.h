/* Copyright 2023-2025 Bryan Wong */
#pragma once

#include "dpl/configuration/builtin_check.h" // IWYU pragma: keep

#ifdef __SIZEOF_INT128__
#  if !DPL_DISABLE_INT128_SUPPORT
#    define DPL_SUPPORTS_INT128 1
#    define DPL_SUPPORTS_EXT_INT128 1
#  endif
#endif

#if !DPL_DISABLE_FP16_SUPPORT
#  ifdef __STDCPP_FLOAT16_T__
#    define DPL_SUPPORTS_STD_FLOAT16 1
#  else
#    define DPL_SUPPORTS_STD_FLOAT16 0
#  endif

#  ifdef __FLT16_MAX__
#    define DPL_SUPPORTS_EXT_FLOAT16 1
#  else
#    define DPL_SUPPORTS_EXT_FLOAT16 0
#  endif

#  if DPL_IS_RESERVED_IDENTIFIER(__f16)
#    define DPL_SUPPORTS_STORAGE_FLOAT16 1
#  else
#    define DPL_SUPPORTS_STORAGE_FLOAT16 0
#  endif

#  define DPL_SUPPORTS_FLOAT16 \
      (DPL_SUPPORTS_STD_FLOAT16 | DPL_SUPPORTS_EXT_FLOAT16)
#else
#  define DPL_SUPPORTS_FLOAT16 0
#endif

#if !DPL_DISABLE_BF16_SUPPORT
#  ifdef __STDCPP_BFLOAT16_T__
#    define DPL_SUPPORTS_STD_BFLOAT16 1
#  else
#    define DPL_SUPPORTS_STD_BFLOAT16 0
#  endif

#  ifdef __BFLT16_MAX__
#    define DPL_SUPPORTS_EXT_BFLOAT16 1
#  else
#    define DPL_SUPPORTS_EXT_BFLOAT16 0
#  endif

#  if DPL_IS_RESERVED_IDENTIFIER(__bf16)
#    define DPL_SUPPORTS_STORAGE_BFLOAT16 1
#  else
#    define DPL_SUPPORTS_STORAGE_BFLOAT16 0
#  endif

#  define DPL_SUPPORTS_BFLOAT16 \
      (DPL_SUPPORTS_STD_BFLOAT16 | DPL_SUPPORTS_EXT_BFLOAT16)
#else
#  define DPL_SUPPORTS_BFLOAT16 0
#endif

#ifdef __STDCPP_FLOAT32_T__
#  define DPL_SUPPORTS_STD_FLOAT32 1
#  define DPL_SUPPORTS_FLOAT32 1
#endif
#ifdef __STDCPP_FLOAT64_T__
#  define DPL_SUPPORTS_STD_FLOAT64 1
#  define DPL_SUPPORTS_FLOAT64 1
#endif
#ifdef __STDCPP_FLOAT128_T__
#  define DPL_SUPPORTS_STD_FLOAT128 1
#  define DPL_SUPPORTS_FLOAT128 1
#endif

#ifdef __cpp_char8_t
#  if __cpp_char8_t >= 201811L
#    define DPL_SUPPORTS_STD_CHAR8_T 1
#    define DPL_SUPPORTS_CHAR8_T 1
#  endif
#endif
