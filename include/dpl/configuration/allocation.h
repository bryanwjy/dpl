/* Copyright 2023-2025 Bryan Wong */

#pragma once

#ifndef __cpp_aligned_new
#  define DPL_SUPPORTS_ALIGNED_ALLOCATION 0
#elif __cpp_aligned_new >= 201606L
#  define DPL_SUPPORTS_ALIGNED_ALLOCATION 1
#else
#  define DPL_SUPPORTS_ALIGNED_ALLOCATION 0
#endif

#ifndef __cpp_sized_deallocation
#  define DPL_SUPPORTS_SIZED_DEALLOCATION 0
#elif __cpp_sized_deallocation >= 201309L
#  define DPL_SUPPORTS_SIZED_DEALLOCATION 1
#else
#  define DPL_SUPPORTS_SIZED_DEALLOCATION 0
#endif

#ifdef __STDCPP_DEFAULT_NEW_ALIGNMENT__
#  define DPL_DEFAULT_NEW_ALIGNMENT __STDCPP_DEFAULT_NEW_ALIGNMENT__
#endif
