/* Copyright 2023-2025 Bryan Wong */

#pragma once

#include "dpl/configuration/abi.h"
#include "dpl/configuration/stl.h" // IWYU pragma: keep

#define DPL_NAMESPACE_BEGIN(NAME) namespace __DPL_PUBLIC_TYPE_VISIBILITY NAME {

#define DPL_NAMESPACE_END(NAME) }

#if DPL_MSVC_STL
#  define DPL_STD_NAMESPACE_BEGIN _STD_BEGIN
#  define DPL_STD_NAMESPACE_END _STD_END
#else
#  ifdef _LIBCPP_ABI_NAMESPACE
#    define __DPL_STD_ABI_NAMESPACE_BEGIN \
        inline namespace _LIBCPP_ABI_NAMESPACE {
#    define __DPL_STD_ABI_NAMESPACE_END }
#  elif defined(_GLIBCXX_BEGIN_NAMESPACE_VERSION)
#    define __DPL_STD_ABI_NAMESPACE_BEGIN _GLIBCXX_BEGIN_NAMESPACE_VERSION
#    define __DPL_STD_ABI_NAMESPACE_END _GLIBCXX_END_NAMESPACE_VERSION
#  endif

#  ifndef __DPL_STD_ABI_NAMESPACE_BEGIN
#    define __DPL_STD_ABI_NAMESPACE_BEGIN
#    define __DPL_STD_ABI_NAMESPACE_END
#  endif

/* extern C++ for MSVC > C++20, no effect anywhere else */
/* Note: Does not apply to all std entities */
#  define DPL_STD_NAMESPACE_BEGIN                               \
      namespace DPL_ATTRIBUTE(TYPE_VISIBILITY("default")) std { \
      __DPL_STD_ABI_NAMESPACE_BEGIN

#  define DPL_STD_NAMESPACE_END   \
      __DPL_STD_ABI_NAMESPACE_END \
      }

#endif
