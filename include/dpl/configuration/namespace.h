/* Copyright 2023-2025 Bryan Wong */

#pragma once

#include "configuration/utl_abi.h"

#define DPL_NAMESPACE_BEGIN(NAME) namespace __DPL_PUBLIC_TYPE_VISIBILITY NAME {

#define DPL_NAMESPACE_END(NAME) }

#include <stddef.h>

#ifdef _LIBCPP_ABI_NAMESPACE
#  define DPL_STD_ABI_NAMESPACE_BEGIN inline namespace _LIBCPP_ABI_NAMESPACE {
#  define DPL_STD_ABI_NAMESPACE_END }
#elif defined(_GLIBCXX_BEGIN_NAMESPACE_VERSION)
#  define DPL_STD_ABI_NAMESPACE_BEGIN _GLIBCXX_BEGIN_NAMESPACE_VERSION
#  define DPL_STD_ABI_NAMESPACE_END _GLIBCXX_END_NAMESPACE_VERSION
#endif

#ifndef DPL_STD_ABI_NAMESPACE_BEGIN
#  define DPL_STD_ABI_NAMESPACE_BEGIN
#  define DPL_STD_ABI_NAMESPACE_END
#endif

/* extern C++ for MSVC > C++20, no effect anywhere else */
#define DPL_STD_NAMESPACE_BEGIN                                              \
    DPL_EXTERN_CXX_BEGIN namespace DPL_ATTRIBUTE(TYPE_VISIBILITY("default")) \
        std {                                                                \
        DPL_STD_ABI_NAMESPACE_BEGIN

#define DPL_STD_NAMESPACE_END \
    DPL_STD_ABI_NAMESPACE_END \
    }                         \
    DPL_EXTERN_CXX_END
