/* Copyright 2023-2025 Bryan Wong */

#pragma once

#include "dpl/configuration/glibcxx.h"  // IWYU pragma: keep
#include "dpl/configuration/standard.h" // IWYU pragma: keep
#include "dpl/preprocessor/concatenation.h"

/**
 * Standard library macros
 */

#ifdef _LIBCPP_VERSION
#  define DPL_LIBCXX 1

/**
 * All arguments must be two character decimal digits, if the value is single
 * digit the argument should be prefixed with '0'
 */
#  define DPL_LIBCXX_AT_LEAST(MAJOR, MINOR, PATCH) \
      (_LIBCPP_VERSION >= DPL_CONCAT(DPL_CONCAT(MAJOR, MINOR), PATCH))
#else
#  define DPL_LIBCXX_AT_LEAST(MAJOR, MINOR, PATCH) 0
#endif

#ifdef __DPL_GLIBCXX_GCC_MAJOR
#  if DPL_LIBCXX
#    error "Invalid environment"
#  endif
/**
 * Only major version is exposed
 */
#  define DPL_LIBSTDCXX_AT_LEAST(MAJOR) (__DPL_GLIBCXX_GCC_MAJOR >= MAJOR)
#  define DPL_LIBSTDCXX_AFTER(YYYY, MM, DD) (__GLIBCXX__ > YYYY##MM##DD)
#  define DPL_LIBSTDCXX 1
#else
#  define DPL_LIBSTDCXX_AFTER(YYYY, MM, DD) 0
#  define DPL_LIBSTDCXX_AT_LEAST(MAJOR) 0
#endif

#ifdef _MSVC_STL_VERSION

#  ifndef _MSVC_STL_UPDATE
#    error "Invalid environment"
#  endif
#  if DPL_LIBCXX | DPL_LIBSTDCXX
#    error "Invalid environment"
#  endif

#  define DPL_MSVC_STL 1

#  define DPL_MSVC_STL_AT_LEAST(VERSION, UPDATE) \
      (_MSVC_STL_VERSION > VERSION ||            \
          (_MSVC_STL_VERSION == VERSION && _MSVC_STL_UPDATE >= UPDATE))
#else
#  define DPL_MSVC_STL_AT_LEAST(VERSION, UPDATE) 0
#endif
