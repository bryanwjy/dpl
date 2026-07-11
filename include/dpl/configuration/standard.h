/* Copyright 2023-2025 Bryan Wong */

#pragma once

#include "dpl/configuration/compiler.h" // IWYU pragma: keep
#include "dpl/configuration/stl.h"      // IWYU pragma: keep

#if DPL_COMPILER_MSVC
#  ifdef __cplusplus
#    define DPL_CXX _MSVC_LANG
#  endif
#elif defined(__cplusplus)
#  define DPL_CXX __cplusplus
#endif

#ifdef DPL_CXX

#  if DPL_CXX < 201103L
#    error "C++11 is required"
#  endif /* DPL_CXX < 201103L */

#  define DPL_CXX11 1

#  if DPL_CXX >= 201402L
#    define DPL_CXX14 1
#    define DPL_CONSTEXPR_CXX14 constexpr
#  else /* DPL_CXX >= 201402L */
#    define DPL_CONSTEXPR_CXX14
#  endif /* DPL_CXX >= 201402L */

#  if DPL_CXX >= 201703L
#    define DPL_CXX17 1
#    define DPL_CONSTEXPR_CXX17 constexpr
#    define DPL_INLINE_CXX17 inline
#    define DPL_IF_CONSTEXPR if constexpr
#  else /* DPL_CXX >= 201703L */
#    define DPL_INLINE_CXX17
#    define DPL_CONSTEXPR_CXX17
#    define DPL_IF_CONSTEXPR if
#  endif /* DPL_CXX >= 201703L */

#  if DPL_CXX >= 202002L
#    define DPL_CXX20 1
#    define DPL_CONSTEXPR_CXX20 constexpr
#    define DPL_CONSTEVAL consteval

#  else /* DPL_CXX >= 202002L */

#    define DPL_CONSTEXPR_CXX20
#    define DPL_CONSTEVAL constexpr
#  endif /* DPL_CXX >= 202002L */

#  if DPL_CXX >= 202302L
#    define DPL_CXX23 1
#    define DPL_CONSTEXPR_CXX23 constexpr
#    define DPL_CONSTEVAL_CXX23 consteval
#    define DPL_STATIC_CALL static
#    define DPL_CONST_CALL
#  else /* DPL_CXX >= 202302L */
#    define DPL_CONSTEXPR_CXX23
#    define DPL_CONSTEVAL_CXX23 constexpr
#    define DPL_STATIC_CALL
#    define DPL_CONST_CALL const

#  endif /* DPL_CXX >= 202302L */

#  if __cpp_constexpr >= 202306L
#    define DPL_CONSTEXPR_VOIDP constexpr
#  else
#    define DPL_CONSTEXPR_VOIDP
#  endif

#  if DPL_CXX >= 202602L
#    define DPL_CXX26 1
#    define DPL_CONSTEXPR_CXX26 constexpr
#    define DPL_CONSTEVAL_CXX26 consteval
#  else /* DPL_CXX >= 202302L */
#    define DPL_CONSTEXPR_CXX26
#    define DPL_CONSTEVAL_CXX26 constexpr

#  endif /* DPL_CXX >= 202302L */

#else /* ifdef DPL_CXX */

/* Not C++, so only define qualifiers usable on global functions/variables */
#  define DPL_CONSTEXPR_CXX14
#  define DPL_INLINE_CXX17
#  define DPL_CONSTEXPR_CXX17
#  define DPL_CONSTEXPR_CXX20
#  define DPL_CONSTEXPR_CXX23
#  define DPL_CONSTEVAL

#endif /* ifdef DPL_CXX */
