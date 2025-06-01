/* Copyright 2023-2025 Bryan Wong */

#pragma once

#include "configuration/utl_compiler.h"

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
#    define DPL_EXPLICIT_IF(...) explicit(__VA_ARGS__)
#    define DPL_IMPLICIT_IF(...) explicit(!(__VA_ARGS__))

#    define DPL_TEMPLATE_CXX11(...)
#    define DPL_ENABLE_IF_CXX11(TYPE, ...) TYPE
#    define DPL_CONSTRAINT_CXX11(...)
#    define DPL_CONSTRAINT_CXX20(...) requires (__VA_ARGS__)
#    define DPL_CONCEPT_CXX20(...) __VA_ARGS__

#  else /* DPL_CXX >= 202002L */

#    define DPL_CONSTEXPR_CXX20
#    define DPL_CONSTEVAL constexpr
#    define DPL_EXPLICIT_IF(...)
#    define DPL_IMPLICIT_IF(...) explicit
#    define DPL_TEMPLATE_CXX11(...) , __VA_ARGS__
#    define DPL_ENABLE_IF_CXX11(TYPE, ...) __DPL enable_if_t<(__VA_ARGS__), TYPE>
#    define DPL_CONSTRAINT_CXX11(...) , __DPL enable_if_t<(__VA_ARGS__), int> = __LINE__
#    define DPL_CONSTRAINT_CXX20(...)
#    define DPL_CONCEPT_CXX20(...) typename
#  endif /* DPL_CXX >= 202002L */

#  if DPL_CXX >= 202302L
#    define DPL_CXX23 1
#    define DPL_CONSTEXPR_CXX23 constexpr
#    define DPL_CONSTEVAL_CXX23 consteval
#    define DPL_STATIC_CXX23 static
#    define DPL_CONST_CALL
#    define DPL_IF_CONSTEVAL(...) if consteval
#  else /* DPL_CXX >= 202302L */
#    define DPL_CONSTEXPR_CXX23
#    define DPL_CONSTEVAL_CXX23 constexpr
#    define DPL_STATIC_CXX23
#    define DPL_CONST_CALL const
#    define DPL_IF_CONSTEVAL(...) \
        if (DPL_CONSTANT_P(__VA_ARGS__)) /* This requires including 'utl_constant_p.h' */

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
#  define DPL_CONSTEXPR_CXX14 const
#  define DPL_INLINE_CXX17
#  define DPL_CONSTEXPR_CXX17 const
#  define DPL_CONSTEXPR_CXX20 const
#  define DPL_CONSTEXPR_CXX23 const
#  define DPL_CONSTEVAL

#endif /* ifdef DPL_CXX */
