/* Copyright 2023-2025 Bryan Wong */

#pragma once

#include "configuration/utl_compiler.h"
#include "configuration/utl_standard.h"
#include "preprocessor/utl_paste.h"

#define DPL_EVAL1(...) DPL_PASTE(DPL_PASTE(DPL_PASTE(__VA_ARGS__)))
#define DPL_EVAL2(...) DPL_EVAL1(DPL_EVAL1(DPL_EVAL1(__VA_ARGS__)))
#define DPL_EVAL3(...) DPL_EVAL2(DPL_EVAL2(DPL_EVAL2(__VA_ARGS__)))
#define DPL_EVAL4(...) DPL_EVAL3(DPL_EVAL3(DPL_EVAL3(__VA_ARGS__)))
#define DPL_EVAL(...) DPL_EVAL4(DPL_EVAL4(DPL_EVAL4(__VA_ARGS__)))

#if DPL_CXX20 && !DPL_MSVC_PREPROCESSOR

#  define DPL_APPLY_MACRO_PARENTHESIS ()
#  define DPL_APPLY_MACRO_IMPL(F, _0, ...) \
      F(_0)                                \
      __VA_OPT__(DPL_APPLY_MACRO_LOOP DPL_APPLY_MACRO_PARENTHESIS DPL_PASTE((F, __VA_ARGS__)))
#  define DPL_APPLY_MACRO_LOOP() DPL_APPLY_MACRO_IMPL

#  define DPL_APPLY_MACRO(F, ...) __VA_OPT__(DPL_EVAL(DPL_APPLY_MACRO_IMPL(F, __VA_ARGS__)))

#else

#  define DPL_APPLY_MACRO_END(...)
#  define DPL_APPLY_MACRO_OUT
#  define DPL_APPLY_MACRO_COMMA ,

#  define DPL_APPLY_MACRO_GET_END2() 0, DPL_APPLY_MACRO_END
#  define DPL_APPLY_MACRO_GET_END1(...) DPL_APPLY_MACRO_GET_END2
#  define DPL_APPLY_MACRO_GET_END(...) DPL_APPLY_MACRO_GET_END1
#  define DPL_APPLY_MACRO_NEXT0(test, next, ...) next DPL_APPLY_MACRO_OUT
#  if !DPL_MSVC_PREPROCESSOR
#    define DPL_APPLY_MACRO_NEXT1(test, next) DPL_APPLY_MACRO_NEXT0(test, next, 0)
#    define DPL_APPLY_MACRO_NEXT(test, next) \
        DPL_APPLY_MACRO_NEXT1(DPL_APPLY_MACRO_GET_END test, next)
#    define DPL_APPLY_MACRO0(f, x, peek, ...) \
        f(x) DPL_APPLY_MACRO_NEXT(peek, DPL_APPLY_MACRO1)(f, peek, __VA_ARGS__)
#    define DPL_APPLY_MACRO1(f, x, peek, ...) \
        f(x) DPL_APPLY_MACRO_NEXT(peek, DPL_APPLY_MACRO0)(f, peek, __VA_ARGS__)
#  else /* if !DPL_MSVC_PREPROCESSOR */
#    define DPL_APPLY_MACRO_NEXT1(test, next) DPL_PASTE(DPL_APPLY_MACRO_NEXT0(test, next, 0))
#    define DPL_APPLY_MACRO_NEXT(test, next) \
        DPL_PASTE(DPL_APPLY_MACRO_NEXT1(DPL_APPLY_MACRO_GET_END test, next))
#    define DPL_APPLY_MACRO0(f, x, peek, ...) \
        f(x) DPL_PASTE(DPL_APPLY_MACRO_NEXT(peek, DPL_APPLY_MACRO1)(f, peek, __VA_ARGS__))
#    define DPL_APPLY_MACRO1(f, x, peek, ...) \
        f(x) DPL_PASTE(DPL_APPLY_MACRO_NEXT(peek, DPL_APPLY_MACRO0)(f, peek, __VA_ARGS__))
#  endif /* if !DPL_MSVC_PREPROCESSOR */

#  define DPL_APPLY_MACRO_LIST_NEXT1(test, next) \
      DPL_APPLY_MACRO_NEXT0(test, DPL_APPLY_MACRO_COMMA next, 0)
#  define DPL_APPLY_MACRO_LIST_NEXT(test, next) \
      DPL_APPLY_MACRO_LIST_NEXT1(DPL_APPLY_MACRO_GET_END test, next)

#  define DPL_APPLY_MACRO_LIST0(f, x, peek, ...) \
      f(x) DPL_PASTE(DPL_APPLY_MACRO_LIST_NEXT(peek, DPL_APPLY_MACRO_LIST1)(f, peek, __VA_ARGS__))
#  define DPL_APPLY_MACRO_LIST1(f, x, peek, ...) \
      f(x) DPL_PASTE(DPL_APPLY_MACRO_LIST_NEXT(peek, DPL_APPLY_MACRO_LIST0)(f, peek, __VA_ARGS__))

/**
 * Applies the function macro `f` to each of the remaining parameters.
 */
#  define DPL_APPLY_MACRO(f, ...) \
      DPL_EVAL(DPL_APPLY_MACRO1(f, __VA_ARGS__, ()()(), ()()(), ()()(), 0))

#endif /* if DPL_CXX20 */
