/* Copyright 2023-2025 Bryan Wong */

#pragma once

#include "configuration/utl_builtin_check.h"
#include "configuration/utl_compiler.h"
#include "configuration/utl_msvc_builtins.h"
#include "configuration/utl_standard.h"

#if DPL_HAS_BUILTIN(__builtin_launder)
#  define DPL_BUILTIN_launder(X) __builtin_launder(X)
#elif DPL_COMPILER_GCC_AT_LEAST(7, 1, 0)
#  define DPL_BUILTIN_launder(X) __builtin_launder(X)
#elif DPL_COMPILER_ICC_AT_LEAST(1900)
#  define DPL_BUILTIN_launder(X) __builtin_launder(X)
#elif DPL_COMPILER_ICX
#  define DPL_BUILTIN_launder(X) __builtin_launder(X)
#endif /* DPL_HAS_BUILTIN(__builtin_launder) */

#if DPL_HAS_BUILTIN(__builtin_addressof)
#  define DPL_BUILTIN_addressof(X) __builtin_addressof(X)
#elif DPL_COMPILER_GCC_AT_LEAST(7, 1, 0)
#  define DPL_BUILTIN_addressof(X) __builtin_addressof(X)
#elif DPL_COMPILER_MSVC_AT_LEAST(1914)
#  define DPL_BUILTIN_addressof(X) __builtin_addressof(X)
#elif DPL_COMPILER_ICC_AT_LEAST(1900)
#  define DPL_BUILTIN_addressof(X) __builtin_addressof(X)
#endif /* if DPL_HAS_BUILTIN(__builtin_addressof) */

#if DPL_HAS_BUILTIN(__builtin_unreachable)
#  define DPL_BUILTIN_unreachable() __builtin_unreachable()
#elif DPL_COMPILER_GCC_AT_LEAST(4, 5, 0)
#  define DPL_BUILTIN_unreachable() __builtin_unreachable()
#elif DPL_COMPILER_ICC_AT_LEAST(1300)
#  define DPL_BUILTIN_unreachable() __builtin_unreachable()
#elif DPL_COMPILER_MSVC /* DPL_HAS_BUILTIN(__builtin_unreachable) */
#  define DPL_BUILTIN_unreachable() __assume(0)
#else /* DPL_HAS_BUILTIN(__builtin_unreachable) */

#  if DPL_HAS_BUILTIN(__builtin_expect)
#    define DPL_BUILTIN_expect(VALUE, EXP) __builtin_expect(VALUE, EXP)
#  else
#    define DPL_BUILTIN_expect(VALUE, _1) VALUE
#  endif

#  ifdef DPL_CXX
extern "C" void abort(void);
#  else
void abort(void);
#  endif

#  define DPL_BUILTIN_unreachable() abort()
#endif /* DPL_HAS_BUILTIN(__builtin_unreachable) */

#if DPL_HAS_BUILTIN(__builtin_is_constant_evaluated)
#  define DPL_BUILTIN_is_constant_evaluated() __builtin_is_constant_evaluated()
#endif /* DPL_HAS_BUILTIN(__builtin_is_constant_evaluated) */

#if DPL_HAS_BUILTIN(__builtin_bit_cast)
#  define DPL_BUILTIN_bit_cast(...) __builtin_bit_cast(__VA_ARGS__)
#endif /* DPL_HAS_BUILTIN(__builtin_is_constant_evaluated) */

#if DPL_HAS_BUILTIN(__builtin_assume)
#  define DPL_BUILTIN_assume(...) __builtin_assume(__VA_ARGS__)
#elif DPL_COMPILER_MSVC /* DPL_HAS_BUILTIN(__builtin_unreachable) */
#  define DPL_BUILTIN_assume(...) __assume(__VA_ARGS__)
#elif DPL_HAS_BUILTIN(__builtin_unreachable)
#  define DPL_BUILTIN_assume(...) ((__VA_ARGS__) ? (void)0 : __builtin_unreachable())
#else
#  define DPL_BUILTIN_assume(...) (void)0
#endif /* DPL_HAS_BUILTIN(__builtin_assume) */

#if DPL_HAS_BUILTIN(__is_layout_compatible) || DPL_COMPILER_GCC_AT_LEAST(12, 0, 0)
#  define DPL_BUILTIN_is_layout_compatible(...) __is_layout_compatible(__VA_ARGS__)
#endif

#if DPL_HAS_BUILTIN(__is_pointer_interconvertible_base_of) || DPL_COMPILER_GCC_AT_LEAST(12, 0, 0)
#  define DPL_BUILTIN_is_pointer_interconvertible_base_of(...) \
      __is_pointer_interconvertible_base_of(__VA_ARGS__)
#endif

#if DPL_HAS_BUILTIN(__builtin_source_location)
#  define DPL_BUILTIN_source_location() __builtin_source_location()
#endif

#if DPL_HAS_BUILTIN(__builtin_FILE)
#  define DPL_BUILTIN_FILE() __builtin_FILE()
#endif

#if DPL_HAS_BUILTIN(__builtin_FUNCTION)
#  define DPL_BUILTIN_FUNCTION() __builtin_FUNCTION()
#endif

#if DPL_HAS_BUILTIN(__builtin_LINE)
#  define DPL_BUILTIN_LINE() __builtin_LINE()
#endif

#if DPL_HAS_BUILTIN(__builtin_COLUMN)
#  define DPL_BUILTIN_COLUMN() __builtin_COLUMN()
#endif

#if DPL_HAS_BUILTIN(__builtin_FUNCSIG)
#  define DPL_BUILTIN_FUNCSIG() __builtin_FUNCSIG()
#endif
