/* Copyright 2023-2025 Bryan Wong */

#pragma once

#include "configuration/utl_compiler.h"
#include "configuration/utl_standard.h"

#ifdef __is_identifier
#  define DPL_IS_RESERVED_IDENTIFIER(X) !__is_identifier(X)
#else
#  define DPL_IS_RESERVED_IDENTIFIER(...) 0
#endif

#ifdef __has_builtin
#  define DPL_HAS_BUILTIN(BUILTIN) (__has_builtin(BUILTIN) || DPL_IS_RESERVED_IDENTIFIER(BUILTIN))
#else
#  define DPL_HAS_BUILTIN(...) 0
#endif /* ifdef __has_builtin */

#define __DPL_SHOULD_USE_BUILTIN(BUILTIN) \
    DPL_HAS_BUILTIN(__##BUILTIN) && !DPL_DISABLE_BUILTIN_##BUILTIN
