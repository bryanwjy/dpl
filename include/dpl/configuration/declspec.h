/* Copyright 2023-2025 Bryan Wong */

#pragma once

#include "configuration/utl_compiler.h"

#include "preprocessor/utl_concatenation.h"
#include "preprocessor/utl_is_empty.h"

#if DPL_COMPILER_MSVC
#  define DPL_HAS_DECLSPEC(NAME) DPL_IS_EMPTY(DPL_CONCAT(DPL_DECLSPEC_, NAME))

#  define DPL_DECLSPEC_dllexport
#  define DPL_DECLSPEC_dllimport
#  define DPL_DECLSPEC_allocator
#  define DPL_DECLSPEC_deprecated
#  define DPL_DECLSPEC_empty_bases
#  define DPL_DECLSPEC_noalias
#  define DPL_DECLSPEC_noinline
#  define DPL_DECLSPEC_noreturn
#  define DPL_DECLSPEC_novtable
#  define DPL_DECLSPEC_no_sanitize_address
#  define DPL_DECLSPEC_restrict

#elif DPL_COMPILER_ARMCC /* NOT DEFINED */

#  define DPL_HAS_DECLSPEC(NAME) DPL_IS_EMPTY(DPL_CONCAT(DPL_DECLSPEC_, NAME))

#  define DPL_DECLSPEC_dllexport
#  define DPL_DECLSPEC_dllimport
#  define DPL_DECLSPEC_noinline
#  define DPL_DECLSPEC_noreturn
#  define DPL_DECLSPEC_notshared

#else
#  define DPL_HAS_DECLSPEC(NAME) 0
#endif
