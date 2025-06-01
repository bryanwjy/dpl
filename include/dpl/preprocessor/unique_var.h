/* Copyright 2023-2025 Bryan Wong */

#pragma once

#include "configuration/utl_compiler.h"
#include "preprocessor/utl_concatenation.h"

#if DPL_COMPILER_CLANG | DPL_COMPILER_GCC | DPL_COMPILER_MSVC | DPL_COMPILER_INTEL

#  define DPL_UNIQUE_VAR(var) DPL_CONCAT(var, __COUNTER__)

#else

/* On unrecognized platforms, use __LINE__ */
#  define DPL_UNIQUE_VAR(var) DPL_CONCAT(var, __LINE__)

#endif /* if DPL_COMPILER_CLANG */
