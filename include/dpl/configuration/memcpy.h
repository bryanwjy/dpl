/* Copyright 2023-2025 Bryan Wong */

#pragma once

#include "dpl/configuration/builtin_check.h"
#include "dpl/configuration/compiler.h"
#include "dpl/configuration/keywords.h"
#include "dpl/configuration/standard.h"

#if DPL_HAS_BUILTIN(__builtin_memcpy)

#  define __DPL_MEMCPY __builtin_memcpy

#else

#  if DPL_CXX
extern "C" void* memcpy(
    void* DPL_RESTRICT, void const* DPL_RESTRICT, decltype(sizeof(0)));
#  else
#    include <stddef.h>
void* memcpy(void* DPL_RESTRICT, void const* DPL_RESTRICT, size_t);
#  endif

#  if DPL_COMPILER_MSVC
#    pragma intrinsic(memcpy)
#  endif
#  define __DPL_MEMCPY ::memcpy

#endif
