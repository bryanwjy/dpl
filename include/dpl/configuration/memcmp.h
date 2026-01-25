/* Copyright 2023-2025 Bryan Wong */

#pragma once

#include "dpl/configuration/builtin_check.h"

#if DPL_HAS_BUILTIN(__builtin_memcmp)

#  define __DPL_MEMCMP __builtin_memcmp

#else

#  include "dpl/configuration/compiler.h"
#  include "dpl/configuration/standard.h"

#  if DPL_CXX
extern "C" int memcmp(void const*, void const*, decltype(sizeof(0)));
#  else
#    include <stddef.h>
int memcmp(void const*, void const*, size_t);
#  endif

#  if DPL_COMPILER_MSVC
#    pragma intrinsic(memcmp)
#  endif
#  define __DPL_MEMCMP ::memcmp

#endif
