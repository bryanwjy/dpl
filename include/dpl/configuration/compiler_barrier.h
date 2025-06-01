/* Copyright 2023-2025 Bryan Wong */

#pragma once

#include "configuration/utl_compiler.h"
#include "configuration/utl_pragma.h"
#include "configuration/utl_standard.h"

#if DPL_SUPPORTS_GNU_ASM
#  define DPL_COMPILER_BARRIER() __asm__ __volatile__("" ::: "memory")
#elif DPL_COMPILER_MSVC

#  ifdef DPL_CXX
extern "C" void _ReadWriteBarrier();
#  else
void _ReadWriteBarrier();
#  endif
#  pragma intrinsic(_ReadWriteBarrier)
#  define DPL_COMPILER_BARRIER() _ReadWriteBarrier()
#else
DPL_PRAGMA_WARN("Unrecognize compiler")
#  define DPL_COMPILER_BARRIER()
#endif
