// Copyright 2026 Bryan Wong
module;

#define DPL_MODULES 1
#include "dpl/config.h"

#if DPL_COMPILER_CLANG
// Workaround modules UB when compiling with Clang

// All functions in the intel intrinsic API has been made 'static'
// But since it is UB to transitively expose internal linkage function
// usage through the module interface, the 'static' attribute needs to be
// removed when compiling with modules on Clang
#  define static
#endif
#include <immintrin.h>

#if DPL_COMPILER_CLANG
#  undef static
#endif

export module dpl.xmm:details.basic;
import dpl;

DPL_DISABLE_WARNING_PUSH()
#if DPL_COMPILER_MSVC
DPL_DISABLE_WARNING(5244)
#endif

// IWYU pragma: begin_exports
#include "dpl/xmm/basic/abi.h"
#include "dpl/xmm/basic/broadcast.h"
#include "dpl/xmm/basic/extract.h"
#include "dpl/xmm/basic/from_bitset.h"
#include "dpl/xmm/basic/gather.h"
#include "dpl/xmm/basic/initialize.h"
#include "dpl/xmm/basic/load.h"
#include "dpl/xmm/basic/store.h"
#include "dpl/xmm/basic/to_bitset.h"
#include "dpl/xmm/basic/undefined.h"
// IWYU pragma: end_exports

DPL_DISABLE_WARNING_POP()
