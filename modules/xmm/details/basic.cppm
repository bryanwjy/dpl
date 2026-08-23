// Copyright 2026 Bryan Wong
module;

#define DPL_MODULES 1
#include "dpl/config.h"

#include <immintrin.h>
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
