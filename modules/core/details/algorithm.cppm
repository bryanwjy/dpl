// Copyright 2025 Bryan Wong
module;

#define DPL_MODULES 1
#include "dpl/config.h"

export module dpl:core.details.algorithm;
export import :core.fwd;
export import :core.operations;
import :core.type_traits;
import :std.concepts;
import :std.bit;
import :std.type_traits;
import :std.utility;
import :core.basic;
import :core.concepts;
import :core.type_traits;
import :core.utility;
import :core.math;

DPL_DISABLE_WARNING_PUSH()
#if DPL_COMPILER_MSVC
DPL_DISABLE_WARNING(5244)
#endif

// IWYU pragma: begin_exports
#include "dpl/core/algorithm/clamp.h"
#include "dpl/core/algorithm/compress.h"
#include "dpl/core/algorithm/expand.h"
#include "dpl/core/algorithm/lerp.h"
#include "dpl/core/algorithm/lookup.h"
#include "dpl/core/algorithm/reduce.h"
#include "dpl/core/algorithm/rotate.h"
#include "dpl/core/algorithm/scan.h"
#include "dpl/core/algorithm/shift.h"
#include "dpl/core/algorithm/slide.h"
#include "dpl/core/algorithm/splice.h"
// IWYU pragma: end_exports

DPL_DISABLE_WARNING_POP()
