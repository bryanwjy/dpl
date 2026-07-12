// Copyright 2026 Bryan Wong
module;

#define DPL_MODULES 1
#include "dpl/config.h"

export module dpl:core.details.numbers;
import :std.concepts;
import :std.bit;
import :std.type_traits;
import :std.utility;

DPL_DISABLE_WARNING_PUSH()
#if DPL_COMPILER_MSVC
DPL_DISABLE_WARNING(5244)
#endif

// IWYU pragma: begin_exports
#include "dpl/core/numbers/details/binary_layout_floating_point.h"
#include "dpl/core/numbers/ext.h"
#include "dpl/core/numbers/floating_point_like.h"
#include "dpl/core/numbers/floating_point_traits.h"
// IWYU pragma: end_exports

DPL_DISABLE_WARNING_POP()
