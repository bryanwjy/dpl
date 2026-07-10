// Copyright 2026 Bryan Wong
module;

#define DPL_MODULES 1
#include "dpl/config.h"

export module dpl:core.details.numbers;
import :std.concepts;
import :std.bit;
import :std.type_traits;
import :std.utility;

// IWYU pragma: begin_exports
#include "dpl/core/numbers/details/fwd.h"

#include "dpl/core/numbers/details/binary_layout_floating_point.h"
#include "dpl/core/numbers/details/extended_floating_point.h"
#include "dpl/core/numbers/details/extended_floating_point_operations.h"
#include "dpl/core/numbers/details/floating_point_like.h"
#include "dpl/core/numbers/details/promotable.h"
#include "dpl/core/numbers/details/storage16.h"
#include "dpl/core/numbers/details/xfp.h"
// IWYU pragma: end_exports
