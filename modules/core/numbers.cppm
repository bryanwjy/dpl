// Copyright 2026 Bryan Wong
module;

#define DPL_MODULES 1
#include "dpl/config.h"

export module dpl:core.numbers;
import :core.details.numbers;
import :std.concepts;
import :std.bit;
import :std.type_traits;
import :std.utility;

// IWYU pragma: begin_exports
#include "dpl/core/numbers/ext.h"
#include "dpl/core/numbers/floating_point_like.h"
#include "dpl/core/numbers/floating_point_traits.h"
// IWYU pragma: end_exports
