// Copyright 2026 Bryan Wong
module;

#define DPL_MODULES 1
#include "dpl/config.h"

export module dpl:core.details.type_traits;
export import :core.fwd;
export import :core.type_interface;
import :std.concepts;
import :std.bit;
import :std.type_traits;
import :std.utility;

// IWYU pragma: begin_exports
#include "dpl/core/details/type_traits.h"
// IWYU pragma: end_exports
