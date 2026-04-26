// Copyright 2026 Bryan Wong
module;

#define DPL_MODULES 1
#include "dpl/config.h"

export module dpl:core.utility;
export import :core.fwd;
import :std.concepts;
import :std.bit;
import :std.type_traits;
import :std.utility;
import :core.basic;
import :core.concepts;
import :core.constants;
import :core.operations;
import :core.type_traits;

// IWYU pragma: begin_exports
#include "dpl/core/utility/fpfix.h"
#include "dpl/core/utility/rounding.h"
#include "dpl/core/utility/to_basic_type.h"
#include "dpl/core/utility/to_signed.h"
#include "dpl/core/utility/to_underlying.h"
#include "dpl/core/utility/to_unsigned.h"
// IWYU pragma: end_exports
