// Copyright 2026 Bryan Wong
module;

#define DPL_MODULES 1
#include "dpl/config.h"

export module dpl:core.constants;
export import :core.fwd;
import :std.concepts;
import :std.bit;
import :std.type_traits;
import :std.utility;
import :core.basic;
import :core.concepts;
import :core.type_traits;

// IWYU pragma: begin_exports
#include "dpl/core/constants/all_bits.h"
#include "dpl/core/constants/digits.h"
#include "dpl/core/constants/epsilon.h"
#include "dpl/core/constants/exponent_bits.h"
#include "dpl/core/constants/infinity.h"
#include "dpl/core/constants/ln2.h"
#include "dpl/core/constants/lsb.h"
#include "dpl/core/constants/mantissa_bits.h"
#include "dpl/core/constants/max_value.h"
#include "dpl/core/constants/min_value.h"
#include "dpl/core/constants/msb.h"
#include "dpl/core/constants/nan.h"
#include "dpl/core/constants/one.h"
#include "dpl/core/constants/rcp_pi.h"
#include "dpl/core/constants/value_bits.h"
#include "dpl/core/constants/zero.h"
// IWYU pragma: end_exports
