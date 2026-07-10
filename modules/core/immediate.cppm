// Copyright 2026 Bryan Wong
module;

#define DPL_MODULES 1
#include "dpl/config.h"

export module dpl:core.immediate;
export import :core.fwd;
import :core.details.type_traits;
import :std.concepts;
import :std.bit;
import :std.type_traits;
import :std.utility;
import :core.concepts;
import :core.type_traits;

// IWYU pragma: begin_exports
#include "dpl/core/immediate/broadcastable_base.h"
#include "dpl/core/immediate/const_mask.h"
#include "dpl/core/immediate/constants/all_bits.h"
#include "dpl/core/immediate/constants/digits.h"
#include "dpl/core/immediate/constants/epsilon.h"
#include "dpl/core/immediate/constants/exponent_bias.h"
#include "dpl/core/immediate/constants/exponent_bits.h"
#include "dpl/core/immediate/constants/exponent_mask.h"
#include "dpl/core/immediate/constants/infinity.h"
#include "dpl/core/immediate/constants/inv_pi.h"
#include "dpl/core/immediate/constants/ln2.h"
#include "dpl/core/immediate/constants/lsb.h"
#include "dpl/core/immediate/constants/mantissa_bits.h"
#include "dpl/core/immediate/constants/mantissa_width.h"
#include "dpl/core/immediate/constants/max_value.h"
#include "dpl/core/immediate/constants/min_value.h"
#include "dpl/core/immediate/constants/msb.h"
#include "dpl/core/immediate/constants/nan.h"
#include "dpl/core/immediate/constants/one.h"
#include "dpl/core/immediate/constants/value_bits.h"
#include "dpl/core/immediate/constants/zero.h"
#include "dpl/core/immediate/immediate.h"
// IWYU pragma: end_exports
