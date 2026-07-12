// Copyright 2026 Bryan Wong
module;

#define DPL_MODULES 1
#include "dpl/config.h"

export module dpl:core.details.math;
export import :core.fwd;
import :core.details.type_traits;
import :std.concepts;
import :std.bit;
import :std.type_traits;
import :std.utility;
import :core.basic;
import :core.concepts;
import :core.immediate;
import :core.numbers;
import :core.operations;
import :core.type_traits;
import :core.utility;

DPL_DISABLE_WARNING_PUSH()
#if DPL_COMPILER_MSVC
DPL_DISABLE_WARNING(5244)
#endif

// IWYU pragma: begin_exports
#include "dpl/core/math/details/fwd.h"

#include "dpl/core/math/details/accuracy.h"
#include "dpl/core/math/details/compliance.h"
#include "dpl/core/math/details/constants.h"
#include "dpl/core/math/details/floating_point_simd.h"
#include "dpl/core/math/details/fpfix.h"
#include "dpl/core/math/details/gather.h"
#include "dpl/core/math/details/ilogb.h"
#include "dpl/core/math/details/ldexp.h"
#include "dpl/core/math/details/pair.h"
#include "dpl/core/math/details/polynomial.h"
#include "dpl/core/math/details/rempi_table.h"
#include "dpl/core/math/details/rounding.h"
#include "dpl/core/math/details/rsqrt2.h"
// IWYU pragma: end_exports

DPL_DISABLE_WARNING_POP()
