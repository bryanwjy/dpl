// Copyright 2026 Bryan Wong
module;

#define DPL_MODULES 1
#include "dpl/config.h"

export module dpl:core.details.math;
export import :core.fwd;
import :core.type_traits;
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
#include "dpl/core/math/addsub.h"
#include "dpl/core/math/ceil.h"
#include "dpl/core/math/cmp.h"
#include "dpl/core/math/copysign.h"
#include "dpl/core/math/dot_product.h"
#include "dpl/core/math/exp.h"
#include "dpl/core/math/exp2.h"
#include "dpl/core/math/fixup.h"
#include "dpl/core/math/floor.h"
#include "dpl/core/math/fma.h"
#include "dpl/core/math/fpfix.h"
#include "dpl/core/math/inner_product.h"
#include "dpl/core/math/isfinite.h"
#include "dpl/core/math/isinf.h"
#include "dpl/core/math/isnan.h"
#include "dpl/core/math/isnanq.h"
#include "dpl/core/math/isnans.h"
#include "dpl/core/math/isnormal.h"
#include "dpl/core/math/ldexp.h"
#include "dpl/core/math/lerp.h"
#include "dpl/core/math/log2.h"
#include "dpl/core/math/mulx.h"
#include "dpl/core/math/pow.h"
#include "dpl/core/math/rcp.h"
#include "dpl/core/math/round.h"
#include "dpl/core/math/rounding.h"
#include "dpl/core/math/rsqrt.h"
#include "dpl/core/math/sign.h"
#include "dpl/core/math/signbit.h"
#include "dpl/core/math/sincos.h"
#include "dpl/core/math/sinhcosh.h"
#include "dpl/core/math/sqrt.h"
#include "dpl/core/math/trunc.h"
// IWYU pragma: end_exports

DPL_DISABLE_WARNING_POP()
