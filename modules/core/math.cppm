
// Copyright 2026 Bryan Wong
module;

#define DPL_MODULES 1
#include "dpl/config.h"

export module dpl:core.math;
import :core.details.math;

// NOLINTBEGIN(misc-unused-using-decls,misc-unused-alias-decls)
__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {
namespace rounding {
using __DPL datapar::rounding::current_dir;
using __DPL datapar::rounding::current_dir_t;
using __DPL datapar::rounding::no_exc;
using __DPL datapar::rounding::no_exc_t;
using __DPL datapar::rounding::to_nearest_int;
using __DPL datapar::rounding::to_nearest_int_t;
using __DPL datapar::rounding::to_neg_inf;
using __DPL datapar::rounding::to_neg_inf_t;
using __DPL datapar::rounding::to_pos_inf;
using __DPL datapar::rounding::to_pos_inf_t;
using __DPL datapar::rounding::to_zero;
using __DPL datapar::rounding::to_zero_t;
} // namespace rounding

namespace fpfix {
using __DPL datapar::fpfix::all;
using __DPL datapar::fpfix::finite;
using __DPL datapar::fpfix::infinity;
using __DPL datapar::fpfix::nan;
using __DPL datapar::fpfix::neg_inf;
using __DPL datapar::fpfix::negative;
using __DPL datapar::fpfix::none;
using __DPL datapar::fpfix::one;
using __DPL datapar::fpfix::pos_inf;
using __DPL datapar::fpfix::positive;
using __DPL datapar::fpfix::qnan;
using __DPL datapar::fpfix::revert;
using __DPL datapar::fpfix::signed_inf;
using __DPL datapar::fpfix::snan;
using __DPL datapar::fpfix::zero;
//
using __DPL datapar::fpfix::condition_set;
using __DPL datapar::fpfix::condition_set_for;
using __DPL datapar::fpfix::fpclass_type;
using __DPL datapar::fpfix::popcount;
using __DPL datapar::fpfix::result_subset_of;
//
using __DPL datapar::fpfix::condition;
} // namespace fpfix

inline namespace cpo {
using __DPL datapar::cpo::fmacc;
using __DPL datapar::cpo::fmadd;
using __DPL datapar::cpo::fmaddsac;
using __DPL datapar::cpo::fmaddsub;
using __DPL datapar::cpo::fmsac;
using __DPL datapar::cpo::fmsub;
using __DPL datapar::cpo::fmsubacc;
using __DPL datapar::cpo::fmsubadd;
using __DPL datapar::cpo::fnmacc;
using __DPL datapar::cpo::fnmadd;
using __DPL datapar::cpo::fnmsac;
using __DPL datapar::cpo::fnmsub;
//
using __DPL datapar::cpo::mulacc;
using __DPL datapar::cpo::muladd;
using __DPL datapar::cpo::muladdsac;
using __DPL datapar::cpo::muladdsub;
using __DPL datapar::cpo::mulsac;
using __DPL datapar::cpo::mulsub;
using __DPL datapar::cpo::mulsubacc;
using __DPL datapar::cpo::mulsubadd;
using __DPL datapar::cpo::nmulacc;
using __DPL datapar::cpo::nmuladd;
using __DPL datapar::cpo::nmulsac;
using __DPL datapar::cpo::nmulsub;
//
using __DPL datapar::cpo::ceil;
using __DPL datapar::cpo::floor;
using __DPL datapar::cpo::round;
using __DPL datapar::cpo::trunc;
//
using __DPL datapar::cpo::copysign;
using __DPL datapar::cpo::sign;
using __DPL datapar::cpo::signbit;
//
using __DPL datapar::cpo::dot_product;
//
// TODO: cleanup and export frexp
using __DPL datapar::cpo::exp;
using __DPL datapar::cpo::exp2;
using __DPL datapar::cpo::fixup;
using __DPL datapar::cpo::ldexp;
using __DPL datapar::cpo::log2;
using __DPL datapar::cpo::pow;
using __DPL datapar::cpo::rcp;
using __DPL datapar::cpo::rsqrt;
using __DPL datapar::cpo::sqrt;
//

using __DPL datapar::cpo::isfinite;
using __DPL datapar::cpo::isinf;
using __DPL datapar::cpo::isnan;
using __DPL datapar::cpo::isnanq;
using __DPL datapar::cpo::isnans;
using __DPL datapar::cpo::isnormal;
//
using __DPL datapar::cpo::cos;
using __DPL datapar::cpo::cosh;
using __DPL datapar::cpo::sin;
using __DPL datapar::cpo::sincos;
using __DPL datapar::cpo::sincosi;
using __DPL datapar::cpo::sinh;
using __DPL datapar::cpo::sinhcosh;
using __DPL datapar::cpo::sinhcoshi;

} // namespace cpo
} // namespace datapar

namespace rounding = __DPL datapar::rounding;
namespace fpfix = __DPL datapar::fpfix;
__DPL_DEFAULT_NAMESPACE_END
// NOLINTEND(misc-unused-using-decls,misc-unused-alias-decls)
