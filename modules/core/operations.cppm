// Copyright 2025 Bryan Wong
module;

#define DPL_MODULES 1
#include "dpl/config.h"

export module dpl:core.operations;
export import :core.fwd;
import :core.details.operations;

__DPL_DEFAULT_NAMESPACE_BEGIN
// NOLINTBEGIN(misc-unused-using-decls)
namespace datapar {
using __DPL datapar::define_promotion;
using __DPL datapar::demote_abi;
using __DPL datapar::demote_abi_t;
using __DPL datapar::promote_abi;
using __DPL datapar::promote_abi_t;
inline namespace cpo {
using __DPL datapar::cpo::abs;
using __DPL datapar::cpo::add;
using __DPL datapar::cpo::addsub;
using __DPL datapar::cpo::divide;
using __DPL datapar::cpo::fmadd;
using __DPL datapar::cpo::fmaddsub;
using __DPL datapar::cpo::fmsub;
using __DPL datapar::cpo::fmsubadd;
using __DPL datapar::cpo::fnmadd;
using __DPL datapar::cpo::fnmsub;
using __DPL datapar::cpo::multiply;
using __DPL datapar::cpo::negate;
using __DPL datapar::cpo::subadd;
using __DPL datapar::cpo::subtract;
//
using __DPL datapar::cpo::bwand;
using __DPL datapar::cpo::bwandnot;
using __DPL datapar::cpo::bwnot;
using __DPL datapar::cpo::bwor;
using __DPL datapar::cpo::bwornot;
using __DPL datapar::cpo::bwshift_left;
using __DPL datapar::cpo::bwshift_lefti;
using __DPL datapar::cpo::bwshift_right;
using __DPL datapar::cpo::bwshift_righti;
using __DPL datapar::cpo::bwxor;
//
using __DPL datapar::cpo::byteswap;
using __DPL datapar::cpo::countl_one;
using __DPL datapar::cpo::countl_zero;
using __DPL datapar::cpo::countr_one;
using __DPL datapar::cpo::countr_zero;
using __DPL datapar::cpo::popcount;
using __DPL datapar::cpo::rotl;
using __DPL datapar::cpo::rotli;
using __DPL datapar::cpo::rotr;
using __DPL datapar::cpo::rotri;
//
using __DPL datapar::cpo::abi_cast;
using __DPL datapar::cpo::element_cast;
using __DPL datapar::cpo::simd_cast;
//
using __DPL datapar::cpo::cmpeq;
using __DPL datapar::cpo::cmpge;
using __DPL datapar::cpo::cmpgt;
using __DPL datapar::cpo::cmple;
using __DPL datapar::cpo::cmplt;
using __DPL datapar::cpo::cmpneq;
using __DPL datapar::cpo::max;
using __DPL datapar::cpo::min;
//
using __DPL datapar::cpo::all_of;
using __DPL datapar::cpo::any_of;
using __DPL datapar::cpo::logical_and;
using __DPL datapar::cpo::logical_not;
using __DPL datapar::cpo::logical_or;
using __DPL datapar::cpo::none_of;
using __DPL datapar::cpo::some_of;
//
using __DPL datapar::cpo::broadcast_lane;
using __DPL datapar::cpo::broadcast_lanei;
using __DPL datapar::cpo::concat;
using __DPL datapar::cpo::permute;
using __DPL datapar::cpo::permutei;
//
using __DPL datapar::cpo::reinterpret;
using __DPL datapar::cpo::select;
using __DPL datapar::cpo::selecti;
using __DPL datapar::cpo::split;
using __DPL datapar::cpo::to_simd_mask;
} // namespace cpo

using __DPL datapar::assume_normalized_mask;
using __DPL datapar::assume_normalized_mask_t;
using __DPL datapar::split_result;

using __DPL datapar::arithmetic_vector_interface;
inline namespace operators {
using __DPL datapar::operators::operator+;
using __DPL datapar::operators::operator-;
using __DPL datapar::operators::operator*;
using __DPL datapar::operators::operator/;
using __DPL datapar::operators::operator+=;
using __DPL datapar::operators::operator-=;
using __DPL datapar::operators::operator*=;
using __DPL datapar::operators::operator/=;
} // namespace operators

using __DPL datapar::logical_simd_interface;
inline namespace operators {
using __DPL datapar::operators::operator||;
using __DPL datapar::operators::operator&&;
using __DPL datapar::operators::operator!;
} // namespace operators

using __DPL datapar::bitwise_simd_interface;
inline namespace operators {
using __DPL datapar::operators::operator|;
using __DPL datapar::operators::operator&;
using __DPL datapar::operators::operator^;
using __DPL datapar::operators::operator<<;
using __DPL datapar::operators::operator>>;
using __DPL datapar::operators::operator|=;
using __DPL datapar::operators::operator&=;
using __DPL datapar::operators::operator^=;
using __DPL datapar::operators::operator<<=;
using __DPL datapar::operators::operator>>=;
using __DPL datapar::operators::operator~;
} // namespace operators

using __DPL datapar::comparison_simd_interface;
inline namespace operators {
using __DPL datapar::operators::operator==;
using __DPL datapar::operators::operator!=;
using __DPL datapar::operators::operator<;
using __DPL datapar::operators::operator>;
using __DPL datapar::operators::operator<=;
using __DPL datapar::operators::operator>=;
} // namespace operators
} // namespace datapar
// NOLINTEND(misc-unused-using-decls)
__DPL_DEFAULT_NAMESPACE_END
