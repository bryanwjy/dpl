// Copyright 2025 Bryan Wong
#pragma once

#include "dpl/config.h"

#if DPL_SIMD_X86_SSE4_2

#  if !DPL_MODULES
#    include "dpl/core/numbers/floating_point_like.h"
#    include "dpl/xmm/basic/abi.h"

#    include <immintrin.h>
#  endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {

template <simd_element To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<To> element_cast(type_identity_t<vector<To>> val) noexcept {
    return val;
}

template <signed_integral To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<To> element_cast(vector<make_unsigned_t<To>> val) noexcept {
    return +val;
}

template <unsigned_integral To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<To> element_cast(vector<make_signed_t<To>> val) noexcept {
    return +val;
}

#  define __DPL_CAST_FWD(FROM, TO)                     \
      template <same_as<TO> To>                        \
      DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD) \
      inline vector<To> DPL_VECTORCALL element_cast(vector<FROM>) noexcept

__DPL_CAST_FWD(int8, int16);
__DPL_CAST_FWD(int8, int32);
__DPL_CAST_FWD(int8, int64);
__DPL_CAST_FWD(uint8, int16);
__DPL_CAST_FWD(uint8, int32);
__DPL_CAST_FWD(uint8, int64);
__DPL_CAST_FWD(int16, int32);
__DPL_CAST_FWD(int16, int64);
__DPL_CAST_FWD(uint16, int32);
__DPL_CAST_FWD(uint16, int64);
__DPL_CAST_FWD(int32, int64);
__DPL_CAST_FWD(uint32, int64);
//
__DPL_CAST_FWD(int64, int32);
__DPL_CAST_FWD(int64, int16);
__DPL_CAST_FWD(int64, int8);
//
__DPL_CAST_FWD(int32, int16);
__DPL_CAST_FWD(int32, int8);
//
__DPL_CAST_FWD(int16, int8);
#  undef __DPL_CAST_FWD

template <integral To, floating_point_like From>
requires (sizeof(From) >= sizeof(float) && sizeof(To) < sizeof(float))
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<To> element_cast(vector<From> val) noexcept
requires requires { xmm::element_cast<int>(val); }
{
    return xmm::element_cast<To>(xmm::element_cast<int>(val));
}

template <integral To, floating_point_like From>
requires (sizeof(From) == sizeof(int16) && sizeof(To) < sizeof(int16))
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<To> element_cast(vector<From> val) noexcept
requires requires { xmm::element_cast<int16>(val); }
{
    return xmm::element_cast<To>(xmm::element_cast<int16>(val));
}

// Truncating unsigned-to-unsigned conversions
template <unsigned_integral To, unsigned_integral From>
requires (sizeof(From) > sizeof(To))
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<To> element_cast(vector<From> val) noexcept
requires requires {
    xmm::element_cast<make_signed_t<To>>(vector<make_signed_t<From>>(+val));
}
{
    return +xmm::element_cast<make_signed_t<To>>(
        vector<make_signed_t<From>>(+val));
}

// Truncating unsigned-to-signed conversions
template <signed_integral To, unsigned_integral From>
requires (sizeof(From) > sizeof(To)) &&
    requires(
        vector<make_signed_t<From>> inter) { xmm::element_cast<To>(inter); }
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<To> element_cast(vector<From> val) noexcept {
    return xmm::element_cast<To>(vector<make_signed_t<From>>(+val));
}

// Truncating signed-to-unsigned conversions
template <unsigned_integral To, signed_integral From>
requires (sizeof(From) > sizeof(To))
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<To> element_cast(vector<From> val) noexcept
requires requires { xmm::element_cast<make_signed_t<To>>(val); }
{
    return +xmm::element_cast<make_signed_t<To>>(val);
}

// Extending to unsigned conversions
template <unsigned_integral To, integral From>
requires (sizeof(From) < sizeof(To))
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<To> element_cast(vector<From> val) noexcept
requires requires { xmm::element_cast<make_signed_t<To>>(val); }
{
    return +xmm::element_cast<make_signed_t<To>>(val);
}

} // namespace datapar::xmm

__DPL_DEFAULT_NAMESPACE_END

#endif
