// Copyright 2025 Bryan Wong
#pragma once

#include "dpl/config.h"

#if DPL_SIMD_X86_SSE2

// IWYU pragma: begin_exports
#  include "dpl/xmm/operations/element_cast/fallback.h"
#  include "dpl/xmm/operations/element_cast/to_bfloat16.h"
#  include "dpl/xmm/operations/element_cast/to_double.h"
#  include "dpl/xmm/operations/element_cast/to_float.h"
#  include "dpl/xmm/operations/element_cast/to_float16.h"
#  include "dpl/xmm/operations/element_cast/to_int16.h"
#  include "dpl/xmm/operations/element_cast/to_int32.h"
#  include "dpl/xmm/operations/element_cast/to_int64.h"
#  include "dpl/xmm/operations/element_cast/to_int8.h"
#  include "dpl/xmm/operations/element_cast/to_uint16.h"
#  include "dpl/xmm/operations/element_cast/to_uint32.h"
#  include "dpl/xmm/operations/element_cast/to_uint64.h"
#  include "dpl/xmm/operations/element_cast/to_uint8.h"
// IWYU pragma: end_exports

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {

template <integral To, imask_t<To> M, floating_point_like From>
requires (sizeof(From) >= sizeof(float) && sizeof(To) < sizeof(float))
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<To> element_cast(
    src_vector_t<To> src, mask<To> mask, vector<From> val) noexcept
requires requires {
    xmm::element_cast<To>(src, mask, xmm::element_cast<int>(val));
}
{
    return xmm::element_cast<To>(src, mask, xmm::element_cast<int>(val));
}

template <integral To, imask_t<To> M, floating_point_like From>
requires (sizeof(From) >= sizeof(float) && sizeof(To) < sizeof(float))
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<To> element_cast(
    dx::zero_t zero, cmask_t<To, M> mask, vector<From> val) noexcept
requires requires {
    xmm::element_cast<To>(zero, mask, xmm::element_cast<int>(val));
}
{
    return xmm::element_cast<To>(zero, mask, xmm::element_cast<int>(val));
}

template <integral To, imask_t<To> M, floating_point_like From>
requires (sizeof(From) == sizeof(int16) && sizeof(To) < sizeof(int16))
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<To> element_cast(
    src_vector_t<To> src, cmask_t<To, M> mask, vector<From> val) noexcept
requires requires {
    xmm::element_cast<To>(src, mask, xmm::element_cast<int16>(val));
}
{
    return xmm::element_cast<To>(src, mask, xmm::element_cast<int16>(val));
}

template <integral To, imask_t<To> M, floating_point_like From>
requires (sizeof(From) == sizeof(int16) && sizeof(To) < sizeof(int16))
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<To> element_cast(
    dx::zero_t zero, cmask_t<To, M> mask, vector<From> val) noexcept
requires requires {
    xmm::element_cast<To>(zero, mask, xmm::element_cast<int16>(val));
}
{
    return xmm::element_cast<To>(zero, mask, xmm::element_cast<int16>(val));
}
template <signed_integral To, imask_t<To> M, unsigned_integral From>
requires (sizeof(From) > sizeof(To))
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<To> element_cast(
    src_vector_t<To> src, cmask_t<To, M> mask, vector<From> val) noexcept
requires requires {
    xmm::element_cast<To>(
        src, mask, xmm::element_cast<make_signed_t<From>>(val));
}
{
    return xmm::element_cast<To>(
        src, mask, xmm::element_cast<make_signed_t<From>>(val));
}

template <signed_integral To, imask_t<To> M, unsigned_integral From>
requires (sizeof(From) > sizeof(To))
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<To> element_cast(
    dx::zero_t zero, cmask_t<To, M> mask, vector<From> val) noexcept
requires requires {
    xmm::element_cast<To>(
        zero, mask, xmm::element_cast<make_signed_t<From>>(val));
}
{
    return xmm::element_cast<To>(
        zero, mask, xmm::element_cast<make_signed_t<From>>(val));
}

template <unsigned_integral To, imask_t<To> M, signed_integral From>
requires (sizeof(From) > sizeof(To))
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<To> element_cast(
    src_vector_t<To> src, cmask_t<To, M> mask, vector<From> val) noexcept
requires requires { xmm::element_cast<make_signed_t<To>>(+src, mask, val); }
{
    return +xmm::element_cast<make_signed_t<To>>(+src, mask, val);
}

template <unsigned_integral To, imask_t<To> M, signed_integral From>
requires (sizeof(From) > sizeof(To))
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<To> element_cast(
    dx::zero_t zero, cmask_t<To, M> mask, vector<From> val) noexcept
requires requires { xmm::element_cast<make_signed_t<To>>(zero, mask, val); }
{
    return +xmm::element_cast<make_signed_t<To>>(zero, mask, val);
}

template <unsigned_integral To, imask_t<To> M, integral From>
requires (sizeof(From) < sizeof(To))
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<To> element_cast(
    src_vector_t<To> src, cmask_t<To, M> mask, vector<From> val) noexcept
requires requires { xmm::element_cast<make_signed_t<To>>(+src, mask, val); }
{
    return +xmm::element_cast<make_signed_t<To>>(+src, mask, val);
}

template <unsigned_integral To, imask_t<To> M, integral From>
requires (sizeof(From) < sizeof(To))
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<To> element_cast(
    dx::zero_t zero, cmask_t<To, M> mask, vector<From> val) noexcept
requires requires { xmm::element_cast<make_signed_t<To>>(zero, mask, val); }
{
    return +xmm::element_cast<make_signed_t<To>>(zero, mask, val);
}

template <simd_element To, simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<To> element_cast(abi_tag, vector<E> arg) noexcept
requires requires { xmm::element_cast<To>(arg); }
{
    return xmm::element_cast<To>(arg);
}

template <simd_element To, imask_t<To> M, simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<To> element_cast(
    abi_tag, src_vector_t<To> src, cmask_t<To, M> mask, vector<E> arg) noexcept
requires requires { xmm::element_cast<To>(src, mask, arg); }
{
    return xmm::element_cast<To>(src, mask, arg);
}

template <simd_element To, imask_t<To> M, simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<To> element_cast(
    abi_tag, dx::zero_t zero, cmask_t<To, M> mask, vector<E> arg) noexcept
requires requires { xmm::element_cast<To>(zero, mask, arg); }
{
    return xmm::element_cast<To>(zero, mask, arg);
}

} // namespace datapar::xmm

__DPL_DEFAULT_NAMESPACE_END

#endif
