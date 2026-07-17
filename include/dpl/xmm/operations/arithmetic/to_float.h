// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/xmm/basic/abi.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::xmm {
namespace details {
template <typename>
struct convert_t;

template <typename From, typename To>
concept convert_to = simd_element<From> && simd_element<To> &&
    requires(convert_t<To> cvt, vector<From> src) { cvt(src); };
} // namespace details

template <simd_element To, details::convert_to<To> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<To> element_cast(vector<E> src) noexcept;

DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<float> to_float(vector<ext::bfloat16> arg) noexcept;
} // namespace datapar::xmm
__DPL_DEFAULT_NAMESPACE_END
