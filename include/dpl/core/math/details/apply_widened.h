// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/details/fwd.h" // IWYU pragma: export

#if !DPL_MODULES
#  include "dpl/core/algorithm/rotate.h"
#  include "dpl/core/algorithm/slide.h"
#  include "dpl/core/concepts/canonical.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/numbers/ext.h"
#  include "dpl/core/operations/element_cast.h"
#  include "dpl/core/type_traits/simd_abi_traits.h"
#  include "dpl/std/type_traits/type_identity.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::fmath {
namespace dx = __DPL datapar;

template <typename E, canonical_vector T, typename F>
requires (sizeof(E) / sizeof(simd_element_type_t<T>) == 2) &&
    simd_element_for<E, simd_abi_type_t<T>>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr auto apply_widened(F&& func, T val) noexcept {
    auto const shift = [] {
        if constexpr (fixed_width_abi<simd_abi_type_t<T>>) {
            return dx::imm<dx::simd_abi_traits<T>::size() / 2>;
        } else {
            return dx::simd_abi_traits<T>::size() / 2;
        }
    }();
    auto const lo = dx::element_cast<E>(val);
    auto const hi = dx::element_cast<E>(dx::shift_left(val, shift));
    auto const r_lo = dx::element_cast<simd_element_type_t<T>>(func(lo));
    auto const r_hi = dx::element_cast<simd_element_type_t<T>>(func(hi));
    return dx::slide_left(dx::rotate_right(r_lo, shift), r_hi, shift);
}

template <typename E, canonical_vector T, same_as<T> U, typename F>
requires (sizeof(E) / sizeof(simd_element_type_t<T>) == 2) &&
    simd_element_for<E, simd_abi_type_t<T>>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr auto apply_widened(F&& func, T lhs, U rhs) noexcept {
    auto const shift = [] {
        if constexpr (fixed_width_abi<simd_abi_type_t<T>>) {
            return dx::imm<dx::simd_abi_traits<T>::size() / 2>;
        } else {
            return dx::simd_abi_traits<T>::size() / 2;
        }
    }();
    auto left = dx::element_cast<E>(lhs);
    auto right = dx::element_cast<E>(rhs);
    auto const lo = dx::element_cast<simd_element_type_t<T>>(func(left, right));

    left = dx::element_cast<E>(dx::shift_left(lhs, shift));
    right = dx::element_cast<E>(dx::shift_left(rhs, shift));

    auto const hi = dx::element_cast<simd_element_type_t<T>>(func(left, right));
    return dx::slide_left(dx::rotate_right(lo, shift), hi, shift);
}

} // namespace datapar::fmath
__DPL_DEFAULT_NAMESPACE_END
