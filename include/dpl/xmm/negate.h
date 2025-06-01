// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h" // IWYU pragma: keep

// IWYU pragma: begin_exports
#include "dpl/xmm/fwd.h"

#include "dpl/xmm/native.h"
#include "dpl/xmm/scalar.h"
#include "dpl/xmm/vector.h"
#include "dpl/xmm/vector_mask.h"
// IWYU pragma: end_exports

#include "dpl/xmm/concepts.h"
#include "dpl/xmm/constants.h"
#include "dpl/xmm/immediate_mask.h"

#if !DPL_MODULES
#  include <immintrin.h>

#  include <concepts>
#  include <type_traits>
#endif

DPL_NAMESPACE_BEGIN

DPL_SIMD_ABI_NAMESPACE_BEGIN(xmm)

DPL_EXPORT template <xmm_element T>
requires std::signed_integral<T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) inline vector<T> sign(
    vector<T> val, std::type_identity_t<vector<T>> ctrl) noexcept {
    if constexpr (sizeof(T) == 1) {
        return _mm_sign_epi8(val, +ctrl);
    } else if constexpr (sizeof(T) == 2) {
        return _mm_sign_epi16(val, +ctrl);
    } else if constexpr (sizeof(T) == 4) {
        return _mm_sign_epi32(val, +ctrl);
    } else {
        static_assert(sizeof(T) == 4, "unsupported operation");
        DPL_BUILTIN_unreachable();
    }
}

namespace details::negate {

DPL_EXPORT template <xmm_element T, bit_type_for<T> N>
requires std::signed_integral<T>
class negation {
    using element_type = T;
    using native_type = native_type_t<T>;
    using mask_type = mask_for<T>;
    using vector_type = vector<T>;

public:
    explicit inline negation(vector<T> val) noexcept : value_{val} {}

private:
    __DPL_HIDE_FROM_ABI static consteval auto sign_mask() noexcept {
        return bitset_for<T>{N} * all<T> | ~bitset_for<T>{N} * lsb<T>;
    }

    DPL_ATTRIBUTES(
            _HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) friend inline vector_type
    evaluate(negation self) noexcept {
        if constexpr (!N) {
            return self.value_;
        } else {
            static constexpr auto mask = +sign_mask();
            return sign(self.value_, reinterpret_as<T>(vector{mask}));
        }
    }

    template <bit_type_for<T> S>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) friend inline vector_type
    select(mask_for<T, S>, negation self) noexcept {
        static constexpr auto mask = +(sign_mask() & bitset_for<T>{S} * all<T>);

        return sign(self.value_, reinterpret_as<T>(vector{mask}));
    }

    vector_type value_;
};

} // namespace details::negate

DPL_EXPORT template <std::integral auto V, xmm_element T>
requires requires { typename mask_for<T, V>; } &&
    (std::floating_point<T> || std::signed_intergral<T>)
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) inline auto negate(
    vector<T> val, mask_for<T, V> = {}) noexcept {
    if constexpr (std::signed_intergral<T>) {
        return details::negate::negation<T, V>{val};
    } else {
        using bitset_type = bitset_for<T>;
        static constexpr auto signs = +(bitset_type{V} * msb<T>);
        return reinterpret_as<T>(vector{signs}) ^ val;
    }
}

DPL_EXPORT template <xmm_element T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) inline vector<T>
vector<T>::operator-(this vector<T> self) noexcept {
    return negate(self.val, mask(all<T>));
}

DPL_EXPORT template <xmm_element T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) inline scalar<T>
scalar<T>::operator-(this scalar<T> self) noexcept {
    return scalar{negate<1>(vector{self})};
}

DPL_SIMD_ABI_NAMESPACE_END(xmm)

DPL_NAMESPACE_END