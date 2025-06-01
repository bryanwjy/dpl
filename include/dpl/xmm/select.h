// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h" // IWYU pragma: keep

#include "dpl/xmm/concepts.h"

// IWYU pragma: begin_exports
#include "dpl/xmm/fwd.h"

#include "dpl/xmm/constants.h"
#include "dpl/xmm/native.h"
#include "dpl/xmm/vector.h"
#include "dpl/xmm/vector_mask.h"
// IWYU pragma: end_exports

#if !DPL_MODULES
#  include <immintrin.h>

#  include <climits>
#  include <concepts>
#  include <type_traits>
#endif

DPL_NAMESPACE_BEGIN

DPL_SIMD_ABI_NAMESPACE_BEGIN(xmm)

namespace details::select {

template <xmm_element T, bit_type_for<T> V>
class selector_t {
public:
    using element_type = T;
    using native_type = native_type_t<T>;
    using mask_type = mask_for<T>;
    using vector_type = vector<T>;

    __DPL_HIDE_FROM_ABI explicit consteval selector_t() noexcept = default;

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) static inline vector_type
    operator()(vector_type other) noexcept {
        return self(other, +zero<T>);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, VECTORCALL, NODISCARD) static inline vector_type
    operator()(vector_type if_false, vector_type if_true) noexcept {
        if constexpr (sizeof(T) == 16) {
            return V == 1 ? if_true : if_false;
        } else if constexpr (sizeof(T) == 8) {
            return native_cast<T, double>(
                _mm_blend_pd(native_cast<double, T>(+if_false),
                    native_cast<double, T>(+if_true), V));
        } else if constexpr (sizeof(T) == 4) {
            return _mm_blend_epi32(+if_false, +if_true, V);
        } else if constexpr (sizeof(T) == 2) {
            return _mm_blend_epi16(+if_false, +if_true, V);
        } else {
            static constexpr auto set =
                +bitset<xmm_abi::size>{V}.template repeat<CHAR_BIT>();
            auto const mask =
                _mm_load_si128(reinterpret_cast<__m128i const*>(&set));
            return _mm_blendv_epi8(+if_false, +if_true, mask);
        }
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) static inline vector_type
    operator()(vector_type if_false, vector_type if_true) noexcept
    requires (sizeof(T) == 1)
    {
        static constexpr auto set =
            +bitset<xmm_abi::size>{V}.template repeat<CHAR_BIT>();
        auto const mask =
            _mm_load_si128(reinterpret_cast<__m128i const*>(&set));
        return _mm_blendv_epi8(+if_false, +if_true, mask);
    }

#if DPL_SIMD_X86_AVX512FP16 & DPL_SUPPORTS_FLOAT16
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, VECTORCALL, NODISCARD) static vector_type
    operator()(vector_type if_false, vector_type if_true) noexcept
    requires std::same_as<float16, T>
    {
        return _mm_mask_blend_ph(static_cast<__mmask8>(V), +if_false, +if_true);
    }
#endif
};

} // namespace details::select

DPL_EXPORT_BEGIN

template <std::integral auto V, xmm_element T>
requires requires { typename mask_for<T, V>; } &&
    requires(
        details::select::selector_t<T, V> impl, vector<T> val) { impl(val); }
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) inline auto select(
    vector<T> val, mask_for<T, V> = {}) noexcept {
    static constexpr selector_t<T, V> impl{};
    return impl(val);
}

template <std::integral auto V, xmm_element T>
requires requires { typename mask_for<T, V>; } &&
    requires(details::select::selector_t<T, V> impl, vector<T> if_false,
        vector<T> if_true) { impl(if_false, if_true); }
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) inline auto select(
    vector<T> if_false, vector<T> if_true, mask_for<T, V> = {}) noexcept {
    static constexpr selector_t<T, V> impl{};
    return impl(if_false, if_true);
}

template <xmm_element T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, VECTORCALL, NODISCARD) inline vector<T> select(
    vector<T> if_false, vector<T> if_true,
    std::type_identity_t<vector_mask<T>> selector) noexcept {
    if constexpr (sized_floating_point<T, 4>) {
        return _mm_blendv_ps(+if_false, +if_true, +selector);
    } else if constexpr (sized_floating_point<T, 8>) {
        return _mm_blendv_pd(+if_false, +if_true, +selector);
    } else if (sizeof(T) == 8) {
        return native_cast<T, double>(_mm_blendv_pd(
            native_cast<double, T>(+if_false), native_cast<double, T>(+if_true),
            native_cast<double, T>(+selector)));
    } else if (sizeof(T) == 4) {
        return native_cast<T, float>(_mm_blendv_ps(
            native_cast<float, T>(+if_false), native_cast<float, T>(+if_true),
            native_cast<float, T>(+selector)));
    } else {
        // Users should manually reinterpret to 1 byte data, and select from
        // there
        static_assert(sizeof(T) != 2, "unsupported operation");
        return _mm_blendv_epi8(+if_false, +if_true, +selector);
    }
}

template <xmm_element T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline vector<T>
vector<T>::operator[](this vector<T> self, vector_mask<T> where) noexcept {
    return select(zero<element_type>, self, where);
}

DPL_EXPORT_END

DPL_SIMD_ABI_NAMESPACE_END(xmm)

DPL_NAMESPACE_END