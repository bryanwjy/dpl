// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

#if !DPL_ARCH_x86_64 || !DPL_SIMD_X86_SSE4_2
#  error "Unsupported platform"
#endif

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/core/basic/immediate_mask.h"
#  include "dpl/core/concepts/common_arithmetic_with.h"
#  include "dpl/core/concepts/common_integral_with.h"
#  include "dpl/core/concepts/common_size_with.h"
#  include "dpl/core/type_traits/common_size_type.h"
#  include "dpl/xmm/basic/abi.h"
#  include "dpl/xmm/basic/reinterpret.h"

#  include <immintrin.h>
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::xmm {

template <typename L, typename R>
using ternary_result_t DPL_NODEBUG =
    __DPL decay_t<decltype(false ? __DPL declval<L>()
                                   : __DPL declval<R>())>;

template <typename L, typename R>
struct ternary_type {};
template <typename L, typename R>
using ternary_type_t DPL_NODEBUG = typename ternary_type<L, R>::type;

template <simd_type L, simd_element R>
struct ternary_type<L, R> : ternary_type<typename L::value_type, R> {};

template <simd_element L, simd_type R>
struct ternary_type<L, R> : ternary_type<L, typename R::value_type> {};

template <simd_type L, simd_type R>
struct ternary_type<L, R> :
    ternary_type<typename L::value_type, typename R::value_type> {};

template <simd_element L, simd_element R>
requires requires {
    typename ternary_result_t<L, R>;
    requires common_arithmetic_with<ternary_result_t<L, R>, L>;
    requires common_arithmetic_with<ternary_result_t<L, R>, R>;
}
struct ternary_type<L, R> {
    using type DPL_NODEBUG = ternary_result_t<L, R>;
};

template <template_barrier_t = barrier, simd_element C, simd_element T,
    simd_element F>
requires common_size_with<T, F> && common_size_with<T, C> &&
    common_size_with<F, C> &&
    common_size_with<ternary_type_t<T, F>, common_size_type_t<T, F>>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<ternary_type_t<T, F>> select(
    abi_tag tag, simd<C> condition, simd<T> if_true, simd<F> or_else) noexcept {
    using V = ternary_type_t<T, F>;
    if constexpr (common_float_with<float, V>) {
        return _mm_blendv_ps(
            +or_else, +if_true, +xmm::reinterpret<float>(tag, condition));
    } else if constexpr (common_float_with<double, V>) {
        return _mm_blendv_pd(
            +or_else, +if_true, +xmm::reinterpret<double>(tag, condition));
    } else if constexpr (common_size_with<float, V>) {
        return xmm::reinterpret<V>(tag,
            xmm::select(tag, condition, xmm::reinterpret<float>(tag, if_true),
                xmm::reinterpret<float>(tag, or_else)));
    } else if constexpr (common_size_with<double, V>) {
        return xmm::reinterpret<V>(tag,
            xmm::select(tag, condition, xmm::reinterpret<double>(tag, if_true),
                xmm::reinterpret<double>(tag, or_else)));
    } else if constexpr (common_integral_with<V, int16>) {
        return _mm_blendv_epi8(
            +or_else, +if_true, +xmm::reinterpret<int16>(tag, condition));
    } else if constexpr (common_size_with<int16, V>) {
        return xmm::reinterpret<V>(tag,
            xmm::select(tag, condition, xmm::reinterpret<int16>(tag, if_true),
                xmm::reinterpret<int16>(tag, or_else)));
    } else {
        static_assert(sizeof(T) == sizeof(int8));
        return _mm_blendv_epi8(+or_else, +if_true, +condition);
    }
}

template <template_barrier_t = barrier, simd_element C, simd_element T,
    simd_element F>
requires common_size_with<T, F> && common_size_with<T, C> &&
    common_size_with<F, C>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<common_size_type_t<T, F>> select(
    abi_tag tag, mask<C> condition, mask<T> if_true, mask<F> or_else) noexcept {
    using V = common_size_type_t<T, F>;
    return +xmm::select(tag, condition, simd<V>(+xmm::reinterpret<V>(if_true)),
        simd<V>(+xmm::reinterpret<V>(or_else)));
}

namespace details {
template <auto M>
inline constexpr auto imm16 = []<size_t... Is>(index_sequence<Is...>) {
    return xmm::initialize<uint8>(xmm::abi, M[Is]...);
}(iota_sequence<uint8, abi_tag>);
} // namespace details

template <integral auto C, simd_element T, simd_element F>
requires common_size_with<T, F> &&
    common_size_with<ternary_type_t<T, F>, common_size_type_t<T, F>>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<ternary_type_t<T, F>> select(
    abi_tag tag, simd<T> if_true, simd<F> or_else) noexcept {
    using V = ternary_type_t<T, F>;
    constexpr immediate_mask<element_count<V, abi_tag>, C> mask{};
    constexpr auto imm8 = static_cast<int>(mask);
    if constexpr (common_float_with<float, V>) {
        return _mm_blend_ps(+or_else, +if_true, imm8);
    } else if constexpr (common_float_with<double, V>) {
        return _mm_blend_pd(+or_else, +if_true, imm8);
    } else if constexpr (common_size_with<float, V>) {
        return xmm::reinterpret<V>(tag,
            xmm::select<C>(tag, xmm::reinterpret<float>(tag, if_true),
                xmm::reinterpret<float>(tag, or_else)));
    } else if constexpr (common_size_with<double, V>) {
        return xmm::reinterpret<V>(tag,
            xmm::select<C>(tag, xmm::reinterpret<double>(tag, if_true),
                xmm::reinterpret<double>(tag, or_else)));
    } else if constexpr (common_integral_with<V, int16>) {
        return _mm_blend_epi16(+or_else, +if_true, imm8);
    } else if constexpr (common_size_with<int16, V>) {
        return xmm::reinterpret<V>(tag,
            xmm::select<C>(tag, xmm::reinterpret<int16>(tag, if_true),
                xmm::reinterpret<int16>(tag, or_else)));
    } else {
        static_assert(sizeof(T) == sizeof(int8));
        return _mm_blendv_epi8(+or_else, +if_true, +details::imm16<mask>);
    }
}

template <integral auto C, simd_element T, simd_element F>
requires common_size_with<T, F>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<common_size_type_t<T, F>> select(
    abi_tag tag, mask<T> if_true, mask<F> or_else) noexcept {
    using V = common_size_type_t<T, F>;
    return +xmm::select<C>(tag, simd<V>(+xmm::reinterpret<V>(if_true)),
        simd<V>(+xmm::reinterpret<V>(or_else)));
}

} // namespace datapar::xmm

DPL_DEFAULT_NAMESPACE_END
