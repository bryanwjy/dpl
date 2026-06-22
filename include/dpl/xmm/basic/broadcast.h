// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

#if !DPL_ARCH_x86_64 || !DPL_SIMD_X86_SSE4_2
#  error "Unsupported platform"
#endif

#include "dpl/xmm/basic/abi.h"
#include "dpl/xmm/basic/initialize.h"

#if !DPL_MODULES
#  include "dpl/core/immediate/constants/all_bits.h"
#  include "dpl/core/immediate/constants/zero.h"
#  include "dpl/std/bit/bit_cast.h"
#  include "dpl/std/type_traits/is_const.h"
#  include "dpl/std/type_traits/is_volatile.h"
#  include "dpl/std/type_traits/type_identity.h"
#  include "dpl/std/utility/sequence.h"

#  include <immintrin.h>
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {
DPL_EXPORT template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr simd<E> broadcast(abi_tag tag, type_identity_t<E> scalar) noexcept {
    static_assert(!is_const_v<E> && !is_volatile_v<E>);
    if consteval {
        constexpr auto seq = make_index_sequence<abi_tag::size / sizeof(E)>{};
        constexpr auto forward = [](auto, E scalar) { return scalar; };
        return [forward]<size_t... Is>(
                   index_sequence<Is...>, abi_tag tag, E data) {
            return dx::xmm::initialize<E>(tag, forward(Is, data)...);
        }(iota<E>, tag, scalar);
    } else {
        if constexpr (same_as<float, representation_t<E>>) {
            return _mm_set1_ps(__DPL bit_cast<float>(scalar));
        } else if constexpr (same_as<double, representation_t<E>>) {
            return _mm_set1_pd(__DPL bit_cast<double>(scalar));
        } else if constexpr (bfloat16_like<representation_t<E>> ||
            float16_like<representation_t<E>>) {
            return __DPL bit_cast<native_vector_t<E>>(
                _mm_set1_epi16(__DPL bit_cast<int16>(scalar)));
        } else {
            static_assert(integral<representation_t<E>>);
            if constexpr (sizeof(E) == sizeof(int32)) {
                return _mm_set1_epi32(__DPL bit_cast<int32>(scalar));
            } else if constexpr (sizeof(E) == sizeof(int16)) {
                return _mm_set1_epi16(__DPL bit_cast<int16>(scalar));
            } else if constexpr (sizeof(E) == sizeof(int8)) {
                return _mm_set1_epi8(__DPL bit_cast<int8>(scalar));
            } else {
                static_assert(sizeof(E) == sizeof(int64));
                return _mm_set1_epi64x(__DPL bit_cast<int64>(scalar));
            }
        }
    }
}

DPL_EXPORT template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr simd<E> broadcast(abi_tag tag, dx::zero_t) noexcept {
    static_assert(!is_const_v<E> && !is_volatile_v<E>);
    if consteval {
        return xmm::broadcast<E>(tag, 0);
    } else {
        if constexpr (same_as<float, representation_t<E>>) {
            return _mm_setzero_ps();
        } else if constexpr (same_as<double, representation_t<E>>) {
            return _mm_setzero_pd();
        } else if constexpr (bfloat16_like<representation_t<E>> ||
            float16_like<representation_t<E>>) {
            return __DPL bit_cast<native_vector_t<E>>(_mm_setzero_si128());
        } else {
            static_assert(integral<E>);
            return _mm_setzero_si128();
        }
    }
}

DPL_EXPORT template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr simd<E> broadcast(abi_tag tag, dx::all_bits_t) noexcept {
    static_assert(!is_const_v<E> && !is_volatile_v<E>);
    if consteval {
        return xmm::broadcast<E>(tag, dx::all_bits_v<E>);
    } else {
        auto xmm0 = _mm_undefined_si128();
        xmm0 = _mm_cmpeq_epi32(xmm0, xmm0);
        if constexpr (same_as<float, representation_t<E>>) {
            return _mm_castsi128_ps(xmm0);
        } else if constexpr (same_as<double, representation_t<E>>) {
            return _mm_castsi128_pd(xmm0);
        } else if constexpr (bfloat16_like<representation_t<E>> ||
            float16_like<representation_t<E>>) {
            return __DPL bit_cast<native_vector_t<E>>(xmm0);
        } else {
            static_assert(integral<E>);
            return xmm0;
        }
    }
}

DPL_EXPORT template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr mask<E> broadcast(abi_tag tag, same_as<bool> auto scalar) noexcept {
    static_assert(!is_const_v<E> && !is_volatile_v<E>);
    return scalar ? +xmm::broadcast<E>(tag, dx::all_bits)
                  : +xmm::broadcast<E>(tag, dx::zero);
}

DPL_EXPORT template <simd_element E, integral_constant_like V>
requires same_as<typename V::value_type, bool>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr mask<E> broadcast(abi_tag tag, V) noexcept {
    static_assert(!is_const_v<E> && !is_volatile_v<E>);
    if constexpr (V::value) {
        return +xmm::broadcast<E>(tag, dx::all_bits);
    } else {
        return +xmm::broadcast<E>(tag, dx::zero);
    }
}

DPL_EXPORT template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr simd<E> broadcast(type_identity_t<E> scalar) noexcept {
    return xmm::broadcast<E>(xmm::abi, scalar);
}

DPL_EXPORT template <simd_element E, same_as<bool> T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr mask<E> broadcast(T boolean) noexcept {
    return xmm::broadcast<E>(xmm::abi, boolean);
}

DPL_EXPORT template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr simd<E> broadcast(dx::zero_t zero) noexcept {
    return xmm::broadcast<E>(xmm::abi, zero);
}

DPL_EXPORT template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr simd<E> broadcast(dx::all_bits_t all) noexcept {
    return xmm::broadcast<E>(xmm::abi, all);
}

} // namespace datapar::xmm

DPL_DEFAULT_NAMESPACE_END
