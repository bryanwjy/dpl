// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

#if DPL_SIMD_X86_SSE4_2

#  include "dpl/xmm/basic/abi.h"
#  include "dpl/xmm/basic/initialize.h"
#  include "dpl/xmm/basic/inl/compare.h"

#  if !DPL_MODULES
#    include "dpl/core/immediate/constants/all_bits.h"
#    include "dpl/core/immediate/constants/zero.h"
#    include "dpl/std/bit/bit_cast.h"
#    include "dpl/std/type_traits/sequence.h"
#    include "dpl/std/type_traits/type_identity.h"

#    include <immintrin.h>
#  endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr vector<E> broadcast(type_identity_t<E> scalar) noexcept {
    if consteval {
        constexpr auto forward = [](auto, E scalar) { return scalar; };
        return [forward]<size_t... Is>(index_sequence<Is...>, E data) {
            return xmm::initialize<E>(forward(Is, data)...);
        }(iota<E>, scalar);
    } else {
        if constexpr (is_same_v<native_vector_t<E>, __m128>) {
            return _mm_set1_ps(__DPL bit_cast<float>(scalar));
        } else if constexpr (is_same_v<native_vector_t<E>, __m128d>) {
            return _mm_set1_pd(__DPL bit_cast<double>(scalar));
        } else if constexpr (!is_same_v<native_vector_t<E>, __m128i>) {
            return __DPL bit_cast<native_vector_t<E>>(
                _mm_set1_epi16(__DPL bit_cast<int16>(scalar)));
        } else {
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

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr vector<E> broadcast(dx::zero_t) noexcept {
    if consteval {
        return xmm::broadcast<E>(0);
    } else {
        if constexpr (is_same_v<native_vector_t<E>, __m128>) {
            return _mm_setzero_ps();
        } else if constexpr (is_same_v<native_vector_t<E>, __m128d>) {
            return _mm_setzero_pd();
        } else if constexpr (!is_same_v<native_vector_t<E>, __m128i>) {
            return __DPL bit_cast<native_vector_t<E>>(_mm_setzero_si128());
        } else {
            static_assert(integral<E>);
            return _mm_setzero_si128();
        }
    }
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr vector<E> broadcast(dx::all_bits_t) noexcept {
    if consteval {
        return xmm::broadcast<E>(dx::all_bits_v<E>);
    } else {
        auto xmm0 = _mm_undefined_si128();
        xmm0 = fwd::cmpeq_epi32(xmm0, xmm0);
        if constexpr (is_same_v<native_vector_t<E>, __m128>) {
            return _mm_castsi128_ps(xmm0);
        } else if constexpr (is_same_v<native_vector_t<E>, __m128d>) {
            return _mm_castsi128_pd(xmm0);
        } else if constexpr (!is_same_v<native_vector_t<E>, __m128i>) {
            return __DPL bit_cast<native_vector_t<E>>(xmm0);
        } else {
            static_assert(integral<E>);
            return xmm0;
        }
    }
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr mask<E> broadcast(same_as<bool> auto boolean) noexcept {
    return boolean ? +xmm::broadcast<E>(dx::all_bits)
                   : +xmm::broadcast<E>(dx::zero);
}

template <simd_element E, bool_constant_like V>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr mask<E> broadcast(V) noexcept {
    if constexpr (V::value) {
        return +xmm::broadcast<E>(dx::all_bits);
    } else {
        return +xmm::broadcast<E>(dx::zero);
    }
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr vector<E> broadcast(abi_tag, type_identity_t<E> scalar) noexcept {
    return xmm::broadcast<E>(scalar);
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr vector<E> broadcast(abi_tag, dx::zero_t zero) noexcept {
    return xmm::broadcast<E>(zero);
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr vector<E> broadcast(abi_tag, dx::all_bits_t all_bits) noexcept {
    return xmm::broadcast<E>(all_bits);
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr mask<E> broadcast(abi_tag, same_as<bool> auto boolean) noexcept {
    return xmm::broadcast<E>(boolean);
}

template <simd_element E, bool_constant_like V>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr mask<E> broadcast(abi_tag, V constant) noexcept {
    return xmm::broadcast<E>(constant);
}

} // namespace datapar::xmm

__DPL_DEFAULT_NAMESPACE_END

#endif
