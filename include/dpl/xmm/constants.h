// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h" // IWYU pragma: keep

#include "dpl/constants.h"
#include "dpl/xmm/concepts.h"
#include "dpl/xmm/native.h"

#if !DPL_MODULES
#  include <climits>
#  include <utility>
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

template <xmm_element T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD, CONST) native_type_t<T> operator+(
    zero_t<T>) noexcept {
    if constexpr (sized_floating_point<T, 4>) {
        return _mm_setzero_ps();
    } else if constexpr (sized_floating_point<T, 8>) {
        return _mm_setzero_pd();
    } else if constexpr (sized_floating_point<T, 2>) {
        return native_cast<T, float>(result);
    } else {
        return _mm_setzero_si128();
    }
}

template <xmm_element T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD, CONST) native_type_t<T> operator+(
    all_t<T>) noexcept {
    auto const any = _mm_undefined_ps();
    auto const result = _mm_cmpeq_ps(any, any);
    if constexpr (sized_floating_point<T, 4>) {
        return result;
    } else {
        return native_cast<T, float>(result);
    }
}

template <xmm_element T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD, CONST) native_type_t<T> operator+(
    msb_t<T>) noexcept {
    if constexpr (sizeof(T) == 1) {
        native_type_t<short> const val = +msb<short>;
        return _mm_or_si128(val, _mm_bsrli_si128(val, 1));
    } else if constexpr (sizeof(T) == 2) {
        return native_cast<T, short>(
            _mm_slli_epi16(+all<T>, sizeof(T) * CHAR_BIT - 1));
    } else if constexpr (sizeof(T) == 4) {
        return native_cast<T, int>(
            _mm_slli_epi32(+all<T>, sizeof(T) * CHAR_BIT - 1));
    } else {
        static_assert(sizeof(T) == 8, "unsupported operation");
        return native_cast<T, int64_t>(
            _mm_slli_epi64(+all<T>, sizeof(T) * CHAR_BIT - 1));
    }
}

template <xmm_element T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD, CONST) native_type_t<T> operator+(
    lsb_t<T>) noexcept {
    if constexpr (sizeof(T) == 1) {
        native_type_t<short> const val = lsb<short>;
        return _mm_or_si128(val, _mm_bslli_si128(val, 1));
    } else if constexpr (sizeof(T) == 2) {
        return native_cast<T, short>(
            _mm_srli_epi16(all<T>, sizeof(T) * CHAR_BIT - 1));
    } else if constexpr (sizeof(T) == 4) {
        return native_cast<T, int>(
            _mm_srli_epi32(all<T>, sizeof(T) * CHAR_BIT - 1));
    } else {
        static_assert(sizeof(T) == 8, "unsupported operation");
        return native_cast<T, int64_t>(
            _mm_srli_epi64(all<T>, sizeof(T) * CHAR_BIT - 1));
    }
}

template <xmm_element T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD, CONST) native_type_t<T> operator+(
    iota_t<T>) noexcept {
    struct alignas(xmm_abi::alignment) iota_data {
        T data[element_count_v<T>];
    };

    static constexpr auto const data = []<T... Vs>(
                                           std::integer_sequence<T, Vs...>) {
        return iota_data{static_cast<T>(Vs)...};
    }(std::make_integer_sequence<T, element_count_v<T>>{});

    return _mm_load_si128(reinterpret_cast<__m128i const*>(&data));
}

DPL_DEFAULT_NAMESPACE_END
