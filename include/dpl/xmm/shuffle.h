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

#  include <bit>
#endif

DPL_NAMESPACE_BEGIN

DPL_SIMD_ABI_NAMESPACE_BEGIN(xmm)

namespace details::shuffle {

template <size_t N>
consteval bool is_iota(std::make_index_sequence<N>) noexcept {
    return true;
}

template <size_t N, size_t... Is>
consteval bool is_iota(std::index_sequence<Is...>) noexcept {
    static_assert(N == sizeof...(Is));
    return false;
}

template <size_t... Is>
class shuffle_t {
    static constexpr auto index_count = sizeof...(Is);
    static constexpr bool all_in_range = (... && (Is < index_count));
    template <xmm_element T>
    static constexpr bool valid_for =
        std::has_single_bit(index_count) && all_in_range &&
        (sizeof...(Is) == element_count<T>)&&(sizeof(T) < xmm_abi::size);

public:
    using element_type = T;
    using native_type = native_type_t<T>;
    using mask_type = mask_for<T>;

    __DPL_HIDE_FROM_ABI explicit consteval shuffle_t() noexcept = default;

    template <xmm_element T>
    requires valid_for<T>
    DPL_ATTRIBUTES(
        _HIDE_FROM_ABI, CONST, VECTORCALL, NODISCARD) static vector<T>
    operator()(vector<T> lower, vector<T> upper) noexcept {
        static constexpr int imm = static_cast<int>(+packed_indices{Is...});
        if constexpr (sized_floating_point<T, 8>) {
            return _mm_shuffle_pd(lower, upper, imm);
        } else if constexpr (sized_integral<T, 8>) {
            return native_cast<T, double>(
                _mm_shuffle_pd(native_cast<double, T>(lower),
                    native_cast<double, T>(upper), imm));
        } else if constexpr (sized_floating_point<T, 4>) {
            return _mm_shuffle_ps(lower, upper, imm);
        } else if constexpr (sized_integral<T, 4>) {
            return native_cast<T, float>(
                _mm_shuffle_ps(native_cast<float, T>(lower),
                    native_cast<float, T>(upper), imm));
        } else {
            static_assert(sized_integral<T, 4>, "unsupported operation");
            DPL_BUILTIN_unreachable();
        }
    }

    template <xmm_element T>
    requires valid_for<T>
    DPL_ATTRIBUTES(
        _HIDE_FROM_ABI, CONST, VECTORCALL, NODISCARD) static vector<T>
    operator()(vector<T> val) noexcept {
        if constexpr (sizeof(T) == 1) {
            static constexpr int imm = static_cast<int>(+packed_indices{Is...});
            return native_cast<T, char>(
                _mm_shuffle_epi8(native_cast<char, T>(val), imm));
        } else if constexpr (sizeof(T) == 2) {
            static constexpr int imm =
                static_cast<int>(+packed_indices{Is...}.template repeat<2>());
            return native_cast<T, char>(
                _mm_shuffle_epi8(native_cast<char, T>(val), imm));
        } else {
            return shuffle_t::operator()(val, val);
        }
    }

    template <xmm_element T>
    requires valid_for<T> &&
        (is_iota<element_count<T>>(std::index_sequence<Is...>{}))
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) static vector<T> operator()(
        vector<T> val) noexcept {
        return val;
    }

#if DPL_SIMD_X86_AVX512FP16 & DPL_SUPPORTS_FLOAT16
    template <xmm_element T>
    requires std::same_as<float16, T> && valid_for<T> &&
        (is_iota<element_count<T>>(std::index_sequence<Is...>{}))
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) static vector<T> operator()(
        vector<T> val) noexcept {
        return val;
    }

    template <xmm_element T>
    requires std::same_as<float16, T> && valid_for<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) static vector<T> operator()(
        vector<T> val) noexcept {
        return shuffle_t::operator()(val);
    }

    template <xmm_element T>
    requires std::same_as<float16, T> && valid_for<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) static vector<T> operator()(
        vector<T> lower, vector<T> upper) noexcept {
        static constexpr struct alignas(xmm_abi::alignment) idx_t {
            unsigned short data[element_count<unsigned short>];
        } idx{(Is + (Is >= 4) * 0b100)...};

        auto const vidx = _mm_load_si128(reinterpret_cast<__m128i const*>(idx));
        return _mm_permutex2var_ph(+lower, vidx, +upper);
    }
#endif
};

} // namespace details::shuffle

DPL_EXPORT_BEGIN

template <size_t... Is, xmm_element T>
requires requires(details::shuffle::shuffle_t<Is...> impl, vector<T> lower,
    vector<T> upper) { impl(lower, upper); }
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) inline auto shuffle(
    vector<T> lower, vector<T> upper) noexcept {
    static constexpr details::shuffle::shuffle_t<Is...> impl{};
    return impl(lower, upper);
}

template <size_t... Is, xmm_element T>
requires requires(
    details::shuffle::shuffle_t<Is...> impl, vector<T> val) { impl(val); }
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) inline auto permute(
    vector<T> val) noexcept {
    static constexpr details::shuffle::shuffle_t<Is...> impl{};
    return impl(val);
}

template <size_t I, xmm_element T>
requires (I < element_count<T>)
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) inline auto broadcast(
    vector<T> val) noexcept {
    static constexpr auto impl = []<size_t... Is>(std::index_sequence<Is...>) {
        return detail::shuffle::shuffle_t<(Is, I)...>{};
    }(std::make_index_sequence<element_count<T>>{});
    return impl(val);
}

// TODO: permute with runtime indices

DPL_EXPORT_END

DPL_SIMD_ABI_NAMESPACE_END(xmm)

DPL_NAMESPACE_END
