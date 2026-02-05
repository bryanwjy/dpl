// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/algorithm/reduce.h"

#if !DPL_MODULES
#  include "dpl/core/operations/minmax.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

void hmin(...) noexcept = delete;
template <auto>
void hmin(...) noexcept = delete;

template <typename A, typename T>
concept unqualified_hmin = requires(T val) { hmin(internal::abi<A>, val); };

template <typename M, typename A, typename T>
concept unqualified_hmini = immediate_mask_for<M, T> &&
    requires(T val) { hmin<immediate_mask_v<T, M>>(internal::abi<A>, val); };

template <typename T>
concept has_fallback_hmin =
    simd_type<T> && requires(basic_type_t<T> val) { dx::reduce(val, dx::min); };

template <typename M, typename T>
concept has_fallback_hmini = simd_type<T> && immediate_mask_for<M, T> &&
    requires(basic_type_t<T> val, M mask) { dx::reduce(mask, val, dx::min); };

struct hmin_t {
    template <basic_simd_type T>
    requires arithmetic_simd<T> && has_fallback_hmin<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(T arg) noexcept {
        if constexpr (unqualified_hmin<T, T>) {
            if consteval {
                return static_cast<T>(dx::reduce(arg, dx::min));
            } else {
                return static_cast<T>(hmin(internal::abi<T>, arg));
            }
        } else {
            return dx::reduce(arg, dx::min);
        }
    }

    template <arithmetic_simd T>
    requires (!basic_simd_type<T>) && unqualified_hmin<T, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(T arg) noexcept
        -> equivalent_simd_as<T> auto {
        return hmin(internal::abi<T>, arg);
    }

    template <arithmetic_simd T>
    requires (!basic_simd_type<T> && !unqualified_hmin<T, T>) &&
        has_fallback_hmin<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto operator()(T arg) noexcept
        -> equivalent_simd_as<T> auto {
        return operator()(dx::to_basic_type(arg));
    }

    template <basic_simd_type T, immediate_mask_for<T> M>
    requires arithmetic_simd<T> && has_fallback_hmini<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(M mask, T arg) noexcept {
        if constexpr (unqualified_hmini<M, T, T>) {
            if consteval {
                return static_cast<T>(dx::reduce(mask, arg, dx::min));
            } else {
                constexpr auto V = immediate_mask_v<T, M>;
                return static_cast<T>(hmin<V>(internal::abi<T>, arg));
            }
        } else {
            return dx::reduce(mask, arg, dx::min);
        }
    }

    template <arithmetic_simd T, immediate_mask_for<T> M>
    requires (!basic_simd_type<T>) && unqualified_hmini<M, T, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(M mask, T arg) noexcept
        -> equivalent_simd_as<T> auto {
        constexpr auto V = immediate_mask_v<T, M>;
        return hmin<V>(internal::abi<T>, arg);
    }

    template <arithmetic_simd T, immediate_mask_for<T> M>
    requires (!basic_simd_type<T> && !unqualified_hmini<M, T, T>) &&
        has_fallback_hmini<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto operator()(M mask, T arg) noexcept
        -> equivalent_simd_as<T> auto {
        return operator()(mask, dx::to_basic_type(arg));
    }
};

template <auto V>
struct hmini_t;

template <integral auto V>
struct hmini_t<V> {
private:
    template <typename T>
    using mask_type DPL_NODEBUG = immediate_mask<element_count<T>, V>;

public:
    template <arithmetic_simd T>
    requires requires {
        typename mask_type<T>;
        requires regular_invocable<hmin_t, mask_type<T>, T>;
    }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto operator()(T arg) noexcept {
        constexpr mask_type<T> mask{};
        return hmin_t::operator()(mask, arg);
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::hmin_t hmin{};
DPL_EXPORT template <auto V>
inline constexpr internal::hmini_t<V> hmini{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
