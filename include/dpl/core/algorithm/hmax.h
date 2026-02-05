// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/algorithm/reduce.h"

#if !DPL_MODULES
#  include "dpl/core/operations/minmax.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

void hmax(...) noexcept = delete;
template <auto>
void hmax(...) noexcept = delete;

template <typename A, typename T>
concept unqualified_hmax = requires(T val) { hmax(internal::abi<A>, val); };

template <typename M, typename A, typename T>
concept unqualified_hmaxi = immediate_mask_for<M, T> &&
    requires(T val) { hmax<immediate_mask_v<T, M>>(internal::abi<A>, val); };

template <typename T>
concept has_fallback_hmax =
    simd_type<T> && requires(basic_type_t<T> val) { dx::reduce(val, dx::min); };

template <typename M, typename T>
concept has_fallback_hmaxi = simd_type<T> && immediate_mask_for<M, T> &&
    requires(basic_type_t<T> val, M mask) { dx::reduce(mask, val, dx::min); };

struct hmax_t {
    template <basic_simd_type T>
    requires arithmetic_simd<T> && has_fallback_hmax<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(T arg) noexcept {
        if constexpr (unqualified_hmax<T, T>) {
            if consteval {
                return static_cast<T>(dx::reduce(arg, dx::max));
            } else {
                return static_cast<T>(hmax(internal::abi<T>, arg));
            }
        } else {
            return dx::reduce(arg, dx::max);
        }
    }

    template <arithmetic_simd T>
    requires (!basic_simd_type<T>) && unqualified_hmax<T, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(T arg) noexcept
        -> equivalent_simd_as<T> auto {
        return hmax(internal::abi<T>, arg);
    }

    template <arithmetic_simd T>
    requires (!basic_simd_type<T> && !unqualified_hmax<T, T>) &&
        has_fallback_hmax<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto operator()(T arg) noexcept
        -> equivalent_simd_as<T> auto {
        return operator()(dx::to_basic_type(arg));
    }

    template <basic_simd_type T, immediate_mask_for<T> M>
    requires arithmetic_simd<T> && has_fallback_hmaxi<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(M mask, T arg) noexcept {
        if constexpr (unqualified_hmaxi<M, T, T>) {
            if consteval {
                return static_cast<T>(dx::reduce(mask, arg, dx::max));
            } else {
                constexpr auto V = immediate_mask_v<T, M>;
                return static_cast<T>(hmax<V>(internal::abi<T>, arg));
            }
        } else {
            return dx::reduce(mask, arg, dx::max);
        }
    }

    template <arithmetic_simd T, immediate_mask_for<T> M>
    requires (!basic_simd_type<T>) && unqualified_hmaxi<M, T, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(M mask, T arg) noexcept
        -> equivalent_simd_as<T> auto {
        constexpr auto V = immediate_mask_v<T, M>;
        return hmax<V>(internal::abi<T>, arg);
    }

    template <arithmetic_simd T, immediate_mask_for<T> M>
    requires (!basic_simd_type<T> && !unqualified_hmaxi<M, T, T>) &&
        has_fallback_hmaxi<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto operator()(M mask, T arg) noexcept
        -> equivalent_simd_as<T> auto {
        return operator()(mask, dx::to_basic_type(arg));
    }
};

template <auto V>
struct hmaxi_t;

template <integral auto V>
struct hmaxi_t<V> {
private:
    template <typename T>
    using mask_type DPL_NODEBUG = immediate_mask<element_count<T>, V>;

public:
    template <arithmetic_simd T>
    requires requires {
        typename mask_type<T>;
        requires regular_invocable<hmax_t, mask_type<T>, T>;
    }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto operator()(T arg) noexcept {
        constexpr mask_type<T> mask{};
        return hmax_t::operator()(mask, arg);
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::hmax_t hmax{};
DPL_EXPORT template <auto V>
inline constexpr internal::hmaxi_t<V> hmaxi{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
