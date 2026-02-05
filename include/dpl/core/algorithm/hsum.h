// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/algorithm/reduce.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/arithmetic_type.h"
#  include "dpl/core/operations/arithmetic.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

void hsum(...) noexcept = delete;
template <auto>
void hsum(...) noexcept = delete;

template <typename A, typename T>
concept unqualified_hsum = requires(T val) { hsum(internal::abi<A>, val); };

template <typename M, typename A, typename T>
concept unqualified_hsumi = immediate_mask_for<M, T> &&
    requires(T val) { hsum<immediate_mask_v<T, M>>(internal::abi<A>, val); };

template <typename T, typename... Args>
concept basic_hsum = (simd_type<T> && ... && immediate_mask_for<Args, T>) &&
    requires(basic_type_t<T> val, Args... mask) {
        dx::reduce(mask..., val, dx::add);
    };

struct hsum_t {
    template <basic_simd_type T>
    requires arithmetic_simd<T> && basic_hsum<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(T arg) noexcept {
        if constexpr (unqualified_hsum<T, T>) {
            if consteval {
                return static_cast<T>(dx::reduce(arg, dx::add));
            } else {
                return static_cast<T>(hsum(internal::abi<T>, arg));
            }
        } else {
            return dx::reduce(arg, dx::add);
        }
    }

    template <arithmetic_simd T>
    requires (!basic_simd_type<T>) && unqualified_hsum<T, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(T arg) noexcept
        -> equivalent_simd_as<T> auto {
        return hsum(internal::abi<T>, arg);
    }

    template <arithmetic_simd T>
    requires (!basic_simd_type<T> && !unqualified_hsum<T, T>) && basic_hsum<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto operator()(T arg) noexcept
        -> equivalent_simd_as<T> auto {
        return operator()(dx::to_basic_type(arg));
    }

    template <basic_simd_type T, immediate_mask_for<T> M>
    requires arithmetic_simd<T> && basic_hsum<T, M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(M mask, T arg) noexcept {
        if constexpr (unqualified_hsumi<M, T, T>) {
            if consteval {
                return static_cast<T>(dx::reduce(mask, arg, dx::add));
            } else {
                constexpr auto V = immediate_mask_v<T, M>;
                return static_cast<T>(hsum<V>(internal::abi<T>, arg));
            }
        } else {
            return dx::reduce(mask, arg, dx::add);
        }
    }

    template <arithmetic_simd T, immediate_mask_for<T> M>
    requires (!basic_simd_type<T>) && unqualified_hsumi<M, T, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(M mask, T arg) noexcept
        -> equivalent_simd_as<T> auto {
        constexpr auto V = immediate_mask_v<T, M>;
        return hsum<V>(internal::abi<T>, arg);
    }

    template <arithmetic_simd T, immediate_mask_for<T> M>
    requires (!basic_simd_type<T> && !unqualified_hsumi<M, T, T>) &&
        basic_hsum<T, M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto operator()(M mask, T arg) noexcept
        -> equivalent_simd_as<T> auto {
        return operator()(mask, dx::to_basic_type(arg));
    }
};

template <auto V>
struct hsumi_t;

template <integral auto V>
struct hsumi_t<V> {
private:
    template <typename T>
    using mask_type DPL_NODEBUG = immediate_mask<element_count<T>, V>;

public:
    template <arithmetic_simd T>
    requires requires {
        typename mask_type<T>;
        requires regular_invocable<hsum_t, mask_type<T>, T>;
    }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto operator()(T arg) noexcept {
        constexpr mask_type<T> mask{};
        return hsum_t::operator()(mask, arg);
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::hsum_t hsum{};
DPL_EXPORT template <auto V>
inline constexpr internal::hsumi_t<V> hsumi{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
