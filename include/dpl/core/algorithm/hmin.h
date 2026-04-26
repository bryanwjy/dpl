// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/algorithm/reduce.h"

#if !DPL_MODULES
#  include "dpl/core/operations/minmax.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

template <typename T>
concept unqualified_hmin = requires(T val) {
    { hmin(internal::abi<T>, val) } -> equivalent_simd_as<T>;
};

template <typename M, typename T>
concept unqualified_hmini = immediate_mask_for<M, T> && requires(T val) {
    {
        hmin<immediate_mask_v<T, M>>(internal::abi<T>, val)
    } -> equivalent_simd_as<T>;
};

struct hmin_t {
private:
    template <simd_element E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd<E, A> val) noexcept {
        return dx::reduce(val, dx::max);
    }

    template <auto V, simd_element E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallbacki(basic_simd<E, A> val) noexcept {
        return dx::reducei<V>(val, dx::max);
    }

public:
    template <ordered_simd T>
    requires fixed_width_simd<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T arg) noexcept {
        if constexpr (unqualified_hmin<T>) {
            if constexpr (basic_simd_type<T>) {
                if consteval {
                    return fallback(arg);
                } else {
                    return hmin(internal::abi<T>, arg);
                }
            } else {
                return hmin(internal::abi<T>, arg);
            }
        } else if constexpr (basic_simd_type<T>) {
            return fallback(arg);
        } else {
            return operator()(dx::to_basic_type(arg));
        }
    }

    template <ordered_simd T>
    requires scalable_simd<T> &&
        (unqualified_hmin<T> || unqualified_hmin<basic_type_t<T>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T arg) noexcept {
        if constexpr (unqualified_hmin<T>) {
            return hmin(internal::abi<T>, arg);
        } else {
            return hmin(internal::abi<T>, dx::to_basic_type(arg));
        }
    }

    template <ordered_simd T, immediate_mask_for<T> M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T arg) noexcept {
        constexpr auto V = immediate_mask_v<T, M>;
        if constexpr (unqualified_hmini<M, T>) {
            if constexpr (basic_simd_type<T>) {
                if consteval {
                    return fallbacki<V>(arg);
                } else {
                    return hmin<V>(internal::abi<T>, arg);
                }
            } else {
                return hmin<V>(internal::abi<T>, arg);
            }
        } else if constexpr (basic_simd_type<T>) {
            return fallbacki<V>(arg);
        } else {
            return operator()(mask, dx::to_basic_type(arg));
        }
    }
};

template <auto V>
struct hmini_t {};

template <integral auto V>
struct hmini_t<V> {
private:
    template <typename T>
    using mask_type DPL_NODEBUG = immediate_mask<element_count<T>, V>;

public:
    template <arithmetic_simd T>
    requires fixed_width_simd<T> && requires {
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
