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

template <typename T>
concept unqualified_hmax = requires(T val) {
    { hmax(internal::abi<T>, val) } -> equivalent_simd_as<T>;
};

template <typename M, typename T>
concept unqualified_hmaxi = immediate_mask_for<M, T> && requires(T val) {
    {
        hmax<immediate_mask_v<T, M>>(internal::abi<T>, val)
    } -> equivalent_simd_as<T>;
};

struct hmax_t {
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
        if constexpr (unqualified_hmax<T>) {
            if constexpr (basic_simd_type<T>) {
                if consteval {
                    return fallback(arg);
                } else {
                    return hmax(internal::abi<T>, arg);
                }
            } else {
                return hmax(internal::abi<T>, arg);
            }
        } else if constexpr (basic_simd_type<T>) {
            return fallback(arg);
        } else {
            return operator()(dx::to_basic_type(arg));
        }
    }

    template <ordered_simd T>
    requires scalable_simd<T> &&
        (unqualified_hmax<T> || unqualified_hmax<basic_type_t<T>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T arg) noexcept {
        if constexpr (unqualified_hmax<T>) {
            return hmax(internal::abi<T>, arg);
        } else {
            return hmax(internal::abi<T>, dx::to_basic_type(arg));
        }
    }

    template <ordered_simd T, immediate_mask_for<T> M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T arg) noexcept {
        constexpr auto V = immediate_mask_v<T, M>;
        if constexpr (unqualified_hmaxi<M, T>) {
            if constexpr (basic_simd_type<T>) {
                if consteval {
                    return fallbacki<V>(arg);
                } else {
                    return hmax<V>(internal::abi<T>, arg);
                }
            } else {
                return hmax<V>(internal::abi<T>, arg);
            }
        } else if constexpr (basic_simd_type<T>) {
            return fallbacki<V>(arg);
        } else {
            return operator()(mask, dx::to_basic_type(arg));
        }
    }
};

template <auto V>
struct hmaxi_t {};

template <integral auto V>
struct hmaxi_t<V> {
private:
    template <typename T>
    using mask_type DPL_NODEBUG = immediate_mask<element_count<T>, V>;

public:
    template <arithmetic_simd T>
    requires fixed_width_simd<T> && requires {
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
