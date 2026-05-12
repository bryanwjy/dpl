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

template <typename T>
concept unqualified_hsum = requires(T val) {
    { hsum(internal::abi<T>, val) } -> equivalent_simd_as<T>;
};

template <typename M, typename T>
concept unqualified_hsumi = immediate_mask_for<M, T> && requires(T val) {
    {
        hsum<immediate_mask_v<T, M>>(internal::abi<T>, val)
    } -> equivalent_simd_as<T>;
};

struct hsum_t {
private:
    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd<E, A> val) noexcept {
        return dx::reduce(val, dx::add);
    }

    template <auto V, typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallbacki(basic_simd<E, A> val) noexcept {
        return dx::reducei<V>(val, dx::add);
    }

public:
    template <arithmetic_simd T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T arg) noexcept {
        if constexpr (unqualified_hsum<T>) {
            if constexpr (basic_simd_type<T>) {
                if consteval {
                    using RT = decltype(hsum(internal::abi<T>, arg));
                    return dx::reinterpret<RT>(fallback(arg));
                } else {
                    return hsum(internal::abi<T>, arg);
                }
            } else {
                return hsum(internal::abi<T>, arg);
            }
        } else if constexpr (basic_simd_type<T>) {
            return fallback(arg);
        } else {
            return operator()(dx::to_basic_type(arg));
        }
    }

    template <ordered_simd T>
    requires scalable_simd<T> &&
        (unqualified_hsum<T> || unqualified_hsum<basic_type_t<T>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T arg) noexcept {
        if constexpr (unqualified_hsum<T>) {
            return hsum(internal::abi<T>, arg);
        } else {
            return hsum(internal::abi<T>, dx::to_basic_type(arg));
        }
    }

    template <arithmetic_simd T, immediate_mask_for<T> M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T arg) noexcept {
        constexpr auto V = immediate_mask_v<T, M>;
        if constexpr (unqualified_hsumi<M, T>) {
            if constexpr (basic_simd_type<T>) {
                if consteval {
                    using RT = decltype(hsum<V>(internal::abi<T>, arg));
                    return dx::reinterpret<RT>(fallbacki<V>(arg));
                } else {
                    return hsum<V>(internal::abi<T>, arg);
                }
            } else {
                return hsum<V>(internal::abi<T>, arg);
            }
        } else if constexpr (basic_simd_type<T>) {
            return fallbacki<V>(arg, dx::add);
        } else {
            return operator()(dx::to_basic_type(arg));
        }
    }
};

template <auto V>
struct hsumi_t {};

template <integral auto V>
struct hsumi_t<V> {
private:
    template <typename T>
    using mask_type DPL_NODEBUG = make_immediate_mask_t<T, V>;

public:
    template <arithmetic_simd T>
    requires fixed_width_simd<T> && requires {
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
