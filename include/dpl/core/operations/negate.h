// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/operations/arithmetic.h"
#include "dpl/core/operations/bit.h"
#include "dpl/core/operations/bitwise.h"

#if !DPL_MODULES
#  include "dpl/core/basic/immediate_mask.h"
#  include "dpl/core/basic/reinterpret.h"
#  include "dpl/core/concepts/arithmetic_type.h"
#  include "dpl/core/constants/msb.h"
#  include "dpl/core/type_traits/common_arithmetic_type.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

template <auto>
void negate(...) noexcept = delete;

struct negate_t {
private:
    template <typename T>
    using negated_type DPL_NODEBUG = common_arithmetic_type_t<T, T>;
    template <typename T>
    using negated_simd DPL_NODEBUG = common_arithmetic_simd_t<T, T>;

    template <arithmetic_type E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd<E, A> val) noexcept {
        using T = negated_type<E>;
        return dx::reinterpret<T>(dx::sub(dx::zero, val));
    }

    template <integral auto V, arithmetic_type E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallbacki(basic_simd<E, A> val) noexcept {
        static constexpr immediate_mask<element_count<E, A>, V> mask{};
        if constexpr (all_of(mask)) {
            return fallback(val);
        } else if constexpr (none_of(mask)) {
            using T = negated_type<E>;
            return dx::reinterpret<T>(val);
        } else {
            using T = negated_type<E>;
            return dx::selecti<V>(dx::reinterpret<T>(dx::sub(dx::zero, val)),
                dx::reinterpret<T>(val));
        }
    }

#ifndef DPL_DISABLE_IEC559_FALLBACK

    template <arithmetic_type E, simd_abi A>
    requires floating_point<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd<E, A> val) noexcept {
        return dx::reinterpret<E>(dx::bwxor(val, dx::msb));
    }

    template <integral auto V, arithmetic_type E, simd_abi A>
    requires floating_point<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallbacki(basic_simd<E, A> val) noexcept {
        static constexpr immediate_mask<element_count<E, A>, V> mask{};
        if constexpr (all_of(mask)) {
            return fallback(val);
        } else if constexpr (none_of(mask)) {
            using T = negated_type<E>;
            return dx::reinterpret<T>(val);
        } else {
            constexpr auto nmask = ~mask;
            constexpr auto msb =
                dx::bit_drop(nmask, dx::msb_v<basic_simd<E, A>>);
            return dx::reinterpret<E>(dx::bwxor(val, msb));
        }
    }

#endif

public:
    template <basic_simd_type T>
    requires arithmetic_simd<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr negated_simd<T> operator()(T val) noexcept {
        if constexpr (requires { negate(internal::abi<T>, val); }) {
            if consteval {
                return fallback(val);
            } else {
                return negate(internal::abi<T>, val);
            }
        } else {
            return fallback(val);
        }
    }

    template <arithmetic_simd T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val) noexcept
        -> equivalent_simd_as<negated_simd<T>> auto {
        if constexpr (requires { negate(internal::abi<T>, val); }) {
            return negate(internal::abi<T>, val);
        } else {
            return operator()(dx::to_basic_type(val));
        }
    }

    template <basic_simd_mask_type M, basic_simd_type T>
    requires arithmetic_simd<T> && compatible_mask_for<M, T> &&
        same_abi_simd_as<T, M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr negated_simd<T> operator()(M mask, T val) noexcept {
        if constexpr (requires { negate(internal::abi<T>, mask, val); }) {
            if consteval {
                return dx::select(
                    mask, fallback(val), dx::reinterpret<negated_type<T>>(val));
            } else {
                return negate(internal::abi<T>, mask, val);
            }
        } else {
            return dx::select(
                mask, fallback(val), dx::reinterpret<negated_type<T>>(val));
        }
    }

    template <basic_simd_mask_type M, basic_simd_type T>
    requires arithmetic_simd<T> && compatible_mask_for<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val) noexcept
        -> equivalent_simd_as<negated_simd<T>> auto {
        using A = common_abi_t<M, T>;
        return negate(internal::abi<A>, mask, val);
    }

    template <simd_type T, compatible_mask_for<T> M>
    requires arithmetic_simd<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val) noexcept
        -> equivalent_simd_as<negated_simd<T>> auto {
        if constexpr (requires { negate(internal::abi<T>, mask, val); }) {
            return negate(internal::abi<T>, mask, val);
        } else {
            return operator()(dx::to_basic_type(mask), dx::to_basic_type(val));
        }
    }

    template <basic_simd_type T, immediate_mask_for<T> M>
    requires arithmetic_simd<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr negated_simd<T> operator()(M mask, T val) noexcept {
        constexpr auto V = decltype(dx::to_immediate_mask<T>(mask))::value;
        if constexpr (requires { negate<V>(internal::abi<T>, val); }) {
            if consteval {
                return fallbacki<V>(val);
            } else {
                return negate<V>(internal::abi<T>, val);
            }
        } else {
            return fallbacki<V>(val);
        }
    }

    template <arithmetic_simd T, immediate_mask_for<T> M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val) noexcept
        -> equivalent_simd_as<negated_simd<T>> auto {
        constexpr auto V = decltype(dx::to_immediate_mask<T>(mask))::value;
        if constexpr (requires { negate<V>(internal::abi<T>, val); }) {
            return negate<V>(internal::abi<T>, val);
        } else {
            return operator()(mask, dx::to_basic_type(val));
        }
    }
};

template <auto V>
struct negatei_t {
private:
    template <typename T>
    using mask_type DPL_NODEBUG = immediate_mask<element_count<T>, V>;

public:
    template <simd_class T>
    requires requires {
        typename mask_type<T>;
        requires regular_invocable<negate_t, mask_type<T>, T>;
    }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T arg) noexcept {
        constexpr mask_type<T> mask{};
        return negate_t::operator()(mask, arg);
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::negate_t negate{};
DPL_EXPORT template <auto V>
inline constexpr internal::negatei_t<V> negatei{};
} // namespace cpo
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
