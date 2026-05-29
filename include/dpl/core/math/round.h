// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/ceil.h"
#include "dpl/core/math/copysign.h"
#include "dpl/core/math/floor.h"
#include "dpl/core/math/internal/constants.h"
#include "dpl/core/math/internal/floating_point_simd.h"
#include "dpl/core/math/isfinite.h"
#include "dpl/core/math/trunc.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/basic_type.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/constants/one.h"
#  include "dpl/core/operations/arithmetic.h"
#  include "dpl/core/operations/cast.h"
#  include "dpl/core/operations/compare.h"
#  include "dpl/core/operations/select.h"
#  include "dpl/core/type_traits/representation.h"
#  include "dpl/core/utility/rounding.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
namespace mx = datapar::fmath;

struct round_t;

template <typename T>
concept unqualified_canonical_cmath_round = requires(T val) {
    {
        round(internal::abi<T>, val)
    } -> canonical_arithmetic_result<T, T, typename T::abi_type>;
};

template <typename T>
concept unqualified_extended_cmath_round = requires(T val) {
    { round(val) } -> extended_arithmetic_result<T, T, typename T::abi_type>;
};

template <typename T>
concept unqualified_cmath_round = unqualified_extended_cmath_round<T> ||
    (decayable_vector_for<T, operation_category::lane_agnostic> &&
        regular_invocable<round_t, canonical_type_t<T>>);

template <typename T, rounding_flags R>
concept unqualified_canonical_round = requires(T val) {
    {
        round(internal::abi<T>, val, rounding_v<R>)
    } -> canonical_arithmetic_result<T, T, typename T::abi_type>;
};

template <typename T, rounding_flags R>
concept unqualified_extended_round = requires(T val) {
    {
        round(val, rounding_v<R>)
    } -> extended_arithmetic_result<T, T, typename T::abi_type>;
};

template <typename T, rounding_flags R>
concept unqualified_round = unqualified_extended_round<T, R> ||
    (decayable_vector_for<T, operation_category::lane_agnostic> &&
        regular_invocable<round_t, canonical_type_t<T>, rounding_t<R>>);

struct round_t : mx::masked_operation<round_t> {
private:
    friend mx::masked_operation<round_t>;

    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_vector<E, A> val) noexcept {
        auto const isfinite = dx::isfinite(val);
        auto const finite = dx::select(isfinite, val, dx::zero);
        auto x = finite + mx::half;
        auto fr = x - dx::trunc(x);
        auto one = dx::broadcast<E, A>(dx::one);
        x -= dx::select(x <= dx::zero && fr == dx::zero, one, dx::zero);
        fr += dx::select(fr < dx::zero, one, dx::zero);

        x = dx::select(fr != mx::underhalf, x, dx::zero);
        return dx::select(isfinite && dx::abs(val) < mx::maxint,
            dx::copysign(x - fr, finite), val);
    }

    template <typename E, typename A, rounding_flags R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_vector<E, A> val, rounding_t<R>) noexcept {
        constexpr auto opt = rounding_v<R>;
        static_assert(opt);
        if constexpr (opt.has(rounding::to_zero | rounding::no_exc)) {
            return dx::trunc(val, rounding::no_exc);
        } else if constexpr (opt.has(rounding::to_zero)) {
            return dx::trunc(val);
        } else if constexpr (opt.has(rounding::to_pos_inf | rounding::no_exc)) {
            return dx::ceil(val, rounding::no_exc);
        } else if constexpr (opt.has(rounding::to_pos_inf)) {
            return dx::ceil(val);
        } else if constexpr (opt.has(rounding::to_neg_inf | rounding::no_exc)) {
            return dx::floor(val, rounding::no_exc);
        } else if constexpr (opt.has(rounding::to_neg_inf)) {
            return dx::floor(val);
        } else {
            // R to nearest int, tie to nearest even
            auto const isfinite = dx::isfinite(val);
            auto const finite = dx::select(isfinite, val, dx::zero);
            auto x = finite;
            auto i = dx::floor(x);
            auto one = dx::broadcast<E, A>(dx::one);
            auto fr = x - i;
            x = dx::add(x, fr > mx::half, x, one);

            // there are bit tricks alternatives to casting available but
            // they usually just add more instructions
            using sint = signed_representation_t<E>;
            auto const isodd = (dx::element_cast<sint>(i) & dx::one) == dx::one;
            i = dx::add(i, isodd, i, one);
            return dx::select(isfinite && dx::abs(val) < mx::maxint,
                dx::copysign(i, finite), val);
        }
    }

    template <simd_vector S, typename M, simd_vector T>
    requires mx::maskable_operator<round_t, S, M, T> &&
        mx::canonical_operator_args<S, M, T> && requires(S src, M mask, T val) {
            round(internal::abi<T>, src, mask, val);
        }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(S src, M mask, T val) noexcept {
        return round(internal::abi<T>, src, mask, val);
    }

    template <simd_vector S, typename M, simd_vector T>
    requires mx::maskable_operator<round_t, S, M, T> &&
        (!mx::canonical_operator_args<S, M, T>) &&
        requires(S src, M mask, T val) { round(src, mask, val); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(S src, M mask, T val) noexcept {
        return round(src, mask, val);
    }

    template <typename M, simd_vector T>
    requires mx::maskable_zoperator<round_t, M, T> &&
        mx::canonical_zoperator_args<round_t, M, T> && requires(M mask, T val) {
            round(internal::abi<T>, dx::zero, mask, val);
        }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(M mask, T val) noexcept {
        return round(internal::abi<T>, dx::zero, mask, val);
    }

    template <typename M, simd_vector T>
    requires mx::maskable_zoperator<round_t, M, T> &&
        (!mx::canonical_zoperator_args<round_t, M, T>) &&
        requires(M mask, T val) { round(dx::zero, mask, val); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(M mask, T val) noexcept {
        return round(dx::zero, mask, val);
    }

    template <simd_vector S, typename M, simd_vector T, rounding_flags R>
    requires mx::maskable_operator<round_t, S, M, T, rounding_t<R>> &&
        mx::canonical_operator_args<S, M, T, rounding_t<R>> &&
        requires(S src, M mask, T val) {
            round(internal::abi<T>, src, mask, val, rounding_v<R>);
        }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(
        S src, M mask, T val, rounding_t<R> flags) noexcept {
        return round(internal::abi<T>, src, mask, val, flags);
    }

    template <simd_vector S, typename M, simd_vector T, rounding_flags R>
    requires mx::maskable_operator<round_t, S, M, T> &&
        (!mx::canonical_operator_args<S, M, T>) &&
        requires(S src, M mask, T val) { round(src, mask, val, rounding_v<R>); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(
        S src, M mask, T val, rounding_t<R> flags) noexcept {
        return round(src, mask, val, flags);
    }

    template <typename M, simd_vector T, rounding_flags R>
    requires mx::maskable_zoperator<round_t, M, T> &&
        mx::canonical_zoperator_args<round_t, M, T> && requires(M mask, T val) {
            round(internal::abi<T>, dx::zero, mask, val, rounding_v<R>);
        }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(
        M mask, T val, rounding_t<R> flags) noexcept {
        return round(internal::abi<T>, dx::zero, mask, val, flags);
    }

    template <typename M, simd_vector T, rounding_flags R>
    requires mx::maskable_zoperator<round_t, M, T> &&
        (!mx::canonical_zoperator_args<round_t, M, T>) &&
        requires(M mask, T val) { round(dx::zero, mask, val, rounding_v<R>); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(
        M mask, T val, rounding_t<R> flags) noexcept {
        return round(dx::zero, mask, val, flags);
    }

public:
    template <simd_abi A, simd_element_for<A> E>
    requires floating_point<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_cmath_round<basic_vector<E, A>>) {
            if consteval {
                return fallback(val);
            } else {
                return round(internal::abi<A>, val);
            }
        } else {
            return fallback(val);
        }
    }

    template <simd_abi A, simd_element_for<A> E>
    requires (!floating_point<E>) &&
        unqualified_canonical_cmath_round<basic_vector<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val) noexcept {
        return round(internal::abi<A>, val);
    }

    template <extended_vector T>
    requires unqualified_cmath_round<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val) noexcept {
        if constexpr (unqualified_extended_cmath_round<T>) {
            return round(val);
        } else {
            return operator()(dx::to_canonical(val));
        }
    }

    template <simd_abi A, simd_element_for<A> E, rounding_flags R>
    requires floating_point<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val, rounding_t<R> flags) noexcept {
        if constexpr (unqualified_canonical_round<basic_vector<E, A>, R>) {
            if consteval {
                return fallback(val, flags);
            } else {
                return round(internal::abi<A>, val, flags);
            }
        } else {
            return fallback(val, flags);
        }
    }

    template <simd_abi A, simd_element_for<A> E, rounding_flags R>
    requires (!floating_point<E>) &&
        unqualified_canonical_round<basic_vector<E, A>, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val, rounding_t<R> flags) noexcept {
        return round(internal::abi<A>, val, flags);
    }

    template <extended_vector T, rounding_flags R>
    requires unqualified_round<T, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val, rounding_t<R> flags) noexcept {
        if constexpr (unqualified_extended_round<T, R>) {
            return round(val, flags);
        } else {
            return operator()(dx::to_canonical(val), flags);
        }
    }

    using mx::masked_operation<round_t>::operator();
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::round_t round{};
}
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
