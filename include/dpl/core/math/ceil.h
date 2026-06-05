// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/copysign.h"
#include "dpl/core/math/internal/constants.h"
#include "dpl/core/math/internal/floating_point_simd.h"
#include "dpl/core/math/internal/masked_op.h"
#include "dpl/core/math/isfinite.h"
#include "dpl/core/math/trunc.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/canonical.h"
#  include "dpl/core/concepts/extended.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/constants/one.h"
#  include "dpl/core/operations/arithmetic.h"
#  include "dpl/core/operations/compare.h"
#  include "dpl/core/operations/select.h"
#  include "dpl/core/utility/rounding.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

struct ceil_t;

template <typename T>
concept unqualified_canonical_ceil = requires(T val) {
    {
        round(internal::abi<T>, val, rounding::to_pos_inf)
    } -> canonical_arithmetic_result<T, T, typename T::abi_type>;
};

template <typename T>
concept unqualified_extended_ceil = requires(T val) {
    {
        round(val, rounding::to_pos_inf)
    } -> vector_with_common_abi<typename T::abi_type>;
};

template <typename T>
concept expression_ceil = simd_expression<T> &&
    regular_invocable<ceil_t, simd_expression_result_t<T>>;

template <typename T>
concept decayable_ceil =
    decayable_vector_for<T, operation_category::lane_agnostic> &&
    regular_invocable<ceil_t, canonical_type_t<T>>;

template <typename T>
concept extended_ceil =
    unqualified_extended_ceil<T> || expression_ceil<T> || decayable_ceil<T>;

template <typename T>
concept unqualified_canonical_ceilne = requires(T val) {
    {
        round(internal::abi<T>, val, rounding::to_pos_inf | rounding::no_exc)
    } -> equivalent_simd_type_with<T>;
};

template <typename T>
concept unqualified_extended_ceilne = requires(T val) {
    {
        round(val, rounding::to_pos_inf | rounding::no_exc)
    } -> vector_with_common_abi<typename T::abi_type>;
};

template <typename T>
concept expression_ceilne = simd_expression<T> &&
    regular_invocable<ceil_t, simd_expression_result_t<T>, rounding::no_exc_t>;

template <typename T>
concept decayable_ceilne =
    decayable_vector_for<T, operation_category::lane_agnostic> &&
    regular_invocable<ceil_t, canonical_type_t<T>, rounding::no_exc_t>;

template <typename T>
concept extended_ceilne =
    unqualified_extended_ceil<T> || expression_ceilne<T> || decayable_ceilne<T>;

struct ceil_t : private mx::masked_operation<ceil_t> {
private:
    friend mx::masked_operation<ceil_t>;
    static constexpr auto noexc = rounding::to_pos_inf | rounding::no_exc;
    static constexpr auto exc = rounding::to_pos_inf;

    template <floating_point E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_vector<E, A> val) noexcept {
        auto const isfinite = dx::isfinite(val);
        auto const finite = dx::select(isfinite, val, dx::zero);
        auto fr = finite - dx::trunc(finite);
        fr = dx::subtract(fr, fr > dx::zero, fr, dx::broadcast<E, A>(dx::one));
        return dx::select(isfinite && dx::abs(val) < mx::maxint,
            dx::copysign(finite - fr, finite), val);
    }

    template <simd_vector S, typename M, simd_vector T>
    requires mx::canonical_masked_math_operator<ceil_t, S, M, T> &&
        requires(S src, M mask, T val) {
            round(internal::abi<T>, src, mask, val, ceil_t::exc);
        }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(S src, M mask, T val) noexcept {
        return round(internal::abi<T>, src, mask, val, ceil_t::exc);
    }

    template <simd_vector S, typename M, simd_vector T>
    requires mx::extended_masked_math_operator<ceil_t, S, M, T> &&
        requires(S src, M mask, T val) { round(src, mask, val, ceil_t::exc); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(S src, M mask, T val) noexcept {
        return round(src, mask, val, ceil_t::exc);
    }

    template <typename M, simd_vector T>
    requires mx::canonical_masked_math_zoperator<ceil_t, M, T> &&
        requires(M mask, T val) {
            round(internal::abi<T>, dx::zero, mask, val, ceil_t::exc);
        }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(M mask, T val) noexcept {
        return round(internal::abi<T>, dx::zero, mask, val, ceil_t::exc);
    }

    template <typename M, simd_vector T>
    requires mx::extended_masked_math_zoperator<ceil_t, M, T> &&
        requires(M mask, T val) { round(dx::zero, mask, val, ceil_t::exc); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(M mask, T val) noexcept {
        return round(dx::zero, mask, val, ceil_t::exc);
    }

    template <simd_vector S, typename M, simd_vector T>
    requires mx::canonical_masked_math_operator<ceil_t, S, M, T,
                 rounding::no_exc_t> &&
        requires(S src, M mask, T val) {
            round(internal::abi<T>, src, mask, val, ceil_t::noexc);
        }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(
        S src, M mask, T val, rounding::no_exc_t) noexcept {
        return round(internal::abi<T>, src, mask, val, ceil_t::noexc);
    }

    template <simd_vector S, typename M, simd_vector T>
    requires mx::extended_masked_math_operator<ceil_t, S, M, T,
                 rounding::no_exc_t> &&
        requires(S src, M mask, T val) { round(src, mask, val, ceil_t::noexc); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(
        S src, M mask, T val, rounding::no_exc_t) noexcept {
        return round(src, mask, val, ceil_t::noexc);
    }

    template <typename M, simd_vector T>
    requires mx::canonical_masked_math_zoperator<ceil_t, M, T,
                 rounding::no_exc_t> &&
        requires(M mask, T val) {
            round(internal::abi<T>, dx::zero, mask, val, ceil_t::noexc);
        }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(
        M mask, T val, rounding::no_exc_t) noexcept {
        return round(internal::abi<T>, dx::zero, mask, val, ceil_t::noexc);
    }

    template <typename M, simd_vector T>
    requires mx::extended_masked_math_zoperator<ceil_t, M, T,
                 rounding::no_exc_t> &&
        requires(M mask, T val) { round(dx::zero, mask, val, ceil_t::noexc); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(
        M mask, T val, rounding::no_exc_t) noexcept {
        return round(dx::zero, mask, val, ceil_t::noexc);
    }

public:
    template <simd_abi A, simd_element_for<A> E>
    requires floating_point<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_ceil<basic_vector<E, A>>) {
            if consteval {
                return fallback(val);
            } else {
                return round(internal::abi<A>, val, ceil_t::exc);
            }
        } else {
            return fallback(val);
        }
    }

    template <simd_abi A, simd_element_for<A> E>
    requires (!floating_point<E>) &&
        unqualified_canonical_ceil<basic_vector<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val) noexcept {
        return round(internal::abi<A>, val, ceil_t::exc);
    }

    template <extended_vector T>
    requires extended_ceil<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val) noexcept {
        if constexpr (unqualified_extended_ceil<T>) {
            return round(val, ceil_t::exc);
        } else if constexpr (expression_ceil<T>) {
            return operator()(dx::evaluate(val));
        } else {
            return operator()(dx::to_canonical(val));
        }
    }

    template <simd_abi A, simd_element_for<A> E>
    requires floating_point<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val, rounding::no_exc_t) noexcept {
        if constexpr (unqualified_canonical_ceilne<basic_vector<E, A>>) {
            if consteval {
                return fallback(val);
            } else {
                return round(internal::abi<A>, val, ceil_t::noexc);
            }
        } else {
            return fallback(val);
        }
    }

    template <simd_abi A, simd_element_for<A> E>
    requires (!floating_point<E>) &&
        unqualified_canonical_ceilne<basic_vector<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val, rounding::no_exc_t) noexcept {
        return round(internal::abi<A>, val, ceil_t::noexc);
    }

    template <extended_vector T>
    requires extended_ceilne<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val, rounding::no_exc_t) noexcept {
        if constexpr (unqualified_extended_ceilne<T>) {
            return round(val, ceil_t::noexc);
        } else if constexpr (expression_ceilne<T>) {
            return operator()(dx::evaluate(val), rounding::no_exc);
        } else {
            return operator()(dx::to_canonical(val), rounding::no_exc);
        }
    }

    using mx::masked_operation<ceil_t>::operator();
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::ceil_t ceil{};
}
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
