// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/internal/floating_point_simd.h"
#include "dpl/core/math/internal/masked_op.h"
#include "dpl/core/math/isfinite.h"
#include "dpl/core/math/isinf.h"
#include "dpl/core/math/isnan.h"
#include "dpl/core/math/isnanq.h"
#include "dpl/core/math/isnans.h"
#include "dpl/core/math/signbit.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/basic_type.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_traits.h"
#  include "dpl/core/concepts/simd_vector.h"
#  include "dpl/core/constants/nan.h"
#  include "dpl/core/constants/zero.h"
#  include "dpl/core/operations/arithmetic.h" // IWYU pragma: keep
#  include "dpl/core/operations/bitwise.h"    // IWYU pragma: keep
#  include "dpl/core/operations/compare.h"    // IWYU pragma: keep
#  include "dpl/core/operations/select.h"
#  include "dpl/core/utility/fpfix.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::internal {
void fixup(...) noexcept = delete;

struct fixup_t;

template <typename L, typename R, typename C, typename A = common_abi_t<L, R>>
concept unqualified_canonical_fixup = requires(L lhs, R rhs, C conditions) {
    {
        fixup(internal::abi<A>, lhs, rhs, conditions)
    } -> canonical_arithmetic_result<L, R, A>;
};

template <typename L, typename R, typename C, typename A = common_abi_t<L, R>>
concept unqualified_extended_fixup = requires(L lhs, R rhs, C conditions) {
    { fixup(lhs, rhs, conditions) } -> extended_operation_vector<A>;
};

template <typename L, typename R, typename C, typename A = common_abi_t<L, R>>
concept expression_fixup = (simd_expression<L> || simd_expression<R>) &&
    invocable<fixup_t, simd_expression_result_t<L>, simd_expression_result_t<R>,
        C>;

template <typename L, typename R, typename C, typename A = common_abi_t<L, R>>
concept decayable_fixup =
    decayable_vector_for<L, operation_category::lane_agnostic> &&
    decayable_vector_for<R, operation_category::lane_agnostic> &&
    regular_invocable<fixup_t, canonical_type_t<L>, canonical_type_t<R>, C>;

template <typename L, typename R, typename C, typename A = common_abi_t<L, R>>
concept extended_fixup = unqualified_extended_fixup<L, R, C, A> ||
    expression_fixup<L, R, C, A> || decayable_fixup<L, R, C, A>;

struct fixup_t : private mx::masked_assignment<fixup_t> {
private:
    friend mx::masked_assignment<fixup_t>;
    struct sets {
        static constexpr auto finite_gezero =
            fpfix::positive | fpfix::zero | fpfix::one;
        static constexpr auto finite_lezero = fpfix::negative | fpfix::zero;
        static constexpr auto finite_gtzero = fpfix::positive | fpfix::one;
        static constexpr auto finite_ltzero = fpfix::negative;
        static constexpr auto full_gezero = finite_gezero | fpfix::pos_inf;
        static constexpr auto full_lezero = finite_lezero | fpfix::neg_inf;
        static constexpr auto nonfinite = fpfix::nan | fpfix::infinity;
        static constexpr auto posneg = fpfix::positive | fpfix::negative;
    };

    template <floating_point E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto match_disjoint(
        basic_vector<E, A> src, auto flags) noexcept {
        static_assert((flags & fpfix::nan) != fpfix::nan);
        static_assert((flags & fpfix::infinity) != fpfix::infinity);
        static_assert((flags & fpfix::finite) != fpfix::finite);
        static_assert(!(flags & fpfix::negative));
        using flag_type = decltype(flags);
        constexpr flag_type F{};

        if constexpr (F & fpfix::snan) {
            if constexpr (F == fpfix::snan) {
                return dx::isnans(src);
            } else {
                return dx::isnans(src) || match_disjoint(src, F & ~fpfix::snan);
            }
        } else if constexpr (F & fpfix::qnan) {
            if constexpr (F == fpfix::qnan) {
                return dx::isnanq(src);
            } else {
                return dx::isnanq(src) || match_disjoint(src, F & ~fpfix::qnan);
            }
        } else if constexpr (F & fpfix::positive) {
            auto if_pos = [&]() {
                if constexpr (F & fpfix::zero) {
                    return dx::cmpge(src, dx::zero);
                } else {
                    return dx::cmpgt(src, dx::zero);
                }
            };
            if constexpr (!(F & fpfix::one))
                if_pos &= dx::cmpneq(src, dx::one);

            if constexpr (!(F & fpfix::pos_inf))
                if_pos &= dx::cmplt(src, dx::infinity);

            if constexpr (auto remainder = F & ~sets::full_gezero; remainder) {
                return if_pos || match_disjoint(src, remainder);
            } else {
                return if_pos;
            }
        } else if constexpr (F & fpfix::pos_inf) {
            if constexpr (F == fpfix::pos_inf) {
                return src == dx::infinity;
            } else {
                return src == dx::infinity ||
                    match_disjoint(src, F & ~fpfix::pos_inf);
            }
        } else if constexpr (F & fpfix::neg_inf) {
            if constexpr (F == fpfix::neg_inf) {
                return src == -dx::infinity;
            } else {
                return src == -dx::infinity ||
                    match_disjoint(src, F & ~fpfix::neg_inf);
            }
        } else if constexpr (F & fpfix::zero) {
            if constexpr (F == fpfix::zero) {
                return dx::cmpeq(src, dx::zero);
            } else {
                return dx::cmpeq(src, dx::zero) ||
                    match_disjoint(src, F & ~fpfix::zero);
            }
        } else if constexpr (F & fpfix::one) {
            if constexpr (F == fpfix::one) {
                return dx::cmpeq(src, dx::one);
            } else {
                return dx::cmpeq(src, dx::one) ||
                    match_disjoint(src, F & ~fpfix::one);
            }
        } else {
            return dx::broadcast<E, A>(false);
        }
    }

    template <floating_point E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto match_posneg(
        basic_vector<E, A> src, auto flags) noexcept {
        using flag_type = decltype(flags);
        constexpr flag_type F{};
        static_assert((F & fpfix::finite) != fpfix::finite);
        static_assert((F & sets::posneg) == sets::posneg);

        auto initial = [&]() {
            if constexpr (!(F & fpfix::zero) && !(F & fpfix::one)) {
                return dx::cmpneq(src, dx::one) && dx::cmpneq(src, dx::zero);
            } else if constexpr (!(F & fpfix::zero)) {
                return dx::cmpneq(src, dx::zero);
            } else if constexpr (!(F & fpfix::one)) {
                return dx::cmpneq(src, dx::one);
            } else {
                return dx::broadcast<E, A>(true);
            }
        }();

        constexpr auto remainder =
            F & ~(sets::posneg | fpfix::one | fpfix::zero);
        if constexpr (remainder == sets::nonfinite) {
            return initial;
        } else if constexpr ((remainder & fpfix::nan) == fpfix::nan) {
            constexpr auto nonnan = remainder & ~fpfix::nan;
            static_assert(nonnan != fpfix::infinity);
            if constexpr (!nonnan) {
                return initial && !dx::isinf(src);
            } else if constexpr (nonnan == fpfix::pos_inf) {
                return initial && dx::cmpneq(src, -dx::infinity);
            } else {
                return initial && dx::cmpneq(src, dx::infinity);
            }
        } else if constexpr ((remainder & fpfix::infinity) == fpfix::infinity) {
            constexpr auto noninf = remainder & ~fpfix::infinity;
            static_assert(noninf != fpfix::nan);
            if constexpr (!noninf) {
                return initial && !dx::isnan(src);
            } else if constexpr (noninf == fpfix::snan) {
                return initial && !dx::isnanq(src);
            } else {
                return initial && !dx::isnans(src);
            }
        } else {
            constexpr auto nonfin = remainder & sets::nonfinite;
            if constexpr (!nonfin) {
                return initial && dx::isfinite(src);
            } else if constexpr (nonfin == (fpfix::pos_inf | fpfix::snan)) {
                return initial && dx::cmpneq(src, -dx::infinity) &&
                    !dx::isnanq(src);
            } else if constexpr (nonfin == (fpfix::pos_inf | fpfix::qnan)) {
                return initial && dx::cmpneq(src, -dx::infinity) &&
                    !dx::isnans(src);
            } else if constexpr (nonfin == (fpfix::neg_inf | fpfix::snan)) {
                return initial && dx::cmpneq(src, dx::infinity) &&
                    !dx::isnanq(src);
            } else if constexpr (nonfin == (fpfix::neg_inf | fpfix::qnan)) {
                return initial && dx::cmpneq(src, dx::infinity) &&
                    !dx::isnans(src);
            } else {
                return (initial && dx::isfinite(src)) ||
                    match_disjoint(src, nonfin);
            }
        }
    }

    template <floating_point E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto match(basic_vector<E, A> src, auto flags) noexcept {
        using flag_type = decltype(flags);
        constexpr flag_type F{};
        if constexpr (F == fpfix::all) {
            return dx::broadcast<E, A>(true);
        } else if constexpr (F == fpfix::none) {
            return dx::broadcast<E, A>(false);
        } else if constexpr (F == (fpfix::infinity | fpfix::finite)) {
            return !dx::isnan(src);
        } else if constexpr (F == sets::nonfinite) {
            return !dx::isfinite(src);
        } else if constexpr (F == (fpfix::finite | fpfix::nan)) {
            return !dx::isinf(src);
        } else if constexpr (F == fpfix::nan) {
            return dx::isnan(src);
        } else if constexpr (F == fpfix::infinity) {
            return dx::isinf(src);
        } else if constexpr (F == fpfix::finite) {
            return dx::isfinite(src);
        } else if constexpr ((F & fpfix::finite) == fpfix::finite) {
            return dx::isfinite(src) || match(src, F & ~fpfix::finite);
        } else if constexpr ((F & sets::posneg) == sets::posneg) {
            return match_posneg(src, F);
        } else if constexpr ((F & sets::finite_gtzero) == sets::finite_gtzero) {
            static_assert(!(F & fpfix::negative));
            auto initial = [&]() {
                if constexpr (F & fpfix::zero)
                    return dx::cmpge(src, dx::zero);
                else
                    return dx::cmpgt(src, dx::zero);
            }();

            if constexpr (!(F & fpfix::pos_inf))
                initial &= dx::cmplt(src, dx::infinity);

            if constexpr (!(F & ~sets::full_gezero))
                return initial;
            else
                return initial || match_disjoint(src, F & ~sets::full_gezero);
        } else if constexpr ((F & sets::finite_ltzero) == sets::finite_ltzero) {
            static_assert(!(F & fpfix::positive));
            auto initial = [&]() {
                if constexpr (F & fpfix::zero)
                    return dx::cmple(src, dx::zero);
                else
                    return dx::cmplt(src, dx::zero);
            }();

            if constexpr (!(F & fpfix::neg_inf))
                initial &= dx::cmpgt(src, -dx::infinity);

            if constexpr (!(F & ~sets::full_gezero))
                return initial;
            else
                return initial || match_disjoint(src, F & ~sets::full_gezero);
        } else if constexpr ((F & sets::nonfinite) == sets::nonfinite) {
            static_assert(!(F & fpfix::negative));
            return !dx::isfinite(src) ||
                match_disjoint(src, F & ~sets::nonfinite);
        } else if constexpr ((F & fpfix::nan) == fpfix::nan) {
            static_assert(!(F & fpfix::negative));
            return dx::isnan(src) || match_disjoint(src, F & ~fpfix::nan);
        } else if constexpr ((F & fpfix::infinity) == fpfix::infinity) {
            static_assert(!(F & fpfix::negative));
            return dx::isinf(src) || match_disjoint(src, F & ~fpfix::infinity);
        } else {
            return match_disjoint(src, F);
        }
    }

    template <floating_point E, simd_abi A, fpfix::condition_set F>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto fallback(basic_vector<E, A> src,
        basic_vector<E, A> result, F conditions) noexcept {
        auto mask = dx::broadcast<E, A>(false);
        fpfix::template_for(
            [&](auto flags, auto val) {
                using simd = basic_vector<E, A>;
                if constexpr (val == fpfix::signed_inf) {
                    result = dx::select(match(src, flags),
                        dx::negate(dx::signbit(src), dx::infinity_v<simd>),
                        result);
                } else if constexpr (val == fpfix::revert) {
                    result = dx::select(match(src, flags), src, result);
                } else if constexpr (same_as<decltype(val),
                                         decltype(dx::nan)>) {
                    result =
                        dx::select(match(src, flags), dx::all_bits, result);
                } else if constexpr (same_as<decltype(val),
                                         decltype(dx::zero)>) {
                    result = dx::select(match(src, flags), dx::zero, result);
                } else {
                    result = dx::select(match(src, flags), val, result);
                }
            },
            conditions);
        return result;
    }

    template <simd_vector S, typename M, simd_vector R,
        fpfix::condition_set_for<typename R::value_type> F>
    requires mx::canonical_masked_math_assignment<fixup_t, S, M, R, F> &&
        requires(S src, M mask, R val, F flags) {
            fixup(internal::abi<common_abi_t<S, R>>, src, mask, val, flags);
        }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(
        S src, M mask, R val, F flags) noexcept {
        return fixup(internal::abi<common_abi_t<S, R>>, src, mask, val, flags);
    }

    template <simd_vector S, typename M, simd_vector R,
        fpfix::condition_set_for<typename R::value_type> F>
    requires mx::extended_masked_math_assignment<fixup_t, S, M, R, F> &&
        requires(
            S src, M mask, R val, F flags) { fixup(src, mask, val, flags); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(
        S src, M mask, R val, F flags) noexcept {
        return fixup(src, mask, val, flags);
    }

    template <typename M, simd_vector S, simd_vector R,
        fpfix::condition_set_for<typename R::value_type> F>
    requires mx::canonical_masked_math_zassignment<fixup_t, M, S, R, F> &&
        requires(M mask, S src, R val, F flags) {
            fixup(internal::abi<common_abi_t<S, R>>, dx::zero, mask, src, val,
                flags);
        }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(
        M mask, S src, R val, F flags) noexcept {
        return fixup(
            internal::abi<common_abi_t<S, R>>, dx::zero, mask, src, val, flags);
    }

    template <typename M, simd_vector S, simd_vector R,
        fpfix::condition_set_for<typename R::value_type> F>
    requires mx::extended_masked_math_zassignment<fixup_t, M, S, R, F> &&
        requires(M mask, S src, R val, F flags) {
            fixup(dx::zero, mask, src, val, flags);
        }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(
        M mask, S src, R val, F flags) noexcept {
        return fixup(dx::zero, mask, src, val, flags);
    }

public:
    template <simd_abi A, simd_element_for<A> E, fpfix::condition_set_for<E> F>
    requires floating_point<E> &&
        fpfix::result_subset_of<F, E, dx::nan, dx::zero, -dx::zero,
            dx::infinity, -dx::infinity>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_vector<E, A> src,
        basic_vector<E, A> result, F conditions) noexcept {
        if constexpr (unqualified_canonical_fixup<basic_vector<E, A>,
                          basic_vector<E, A>, F>) {
            if consteval {
                return fallback(src, result, conditions);
            } else {
                return fixup(internal::abi<A>, src, result, conditions);
            }
        } else {
            return fallback(src, result, conditions);
        }
    }

    template <simd_abi SA, simd_element_for<SA> E, common_abi_with<SA> RA,
        fpfix::condition_set_for<E> F>
    requires simd_element_for<E, RA> &&
        fpfix::result_subset_of<F, E, dx::nan, dx::zero, -dx::zero,
            dx::infinity, -dx::infinity> &&
        (different_from<SA, RA> || !floating_point<E>) &&
        unqualified_canonical_fixup<basic_vector<E, SA>, basic_vector<E, RA>, F>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(basic_vector<E, SA> src,
        basic_vector<E, RA> result, F conditions) noexcept {
        return fixup(
            internal::abi<common_abi_t<SA, RA>>, src, result, conditions);
    }

    template <extended_vector S, extended_vector R,
        fpfix::condition_set_for<typename R::value_type> F>
    requires same_as<typename S::value_type, typename R::value_type> &&
        fpfix::result_subset_of<F, typename R::value_type, dx::nan, dx::zero,
            -dx::zero, dx::infinity, -dx::infinity> &&
        extended_fixup<S, R, F>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, R result, F conditions) noexcept {
        if constexpr (unqualified_extended_fixup<S, R, F>) {
            return fixup(src, result, conditions);
        } else if constexpr (expression_fixup<S, R, F>) {
            return operator()(
                dx::evaluate(src), dx::evaluate(result), conditions);
        } else {
            return operator()(
                dx::to_canonical(src), dx::to_canonical(result), conditions);
        }
    }

    using mx::masked_assignment<fixup_t>::operator();
};
} // namespace datapar::internal

DPL_EXPORT namespace fpfix = datapar::fpfix; // NOLINT

namespace datapar {
inline namespace cpo {
inline constexpr internal::fixup_t fixup{};
}
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
