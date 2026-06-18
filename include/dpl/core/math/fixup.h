// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/isfinite.h"
#include "dpl/core/math/isinf.h"
#include "dpl/core/math/isnan.h"
#include "dpl/core/math/isnanq.h"
#include "dpl/core/math/isnans.h"
#include "dpl/core/math/signbit.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_vector.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/maskable/accumulation.h"
#  include "dpl/core/dispatch/operation/math.h"
#  include "dpl/core/immediate/constants/nan.h"
#  include "dpl/core/immediate/constants/zero.h"
#  include "dpl/core/operations/arithmetic.h" // IWYU pragma: keep
#  include "dpl/core/operations/bitwise.h"    // IWYU pragma: keep
#  include "dpl/core/operations/compare.h"    // IWYU pragma: keep
#  include "dpl/core/operations/select.h"
#  include "dpl/core/utility/fpfix.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::internal {
void fixup(...) noexcept = delete;
struct DPL_EMPTY_BASES fixup_t :
    private math_operation_base<fixup_t>,
    private maskable_accumulation_base<fixup_t> {
    using math_operation_base<fixup_t>::operator();
    using maskable_accumulation_base<fixup_t>::operator();
};

template <typename C, typename E>
concept fixflags_for = fpfix::condition_set_for<C, E> &&
    fpfix::result_subset_of<C, E, dx::nan, dx::zero, -dx::zero, dx::infinity,
        -dx::infinity>;

template <>
struct operation_signature<fixup_t> {
    template <simd_vector L, simd_vector R,
        fpfix::condition_set_for<simd_element_type_t<L>> C>
    requires same_as<simd_element_type_t<L>, simd_element_type_t<R>>
    static consteval void operator()(L&&, R&&, C) noexcept {}
};

template <typename T, typename C>
concept unqualified_canonical_fixup =
    fixflags_for<C, simd_element_type_t<T>> && requires {
        {
            fixup(internal::abi<T>, internal::declarg<T>(),
                internal::declarg<T>(), internal::declarg<C>())
        } -> same_as<T>;
    };

template <typename T, typename C,
    typename M = basic_mask<simd_element_type_t<T>, simd_abi_type_t<T>>>
concept unqualified_canonical_mfixup =
    cpo_invocable<fixup_t, T, T, C> && requires {
        {
            fixup(internal::abi<cpo_result_t<fixup_t, T, T, C>>,
                internal::declarg<T>(), internal::declarg<M>(),
                internal::declarg<T>(), internal::declarg<C>())
        } -> same_as<cpo_result_t<fixup_t, T, T, C>>;
    };

template <typename T, typename C,
    typename M = basic_mask<simd_element_type_t<T>, simd_abi_type_t<T>>>
concept unqualified_canonical_zmfixup =
    cpo_invocable<fixup_t, T, T, C> && requires {
        {
            fixup(internal::abi<cpo_result_t<fixup_t, T, T, C>>, dx::zero,
                internal::declarg<M>(), internal::declarg<T>(),
                internal::declarg<T>(), internal::declarg<C>())
        } -> same_as<cpo_result_t<fixup_t, T, T, C>>;
    };

template <>
struct canonical_impl<fixup_t> {
public:
    template <typename T>
    using mask_t DPL_NODEBUG =
        basic_mask<simd_element_type_t<T>, simd_abi_type_t<T>>;

public:
    template <canonical_vector T, fixflags_for<simd_element_type_t<T>> C>
    requires unqualified_canonical_fixup<T, C>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> lhs, T rhs, C tokens) noexcept {
        return fixup(internal::abi<T>, lhs, rhs, tokens);
    }

    template <canonical_vector T, fixflags_for<simd_element_type_t<T>> C>
    requires unqualified_canonical_mfixup<T, C>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> lhs, mask_t<T> mask, T rhs, C tokens) noexcept {
        return fixup(internal::abi<T>, lhs, mask, rhs, tokens);
    }

    template <fixed_width_vector T, const_mask_for<T> M,
        fixflags_for<simd_element_type_t<T>> C>
    requires canonical_vector<T> &&
        unqualified_canonical_mfixup<T, C, launder_cmask_t<T, M>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> lhs, M cmask, T rhs, C tokens) noexcept {
        return fixup(
            internal::abi<T>, lhs, dx::to_const_mask<T>(cmask), rhs, tokens);
    }

    template <canonical_vector T, fixflags_for<simd_element_type_t<T>> C>
    requires unqualified_canonical_zmfixup<T, C>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(dx::zero_t zero, mask_t<T> mask,
        type_identity_t<T> lhs, T rhs, C tokens) noexcept {
        return fixup(internal::abi<T>, zero, mask, lhs, rhs, tokens);
    }

    template <fixed_width_vector T, const_mask_for<T> M,
        fixflags_for<simd_element_type_t<T>> C>
    requires canonical_vector<T> &&
        unqualified_canonical_zmfixup<T, C, launder_cmask_t<T, M>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(dx::zero_t zero, M cmask,
        type_identity_t<T> lhs, T rhs, C tokens) noexcept {
        return fixup(internal::abi<T>, zero, dx::to_const_mask<T>(cmask), lhs,
            rhs, tokens);
    }
};

template <typename S, typename T, typename C>
concept unqualified_extended_fixup = equivalent_vector_with<S, T> &&
    fixflags_for<C, simd_element_type_t<T>> && requires {
        {
            fixup(internal::declarg<S>(), internal::declarg<T>(),
                internal::declarg<C>())
        } -> equivalent_vector_with<T>;
    };

template <typename S, typename M, typename T, typename C>
concept unqualified_extended_mfixup =
    equivalent_vector_with<S, T> && cpo_invocable<fixup_t, S, T, C> &&
    fixflags_for<C, simd_element_type_t<T>> && requires {
        {
            fixup(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>(), internal::declarg<C>())
        } -> equivalent_vector_with<cpo_result_t<fixup_t, S, T, C>>;
    };

template <typename S, typename M, typename T, typename C>
concept unqualified_extended_zmfixup =
    equivalent_vector_with<S, T> && cpo_invocable<fixup_t, S, T, C> &&
    fixflags_for<C, simd_element_type_t<T>> && requires {
        {
            fixup(dx::zero, internal::declarg<M>(), internal::declarg<S>(),
                internal::declarg<T>(), internal::declarg<C>())
        } -> equivalent_vector_with<cpo_result_t<fixup_t, S, T, C>>;
    };

template <>
struct extended_impl<fixup_t> {
public:
    template <simd_vector S, equivalent_vector_with<S> T,
        fixflags_for<simd_element_type_t<T>> C>
    requires unqualified_extended_fixup<S, T, C>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& lhs, T&& rhs, C tokens) {
        return fixup(__DPL forward<S>(lhs), __DPL forward<T>(rhs), tokens);
    }

    template <simd_vector S, exact_mask_for<S> M, equivalent_vector_with<S> T,
        fixflags_for<simd_element_type_t<T>> C>
    requires unqualified_extended_mfixup<S, M, T, C>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        S&& lhs, M&& mask, T&& rhs, C tokens) noexcept {
        return fixup(__DPL forward<S>(lhs), __DPL forward<M>(mask),
            __DPL forward<T>(rhs), tokens);
    }

    template <simd_vector S, const_mask_for<S> M, equivalent_vector_with<S> T,
        fixflags_for<simd_element_type_t<T>> C>
    requires unqualified_extended_mfixup<S, launder_cmask_t<S, M>, T, C>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        S&& lhs, M cmask, T&& rhs, C tokens) noexcept {
        return fixup( __DPL forward<S>(lhs), dx::to_const_mask<S>(cmask),
            __DPL forward<T>(rhs), tokens);
    }

    template <simd_vector S, exact_mask_for<S> M, equivalent_vector_with<S> T,
        fixflags_for<simd_element_type_t<T>> C>
    requires unqualified_extended_zmfixup<S, M, T, C>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, S&& lhs, T&& rhs, C tokens) noexcept {
        return fixup(zero, __DPL forward<M>(mask),__DPL forward<S>(lhs),
            __DPL forward<T>(rhs), tokens);
    }

    template <simd_vector S, const_mask_for<S> M, equivalent_vector_with<S> T,
        fixflags_for<simd_element_type_t<T>> C>
    requires unqualified_extended_zmfixup<S, launder_cmask_t<S, M>, T, C>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M cmask, S&& lhs, T&& rhs, C tokens) noexcept {
        return fixup(zero, dx::to_const_mask<S>(cmask), __DPL forward<S>(lhs),
            __DPL forward<T>(rhs), tokens);
    }
};

template <>
struct fallback_impl<fixup_t> {
public:
    // TODO enable only if canonical is provided
    template <floating_point E, simd_abi A, fixflags_for<E> C>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_vector<E, A> src,
        basic_vector<E, A> result, C conditions) noexcept {
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

private:
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
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL match_disjoint(
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
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL match_posneg(
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
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL match(
        basic_vector<E, A> src, auto flags) noexcept {
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
};
} // namespace datapar::internal

DPL_EXPORT namespace fpfix = datapar::fpfix; // NOLINT

namespace datapar {
inline namespace cpo {
inline constexpr internal::fixup_t fixup{};
}
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
