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
#  include "dpl/core/concepts/basic_type.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_type.h"
#  include "dpl/core/constants/nan.h"
#  include "dpl/core/constants/zero.h"
#  include "dpl/core/operations/arithmetic.h" // IWYU pragma: keep
#  include "dpl/core/operations/bitwise.h"    // IWYU pragma: keep
#  include "dpl/core/operations/compare.h"    // IWYU pragma: keep
#  include "dpl/core/operations/negate.h"
#  include "dpl/core/operations/select.h"
#  include "dpl/core/utility/fpfix.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::internal {

void fixup(...) noexcept = delete;

template <typename A, typename L, typename R, typename T>
concept unqualified_fixup = floating_point_simd<L> && floating_point_simd<R> &&
    requires(L left, R right, T conditions) {
        fixup(internal::abi<A>, left, right, conditions);
    };

struct fixup_t {
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
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto match_disjoint(
        basic_simd<E, A> src, auto flags) noexcept {
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
        basic_simd<E, A> src, auto flags) noexcept {
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
    static constexpr auto match(basic_simd<E, A> src, auto flags) noexcept {
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
    static constexpr auto fallback(
        basic_simd<E, A> src, basic_simd<E, A> result, F conditions) noexcept {
        auto mask = dx::broadcast<E, A>(false);
        fpfix::template_for(
            [&](auto flags, auto val) {
                using simd = basic_simd<E, A>;
                if constexpr (val == fpfix::signed_inf) {
                    result = dx::select(match(src, flags),
                        dx::negate(dx::signbit(src), dx::infinity_v<simd>),
                        result);
                } else if constexpr (val == fpfix::revert) {
                    result = dx::select(match(src, flags), src, result);
                } else if constexpr (same_as<decltype(val),
                                         decltype(dx::nan)>) {
                    result = dx::bit_fill(match(src, flags), result);
                } else if constexpr (same_as<decltype(val),
                                         decltype(dx::zero)>) {
                    result = dx::bit_drop(match(src, flags), result);
                } else {
                    result = dx::select(match(src, flags), val, result);
                }
            },
            conditions);
        return result;
    }

public:
    template <basic_simd_type T,
        fpfix::condition_set_for<typename T::value_type> F>
    requires floating_point_simd<T> &&
        fpfix::result_subset_of<F, typename T::value_type, dx::nan, dx::zero,
            -dx::zero, dx::infinity, -dx::infinity>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T src, T result, F conditions) noexcept {
        if constexpr (unqualified_fixup<T, T, T, F>) {
            if not consteval {
                return fixup(internal::abi<T>, src, result, conditions);
            } else {
                return fallback(src, result, conditions);
            }
        } else {
            return fallback(src, result, conditions);
        }
    }

    template <floating_point_simd L, common_arithmetic_simd_with<L> R,
        fpfix::condition_set_for<common_arithmetic_type_t<L, R>> F>
    requires unqualified_fixup<common_abi_t<L, R>, L, R, F>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L src, R result, F conditions) noexcept {
        using A = common_abi_t<L, R>;
        return fixup(internal::abi<A>, src, result, conditions);
    }

    template <floating_point_simd L, common_arithmetic_simd_with<L> R,
        fpfix::condition_set_for<common_arithmetic_type_t<L, R>> F>
    requires fpfix::result_subset_of<F,
                 basic_element_t<common_arithmetic_type_t<L, R>>, dx::nan,
                 dx::zero, -dx::zero, dx::infinity, -dx::infinity> &&
        (!basic_simd_type<L> || !basic_simd_type<R> ||
            !same_abi_simd_as<L, R>) &&
        (!unqualified_fixup<common_abi_t<L, R>, L, R, F>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L src, R result, F conditions) noexcept
        -> equivalent_simd_as<common_arithmetic_simd_t<L, R>> auto {
        return operator()(
            dx::to_basic_type(src), dx::to_basic_type(result), conditions);
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
