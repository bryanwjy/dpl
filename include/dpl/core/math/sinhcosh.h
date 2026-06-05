// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/internal/ilogb.h"
#include "dpl/core/math/internal/ldexp.h"
#include "dpl/core/math/internal/masked_op.h"
#include "dpl/core/math/internal/pair.h"
#include "dpl/core/math/internal/polynomial.h"
#include "dpl/core/math/round.h"
#include "dpl/core/math/sign.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/operations/bitwise.h"
#  include "dpl/core/operations/compare.h"
#  include "dpl/core/operations/logical.h"
#  include "dpl/core/operations/select.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
class sinhcosh_base {
private:
    template <floating_point E>
    static constexpr auto polynomial = []() {
        if constexpr (same_as<E, float>) {
            return fmath::polynomial<0.4166637361e-1f, //
                0.8333456703e-2f,                      //
                0.1394256484e-2f,                      //
                0.1980960224e-3f                       //
                >{};
        } else {
            static_assert(same_as<E, double>);
            return fmath::polynomial<0.4166666666666669905e-1,
                0.8333333333333347095e-2, 0.1388888888886763255e-2,
                0.1984126984148071858e-3, 0.2480158735605815065e-4,
                0.2755731892386044373e-5, 0.2755724800902135303e-6,
                0.2505230023782644465e-7, 0.2092255183563157007e-8,
                0.1602472219709932072e-9>{};
        }
    }();

    template <floating_point E, simd_abi A>
    static constexpr auto onesixth = []() {
        return dx::broadcast<A>(static_cast<E>(1) / static_cast<E>(6));
    }();

    template <floating_point E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto is_exp_underflow(basic_vector<E, A> val) noexcept {
        if constexpr (same_as<E, float>) {
            return val < -103.97208f;
        } else {
            static_assert(same_as<E, double>);
            return val < -745.133;
        }
    }

    template <floating_point E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr fmath::pair<E, A>
        DPL_VECTORCALL exp(basic_vector<E, A> arg) noexcept {
        using simdf = basic_vector<E, A>;
        auto const u = arg * fmath::inv_ln2;
        auto const qf =
            dx::round(u, rounding::to_nearest_int | rounding::no_exc);
        constexpr auto nln2 = -fmath::ln2_v<fmath::pair<E, A>>;
        auto s = fmath::single(arg) + qf * nln2.upper;
        s = s + qf * nln2.lower;

        auto const poly = polynomial<E>(s.upper);
        auto t = [&]() {
            if constexpr (same_as<E, float>) {
                auto t = s * poly + onesixth<E, A>;
                t = s * t + fmath::half;
                t = s + fmath::square(s) * t;
                return fmath::fast(dx::one_v<simdf>) + t;
            } else {
                static_assert(same_as<E, double>);
                auto t = fmath::single(dx::broadcast<simdf>(fmath::half)) +
                    s * onesixth<E, A>;
                t = fmath::single(dx::one_v<simdf>) + t * s;
                t = fmath::single(dx::one_v<simdf>) + t * s;
                auto const s4 = [](auto s2) { return s2 * s2; }(
                                    s.upper * s.upper);
                return t + s4;
            }
        }();

        auto const q = dx::element_cast<signed_representation_t<E>>(qf);
        t.upper = fmath::ldexp(fmath::compliance::speed, t.upper, q);
        t.lower = fmath::ldexp(fmath::compliance::speed, t.lower, q);
        auto const underflow = is_exp_underflow(arg);
        t.upper = dx::select(underflow, dx::zero, t.upper);
        t.lower = dx::select(underflow, dx::zero, t.lower);
        return t;
    }

public:
    __DPL_HIDE_FROM_ABI constexpr ~sinhcosh_base() = default;

    template <floating_point E, simd_abi A, typename OpMask>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_vector<E, A>
        DPL_VECTORCALL fallback(
            basic_vector<E, A> const arg, OpMask opmask) noexcept {
        using simdf = basic_vector<E, A>;

        auto const absarg = dx::abs(arg);
        auto const pair = [&](fmath::pair<E, A> p) {
            auto const inv_p = fmath::rcp(p);
            if constexpr (simd_mask<OpMask>) {
                return p -
                    fmath::make_pair(
                        dx::negate(inv_p.upper, opmask, inv_p.upper),
                        dx::negate(inv_p.lower, opmask, inv_p.lower));
            } else if constexpr (dx::all_of(opmask)) {
                return p + inv_p;
            } else if constexpr (dx::none_of(opmask)) {
                return p - inv_p;
            } else {
                return p -
                    fmath::make_pair(
                        dx::negate(inv_p.upper, opmask, inv_p.upper),
                        dx::negate(inv_p.lower, opmask, inv_p.lower));
            }
        }(exp(absarg));

        auto result = (pair.upper + pair.lower) * fmath::half;
        if constexpr (same_as<E, float>) {
            result = dx::select(
                (absarg <= 89.0f) & dx::isfinite(result), result, dx::infinity);
        } else {
            result = dx::select(
                (absarg <= 710.0) & dx::isfinite(result), result, dx::infinity);
        }

        if constexpr (simd_mask<OpMask>) {
            result = dx::sign(result, dx::select(opmask, dx::zero, arg));
        } else if constexpr (dx::none_of(opmask)) {
            result = dx::sign(result, arg);
        } else if constexpr (dx::some_of(opmask)) {
            result = dx::sign(result, dx::select(opmask, dx::zero, arg));
        }

        return dx::select(dx::isnan(arg), dx::all_bits, result);
    }
};

void sinh(...) noexcept = delete;

struct sinh_t;

template <typename T>
concept unqualified_canonical_sinh = requires(T val) {
    {
        sinh(internal::abi<T>, val)
    } -> canonical_arithmetic_result<T, T, typename T::abi_type>;
};

template <typename T>
concept unqualified_extended_sinh = requires(T val) {
    { sinh(val) } -> vector_with_common_abi<typename T::abi_type>;
};

template <typename T>
concept expression_sinh =
    simd_expression<T> && invocable<sinh_t, simd_expression_result_t<T>>;

template <typename T>
concept decayable_sinh =
    decayable_vector_for<T, operation_category::lane_agnostic> &&
    regular_invocable<sinh_t, canonical_type_t<T>>;

template <typename T>
concept extended_sinh =
    unqualified_extended_sinh<T> || expression_sinh<T> || decayable_sinh<T>;

struct sinh_t :
    private internal::sinhcosh_base,
    private mx::masked_operation<sinh_t> {

private:
    friend mx::masked_operation<sinh_t>;

    template <simd_vector S, typename M, simd_vector T>
    requires mx::canonical_masked_math_operator<sinh_t, S, M, T> &&
        requires(
            S src, M mask, T val) { sinh(internal::abi<T>, src, mask, val); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(S src, M mask, T val) noexcept {
        return sinh(internal::abi<T>, src, mask, val);
    }

    template <simd_vector S, typename M, simd_vector T>
    requires mx::extended_masked_math_operator<sinh_t, S, M, T> &&
        requires(S src, M mask, T val) { sinh(src, mask, val); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(S src, M mask, T val) noexcept {
        return sinh(src, mask, val);
    }

    template <typename M, simd_vector T>
    requires mx::canonical_masked_math_zoperator<sinh_t, M, T> &&
        requires(M mask, T val) { sinh(internal::abi<T>, dx::zero, mask, val); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(M mask, T val) noexcept {
        return sinh(internal::abi<T>, dx::zero, mask, val);
    }

    template <typename M, simd_vector T>
    requires mx::extended_masked_math_zoperator<sinh_t, M, T> &&
        requires(M mask, T val) { sinh(dx::zero, mask, val); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(M mask, T val) noexcept {
        return sinh(dx::zero, mask, val);
    }

public:
    template <fixed_width_abi A, simd_element_for<A> E>
    requires same_as<E, float> || same_as<E, double>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val) noexcept {
        constexpr auto opmask =
            dx::to_compatible_const_mask<basic_vector<E, A>>(dx::zero);
        if constexpr (unqualified_canonical_sinh<basic_vector<E, A>>) {
            if consteval {
                return internal::sinhcosh_base::fallback(val, opmask);
            } else {
                return sinh(internal::abi<A>, val);
            }
        } else {
            return internal::sinhcosh_base::fallback(val, opmask);
        }
    }

    template <simd_abi A, simd_element_for<A> E>
    requires (scalable_abi<A> || (!same_as<E, float> && !same_as<E, double>)) &&
        unqualified_canonical_sinh<basic_vector<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val) noexcept {
        return sinh(internal::abi<A>, val);
    }

    template <extended_vector T>
    requires extended_sinh<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val) noexcept {
        if constexpr (unqualified_extended_sinh<T>) {
            return sinh(val);
        } else if constexpr (expression_sinh<T>) {
            return operator()(dx::evaluate(val));
        } else {
            return operator()(dx::to_canonical(val));
        }
    }

    using mx::masked_operation<sinh_t>::operator();
};

void cosh(...) noexcept = delete;

struct cosh_t;

template <typename T>
concept unqualified_canonical_cosh = requires(T val) {
    {
        cosh(internal::abi<T>, val)
    } -> canonical_arithmetic_result<T, T, typename T::abi_type>;
};

template <typename T>
concept unqualified_extended_cosh = requires(T val) {
    { cosh(val) } -> vector_with_common_abi<typename T::abi_type>;
};

template <typename T>
concept expression_cosh =
    simd_expression<T> && invocable<cosh_t, simd_expression_result_t<T>>;

template <typename T>
concept decayable_cosh =
    decayable_vector_for<T, operation_category::lane_agnostic> &&
    regular_invocable<cosh_t, canonical_type_t<T>>;

template <typename T>
concept extended_cosh =
    unqualified_extended_cosh<T> || expression_cosh<T> || decayable_cosh<T>;

struct cosh_t :
    private internal::sinhcosh_base,
    private mx::masked_operation<cosh_t> {
private:
    friend mx::masked_operation<cosh_t>;

    template <simd_vector S, typename M, simd_vector T>
    requires mx::canonical_masked_math_operator<cosh_t, S, M, T> &&
        requires(
            S src, M mask, T val) { cosh(internal::abi<T>, src, mask, val); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(S src, M mask, T val) noexcept {
        return cosh(internal::abi<T>, src, mask, val);
    }

    template <simd_vector S, typename M, simd_vector T>
    requires mx::extended_masked_math_operator<cosh_t, S, M, T> &&
        requires(S src, M mask, T val) { cosh(src, mask, val); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(S src, M mask, T val) noexcept {
        return cosh(src, mask, val);
    }

    template <typename M, simd_vector T>
    requires mx::canonical_masked_math_zoperator<cosh_t, M, T> &&
        requires(M mask, T val) { cosh(internal::abi<T>, dx::zero, mask, val); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(M mask, T val) noexcept {
        return cosh(internal::abi<T>, dx::zero, mask, val);
    }

    template <typename M, simd_vector T>
    requires mx::extended_masked_math_zoperator<cosh_t, M, T> &&
        requires(M mask, T val) { cosh(dx::zero, mask, val); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(M mask, T val) noexcept {
        return cosh(dx::zero, mask, val);
    }

public:
    template <fixed_width_abi A, simd_element_for<A> E>
    requires same_as<E, float> || same_as<E, double>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val) noexcept {
        constexpr auto opmask =
            dx::to_compatible_const_mask<basic_vector<E, A>>(dx::all_bits);
        if constexpr (unqualified_canonical_cosh<basic_vector<E, A>>) {
            if consteval {
                return internal::sinhcosh_base::fallback(val, opmask);
            } else {
                return cosh(internal::abi<A>, val);
            }
        } else {
            return internal::sinhcosh_base::fallback(val, opmask);
        }
    }

    template <simd_abi A, simd_element_for<A> E>
    requires (!same_as<E, float> && !same_as<E, double>) &&
        unqualified_canonical_cosh<basic_vector<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val) noexcept {
        return cosh(internal::abi<A>, val);
    }

    template <extended_vector T>
    requires extended_cosh<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val) noexcept {
        if constexpr (unqualified_extended_cosh<T>) {
            return cosh(val);
        } else if constexpr (expression_cosh<T>) {
            return operator()(dx::evaluate(val));
        } else {
            return operator()(dx::to_canonical(val));
        }
    }

    using mx::masked_operation<cosh_t>::operator();
};

void sinhcosh(...) noexcept = delete;

struct sinhcosh_t;

template <typename L, typename OpMask, typename A = simd_abi_type_t<L>>
concept unqualified_canonical_sinhcosh = requires(L val, OpMask op) {
    {
        sinhcosh(internal::abi<A>, val, op)
    } -> canonical_arithmetic_result<L, L, A>;
};

template <typename L, typename OpMask, typename A = simd_abi_type_t<L>>
concept unqualified_extended_sinhcosh = requires(L val, OpMask op) {
    { sinhcosh(val, op) } -> vector_with_common_abi<A>;
};

template <typename L, typename OpMask>
concept expression_sinhcosh = (simd_expression<L> || simd_expression<OpMask>) &&
    invocable<sinhcosh_t, simd_expression_result_t<L>,
        simd_expression_result_t<OpMask>>;

template <typename L, typename OpMask>
concept decayable_sinhcosh =
    decayable_vector_for<L, operation_category::lane_agnostic> &&
    decayable_mask_for<OpMask, operation_category::lane_agnostic> &&
    regular_invocable<sinhcosh_t, canonical_type_t<L>,
        canonical_type_t<OpMask>>;

template <typename L, typename OpMask, typename A = common_abi_t<L, OpMask>>
concept extended_sinhcosh = unqualified_extended_sinhcosh<L, OpMask, A> ||
    expression_sinhcosh<L, OpMask> || decayable_sinhcosh<L, OpMask>;

template <typename L, typename OpMask>
concept unqualified_canonical_sinhcoshi = requires(L val, OpMask op) {
    {
        sinhcosh(internal::abi<L>, val, dx::to_compatible_const_mask<L>(op))
    } -> canonical_arithmetic_result<L, L, typename L::abi_type>;
};

template <typename L, typename OpMask>
concept unqualified_extended_sinhcoshi = requires(L val, OpMask op) {
    {
        sinhcosh(val, dx::to_compatible_const_mask<L>(op))
    } -> vector_with_common_abi<simd_abi_type_t<L>>;
};

template <typename L, typename OpMask>
concept expression_sinhcoshi = simd_expression<L> &&
    invocable<sinhcosh_t, simd_expression_result_t<L>, OpMask>;

template <typename L, typename OpMask>
concept decayable_sinhcoshi =
    decayable_simd_for<L, operation_category::lane_agnostic> &&
    regular_invocable<sinhcosh_t, canonical_type_t<L>, OpMask>;

template <typename L, typename OpMask>
concept extended_sinhcoshi = unqualified_extended_sinhcoshi<L, OpMask> ||
    expression_sinhcoshi<L, OpMask> || decayable_sinhcoshi<L, OpMask>;

struct sinhcosh_t :
    private internal::sinhcosh_base,
    private mx::masked_operation<sinhcosh_t> {
private:
    friend mx::masked_operation<sinhcosh_t>;

    template <simd_vector S, typename M, simd_vector T, typename OpMask>
    requires mx::canonical_masked_math_operator<sinhcosh_t, S, M, T, OpMask> &&
        requires(S src, M mask, T val, OpMask opmask) {
            sinhcosh(internal::abi<T>, src, mask, val, opmask);
        }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(
        S src, M mask, T val, OpMask opmask) noexcept {
        return sinhcosh(internal::abi<T>, src, mask, val, opmask);
    }

    template <simd_vector S, typename M, simd_vector T, typename OpMask>
    requires mx::extended_masked_math_operator<sinhcosh_t, S, M, T, OpMask> &&
        requires(S src, M mask, T val, OpMask opmask) {
            sinhcosh(src, mask, val, opmask);
        }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(
        S src, M mask, T val, OpMask opmask) noexcept {
        return sinhcosh(src, mask, val, opmask);
    }

    template <typename M, simd_vector T, typename OpMask>
    requires mx::canonical_masked_math_zoperator<sinhcosh_t, M, T, OpMask> &&
        requires(M mask, T val, OpMask opmask) {
            sinhcosh(internal::abi<T>, dx::zero, mask, val, opmask);
        }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(
        M mask, T val, OpMask opmask) noexcept {
        return sinhcosh(internal::abi<T>, dx::zero, mask, val, opmask);
    }

    template <typename M, simd_vector T, typename OpMask>
    requires mx::extended_masked_math_zoperator<sinhcosh_t, M, T, OpMask> &&
        requires(M mask, T val, OpMask opmask) {
            sinhcosh(dx::zero, mask, val, opmask);
        }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(
        M mask, T val, OpMask opmask) noexcept {
        return sinhcosh(dx::zero, mask, val, opmask);
    }

public:
    template <simd_abi A, simd_element_for<A> E,
        const_mask_for<basic_vector<E, A>> OpMask>
    requires same_as<E, float> || same_as<E, double>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val, OpMask op) noexcept {
        constexpr auto opmask =
            dx::to_compatible_const_mask<basic_vector<E, A>>(op);
        if constexpr (unqualified_canonical_sinhcoshi<basic_vector<E, A>,
                          OpMask>) {
            if consteval {
                return internal::sinhcosh_base::fallback(val, opmask);
            } else {
                return sinhcosh(internal::abi<A>, val, opmask);
            }
        } else {
            return internal::sinhcosh_base::fallback(val, opmask);
        }
    }

    template <simd_abi A, simd_element_for<A> E,
        const_mask_for<basic_vector<E, A>> OpMask>
    requires (!same_as<E, float> && !same_as<E, double>) &&
        unqualified_canonical_sinhcoshi<basic_vector<E, A>, OpMask>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val, OpMask op) noexcept {
        return sinhcosh(internal::abi<A>, val, op);
    }

    template <extended_vector L, const_mask_for<L> OpMask>
    requires extended_sinhcoshi<L, OpMask>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L val, OpMask op) noexcept {
        if constexpr (unqualified_extended_sinhcoshi<L, OpMask>) {
            constexpr auto opmask = dx::to_compatible_const_mask<OpMask>(op);
            return sinhcosh(val, opmask);
        } else if constexpr (expression_sinhcoshi<L, OpMask>) {
            return operator()(dx::evaluate(val), op);
        } else {
            return operator()(dx::to_canonical(val), op);
        }
    }

    template <simd_abi A, simd_element_for<A> E, simd_element_for<A> O>
    requires (same_as<E, float> || same_as<E, double>) && common_size_with<O, E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val, basic_mask<O, A> op) noexcept {
        if constexpr (unqualified_canonical_sinhcosh<basic_vector<E, A>,
                          basic_mask<O, A>>) {
            if consteval {
                return internal::sinhcosh_base::fallback(val, op);
            } else {
                return sinhcosh(internal::abi<A>, val, op);
            }
        } else {
            return internal::sinhcosh_base::fallback(val, op);
        }
    }

    template <simd_abi A, simd_element_for<A> E, simd_element_for<A> O>
    requires (!same_as<E, float> && !same_as<E, double>) &&
        common_size_with<O, E> &&
        unqualified_canonical_sinhcosh<basic_vector<E, A>, basic_mask<O, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val, basic_mask<O, A> op) noexcept {
        return sinhcosh(internal::abi<A>, val, op);
    }

    template <simd_vector L, simd_mask O>
    requires common_size_with<simd_element_type_t<L>, simd_element_type_t<O>> &&
        same_as<simd_abi_type_t<L>, simd_abi_type_t<O>> &&
        (extended_vector<L> || extended_mask<O>) && extended_sinhcosh<L, O>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L val, O op) noexcept {
        if constexpr (unqualified_extended_sinhcosh<L, O>) {
            return sinhcosh(val, op);
        } else if constexpr (expression_sinhcosh<L, O>) {
            return operator()(dx::evaluate(val), dx::evaluate(op));
        } else {
            return operator()(dx::to_canonical(val), dx::to_canonical(op));
        }
    }

    using mx::masked_operation<sinhcosh_t>::operator();
};

template <integral auto V>
struct sinhcoshi_t :
    private internal::sinhcosh_base,
    private mx::masked_operation<sinhcoshi_t<V>> {
private:
    friend mx::masked_operation<sinhcoshi_t<V>>;

    template <simd_vector S, typename M, simd_vector T>
    requires invocable<sinhcosh_t, S, M, T, make_const_mask_t<T, V>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(S src, M mask, T val) noexcept {
        constexpr make_const_mask_t<T, V> opmask{};
        return sinhcosh_t::operator()(src, mask, val, opmask);
    }

    template <typename M, simd_vector T>
    requires invocable<sinhcosh_t, M, T, make_const_mask_t<T, V>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(M mask, T val) noexcept {
        constexpr make_const_mask_t<T, V> opmask{};
        return sinhcosh_t::operator()(mask, val, opmask);
    }

public:
    template <simd_vector T>
    requires regular_invocable<sinhcosh_t, T, make_const_mask_t<T, V>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val) noexcept {
        constexpr make_const_mask_t<T, V> op{};
        return sinhcosh_t::operator()(val, op);
    }

    using mx::masked_operation<sinhcoshi_t<V>>::operator();
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::sinh_t sinh{};
DPL_EXPORT inline constexpr internal::cosh_t cosh{};
DPL_EXPORT inline constexpr internal::sinhcosh_t sinhcosh{};
DPL_EXPORT template <integral auto V>
inline constexpr internal::sinhcoshi_t<V> sinhcoshi{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
