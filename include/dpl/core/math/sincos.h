// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/internal/ilogb.h"
#include "dpl/core/math/internal/masked_op.h"
#include "dpl/core/math/internal/pair.h"
#include "dpl/core/math/internal/polynomial.h"
#include "dpl/core/math/internal/rempi_table.h"
#include "dpl/core/math/round.h"
#include "dpl/core/math/sign.h"
#include "dpl/core/math/trunc.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/constants/inv_pi.h"
#  include "dpl/core/constants/zero.h"
#  include "dpl/core/operations/bitwise.h"
#  include "dpl/core/operations/compare.h"
#  include "dpl/core/operations/gather.h"
#  include "dpl/core/operations/logical.h"
#  include "dpl/core/operations/select.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
class sincos_base {
protected:
    __DPL_HIDE_FROM_ABI constexpr ~sincos_base() = default;

    template <floating_point E, simd_abi A>
    static constexpr fmath::pair<E, A> pi_pair = []() {
        // NOLINTBEGIN
        if constexpr (same_as<float, E>) {
            return fmath::make_pair(                          //
                dx::broadcast<A>(3.1415927410125732422f),     //
                dx::broadcast<A>(-8.7422776573475857731e-08f) //
            );
        } else {
            static_assert(same_as<double, E>);
            return fmath::make_pair(                        //
                dx::broadcast<A>(3.141592653589793116),     //
                dx::broadcast<A>(1.2246467991473532072e-16) //
            );
        }
        // NOLINTEND
    }();

    template <floating_point T>
    static constexpr auto polynomial = []() {
        // NOLINTBEGIN
        if constexpr (same_as<float, T>) {
            return fmath::polynomial<-0.166666597127914428710938f, //
                0.00833307858556509017944336f,                     //
                -0.0001981069071916863322258f,                     //
                2.6083159809786593541503e-06f>{};
        } else {
            static_assert(same_as<double, T>);
            return fmath::polynomial<0.00833333333333332974823815, //
                -0.000198412698412696162806809,                    //
                2.75573192239198747630416e-06,                     //
                -2.50521083763502045810755e-08,                    //
                1.60590430605664501629054e-10,                     //
                -7.64712219118158833288484e-13,                    //
                2.81009972710863200091251e-15,                     //
                -7.97255955009037868891952e-18>{};
        }
        // NOLINTEND
    }();

    template <simd_abi A, typename OpMask>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_vector<float, A> rempi_low(basic_vector<float, A> qf,
        basic_vector<float, A> arg, OpMask opmask) noexcept {
        constexpr float a = 3.1414794921875f;           // NOLINT
        constexpr float b = 0.00011315941810607910156f; // NOLINT
        constexpr float c = 1.9841872589410058936e-09f; // NOLINT
        using simdf = basic_vector<float, A>;
        if constexpr (simd_mask<OpMask>) {
            auto scale = dx::select(opmask, fmath::half, dx::one_v<simdf>);
            auto sa = a * scale;
            auto sb = b * scale;
            auto sc = c * scale;

            return dx::fnmadd(
                qf, sc, dx::fnmadd(qf, sb, dx::fnmadd(qf, sa, arg)));

        } else {
            constexpr auto scale =
                dx::select(opmask, fmath::half, dx::one_v<simdf>);
            constexpr auto sa = a * scale;
            constexpr auto sb = b * scale;
            constexpr auto sc = c * scale;

            return dx::fnmadd(
                qf, sc, dx::fnmadd(qf, sb, dx::fnmadd(qf, sa, arg)));
        }
    }

    template <simd_abi A, typename OpMask>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_vector<float, A> rempi_mid(basic_vector<float, A> qf,
        basic_vector<float, A> arg, OpMask opmask) noexcept {
        constexpr float a0 = 3.140625f;               // NOLINT
        constexpr float b0 = 0.0009675025939941406f;  // NOLINT
        constexpr float c0 = 1.7881393432617188e-07f; // NOLINT
        constexpr float d0 = -2.781813535079891e-08f; // NOLINT
        constexpr float a1 = 1.5703125f;              // NOLINT
        constexpr float b1 = 0.0004837512969970703f;  // NOLINT
        constexpr float c1 = 5.960464477539063e-08f;  // NOLINT
        constexpr float d1 = 1.5893254712295857e-08f; // NOLINT
        using simdf = basic_vector<float, A>;
        if constexpr (simd_mask<OpMask>) {
            auto sa = dx::select(opmask, dx::broadcast<A>(a1), a0);
            auto sb = dx::select(opmask, dx::broadcast<A>(b1), b0);
            auto sc = dx::select(opmask, dx::broadcast<A>(c1), c0);
            auto sd = dx::select(opmask, dx::broadcast<A>(d1), d0);

            return dx::fnmadd(qf, sd,
                dx::fnmadd(
                    qf, sc, dx::fnmadd(qf, sb, dx::fnmadd(qf, sa, arg))));

        } else {
            constexpr auto sa = dx::select(opmask, dx::broadcast<A>(a1), a0);
            constexpr auto sb = dx::select(opmask, dx::broadcast<A>(b1), b0);
            constexpr auto sc = dx::select(opmask, dx::broadcast<A>(c1), c0);
            constexpr auto sd = dx::select(opmask, dx::broadcast<A>(d1), d0);

            return dx::fnmadd(qf, sd,
                dx::fnmadd(
                    qf, sc, dx::fnmadd(qf, sb, dx::fnmadd(qf, sa, arg))));
        }
    }

    template <simd_abi A, typename OpMask>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_vector<double, A> rempi_low(
        basic_vector<double, A> qf, basic_vector<double, A> arg,
        OpMask opmask) noexcept {
        constexpr double a = 3.141592653589793116;      // NOLINT
        constexpr double b = 1.2246467991473532072e-16; // NOLINT
        using simdf = basic_vector<float, A>;

        if constexpr (simd_mask<OpMask>) {
            auto scale = dx::select(opmask, fmath::half, dx::one_v<simdf>);
            auto scaled_pi = fmath::scale(pi_pair<double, A>, scale);

            return dx::fnmadd(
                qf, scaled_pi.lower, dx::fnmadd(qf, scaled_pi.upper, arg));
        } else {
            constexpr auto scale =
                dx::select(opmask, fmath::half, dx::one_v<simdf>);
            constexpr auto scaled_pi = fmath::scale(pi_pair<double, A>, scale);

            return dx::fnmadd(
                qf, scaled_pi.lower, dx::fnmadd(qf, scaled_pi.upper, arg));
        }
    }

    template <simd_abi A, typename OpMask>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_vector<double, A> rempi_mid(
        basic_vector<double, A> arg, OpMask opmask) noexcept {
        constexpr auto upper_scale = static_cast<double>(1 << 24);
        constexpr auto pi_scale = dx::inv_pi_v<double> / upper_scale;

        fmath::pair<double, A> const dq{
            .upper = dx::trunc(arg * pi_scale, rounding::no_exc) * upper_scale,
            .lower = dx::round(dx::fmsub(arg, dx::inv_pi, dq.upper),
                rounding::to_nearest_int | rounding::no_exc),
        };

        constexpr double a = 3.1415926218032836914;     // NOLINT
        constexpr double b = 3.1786509424591713469e-08; // NOLINT
        constexpr double c = 1.2246467864107188502e-16; // NOLINT
        constexpr double d = 1.2736634327021899816e-24; // NOLINT
        using simdf = basic_vector<float, A>;
        if constexpr (simd_mask<OpMask>) {
            auto scale = dx::select(opmask, fmath::half, dx::one_v<simdf>);
            auto sa = a * scale;
            auto sb = b * scale;
            auto sc = c * scale;
            auto sd = d * scale;

            return dx::fnmadd(dq.lower + dq.upper, d,
                dx::fnmadd(dq.lower, c,
                    dx::fnmadd(dq.upper, c,
                        dx::fnmadd(dq.lower, b,
                            dx::fnmadd(dq.upper, b,
                                dx::fnmadd(dq.lower, a,
                                    dx::fnmadd(dq.upper, a, arg)))))));
        } else {
            constexpr auto scale =
                dx::select(opmask, fmath::half, dx::one_v<simdf>);
            constexpr auto sa = a * scale;
            constexpr auto sb = b * scale;
            constexpr auto sc = c * scale;
            constexpr auto sd = d * scale;

            return dx::fnmadd(dq.lower + dq.upper, d,
                dx::fnmadd(dq.lower, c,
                    dx::fnmadd(dq.upper, c,
                        dx::fnmadd(dq.lower, b,
                            dx::fnmadd(dq.upper, b,
                                dx::fnmadd(dq.lower, a,
                                    dx::fnmadd(dq.upper, a, arg)))))));
        }
    }

    template <floating_point E>
    static constexpr E threshold_low = []() {
        if constexpr (same_as<float, E>) {
            return 125.0f; // NOLINT
        } else {
            static_assert(same_as<double, E>);
            return 15.0; // NOLINT
        }
    }();

    template <floating_point E>
    static constexpr E threshold_mid = []() {
        if constexpr (same_as<float, E>) {
            return 8.0e6f; // NOLINT
        } else {
            static_assert(same_as<double, E>);
            return 1.0e14; // NOLINT
        }
    }();

    template <floating_point E, simd_abi A>
    struct rempi_single {
        basic_vector<E, A> f;
        basic_vector<signed_representation_t<E>, A> i;
    };
    template <floating_point E, simd_abi A>
    struct rempi_pair {
        fmath::pair<E, A> df;
        basic_vector<signed_representation_t<E>, A> i;
    };

    template <floating_point E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr rempi_single<E, A> quantize_quarters(
        basic_vector<E, A> arg) noexcept {
        // It breaks a value down into its proximity to the nearest quarter
        // (0.25) and identifies which quarter-step it belongs to relative to
        // the nearest whole integer.
        using sint = signed_representation_t<E>;
        constexpr E four = 4.0;
        constexpr E inv_four = 0.25;
        constexpr auto opt = rounding::to_nearest_int | rounding::no_exc;
        auto y = dx::round(arg * four, opt);
        return {
            .f = dx::fnmadd(y, inv_four, arg),
            .i = dx::element_cast<sint>(y - dx::round(arg, opt) * four),
        };
    }

    template <floating_point E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr rempi_pair<E, A> rempi(basic_vector<E, A> arg) noexcept {
        using sint = signed_representation_t<E>;
        using simdi = basic_vector<sint, A>;

        struct expq {
            simdi exp;
            simdi q;
        };

        auto const [exp, q] = [](basic_vector<E, A> arg) {
            constexpr sint n64 = -64;
            if constexpr (same_as<E, double>) {
                auto exp = fmath::ilogb(fmath::compliance::unsafe, arg) - 55;
                return expq{
                    .exp = exp,
                    .q = dx::select(exp > (700 - 55), n64, dx::zero),
                };
            } else {
                static_assert(same_as<E, float>);
                auto exp = fmath::ilogb(fmath::compliance::unsafe, arg) - 25;
                return expq{
                    .exp = exp,
                    .q = dx::select(exp > (90 - 25), n64, dx::zero),
                };
            }
        }(arg);

        return [](basic_vector<sint, A> q, basic_vector<E, A> const arg,
                   basic_vector<sint, A> const exp) {
            auto x =
                fmath::single(arg) * dx::gather(fmath::rempi_table<E>, exp);
            auto di = quantize_quarters(x.upper);
            q = di.i;
            x.upper = di.f;
            x = fmath::normalize(x);

            auto y =
                fmath::single(arg) * dx::gather(fmath::rempi_table<E> + 1, exp);
            x = x + y;
            di = quantize_quarters(x.upper);
            q += di.i;
            x.upper = di.f;
            x = fmath::normalize(x);

            using pair = fmath::pair<E, A>;
            y = pair{
                .upper = dx::gather(fmath::rempi_table<E> + 2, exp),
                .lower = dx::gather(fmath::rempi_table<E> + 3, exp),
            };
            y = y * arg;
            x = x + y;
            x = fmath::normalize(x);
            constexpr E two = 2.0;
            constexpr auto twopi = fmath::scale(pi_pair<E, A>, two);
            x = x * twopi;
            constexpr E p7 = 0.7;
            x = fmath::select(dx::abs(arg) < p7, fmath::make_pair(arg), x);
            return rempi_pair<E, A>{
                .df = x,
                .i = q,
            };
        }(q, fmath::ldexp(fmath::compliance::unsafe, arg, q),
                   dx::bwandnot(dx::signbit(exp), exp) << imm<2>);
    }

    template <floating_point E, simd_abi A, typename OpMask>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_vector<E, A> fallback(
        basic_vector<E, A> const arg, OpMask opmask) noexcept {
        using simdf = basic_vector<E, A>;
        using sint = signed_representation_t<E>;
        using simdi = basic_vector<sint, A>;
        simdf const qf = [&]() {
            if constexpr (simd_mask<OpMask>) {
                constexpr auto one = dx::one_v<simdf>;
                auto shift = dx::select(
                    opmask, dx::broadcast<simdf>(fmath::half), dx::zero);
                auto const qf = dx::round(dx::fmsub(arg, dx::inv_pi, shift),
                    rounding::to_nearest_int | rounding::no_exc);
                auto a = dx::select(opmask, 2.0f, one);
                auto c = dx::select(opmask, one, dx::zero);
                return dx::fmadd(a, qf, c);
            } else if constexpr (dx::none_of(opmask)) {
                return dx::round(arg * dx::inv_pi,
                    rounding::to_nearest_int | rounding::no_exc);
            } else {
                constexpr auto one = dx::one_v<simdf>;
                constexpr auto shift = dx::select(
                    opmask, dx::broadcast<simdf>(fmath::half), dx::zero);
                // when evaluating cosine, minus half
                auto const qf = dx::round(dx::fmsub(arg, dx::inv_pi, shift),
                    rounding::to_nearest_int | rounding::no_exc);

                constexpr auto a = dx::select(opmask, 2.0f, one);
                constexpr auto c = dx::select(opmask, one, dx::zero);
                // when evaluating cos, 2q + 1, when sin, q + 0 (== q)
                // This should be accurate,
                // since the result is only used when
                // magnitude of qf is small < (threshold_mid / pi)
                return dx::fmadd(a, qf, c);
            }
        }();

        auto q = dx::element_cast<sint>(qf);
        auto rem = rempi_low(qf, arg, opmask);
        // Pray to the branch prediction gods
        if (auto is_below = dx::abs(arg) < threshold_low<E>;
            !dx::all_of(is_below)) {
            rem = dx::select(is_below, rem, rempi_mid(qf, arg, opmask));
            if (is_below = dx::abs(arg) < threshold_mid<E>;
                !dx::all_of(is_below)) {
                auto dfi = rempi(arg);
                auto q2 = dfi.i & 3;
                q2 = q2 + q2 + [&]() {
                    if constexpr (simd_mask<OpMask>) {
                        auto base = dx::select(opmask, dx::broadcast<simdi>(8),
                            dx::broadcast<simdi>(2));
                        return base -
                            dx::select(dfi.df.upper <= dx::zero,
                                dx::one_v<decltype(q2)>, dx::zero);
                    } else {
                        constexpr auto base = dx::select(opmask,
                            dx::broadcast<simdi>(8), dx::broadcast<simdi>(2));
                        return base -
                            dx::select(dfi.df.upper <= dx::zero,
                                dx::one_v<decltype(q2)>, dx::zero);
                    }
                }();
                q2 = [&]() {
                    if constexpr (simd_mask<OpMask>) {
                        return dx::select(opmask, q2 >> imm<1>, q2 >> imm<2>);
                    } else if constexpr (dx::none_of(opmask)) {
                        return q2 >> imm<2>;
                    } else if constexpr (dx::all_of(opmask)) {
                        return q2 >> imm<1>;
                    } else {
                        return dx::select(opmask, q2 >> imm<1>, q2 >> imm<2>);
                    }
                }();
                constexpr auto nhalfpi = fmath::scale(pi_pair<E, A>, -0.5);
                auto const x = dfi.df +
                    fmath::make_pair(                          //
                        dx::sign(nhalfpi.upper, dfi.df.upper), //
                        dx::sign(nhalfpi.lower, dfi.df.upper)  //
                    );
                auto const isoddeven = [&](simdi val) {
                    if constexpr (simd_mask<OpMask>) {
                        auto rhs =
                            dx::select(opmask, dx::zero, dx::one_v<simdi>);
                        return (val & dx::one) == rhs;
                    } else if constexpr (dx::none_of(opmask)) {
                        return (val & dx::one) == dx::one;
                    } else if constexpr (dx::all_of(opmask)) {
                        return (val & dx::one) == dx::zero;
                    } else {
                        constexpr auto rhs =
                            dx::select(opmask, dx::zero, dx::one_v<simdi>);
                        return (val & dx::one) == rhs;
                    }
                }(dfi.i);
                dfi.df = fmath::select(isoddeven, x, dfi.df);
                auto const result = dx::select(
                    dx::isfinite(arg), dfi.df.upper + dfi.df.lower, dx::nan);
                q = dx::select(is_below, q, q2);
                rem = dx::select(is_below, rem, result);
            }
        }

        auto const sq_rem = dx::multiply(rem, rem);
        auto const nmask = [&](simdi val) {
            if constexpr (simd_mask<OpMask>) {
                auto m = dx::select(opmask, dx::broadcast<simdi>(2), dx::one);
                auto rhs = dx::select(opmask, dx::zero_v<simdi>, dx::one);
                return (val & m) == rhs;
            } else if constexpr (dx::none_of(opmask)) {
                return (val & dx::one) == dx::one;
            } else if constexpr (dx::all_of(opmask)) {
                return (val & dx::broadcast<simdi>(2)) == dx::zero;
            } else {
                constexpr auto m =
                    dx::select(opmask, dx::broadcast<simdi>(2), dx::one);
                constexpr auto rhs =
                    dx::select(opmask, dx::zero_v<simdi>, dx::one);
                return (val & m) == rhs;
            }
        }(q);

        rem = dx::negate(rem, nmask, rem);

        auto const poly = polynomial<E>(sq_rem);
        auto const result = dx::fmadd(sq_rem, poly * rem, rem);
        if constexpr (simd_mask<OpMask>) {
            return dx::select(
                arg == dx::msb, dx::select(opmask, result, arg), result);
        } else if constexpr (dx::none_of(opmask)) {
            return dx::select(arg == dx::msb, arg, result);
        } else if constexpr (dx::all_of(opmask)) {
            return result;
        } else {
            return dx::select(
                arg == dx::msb, dx::select(opmask, result, arg), result);
        }
    }
};

void sin(...) noexcept = delete;

struct sin_t;

template <typename T>
concept unqualified_canonical_sin = requires(T val) {
    {
        sin(internal::abi<T>, val)
    } -> canonical_arithmetic_result<T, T, typename T::abi_type>;
};

template <typename T>
concept unqualified_extended_sin = requires(T val) {
    { sin(val) } -> vector_with_common_abi<typename T::abi_type>;
};

template <typename T>
concept expression_sin =
    simd_expression<T> && invocable<sin_t, simd_expression_result_t<T>>;

template <typename T>
concept decayable_sin =
    decayable_vector_for<T, operation_category::lane_agnostic> &&
    regular_invocable<sin_t, canonical_type_t<T>>;

template <typename T>
concept extended_sin =
    unqualified_extended_sin<T> || expression_sin<T> || decayable_sin<T>;

struct sin_t :
    private internal::sincos_base,
    private mx::masked_operation<sin_t> {
private:
    friend mx::masked_operation<sin_t>;

    template <simd_vector S, typename M, simd_vector T>
    requires mx::canonical_masked_math_operator<sin_t, S, M, T> &&
        requires(
            S src, M mask, T val) { sin(internal::abi<T>, src, mask, val); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(S src, M mask, T val) noexcept {
        return sin(internal::abi<T>, src, mask, val);
    }

    template <simd_vector S, typename M, simd_vector T>
    requires mx::extended_masked_math_operator<sin_t, S, M, T> &&
        requires(S src, M mask, T val) { sin(src, mask, val); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(S src, M mask, T val) noexcept {
        return sin(src, mask, val);
    }

    template <typename M, simd_vector T>
    requires mx::canonical_masked_math_zoperator<sin_t, M, T> &&
        requires(M mask, T val) { sin(internal::abi<T>, dx::zero, mask, val); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(M mask, T val) noexcept {
        return sin(internal::abi<T>, dx::zero, mask, val);
    }

    template <typename M, simd_vector T>
    requires mx::extended_masked_math_zoperator<sin_t, M, T> &&
        requires(M mask, T val) { sin(dx::zero, mask, val); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(M mask, T val) noexcept {
        return sin(dx::zero, mask, val);
    }

public:
    template <fixed_width_abi A, simd_element_for<A> E>
    requires same_as<E, float> || same_as<E, double>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val) noexcept {
        constexpr auto opmask =
            dx::to_compatible_const_mask<basic_vector<E, A>>(dx::zero);
        if constexpr (unqualified_canonical_sin<basic_vector<E, A>>) {
            if consteval {
                return internal::sincos_base::fallback(val, opmask);
            } else {
                return sin(internal::abi<A>, val);
            }
        } else {
            return internal::sincos_base::fallback(val, opmask);
        }
    }

    template <simd_abi A, simd_element_for<A> E>
    requires (scalable_abi<A> || (!same_as<E, float> && !same_as<E, double>)) &&
        unqualified_canonical_sin<basic_vector<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val) noexcept {
        return sin(internal::abi<A>, val);
    }

    template <extended_vector T>
    requires extended_sin<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val) noexcept {
        if constexpr (unqualified_extended_sin<T>) {
            return sin(val);
        } else if constexpr (expression_sin<T>) {
            return operator()(dx::evaluate(val));
        } else {
            return operator()(dx::to_canonical(val));
        }
    }

    using mx::masked_operation<sin_t>::operator();
};

void cos(...) noexcept = delete;

struct cos_t;

template <typename T>
concept unqualified_canonical_cos = requires(T val) {
    {
        cos(internal::abi<T>, val)
    } -> canonical_arithmetic_result<T, T, typename T::abi_type>;
};

template <typename T>
concept unqualified_extended_cos = requires(T val) {
    { cos(val) } -> vector_with_common_abi<typename T::abi_type>;
};

template <typename T>
concept expression_cos =
    simd_expression<T> && invocable<cos_t, simd_expression_result_t<T>>;

template <typename T>
concept decayable_cos =
    decayable_vector_for<T, operation_category::lane_agnostic> &&
    regular_invocable<cos_t, canonical_type_t<T>>;

template <typename T>
concept extended_cos =
    unqualified_extended_cos<T> || expression_cos<T> || decayable_cos<T>;

struct cos_t :
    private internal::sincos_base,
    private mx::masked_operation<cos_t> {
private:
    friend mx::masked_operation<cos_t>;

    template <simd_vector S, typename M, simd_vector T>
    requires mx::canonical_masked_math_operator<cos_t, S, M, T> &&
        requires(
            S src, M mask, T val) { cos(internal::abi<T>, src, mask, val); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(S src, M mask, T val) noexcept {
        return cos(internal::abi<T>, src, mask, val);
    }

    template <simd_vector S, typename M, simd_vector T>
    requires mx::extended_masked_math_operator<cos_t, S, M, T> &&
        requires(S src, M mask, T val) { cos(src, mask, val); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(S src, M mask, T val) noexcept {
        return cos(src, mask, val);
    }

    template <typename M, simd_vector T>
    requires mx::canonical_masked_math_zoperator<cos_t, M, T> &&
        requires(M mask, T val) { cos(internal::abi<T>, dx::zero, mask, val); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(M mask, T val) noexcept {
        return cos(internal::abi<T>, dx::zero, mask, val);
    }

    template <typename M, simd_vector T>
    requires mx::extended_masked_math_zoperator<cos_t, M, T> &&
        requires(M mask, T val) { cos(dx::zero, mask, val); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(M mask, T val) noexcept {
        return cos(dx::zero, mask, val);
    }

public:
    template <fixed_width_abi A, simd_element_for<A> E>
    requires same_as<E, float> || same_as<E, double>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val) noexcept {
        constexpr auto opmask =
            dx::to_compatible_const_mask<basic_vector<E, A>>(dx::all_bits);
        if constexpr (unqualified_canonical_cos<basic_vector<E, A>>) {
            if consteval {
                return internal::sincos_base::fallback(val, opmask);
            } else {
                return cos(internal::abi<A>, val);
            }
        } else {
            return internal::sincos_base::fallback(val, opmask);
        }
    }

    template <simd_abi A, simd_element_for<A> E>
    requires (!same_as<E, float> && !same_as<E, double>) &&
        unqualified_canonical_cos<basic_vector<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val) noexcept {
        return cos(internal::abi<A>, val);
    }

    template <extended_vector T>
    requires extended_cos<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val) noexcept {
        if constexpr (unqualified_extended_cos<T>) {
            return cos(val);
        } else if constexpr (expression_cos<T>) {
            return operator()(dx::evaluate(val));
        } else {
            return operator()(dx::to_canonical(val));
        }
    }

    using mx::masked_operation<cos_t>::operator();
};

void sincos(...) noexcept = delete;

struct sincos_t;

template <typename L, typename OpMask, typename A = simd_abi_type_t<L>>
concept unqualified_canonical_sincos = requires(L val, OpMask op) {
    {
        sincos(internal::abi<A>, val, op)
    } -> canonical_arithmetic_result<L, L, A>;
};

template <typename L, typename OpMask, typename A = simd_abi_type_t<L>>
concept unqualified_extended_sincos = requires(L val, OpMask op) {
    { sincos(val, op) } -> vector_with_common_abi<A>;
};

template <typename L, typename OpMask>
concept expression_sincos = (simd_expression<L> || simd_expression<OpMask>) &&
    invocable<sincos_t, simd_expression_result_t<L>,
        simd_expression_result_t<OpMask>>;

template <typename L, typename OpMask>
concept decayable_sincos =
    decayable_vector_for<L, operation_category::lane_agnostic> &&
    decayable_mask_for<OpMask, operation_category::lane_agnostic> &&
    regular_invocable<sincos_t, canonical_type_t<L>, canonical_type_t<OpMask>>;

template <typename L, typename OpMask, typename A = common_abi_t<L, OpMask>>
concept extended_sincos = unqualified_extended_sincos<L, OpMask, A> ||
    expression_sincos<L, OpMask> || decayable_sincos<L, OpMask>;

template <typename L, typename OpMask>
concept unqualified_canonical_sincosi = requires(L val, OpMask op) {
    {
        sincos(internal::abi<L>, val, dx::to_compatible_const_mask<L>(op))
    } -> canonical_arithmetic_result<L, L, typename L::abi_type>;
};

template <typename L, typename OpMask>
concept unqualified_extended_sincosi = requires(L val, OpMask op) {
    {
        sincos(val, dx::to_compatible_const_mask<L>(op))
    } -> vector_with_common_abi<simd_abi_type_t<L>>;
};

template <typename L, typename OpMask>
concept expression_sincosi = simd_expression<L> &&
    invocable<sincos_t, simd_expression_result_t<L>, OpMask>;

template <typename L, typename OpMask>
concept decayable_sincosi =
    decayable_simd_for<L, operation_category::lane_agnostic> &&
    regular_invocable<sincos_t, canonical_type_t<L>, OpMask>;

template <typename L, typename OpMask>
concept extended_sincosi = unqualified_extended_sincosi<L, OpMask> ||
    expression_sincosi<L, OpMask> || decayable_sincosi<L, OpMask>;

struct sincos_t :
    private internal::sincos_base,
    private mx::masked_operation<sincos_t> {
private:
    friend mx::masked_operation<sincos_t>;

    template <simd_vector S, typename M, simd_vector T, typename OpMask>
    requires mx::canonical_masked_math_operator<sincos_t, S, M, T, OpMask> &&
        requires(S src, M mask, T val, OpMask opmask) {
            sincos(internal::abi<T>, src, mask, val, opmask);
        }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(
        S src, M mask, T val, OpMask opmask) noexcept {
        return sincos(internal::abi<T>, src, mask, val, opmask);
    }

    template <simd_vector S, typename M, simd_vector T, typename OpMask>
    requires mx::extended_masked_math_operator<sincos_t, S, M, T, OpMask> &&
        requires(S src, M mask, T val, OpMask opmask) {
            sincos(src, mask, val, opmask);
        }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(
        S src, M mask, T val, OpMask opmask) noexcept {
        return sincos(src, mask, val, opmask);
    }

    template <typename M, simd_vector T, typename OpMask>
    requires mx::canonical_masked_math_zoperator<sincos_t, M, T, OpMask> &&
        requires(M mask, T val, OpMask opmask) {
            sincos(internal::abi<T>, dx::zero, mask, val, opmask);
        }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(
        M mask, T val, OpMask opmask) noexcept {
        return sincos(internal::abi<T>, dx::zero, mask, val, opmask);
    }

    template <typename M, simd_vector T, typename OpMask>
    requires mx::extended_masked_math_zoperator<sincos_t, M, T, OpMask> &&
        requires(M mask, T val, OpMask opmask) {
            sincos(dx::zero, mask, val, opmask);
        }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(
        M mask, T val, OpMask opmask) noexcept {
        return sincos(dx::zero, mask, val, opmask);
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
        if constexpr (unqualified_canonical_sincosi<basic_vector<E, A>,
                          OpMask>) {
            if consteval {
                return internal::sincos_base::fallback(val, opmask);
            } else {
                return sincos(internal::abi<A>, val, opmask);
            }
        } else {
            return internal::sincos_base::fallback(val, opmask);
        }
    }

    template <simd_abi A, simd_element_for<A> E,
        const_mask_for<basic_vector<E, A>> OpMask>
    requires (!same_as<E, float> && !same_as<E, double>) &&
        unqualified_canonical_sincosi<basic_vector<E, A>, OpMask>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val, OpMask op) noexcept {
        return sincos(internal::abi<A>, val, op);
    }

    template <extended_vector L, const_mask_for<L> OpMask>
    requires extended_sincosi<L, OpMask>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L val, OpMask op) noexcept {
        if constexpr (unqualified_extended_sincosi<L, OpMask>) {
            constexpr auto opmask = dx::to_compatible_const_mask<OpMask>(op);
            return sincos(val, opmask);
        } else if constexpr (expression_sincosi<L, OpMask>) {
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
        if constexpr (unqualified_canonical_sincos<basic_vector<E, A>,
                          basic_mask<O, A>>) {
            if consteval {
                return internal::sincos_base::fallback(val, op);
            } else {
                return sincos(internal::abi<A>, val, op);
            }
        } else {
            return internal::sincos_base::fallback(val, op);
        }
    }

    template <simd_abi A, simd_element_for<A> E, simd_element_for<A> O>
    requires (!same_as<E, float> && !same_as<E, double>) &&
        common_size_with<O, E> &&
        unqualified_canonical_sincos<basic_vector<E, A>, basic_mask<O, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val, basic_mask<O, A> op) noexcept {
        return sincos(internal::abi<A>, val, op);
    }

    template <simd_vector L, simd_mask O>
    requires common_size_with<simd_element_type_t<L>, simd_element_type_t<O>> &&
        same_as<simd_abi_type_t<L>, simd_abi_type_t<O>> &&
        (extended_vector<L> || extended_mask<O>) && extended_sincos<L, O>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L val, O op) noexcept {
        if constexpr (unqualified_extended_sincos<L, O>) {
            return sincos(val, op);
        } else if constexpr (expression_sincos<L, O>) {
            return operator()(dx::evaluate(val), dx::evaluate(op));
        } else {
            return operator()(dx::to_canonical(val), dx::to_canonical(op));
        }
    }

    using mx::masked_operation<sincos_t>::operator();
};

template <integral auto V>
struct sincosi_t :
    private internal::sincos_base,
    private mx::masked_operation<sincosi_t<V>> {
private:
    friend mx::masked_operation<sincosi_t<V>>;

    template <simd_vector S, typename M, simd_vector T>
    requires invocable<sincos_t, S, M, T, make_const_mask_t<T, V>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(S src, M mask, T val) noexcept {
        constexpr make_const_mask_t<T, V> opmask{};
        return sincos_t::operator()(src, mask, val, opmask);
    }

    template <typename M, simd_vector T>
    requires invocable<sincos_t, M, T, make_const_mask_t<T, V>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(M mask, T val) noexcept {
        constexpr make_const_mask_t<T, V> opmask{};
        return sincos_t::operator()(mask, val, opmask);
    }

public:
    template <simd_vector T>
    requires regular_invocable<sincos_t, T, make_const_mask_t<T, V>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val) noexcept {
        constexpr make_const_mask_t<T, V> op{};
        return sincos_t::operator()(val, op);
    }

    using mx::masked_operation<sincosi_t<V>>::operator();
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::sin_t sin{};
DPL_EXPORT inline constexpr internal::cos_t cos{};
DPL_EXPORT inline constexpr internal::sincos_t sincos{};
DPL_EXPORT template <integral auto V>
inline constexpr internal::sincosi_t<V> sincosi{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
