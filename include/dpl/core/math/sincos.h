// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/internal/ilogb.h"
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
#  include "dpl/core/operations/bit.h"
#  include "dpl/core/operations/bitwise.h"
#  include "dpl/core/operations/compare.h"
#  include "dpl/core/operations/gather.h"
#  include "dpl/core/operations/logic.h"
#  include "dpl/core/operations/negate.h"
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

    template <integral auto V = 0, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_simd<float, A> rempi_low(
        basic_simd<float, A> qf, basic_simd<float, A> arg) noexcept {
        constexpr float a = 3.1414794921875f;           // NOLINT
        constexpr float b = 0.00011315941810607910156f; // NOLINT
        constexpr float c = 1.9841872589410058936e-09f; // NOLINT
        using simdf = basic_simd<float, A>;
        constexpr auto scale = dx::selecti<V>(fmath::half, dx::one_v<simdf>);
        constexpr auto sa = a * scale;
        constexpr auto sb = b * scale;
        constexpr auto sc = c * scale;

        return dx::fnmadd(qf, sc, dx::fnmadd(qf, sb, dx::fnmadd(qf, sa, arg)));
    }

    template <integral auto V = 0, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_simd<float, A> rempi_mid(
        basic_simd<float, A> qf, basic_simd<float, A> arg) noexcept {
        constexpr float a0 = 3.140625f;               // NOLINT
        constexpr float b0 = 0.0009675025939941406f;  // NOLINT
        constexpr float c0 = 1.7881393432617188e-07f; // NOLINT
        constexpr float d0 = -2.781813535079891e-08f; // NOLINT
        constexpr float a1 = 1.5703125f;              // NOLINT
        constexpr float b1 = 0.0004837512969970703f;  // NOLINT
        constexpr float c1 = 5.960464477539063e-08f;  // NOLINT
        constexpr float d1 = 1.5893254712295857e-08f; // NOLINT
        using simdf = basic_simd<float, A>;
        constexpr auto sa = dx::selecti<V>(dx::broadcast<A>(a1), a0);
        constexpr auto sb = dx::selecti<V>(dx::broadcast<A>(b1), b0);
        constexpr auto sc = dx::selecti<V>(dx::broadcast<A>(c1), c0);
        constexpr auto sd = dx::selecti<V>(dx::broadcast<A>(d1), d0);

        return dx::fnmadd(qf, sd,
            dx::fnmadd(qf, sc, dx::fnmadd(qf, sb, dx::fnmadd(qf, sa, arg))));
    }

    template <integral auto V = 0, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_simd<double, A> rempi_low(
        basic_simd<double, A> qf, basic_simd<double, A> arg) noexcept {
        constexpr double a = 3.141592653589793116;      // NOLINT
        constexpr double b = 1.2246467991473532072e-16; // NOLINT
        using simdf = basic_simd<float, A>;
        constexpr auto scale = dx::selecti<V>(fmath::half, dx::one_v<simdf>);
        constexpr auto scaled_pi = fmath::scale(pi_pair<double, A>, scale);

        return dx::fnmadd(
            qf, scaled_pi.lower, dx::fnmadd(qf, scaled_pi.upper, arg));
    }

    template <integral auto V = 0, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_simd<double, A> rempi_mid(
        basic_simd<double, A> arg) noexcept {
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
        using simdf = basic_simd<float, A>;
        constexpr auto scale = dx::selecti<V>(fmath::half, dx::one_v<simdf>);
        constexpr auto sa = a * scale;
        constexpr auto sb = b * scale;
        constexpr auto sc = c * scale;
        constexpr auto sd = d * scale;

        return dx::fnmadd(dq.lower + dq.upper, d,
            dx::fnmadd(dq.lower, c,
                dx::fnmadd(dq.upper, c,
                    dx::fnmadd(dq.lower, b,
                        dx::fnmadd(dq.upper, b,
                            dx::fnmadd(
                                dq.lower, a, dx::fnmadd(dq.upper, a, arg)))))));
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
        basic_simd<E, A> f;
        basic_simd<signed_rep_t<E>, A> i;
    };
    template <floating_point E, simd_abi A>
    struct rempi_pair {
        fmath::pair<E, A> df;
        basic_simd<signed_rep_t<E>, A> i;
    };

    template <floating_point E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr rempi_single<E, A> quantize_quarters(
        basic_simd<E, A> arg) noexcept {
        // It breaks a value down into its proximity to the nearest quarter
        // (0.25) and identifies which quarter-step it belongs to relative to
        // the nearest whole integer.
        using sint = signed_rep_t<E>;
        constexpr E four = 4.0;
        constexpr E inv_four = 0.25;
        constexpr auto opt = rounding::to_nearest_int | rounding::no_exc;
        auto y = dx::round(arg * four, opt);
        return {
            .f = dx::fnmadd(y, inv_four, arg),
            .i = dx::cast<sint>(y - dx::round(arg, opt) * four),
        };
    }

    template <floating_point E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr rempi_pair<E, A> rempi(basic_simd<E, A> arg) noexcept {
        using sint = signed_rep_t<E>;
        using simdi = basic_simd<sint, A>;

        struct expq {
            simdi exp;
            simdi q;
        };

        auto const [exp, q] = [](basic_simd<E, A> arg) {
            constexpr sint n64 = -64;
            if constexpr (common_float_with<E, double>) {
                auto exp = fmath::ilogb(fmath::compliance::unsafe, arg) - 55;
                return expq{
                    .exp = exp,
                    .q = dx::bit_keep(exp > (700 - 55), n64),
                };
            } else {
                static_assert(common_float_with<E, float>);
                auto exp = fmath::ilogb(fmath::compliance::unsafe, arg) - 25;
                return expq{
                    .exp = exp,
                    .q = dx::bit_keep(exp > (90 - 25), n64),
                };
            }
        }(arg);

        return [](basic_simd<sint, A> q, basic_simd<E, A> const arg,
                   basic_simd<sint, A> const exp) {
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

    template <integral auto V, floating_point E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_simd<E, A> fallback(
        basic_simd<E, A> const arg) noexcept {
        using simdf = basic_simd<E, A>;
        using sint = signed_rep_t<E>;
        using simdi = basic_simd<sint, A>;
        static constexpr immediate_mask<element_count<E, A>, V> mask;
        simdf const qf = [](basic_simd<E, A> arg) {
            if constexpr (dx::none_of(mask)) {
                return dx::round(arg * dx::inv_pi,
                    rounding::to_nearest_int | rounding::no_exc);
            } else {
                constexpr auto shift =
                    dx::selecti<V>(fmath::half, dx::zero_v<simdf>);
                // when evaluating cosine, minus half
                auto const qf = dx::round(dx::fmsub(arg, dx::inv_pi, shift),
                    rounding::to_nearest_int | rounding::no_exc);

                constexpr auto a = dx::selecti<V>(2.0f, dx::one_v<simdf>);
                constexpr auto c =
                    dx::selecti<V>(dx::one_v<simdf>, dx::zero_v<simdf>);
                // when evaluating cos, 2q + 1, when sin, q + 0 (== q)
                // This should be accurate,
                // since the result is only used when
                // magnitude of qf is small < (threshold_mid / pi)
                return dx::fmadd(a, qf, c);
            }
        }(arg);

        auto q = dx::cast<sint>(qf);
        auto rem = rempi_low<V>(qf, arg);
        // Pray to the branch prediction gods
        if (auto is_below = dx::abs(arg) < threshold_low<E>;
            !dx::all_of(is_below)) {
            rem = dx::select(is_below, rem, rempi_mid<V>(qf, arg));
            if (is_below = dx::abs(arg) < threshold_mid<E>;
                !dx::all_of(is_below)) {
                auto dfi = rempi(arg);
                auto q2 = dfi.i & 3;
                q2 = q2 + q2 + [upper = dfi.df.upper]() {
                    constexpr simdi base = dx::selecti<V>(
                        dx::broadcast<A>(8), dx::broadcast<A>(2));
                    return base -
                        dx::select(upper <= dx::zero, dx::one_v<decltype(q2)>);
                }();
                q2 = [q2]() {
                    if constexpr (dx::none_of(mask)) {
                        return q2 >> imm<2>;
                    } else if constexpr (dx::all_of(mask)) {
                        return q2 >> imm<1>;
                    } else {
                        return dx::selecti<V>(q2 >> imm<1>, q2 >> imm<2>);
                    }
                }();
                constexpr auto nhalfpi = fmath::scale(pi_pair<E, A>, -0.5);
                auto const x = dfi.df +
                    fmath::make_pair(                          //
                        dx::sign(nhalfpi.upper, dfi.df.upper), //
                        dx::sign(nhalfpi.lower, dfi.df.upper)  //
                    );
                auto const isoddeven = [](simdi val) {
                    if constexpr (dx::none_of(mask)) {
                        return (val & dx::one) == dx::one;
                    } else if constexpr (dx::all_of(mask)) {
                        return (val & dx::one) == dx::zero;
                    } else {
                        constexpr auto rhs = dx::bit_dropi<V>(dx::one_v<simdi>);
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
        auto const nmask = [](simdi val) {
            if constexpr (dx::none_of(mask)) {
                return (val & dx::one) == dx::one;
            } else if constexpr (dx::all_of(mask)) {
                return (val & dx::broadcast<simdi>(2)) == dx::zero;
            } else {
                constexpr auto m =
                    dx::selecti<V>(dx::broadcast<simdi>(2), dx::one);
                constexpr auto rhs = dx::selecti<V>(dx::zero_v<simdi>, dx::one);
                return (val & m) == rhs;
            }
        }(q);

        rem = dx::negate(nmask, rem);

        auto const poly = polynomial<E>(sq_rem);
        auto const result = dx::fmadd(sq_rem, poly * rem, rem);
        if constexpr (dx::none_of(mask)) {
            return dx::select(arg == dx::msb, arg, result);
        } else if constexpr (dx::all_of(mask)) {
            return result;
        } else {
            return dx::select(
                arg == dx::msb, dx::selecti<V>(result, arg), result);
        }
    }

    // TODO
    template <integral auto V, floating_point E, simd_abi A>
    requires (dx::digits_v<E><dx::digits_v<float> || dx::digits_v<E>>
                     dx::digits_v<double>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_simd<E, A> fallback(basic_simd<E, A>) = delete;
};

void sin(...) noexcept = delete;
void cos(...) noexcept = delete;
template <auto>
void sincos(...) noexcept = delete;

struct sin_t : private internal::sincos_base {

    template <floating_point_simd T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val) noexcept {
        if constexpr (requires {
                          {
                              sin(internal::abi<T>, val)
                          } -> equivalent_simd_as<T>;
                      }) {
            if constexpr (basic_simd_type<T>) {
                if not consteval {
                    return sin(internal::abi<T>, val);
                } else {
                    return internal::sincos_base::fallback<0>(val);
                }
            } else {
                return sin(internal::abi<T>, val);
            }
        } else if constexpr (basic_simd_type<T>) {
            return internal::sincos_base::fallback<0>(val);
        } else {
            return operator()(dx::to_basic_type(val));
        }
    }
};

struct cos_t : private internal::sincos_base {

    template <floating_point_simd T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val) noexcept {
        if constexpr (requires {
                          {
                              cos(internal::abi<T>, val)
                          } -> equivalent_simd_as<T>;
                      }) {
            if constexpr (basic_simd_type<T>) {
                if not consteval {
                    return cos(internal::abi<T>, val);
                } else {
                    return internal::sincos_base::fallback<-1>(val);
                }
            } else {
                return cos(internal::abi<T>, val);
            }
        } else if constexpr (basic_simd_type<T>) {
            return internal::sincos_base::fallback<-1>(val);
        } else {
            return operator()(dx::to_basic_type(val));
        }
    }
};

template <integral auto V>
struct sincosi_t : private internal::sincos_base {
    template <floating_point_simd T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val) noexcept {
        if constexpr (requires {
                          {
                              sincos<V>(internal::abi<T>, val)
                          } -> equivalent_simd_as<T>;
                      }) {
            if constexpr (basic_simd_type<T>) {
                if not consteval {
                    return sincos<V>(internal::abi<T>, val);
                } else {
                    return internal::sincos_base::fallback<V>(val);
                }
            } else {
                return sincos<V>(internal::abi<T>, val);
            }
        } else if constexpr (basic_simd_type<T>) {
            return internal::sincos_base::fallback<V>(val);
        } else {
            return operator()(dx::to_basic_type(val));
        }
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::sin_t sin{};
DPL_EXPORT inline constexpr internal::cos_t cos{};
DPL_EXPORT template <integral auto V>
inline constexpr internal::sincosi_t<V> sincosi{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
