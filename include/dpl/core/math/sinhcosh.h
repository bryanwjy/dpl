// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/internal/ilogb.h"
#include "dpl/core/math/internal/ldexp.h"
#include "dpl/core/math/internal/pair.h"
#include "dpl/core/math/internal/polynomial.h"
#include "dpl/core/math/round.h"
#include "dpl/core/math/sign.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/operations/bit.h"
#  include "dpl/core/operations/bitwise.h"
#  include "dpl/core/operations/compare.h"
#  include "dpl/core/operations/logic.h"
#  include "dpl/core/operations/negate.h"
#  include "dpl/core/operations/select.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
class sinhcosh_base {
private:
    template <floating_point E>
    static constexpr auto polynomial = []() {
        if constexpr (common_float_with<E, float>) {
            return fmath::polynomial<0.4166637361e-1f, //
                0.8333456703e-2f,                      //
                0.1394256484e-2f,                      //
                0.1980960224e-3f                       //
                >{};
        } else {
            static_assert(common_float_with<E, double>);
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
    static constexpr auto is_exp_underflow(basic_simd<E, A> val) noexcept {
        if constexpr (common_float_with<E, float>) {
            return val < -103.97208f;
        } else {
            static_assert(common_float_with<E, double>);
            return val < -745.133;
        }
    }

    template <floating_point E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr fmath::pair<E, A> DPL_VECTORCALL
        exp(basic_simd<E, A> arg) noexcept {
        using simdf = basic_simd<E, A>;
        auto const u = arg * fmath::inv_ln2;
        auto const qf =
            dx::round(u, rounding::to_nearest_int | rounding::no_exc);
        constexpr auto nln2 = -fmath::ln2_v<fmath::pair<E, A>>;
        auto s = fmath::single(arg) + qf * nln2.upper;
        s = s + qf * nln2.lower;

        auto const poly = polynomial<E>(s.upper);
        auto t = [&]() {
            if constexpr (common_float_with<E, float>) {
                auto t = s * poly + onesixth<E, A>;
                t = s * t + fmath::half;
                t = s + fmath::square(s) * t;
                return fmath::fast(dx::one_v<simdf>) + t;
            } else {
                static_assert(common_float_with<E, double>);
                auto t = fmath::single(dx::broadcast<simdf>(fmath::half)) +
                    s * onesixth<E, A>;
                t = fmath::single(dx::one_v<simdf>) + t * s;
                t = fmath::single(dx::one_v<simdf>) + t * s;
                auto const s4 = [](auto s2) { return s2 * s2; }(
                                    s.upper * s.upper);
                return t + s4;
            }
        }();

        auto const q = dx::cast<signed_rep_t<E>>(qf);
        t.upper = fmath::ldexp(fmath::compliance::speed, t.upper, q);
        t.lower = fmath::ldexp(fmath::compliance::speed, t.lower, q);
        auto const underflow = is_exp_underflow(arg);
        t.upper = dx::bit_drop(underflow, t.upper);
        t.lower = dx::bit_drop(underflow, t.lower);
        return t;
    }

public:
    __DPL_HIDE_FROM_ABI constexpr ~sinhcosh_base() = default;

    template <integral auto V, floating_point E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_simd<E, A> DPL_VECTORCALL
        fallback(basic_simd<E, A> const arg) noexcept {
        using simdf = basic_simd<E, A>;
        static constexpr make_immediate_mask_t<simdf, V> mask;

        auto const absarg = dx::abs(arg);
        auto const pair = [](fmath::pair<E, A> p) {
            auto const inv_p = fmath::rcp(p);

            if constexpr (dx::all_of(mask)) {
                return p + inv_p;
            } else if constexpr (dx::none_of(mask)) {
                return p - inv_p;
            } else {
                return p -
                    fmath::make_pair(dx::negatei<V>(inv_p.upper),
                        dx::negatei<V>(inv_p.lower));
            }
        }(exp(absarg));

        auto result = (pair.upper + pair.lower) * fmath::half;
        if constexpr (common_float_with<E, float>) {
            result = dx::select(
                (absarg <= 89.0f) & dx::isfinite(result), result, dx::infinity);
        } else {
            result = dx::select(
                (absarg <= 710.0) & dx::isfinite(result), result, dx::infinity);
        }

        if constexpr (dx::none_of(mask)) {
            result = dx::sign(result, arg);
        } else if constexpr (dx::some_of(mask)) {
            result = dx::sign(result, dx::bit_dropi<V>(arg));
        }

        return dx::bit_fill(dx::isnan(arg), result);
    }

    // TODO
    template <integral auto V, floating_point E, simd_abi A>
    requires (dx::digits_v<E><dx::digits_v<float> || dx::digits_v<E>>
                     dx::digits_v<double>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, PURE, NODISCARD)
    static constexpr basic_simd<E, A> fallback(basic_simd<E, A>) = delete;
};

void sinh(...) noexcept = delete;
void cosh(...) noexcept = delete;
template <auto>
void sinhcosh(...) noexcept = delete;

struct sinh_t : private internal::sinhcosh_base {

    template <floating_point_simd T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val) noexcept {
        if constexpr (requires {
                          {
                              sinh(internal::abi<T>, val)
                          } -> equivalent_simd_as<T>;
                      }) {
            if constexpr (basic_simd_type<T>) {
                if not consteval {
                    return sinh(internal::abi<T>, val);
                } else {
                    return internal::sinhcosh_base::fallback<0>(val);
                }
            } else {
                return sinh(internal::abi<T>, val);
            }
        } else if constexpr (basic_simd_type<T>) {
            return internal::sinhcosh_base::fallback<0>(val);
        } else {
            return operator()(dx::to_basic_type(val));
        }
    }
};

struct cosh_t : private internal::sinhcosh_base {

    template <floating_point_simd T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val) noexcept {
        if constexpr (requires {
                          {
                              cosh(internal::abi<T>, val)
                          } -> equivalent_simd_as<T>;
                      }) {
            if constexpr (basic_simd_type<T>) {
                if not consteval {
                    return cosh(internal::abi<T>, val);
                } else {
                    return internal::sinhcosh_base::fallback<-1>(val);
                }
            } else {
                return cosh(internal::abi<T>, val);
            }
        } else if constexpr (basic_simd_type<T>) {
            return internal::sinhcosh_base::fallback<-1>(val);
        } else {
            return operator()(dx::to_basic_type(val));
        }
    }
};

template <integral auto V>
struct sinhcoshi_t : private internal::sinhcosh_base {
    template <floating_point_simd T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val) noexcept {
        if constexpr (requires {
                          {
                              sinhcosh<V>(internal::abi<T>, val)
                          } -> equivalent_simd_as<T>;
                      }) {
            if constexpr (basic_simd_type<T>) {
                if not consteval {
                    return sinhcosh<V>(internal::abi<T>, val);
                } else {
                    return internal::sinhcosh_base::fallback<V>(val);
                }
            } else {
                return sinhcosh<V>(internal::abi<T>, val);
            }
        } else if constexpr (basic_simd_type<T>) {
            return internal::sinhcosh_base::fallback<V>(val);
        } else {
            return operator()(dx::to_basic_type(val));
        }
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::sinh_t sinh{};
DPL_EXPORT inline constexpr internal::cosh_t cosh{};
DPL_EXPORT template <integral auto V>
inline constexpr internal::sinhcoshi_t<V> sinhcoshi{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
