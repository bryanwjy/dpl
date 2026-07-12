// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/details/fwd.h" // IWYU pragma: export

#include "dpl/core/math/details/floating_point_simd.h"
#include "dpl/core/math/fma.h"

#if !DPL_MODULES
#  include "dpl/core/basic/basic_vector.h" // IWYU pragma: export
#  include "dpl/core/basic/broadcast.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_element.h"
#  include "dpl/core/immediate/constants/ln2.h"
#  include "dpl/core/immediate/constants/one.h"
#  include "dpl/core/immediate/constants/zero.h"
#  include "dpl/core/operations/arithmetic/abs.h"
#  include "dpl/core/operations/select.h"
#  include "dpl/std/concepts/convertible_to.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
struct rsqrt_t;
struct sqrt_t;
} // namespace datapar::internal

namespace datapar::fmath {

template <floating_point_like T, simd_abi A>
requires simd_floating_point_for<T, A>
struct pair {
    using value_type = T;
    using abi_type = A;
    using element_type = basic_vector<T, A>;

    basic_vector<T, A> upper;
    basic_vector<T, A> lower;

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr pair DPL_VECTORCALL operator+(
        this pair self, element_type right) noexcept {
        auto s = self.upper + right;
        auto v = s - self.upper;
        return pair{
            .upper = s,
            .lower = self.upper - (s - v) + (right - v) + self.lower,
        };
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    friend constexpr pair DPL_VECTORCALL operator+(
        element_type left, pair right) noexcept {
        auto s = left + right.upper;
        auto v = s - left;
        return pair{
            .upper = s,
            .lower = left - (s - v) + (right.upper - v) + right.lower,
        };
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr pair DPL_VECTORCALL operator+(
        this pair self, pair right) noexcept {
        auto s = self.upper + right.upper;
        auto v = s - self.upper;
        auto t = self.upper - (s - v) + (right.upper - v);
        return pair{
            .upper = s,
            .lower = t + (self.lower + right.lower),
        };
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr pair DPL_VECTORCALL operator-(
        this pair self, element_type right) noexcept {
        auto s = self.upper - right;
        return pair{
            .upper = s,
            .lower = self.upper - s - right + self.lower,
        };
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    friend constexpr pair DPL_VECTORCALL operator-(
        element_type left, pair right) noexcept {
        auto s = left - right.upper;
        return pair{
            .upper = s,
            .lower = left - s - right.upper - right.lower,
        };
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr pair DPL_VECTORCALL operator-(
        this pair self, pair right) noexcept {
        auto s = self.upper - right.upper;
        auto v = s - self.upper;
        auto t = self.upper - (s - v) - (right.upper + v);
        return pair{
            .upper = s,
            .lower = t + (self.lower - right.lower),
        };
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr pair DPL_VECTORCALL operator-(this pair self) noexcept {
        return pair{
            .upper = -self.upper,
            .lower = -self.lower,
        };
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr pair DPL_VECTORCALL operator/(
        this pair self, pair right) noexcept {
        auto t = dx::one / right.upper;
        auto s = self.upper * t;
        auto u = dx::fmsub(t, self.upper, s);
        auto v =
            dx::fnmadd(right.lower, t, dx::fnmadd(right.upper, t, dx::one));
        return pair{
            .upper = s,
            .lower = dx::fmadd(s, v, dx::fmadd(self.lower, t, u)),
        };
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr pair DPL_VECTORCALL operator*(
        this pair self, pair right) noexcept {

        auto const s = self.upper * right.upper;
        return pair{
            .upper = s,
            .lower = dx::fmadd(self.upper, right.lower,
                dx::fmadd(self.lower, right.upper,
                    dx::fmsub(self.upper, right.upper, s))),
        };
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr pair DPL_VECTORCALL operator*(
        this pair self, element_type right) noexcept {
        auto const s = self.upper * right;
        return pair{
            .upper = s,
            .lower =
                dx::fmadd(self.lower, right, dx::fmsub(self.upper, right, s)),
        };
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    friend constexpr pair DPL_VECTORCALL operator*(
        element_type left, pair right) noexcept {
        auto const s = left * right.upper;
        return pair{
            .upper = s,
            .lower =
                dx::fmadd(left, right.lower, dx::fmsub(left, right.upper, s)),
        };
    }
};

template <canonical_vector T>
using pair_of = pair<simd_element_type_t<T>, simd_abi_type_t<T>>;

template <typename T>
inline constexpr bool is_pair = false;
template <simd_abi A, simd_element_for<A> E>
inline constexpr bool is_pair<pair<E, A>> = true;

template <typename T>
concept pair_type = is_pair<T> && requires {
    typename T::value_type;
    typename T::abi_type;
    typename T::element_type;
    requires simd_abi<typename T::abi_type>;
    requires simd_element_for<typename T::value_type, typename T::abi_type>;
    requires simd_vector<typename T::element_type>;
    requires same_as<typename T::element_type,
        basic_vector<typename T::value_type, typename T::abi_type>>;
};

template <simd_vector T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
constexpr pair_of<T>
    DPL_VECTORCALL make_pair(
        T upper, T lower = dx::broadcast<T>(dx::zero)) noexcept {
    return pair_of<T>{
        .upper = upper,
        .lower = lower,
    };
}

template <simd_abi A, simd_floating_point_for<A> E,
    broadcastable_to<basic_vector<E, A>> T1,
    broadcastable_to<basic_vector<E, A>> T2 = dx::zero_t>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
constexpr pair<E, A>
    DPL_VECTORCALL make_pair(T1 upper, T2 lower = {}) noexcept {
    return pair<E, A>{
        .upper = dx::broadcast<E, A>(upper),
        .lower = dx::broadcast<E, A>(lower),
    };
}

template <simd_abi A, simd_floating_point_for<A> E,
    broadcastable_to<basic_vector<E, A>> T2 = dx::zero_t>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
constexpr pair<E, A>
    DPL_VECTORCALL make_pair(E upper, T2 lower = {}) noexcept {
    using simd = basic_vector<E, A>;
    return pair<E, A>{
        .upper = dx::broadcast<E, A>(upper),
        .lower = dx::broadcast<E, A>(lower),
    };
}

template <simd_abi A, simd_floating_point_for<A> E, simd_element_for<A> C>
requires common_size_with<E, C>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
constexpr pair<E, A>
    DPL_VECTORCALL select(
        basic_mask<C, A> mask, pair<E, A> left, pair<E, A> right) noexcept {
    return pair<E, A>{
        .upper = dx::select(mask, left.upper, right.upper),
        .lower = dx::select(mask, left.lower, right.lower),
    };
}

template <pair_type T, broadcastable_to<typename T::element_type> C>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
constexpr T broadcast(C arg) noexcept {
    using simd = typename T::element_type;
    return T{
        .upper = dx::broadcast<simd>(arg),
        .lower = dx::broadcast<simd>(dx::zero),
    };
}

template <simd_abi A, simd_floating_point_for<A> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
constexpr pair<E, A>
    DPL_VECTORCALL abs(pair<E, A> arg) noexcept {
    return pair<E, A>{
        .upper = dx::abs(arg.upper),
        .lower = dx::abs(arg.lower),
    };
}

template <simd_abi A, simd_floating_point_for<A> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
constexpr pair<E, A>
    DPL_VECTORCALL normalize(pair<E, A> arg) noexcept {
    auto upper = arg.upper + arg.lower;
    return pair<E, A>{
        .upper = upper,
        .lower = arg.upper - upper + arg.lower,
    };
}

template <simd_abi A, simd_floating_point_for<A> E, typename S>
requires requires(pair<E, A> p, S scale) {
    p.upper * scale;
    p.lower * scale;
}
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
constexpr pair<E, A>
    DPL_VECTORCALL scale(pair<E, A> arg, S scale) noexcept {
    return pair<E, A>{
        .upper = arg.upper * scale,
        .lower = arg.lower * scale,
    };
}

template <simd_abi A, simd_floating_point_for<A> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
constexpr pair<E, A>
    DPL_VECTORCALL square(pair<E, A> arg) noexcept {
    auto s = arg.upper * arg.upper;
    return pair<E, A>{
        .upper = s,
        .lower = dx::fmadd(        //
            arg.upper + arg.upper, //
            arg.lower,             //
            dx::fmsub(arg.upper, arg.upper, s)),
    };
}

template <simd_abi A, simd_floating_point_for<A> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
constexpr pair<E, A>
    DPL_VECTORCALL rcp(pair<E, A> arg) noexcept {
    auto s = dx::one / arg.upper;
    return pair<E, A>{
        .upper = s,
        .lower =
            s * dx::fnmadd(arg.lower, s, dx::fnmadd(arg.upper, s, dx::one)),
    };
}

struct frsqrt_t : public dx::internal::operation_base<dx::internal::rsqrt_t> {
    using operation_base<dx::internal::rsqrt_t>::operator();
};

template <simd_abi A, simd_floating_point_for<A> E>
requires internal::cpo_invocable<dx::internal::rsqrt_t,
    typename pair<E, A>::element_type>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
constexpr pair<E, A>
    DPL_VECTORCALL sqrt(pair<E, A> arg) noexcept {
    constexpr frsqrt_t rsqrt;
    using simd = typename pair<E, A>::element_type;
    auto x = rsqrt(arg.upper + arg.lower);
    auto r = arg * x;
    constexpr auto n3 = dx::broadcast<simd>(-3.0);
    return fmath::scale(r * (r * x + n3), -0.5);
}

template <typename T>
struct fast;
template <typename T>
struct single;

template <floating_point_like E, simd_abi A>
requires simd_floating_point_for<E, A> &&
    requires { typename basic_vector<E, A>; }
struct single<basic_vector<E, A>> {
    using element_type = basic_vector<E, A>;
    element_type value;
    __DPL_HIDE_FROM_ABI explicit constexpr single(element_type val) noexcept
        : value(val) {}

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    constexpr operator element_type(this single self) noexcept {
        return self.value;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr auto DPL_VECTORCALL operator+(
        this single self, element_type right) noexcept {
        // Knuth's twosum
        auto s = self.value + right;
        auto v = s - self.value;
        return pair<E, A>{
            .upper = s,
            .lower = (self.value - (s - v)) + (right - v),
        };
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    friend constexpr auto DPL_VECTORCALL operator+(
        element_type left, single right) noexcept {
        return single(left) + right.value;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr auto DPL_VECTORCALL operator-(
        this single self, element_type right) noexcept {
        // Knuth's twodiff
        auto s = self.value - right;
        auto v = s - self.value;
        return pair<E, A>{
            .upper = s,
            .lower = (self.value - (s - v)) - (right + v),
        };
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    friend constexpr auto DPL_VECTORCALL operator-(
        element_type left, single right) noexcept {
        return single(left) - right.value;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr auto DPL_VECTORCALL operator*(
        this single self, element_type right) noexcept {
        auto s = self.value * right;
        return pair<E, A>{
            .upper = s,
            .lower = dx::fmsub(self.value, right, s),
        };
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    friend constexpr auto DPL_VECTORCALL operator*(
        element_type left, single right) noexcept {
        return single(left) * right.value;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr auto DPL_VECTORCALL operator*(
        this single self, pair_of<element_type> right) noexcept {
        auto hi = self.value * right.upper;
        auto lo = dx::fmadd(
            self.value, right.lower, dx::fmsub(self.value, right.upper, hi));
        return hi + lo;
    }
};

template <simd_vector T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
constexpr pair_of<T>
    DPL_VECTORCALL rcp(single<T> arg) noexcept {
    auto s = one_v<T> / arg.value;
    return pair_of<T>{
        .upper = s,
        .lower = s * dx::fnmadd(arg.value, s, dx::one),
    };
}

struct fsqrt_t : public dx::internal::operation_base<dx::internal::sqrt_t> {
    using operation_base<dx::internal::sqrt_t>::operator();
};

template <simd_vector T>
requires dx::internal::cpo_invocable<dx::internal::sqrt_t, T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
constexpr pair_of<T>
    DPL_VECTORCALL sqrt(single<T> arg) noexcept {
    constexpr fsqrt_t sqrt;
    auto t = sqrt(arg.value);
    return fmath::scale(
        arg + (single(t) * t) * fmath::rcp(single(t)), dx::broadcast<T>(0.5));
}

template <floating_point_like E, simd_abi A>
requires pair_type<pair<E, A>>
struct single<pair<E, A>> {
    using element_type = pair<E, A>;
    pair<E, A> value;

    __DPL_HIDE_FROM_ABI explicit constexpr single(element_type val) noexcept
        : value(val) {}

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    constexpr operator element_type(this single self) noexcept {
        return self.value;
    }
};

template <floating_point_like E, simd_abi A>
explicit single(pair<E, A>) -> single<pair<E, A>>;
template <floating_point_like E, simd_abi A>
explicit single(basic_vector<E, A>) -> single<basic_vector<E, A>>;

template <pair_type T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
constexpr
    typename T::element_type DPL_VECTORCALL square(single<T> arg) noexcept {
    return dx::fmadd(arg.value.upper, arg.value.upper,
        [](auto val) { return val + val; }(arg.value.upper * arg.value.lower));
}

template <typename T>
struct element_type_if {};

template <typename T>
using element_type_if_t = typename element_type_if<T>::type;

template <typename T>
requires requires { typename T::element_type; }
struct element_type_if<T> {
    using type = typename T::element_type;
};

template <floating_point_like E, simd_abi A>
requires simd_floating_point_for<E, A> &&
    requires { typename basic_vector<E, A>; }
struct fast<basic_vector<E, A>> {
    using element_type = basic_vector<E, A>;
    basic_vector<E, A> value;

    __DPL_HIDE_FROM_ABI explicit constexpr fast(element_type val) noexcept
        : value(val) {}

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    constexpr operator element_type() const noexcept {
        return value;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr auto DPL_VECTORCALL operator+(
        this fast self, element_type right) noexcept {
        // |self.value| > |right|
        auto upper = self.value + right;
        return pair_of<element_type>{
            .upper = upper,
            .lower = self.value - upper + right,
        };
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    friend constexpr auto DPL_VECTORCALL operator+(
        element_type left, fast right) noexcept {
        return fast(left) + right.value;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr auto DPL_VECTORCALL operator-(
        this fast self, element_type right) noexcept {
        // |self.value| > |right|
        auto upper = self.value - right;
        return pair<E, A>{
            .upper = upper,
            .lower = self.value - upper - right,
        };
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr auto DPL_VECTORCALL operator+(
        this fast self, pair_of<element_type> right) noexcept {
        // |self.value| < |right.upper|
        auto upper = self.value + right.upper;
        return pair_of<element_type>{
            .upper = upper,
            .lower = (self.value - upper) + (right.upper + right.lower),
        };
    }
};

template <typename T>
requires pair_type<T>
struct fast<T> {
    static_assert(simd_vector<T> != pair_type<T>);
    using element_type = T;
    T value;

    __DPL_HIDE_FROM_ABI explicit constexpr fast(T val) noexcept : value(val) {}

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    constexpr operator T() const noexcept {
        return value;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr T DPL_VECTORCALL operator+(this fast self, T right) noexcept {
        // |left.upper| >= |right.upper|
        auto s = self.value.upper + right.upper;
        return T{
            .upper = s,
            .lower = (self.value.upper - s) + right.upper +
                (self.value.lower + right.lower),
        };
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr T DPL_VECTORCALL operator+(
        this fast self, element_type_if_t<T> right) noexcept {
        // |self.value.upper| > |right|
        auto s = self.value.upper + right;
        return T{
            .upper = s,
            .lower = (self.value.upper - s) + (right + self.value.lower),
        };
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    friend constexpr T DPL_VECTORCALL operator+(
        typename T::element_type left, fast right) noexcept {
        auto s = left + right.value.upper;
        return T{
            .upper = s,
            .lower = (left - s) + (right.value.upper + right.value.lower),
        };
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr T DPL_VECTORCALL operator-(
        this fast self, typename T::element_type right) noexcept {
        // |x| >= |y|
        auto s = self.value.upper - right;
        return pair{
            .upper = s,
            .lower = self.value.upper - s - right + self.value.lower,
        };
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr T DPL_VECTORCALL operator-(this fast self, T right) noexcept {
        // |x| >= |y|
        auto s = self.value.upper - right.upper;
        auto t = self.value.upper - s;
        t = t - right.upper;
        t = t + self.value.lower;
        return pair{
            .upper = s,
            .lower = t - right.lower,
        };
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    friend constexpr T DPL_VECTORCALL operator-(T left, fast right) noexcept {
        return fast(left) - right.value;
    }
};

template <typename T>
explicit fast(T) -> fast<T>;

struct ln2_t : dx::ln2_t {
    __DPL_HIDE_FROM_ABI explicit constexpr ln2_t() noexcept = default;

    template <simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator pair<float, A>(this ln2_t) noexcept {
        return pair<float, A>{
            .upper = dx::broadcast<A>(0.693145751953125f),
            .lower = dx::broadcast<A>(1.428606765330187045e-06f),
        };
    }

    template <simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator pair<double, A>(this ln2_t) noexcept {
        return pair<double, A>{
            .upper =
                dx::broadcast<A>(0.69314718055966295651160180568695068359375),
            .lower = dx::broadcast<A>(
                0.28235290563031577122588448175013436025525412068e-12),
        };
    }
};

inline constexpr ln2_t ln2{};

template <typename T>
requires explicitly_convertible_to<ln2_t, T>
inline constexpr auto ln2_v = static_cast<T>(ln2);

} // namespace datapar::fmath
__DPL_DEFAULT_NAMESPACE_END
