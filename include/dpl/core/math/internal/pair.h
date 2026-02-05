// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/fma.h"
#include "dpl/core/math/rsqrt.h"
#include "dpl/core/math/sqrt.h"

#if !DPL_MODULES
#  include "dpl/core/basic/basic_simd.h" // IWYU pragma: export
#  include "dpl/core/basic/broadcast.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_element.h"
#  include "dpl/core/constants/ln2.h"
#  include "dpl/core/constants/one.h"
#  include "dpl/core/constants/zero.h"
#  include "dpl/core/operations/abs.h"
#  include "dpl/core/operations/select.h"
#  include "dpl/std/concepts/convertible_to.h"
#  include "dpl/std/concepts/floating_point.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::fmath {
namespace dx = __DPL datapar;

template <typename T, typename... Args>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr auto cpo(Args... args) noexcept {
    constexpr T func;
    return func(args...);
}

template <simd_element T, simd_abi A>
requires floating_point<T>
struct pair {
    using value_type = T;
    using abi_type = A;
    using element_type = basic_simd<T, A>;

    basic_simd<T, A> upper;
    basic_simd<T, A> lower;

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

template <basic_simd_type T>
using pair_of = pair<typename T::value_type, typename T::abi_type>;

template <typename T>
inline constexpr bool is_pair = false;
template <simd_element T, simd_abi A>
requires floating_point<T>
inline constexpr bool is_pair<pair<T, A>> = true;

template <typename T, typename A>
concept abi_float_type = simd_element<T> && simd_abi<A> && floating_point<T> &&
    requires { typename A::template native_type<T>; };

template <typename T>
concept pair_type = is_pair<T> && requires {
    typename T::value_type;
    requires simd_element<typename T::value_type>;
    typename T::abi_type;
    requires simd_abi<typename T::abi_type>;
    typename T::element_type;
    requires simd_type<typename T::element_type>;
    requires same_as<typename T::element_type,
        basic_simd<typename T::value_type, typename T::abi_type>>;
} && abi_float_type<typename T::value_type, typename T::abi_type>;

template <simd_type T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
constexpr pair_of<T> DPL_VECTORCALL
    make_pair(T upper, T lower = dx::broadcast<T>(dx::zero)) noexcept {
    return pair_of<T>{
        .upper = upper,
        .lower = lower,
    };
}

template <simd_element E, simd_abi A, broadcastable_to<basic_simd<E, A>> T1,
    broadcastable_to<basic_simd<E, A>> T2 = dx::zero_t>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
constexpr pair<E, A> DPL_VECTORCALL
    make_pair(T1 upper, T2 lower = {}) noexcept {
    using simd = basic_simd<E, A>;
    return pair_of<simd>{
        .upper = dx::broadcast<simd>(upper),
        .lower = dx::broadcast<simd>(lower),
    };
}

template <simd_abi A, simd_element E,
    broadcastable_to<basic_simd<E, A>> T2 = dx::zero_t>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
constexpr pair<E, A> DPL_VECTORCALL make_pair(E upper, T2 lower = {}) noexcept {
    using simd = basic_simd<E, A>;
    return pair_of<simd>{
        .upper = dx::broadcast<simd>(upper),
        .lower = dx::broadcast<simd>(lower),
    };
}

template <simd_abi A, abi_float_type<A> T, simd_element E>
requires common_size_with<T, E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
constexpr pair<T, A> DPL_VECTORCALL
    select(simd_mask<E, A> mask, pair<T, A> left, pair<T, A> right) noexcept {
    return pair<T, A>{
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

template <simd_abi A, abi_float_type<A> T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
constexpr pair<T, A> DPL_VECTORCALL abs(pair<T, A> arg) noexcept {
    return pair<T, A>{
        .upper = dx::abs(arg.upper),
        .lower = dx::abs(arg.lower),
    };
}

template <simd_abi A, abi_float_type<A> T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
constexpr pair<T, A> DPL_VECTORCALL normalize(pair<T, A> arg) noexcept {
    auto upper = arg.upper + arg.lower;
    return pair<T, A>{
        .upper = upper,
        .lower = arg.upper - upper + arg.lower,
    };
}

template <simd_abi A, abi_float_type<A> T, typename S>
requires requires(pair<T, A> p, S scale) {
    p.upper * scale;
    p.lower * scale;
}
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
constexpr pair<T, A> DPL_VECTORCALL scale(pair<T, A> arg, S scale) noexcept {
    return pair<T, A>{
        .upper = arg.upper * scale,
        .lower = arg.lower * scale,
    };
}

template <simd_abi A, abi_float_type<A> T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
constexpr pair<T, A> DPL_VECTORCALL square(pair<T, A> arg) noexcept {
    auto s = arg.upper * arg.upper;
    return pair<T, A>{
        .upper = s,
        .lower = dx::fmadd(        //
            arg.upper + arg.upper, //
            arg.lower,             //
            dx::fmsub(arg.upper, arg.upper, s)),
    };
}

template <simd_abi A, abi_float_type<A> T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
constexpr pair<T, A> DPL_VECTORCALL rcp(pair<T, A> arg) noexcept {
    auto s = dx::one / arg.upper;
    return pair<T, A>{
        .upper = s,
        .lower =
            s * dx::fnmadd(arg.lower, s, dx::fnmadd(arg.upper, s, dx::one)),
    };
}

template <simd_abi A, abi_float_type<A> T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
constexpr pair<T, A> DPL_VECTORCALL sqrt(pair<T, A> arg) noexcept {
    using simd = typename pair<T, A>::element_type;
    auto x = dx::rsqrt(arg.upper + arg.lower);
    auto r = arg * x;
    constexpr auto n3 = dx::broadcast<simd>(-3.0);
    return fmath::scale(r * (r * x + n3), -0.5);
}

template <typename T>
struct fast;
template <typename T>
struct single;

template <simd_element T, simd_abi A>
requires simd_type<basic_simd<T, A>>
struct single<basic_simd<T, A>> {
    using element_type = basic_simd<T, A>;
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
        auto s = self.value + right;
        auto v = s - self.value;
        return pair<T, A>{
            .upper = s,
            .lower = self.value - (s - v) + (right - v),
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
        auto s = self.value - right;
        return pair<T, A>{
            .upper = s,
            .lower = (self.value - s) - right,
        };
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    friend constexpr auto DPL_VECTORCALL operator-(
        element_type left, single right) noexcept {
        return single(left) + right.value;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr auto DPL_VECTORCALL operator*(
        this single self, element_type right) noexcept {
        auto s = self.value * right;
        return pair<T, A>{
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

template <simd_type T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
constexpr pair_of<T> DPL_VECTORCALL rcp(single<T> arg) noexcept {
    auto s = one_v<T> / arg.value;
    return pair_of<T>{
        .upper = s,
        .lower = s * dx::fnmadd(arg.value, s, dx::one),
    };
}

template <simd_type T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
constexpr pair_of<T> DPL_VECTORCALL sqrt(single<T> arg) noexcept {
    auto t = dx::sqrt(arg.value);
    return fmath::scale(
        arg + (single(t) * t) * fmath::rcp(single(t)), dx::broadcast<T>(0.5));
}

template <simd_element T, simd_abi A>
requires pair_type<pair<T, A>>
struct single<pair<T, A>> {
    using element_type = pair<T, A>;
    pair<T, A> value;

    __DPL_HIDE_FROM_ABI explicit constexpr single(element_type val) noexcept
        : value(val) {}

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    constexpr operator element_type(this single self) noexcept {
        return self.value;
    }
};

template <simd_element T, simd_abi A>
explicit single(pair<T, A>) -> single<pair<T, A>>;
template <simd_element T, simd_abi A>
explicit single(basic_simd<T, A>) -> single<basic_simd<T, A>>;

template <pair_type T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
constexpr typename T::element_type DPL_VECTORCALL
    square(single<T> arg) noexcept {
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

template <simd_element E, simd_abi A>
struct fast<basic_simd<E, A>> {
    using element_type = basic_simd<E, A>;
    basic_simd<E, A> value;

    __DPL_HIDE_FROM_ABI explicit constexpr fast(element_type val) noexcept
        : value(val) {}

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    constexpr operator element_type() const noexcept { return value; }

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
    static_assert(simd_type<T> != pair_type<T>);
    using element_type = T;
    T value;

    __DPL_HIDE_FROM_ABI explicit constexpr fast(T val) noexcept : value(val) {}

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    constexpr operator T() const noexcept { return value; }

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

inline constexpr struct ln2_t : dx::ln2_t {
    __DPL_HIDE_FROM_ABI explicit constexpr ln2_t() noexcept = default;

    template <pair_type T>
    requires common_float_with<typename T::value_type, float>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator T(this ln2_t) noexcept {
        using A = typename T::abi_type;
        return T{
            .upper = dx::broadcast<A>(0.693145751953125f),
            .lower = dx::broadcast<A>(1.428606765330187045e-06f),
        };
    }

    template <pair_type T>
    requires common_float_with<typename T::value_type, double>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator T(this ln2_t) noexcept {
        using A = typename T::abi_type;
        return T{
            .upper =
                dx::broadcast<A>(0.69314718055966295651160180568695068359375),
            .lower = dx::broadcast<A>(
                0.28235290563031577122588448175013436025525412068e-12),
        };
    }
} ln2{};

template <typename T>
requires explicitly_convertible_to<ln2_t, T>
inline constexpr auto ln2_v = static_cast<T>(ln2);

} // namespace datapar::fmath
DPL_DEFAULT_NAMESPACE_END
