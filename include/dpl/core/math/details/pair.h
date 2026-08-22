// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/details/fwd.h" // IWYU pragma: export

#include "dpl/core/math/details/floating_point_simd.h"
#include "dpl/core/math/mulx.h"

#if !DPL_MODULES
#  include "dpl/core/basic/basic_vector.h" // IWYU pragma: export
#  include "dpl/core/basic/broadcast.h"
#  include "dpl/core/basic/tuple_access.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_element.h"
#  include "dpl/core/immediate/constants/ln2.h"
#  include "dpl/core/immediate/constants/one.h"
#  include "dpl/core/immediate/constants/zero.h"
#  include "dpl/core/operations/arithmetic.h"
#  include "dpl/core/operations/select.h"
#  include "dpl/std/concepts/convertible_to.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
struct rsqrt_t;
struct sqrt_t;
} // namespace datapar::internal

namespace datapar::fmath {
template <floating_point_like T, fixed_width_abi A>
struct basic_pair : simd_tuple_base<basic_pair<T, A>> {
    using value_type = T;
    using abi_type = A;
    using element_type = basic_vector<T, A>;

    basic_vector<T, A> upper;
    basic_vector<T, A> lower;

    template <size_t I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr element_type get(this basic_pair const& self) noexcept {
        if constexpr (I == 0) {
            return self.upper;
        } else {
            static_assert(I == 1);
            return self.lower;
        }
    }

    template <size_t I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr element_type& get(this basic_pair& self) noexcept {
        if constexpr (I == 0) {
            return self.upper;
        } else {
            static_assert(I == 1);
            return self.lower;
        }
    }
};

template <canonical_vector T>
struct pair_type {
    using type DPL_NODEBUG =
        basic_pair<simd_element_type_t<T>, simd_abi_type_t<T>>;
};

/**
 * Scalable ABI workaround
 * We don't explicit target scalable abi but check for
 * `same_as<T, simd_native_type_t<T>>`
 */
template <typename T>
concept has_native_pair = canonical_vector<T> &&
    same_as<T, simd_native_type_t<T>> && requires(T vec) {
        typename simd_abi_traits<T>::template native_tuple<2>;
        make_tuple(dx::internal::abi<T>, vec, vec);
    };

template <canonical_vector T>
requires has_native_pair<T>
struct pair_type<T> {
    using type DPL_NODEBUG =
        typename simd_abi_traits<T>::template native_tuple<2>;
};

template <typename T>
using pair_type_t DPL_NODEBUG = typename pair_type<T>::type;

template <typename T>
concept simd_double =
    simd_tuple<T> && same_as<T, pair_type_t<simd_element_type_t<T>>>;

template <canonical_vector T>
requires has_native_pair<T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr pair_type_t<T> make_pair(
    T upper, T lower = dx::broadcast<T>(dx::zero)) noexcept {
    return make_tuple(dx::internal::abi<T>, upper, lower);
}

template <canonical_vector T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
constexpr pair_type_t<T>
    DPL_VECTORCALL make_pair(
        T upper, T lower = dx::broadcast<T>(dx::zero)) noexcept {
    return pair_type_t<T>{
        .upper = upper,
        .lower = lower,
    };
}

template <simd_abi A, floating_point_like E>
requires requires(make_canonical_vector_t<E, A> vec) {
    fmath::make_pair<make_canonical_vector_t<E, A>>(vec, vec);
}
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
constexpr pair_type_t<make_canonical_vector_t<E, A>>
    DPL_VECTORCALL make_pair(E upper, E lower = dx::zero) noexcept {
    return fmath::make_pair(
        dx::broadcast<E, A>(upper), dx::broadcast<E, A>(lower));
}

/**
 * Scalable ABI workaround
 * For ADL operations
 */
template <simd_double T>
struct pair_ref {
    using value_type DPL_NODEBUG = simd_value_type_t<T>;
    using abi_type DPL_NODEBUG = simd_abi_type_t<T>;
    using element_type DPL_NODEBUG = simd_element_type_t<T>;

    __DPL_HIDE_FROM_ABI constexpr pair_ref(
        T const& ref DPL_LIFETIMEBOUND) noexcept
        : value(ref) {}

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) constexpr operator T(
        this pair_ref self) noexcept {
        return self.value;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) constexpr T operator+(
        this pair_ref self) noexcept {
        return self.value;
    }

    template <size_t I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr element_type get(this pair_ref self) noexcept {
        return dx::get_element<I>(self.value);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, PURE, NODISCARD)
    constexpr T DPL_VECTORCALL operator+(
        this pair_ref self, element_type right) noexcept {
        auto const [upper, lower] = self;
        auto s = dx::add(upper, right);
        auto v = dx::subtract(s, upper);
        return fmath::make_pair(s,
            dx::add(lower,
                dx::add(dx::subtract(upper, dx::subtract(s, v)),
                    dx::subtract(right, v))));
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, PURE, NODISCARD)
    friend constexpr T DPL_VECTORCALL operator+(
        element_type left, pair_ref self) noexcept {
        auto const [upper, lower] = self;
        auto s = dx::add(left, upper);
        auto v = dx::subtract(s, left);

        return fmath::make_pair(s,
            dx::add(lower,
                dx::add(dx::subtract(left, dx::subtract(s, v)),
                    dx::subtract(upper, v))));
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, PURE, NODISCARD)
    constexpr T DPL_VECTORCALL operator+(
        this pair_ref left, pair_ref right) noexcept {
        auto const [lupper, llower] = left;
        auto const [rupper, rlower] = right;
        auto s = dx::add(lupper, rupper);
        auto v = dx::subtract(s, lupper);
        auto t = dx::add(
            dx::subtract(rupper, v), dx::subtract(lupper, dx::subtract(s, v)));
        return fmath::make_pair(s, dx::add(t, dx::add(llower, rlower)));
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr T operator+(T left, pair_ref self) noexcept {
        return pair_ref{left} + self;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, PURE, NODISCARD)
    constexpr T DPL_VECTORCALL operator-(
        this pair_ref self, element_type right) noexcept {
        auto const [upper, lower] = self;
        auto s = dx::subtract(upper, right);
        return fmath::make_pair(
            s, dx::add(lower, dx::subtract(dx::subtract(upper, s), right)));
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, PURE, NODISCARD)
    friend constexpr T DPL_VECTORCALL operator-(
        element_type left, pair_ref self) noexcept {
        auto const [upper, lower] = self;
        auto s = dx::subtract(left, upper);
        return fmath::make_pair(
            s, dx::subtract(dx::subtract(dx::subtract(left, s), upper)), lower);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, PURE, NODISCARD)
    constexpr T DPL_VECTORCALL operator-(
        this pair_ref self, pair_ref right) noexcept {
        auto const [lupper, llower] = self;
        auto const [rupper, rlower] = right;

        auto s = dx::subtract(lupper, rupper);
        auto v = dx::subtract(s, lupper);
        auto t = dx::subtract(
            dx::subtract(lupper, dx::subtract(s, v)), dx::add(rupper, v));
        return fmath::make_pair(s, dx::add(t, dx::subtract(llower, rlower)));
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr T operator-(T left, pair_ref self) noexcept {
        return pair_ref{left} - self;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, PURE, NODISCARD)
    constexpr T DPL_VECTORCALL operator-(this pair_ref self) noexcept {
        auto const [upper, lower] = self;
        return fmath::make_pair(dx::negate(upper), dx::negate(lower));
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, PURE, NODISCARD)
    constexpr T DPL_VECTORCALL operator/(
        this pair_ref self, pair_ref right) noexcept {
        auto const [lupper, llower] = self;
        auto const [rupper, rlower] = right;
        auto const one = dx::broadcast<T>(dx::one);

        auto t = dx::divide(one, rupper);
        auto s = dx::multiply(lupper, t);
        auto u = dx::mulsub(t, lupper, s);
        auto v = dx::nmuladd(rlower, t, dx::nmuladd(rupper, t, one));
        return fmath::make_pair(s, dx::muladd(s, v, dx::muladd(llower, t, u)));
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr T operator/(T left, pair_ref self) noexcept {
        return pair_ref{left} / self;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, PURE, NODISCARD)
    constexpr T DPL_VECTORCALL operator*(
        this pair_ref self, element_type right) noexcept {
        auto const [upper, lower] = self;
        auto const s = dx::multiply(upper, right);
        return fmath::make_pair(
            s, dx::muladd(lower, right, dx::mulsub(upper, right, s)));
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, PURE, NODISCARD)
    friend constexpr T DPL_VECTORCALL operator*(
        element_type left, pair_ref self) noexcept {
        auto const [upper, lower] = self;
        auto const s = dx::multiply(left, upper);
        return fmath::make_pair(
            s, dx::muladd(left, lower, dx::mulsub(left, upper, s)));
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, PURE, NODISCARD)
    constexpr T DPL_VECTORCALL operator*(
        this pair_ref self, pair_ref right) noexcept {
        auto const [lupper, llower] = self;
        auto const [rupper, rlower] = right;

        auto const s = dx::multiply(lupper, rupper);
        return fmath::make_pair(s,
            dx::muladd(lupper, rlower,
                dx::muladd(llower, rupper, dx::mulsub(lupper, rupper, s))));
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr T operator*(T left, pair_ref self) noexcept {
        return pair_ref{left} * self;
    }

    T const& value;
};

template <simd_double T>
explicit pair_ref(T const&) -> pair_ref<T>;

} // namespace datapar::fmath

template <floating_point_like E, datapar::fixed_width_abi A>
struct tuple_size<datapar::fmath::basic_pair<E, A>> : size_constant<2> {};

template <size_t I, floating_point_like E, datapar::fixed_width_abi A>
struct tuple_element<I, datapar::fmath::basic_pair<E, A>> :
    type_identity<datapar::basic_vector<E, A>> {};

template <datapar::fmath::simd_double T>
struct tuple_size<datapar::fmath::pair_ref<T>> : size_constant<2> {};

template <size_t I, datapar::fmath::simd_double T>
struct tuple_element<I, datapar::fmath::pair_ref<T>> : tuple_element<I, T> {};

namespace datapar::fmath {

template <simd_double T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
constexpr simd_element_type_t<T>
    DPL_VECTORCALL recombine(T arg) noexcept {
    return dx::add(dx::get_element<0>(arg), dx::get_element<1>(arg));
}

template <simd_double T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr simd_element_type_t<T> recombine(pair_ref<T> arg) noexcept {
    return fmath::recombine(arg.value);
}

template <simd_double T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
constexpr T DPL_VECTORCALL select(
    simd_mask_type_t<simd_element_type_t<T>> mask, T left, T right) noexcept {
    return fmath::make_pair(
        dx::select(mask, dx::get_element<0>(left), dx::get_element<0>(right)),
        dx::select(mask, dx::get_element<1>(left), dx::get_element<1>(right)));
}

template <simd_double T, broadcastable_to<simd_element_type_t<T>> C>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr T broadcast(C arg) noexcept {
    using simd = typename T::element_type;
    return fmath::make_pair(dx::broadcast<simd>(arg));
}

template <simd_double T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
constexpr T DPL_VECTORCALL abs(T arg) noexcept {
    auto const sign = dx::signbit(arg.upper);
    return fmath::make_pair(dx::negate(arg.upper, sign, arg.upper),
        dx::negate(arg.lower, sign, arg.lower));
}

template <simd_double T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
constexpr T DPL_VECTORCALL normalize(T arg) noexcept {
    auto const [in_upper, in_lower] = dx::to_tuple_like(arg);
    auto upper = dx::add(in_upper, in_lower);
    return fmath::make_pair(
        upper, dx::add(dx::subtract(in_upper, upper), in_lower));
}

template <simd_double T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
constexpr T DPL_VECTORCALL scale(T arg, simd_value_type_t<T> scale) noexcept {
    return fmath::make_pair(dx::multiply(dx::get_element<0>(arg), scale),
        dx::multiply(dx::get_element<1>(arg), scale));
}

template <simd_double T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
constexpr T DPL_VECTORCALL square(T arg) noexcept {
    auto const [in_upper, in_lower] = dx::to_tuple_like(arg);
    auto s = dx::multiply(in_upper, in_lower);
    return fmath::make_pair(s,
        dx::muladd(                      //
            dx::add(in_upper, in_upper), //
            in_lower,                    //
            dx::mulsub(in_upper, in_upper, s)));
}

template <simd_double T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
constexpr T DPL_VECTORCALL rcp(T arg) noexcept {
    auto const one = dx::broadcast<simd_element_type_t<T>>(dx::one);
    auto s = dx::divide(one, dx::get_element<0>(arg));
    return fmath::make_pair(s,
        dx::multiply(
            s, dx::nmuladd(arg.lower, s, dx::nmuladd(arg.upper, s, one))));
}

struct frsqrt_t : public dx::internal::operation_base<dx::internal::rsqrt_t> {
    using operation_base<dx::internal::rsqrt_t>::operator();
};

template <simd_double T>
requires internal::cpo_invocable<dx::internal::rsqrt_t, simd_element_type_t<T>>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
constexpr T DPL_VECTORCALL sqrt(T arg) noexcept {
    constexpr frsqrt_t rsqrt;
    using simd = simd_element_type_t<T>;
    auto const [upper, lower] = arg;
    auto x = rsqrt(dx::add(upper, lower));
    T r = pair_ref{arg} * x;
    auto const n3 = dx::broadcast<simd>(-3.0);
    return fmath::scale(pair_ref{r} * (pair_ref{pair_ref{r} * x} + n3), -0.5);
}

template <typename T>
struct fast;
template <typename T>
struct single;

template <canonical_vector T>
struct single<T> {
    using element_type = T;

    element_type const& value;

    __DPL_HIDE_FROM_ABI explicit constexpr single(
        element_type const& val DPL_LIFETIMEBOUND) noexcept
        : value(val) {}

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    constexpr operator element_type(this single self) noexcept {
        return self.value;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, PURE, NODISCARD)
    constexpr pair_type_t<T> DPL_VECTORCALL operator+(
        this single self, element_type right) noexcept {
        // Knuth's twosum
        auto s = dx::add(self.value, right);
        auto v = dx::subtract(s, self.value);
        return fmath::make_pair(s,
            dx::add(dx::subtract(right, v),
                dx::subtract(self.value, dx::subtract(s, v))));
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    friend constexpr pair_type_t<T> DPL_VECTORCALL operator+(
        element_type left, single right) noexcept {
        return single(left) + right.value;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, PURE, NODISCARD)
    constexpr pair_type_t<T> DPL_VECTORCALL operator-(
        this single self, element_type right) noexcept {
        // Knuth's twodiff
        auto s = dx::subtract(self.value, right);
        auto v = dx::subtract(s, self.value);
        return fmath::make_pair(s,
            dx::subtract(dx::subtract(self.value, dx::subtract(s, v)),
                dx::add(right, v)));
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    friend constexpr pair_type_t<T> DPL_VECTORCALL operator-(
        element_type left, single right) noexcept {
        return single(left) - right.value;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, PURE, NODISCARD)
    constexpr pair_type_t<T> DPL_VECTORCALL operator*(
        this single self, element_type right) noexcept {
        auto s = dx::multiply(self.value, right);
        return fmath::make_pair(s, dx::mulsub(self.value, right, s));
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    friend constexpr pair_type_t<T> DPL_VECTORCALL operator*(
        element_type left, single right) noexcept {
        return single(left) * right.value;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, PURE, NODISCARD)
    constexpr T DPL_VECTORCALL operator*(
        this single self, pair_type_t<T> right) noexcept {
        auto const [upper, lower] = right;
        auto hi = dx::multiply(self.value, upper);
        auto lo =
            dx::muladd(self.value, lower, dx::mulsub(self.value, upper, hi));
        return dx::add(hi, lo);
    }
};

template <canonical_vector T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
constexpr pair_type_t<T>
    DPL_VECTORCALL rcp(single<T> arg) noexcept {
    auto const one = dx::broadcast<T>(dx::one);
    auto s = dx::divide(one, arg.value);
    return fmath::make_pair(s, dx::multiply(s, dx::nmuladd(arg.value, s, one)));
}

struct fsqrt_t : public dx::internal::operation_base<dx::internal::sqrt_t> {
    using operation_base<dx::internal::sqrt_t>::operator();
};

template <canonical_vector T>
requires dx::internal::cpo_invocable<dx::internal::sqrt_t, T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
constexpr pair_type_t<T>
    DPL_VECTORCALL sqrt(single<T> arg) noexcept {
    constexpr fsqrt_t sqrt;
    auto t = sqrt(arg.value);
    return fmath::scale(
        arg.value + pair_ref{single{t} * t} * fmath::rcp(single{t}),
        dx::broadcast<T>(0.5));
}

template <simd_double T>
struct single<T> {
    using element_type = T;
    T const& value;

    __DPL_HIDE_FROM_ABI explicit constexpr single(
        T const& val DPL_LIFETIMEBOUND) noexcept
        : value(val) {}

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    constexpr operator element_type(this single self) noexcept {
        return self.value;
    }
};

template <simd_double T>
explicit single(T const&) -> single<T>;
template <canonical_vector T>
explicit single(T const&) -> single<T>;

template <simd_double T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, PURE, NODISCARD)
constexpr simd_element_type_t<T>
    DPL_VECTORCALL square(single<T> arg) noexcept {
    auto const [upper, lower] = arg.value;
    return dx::muladd(upper, upper,
        [](auto val) { return dx::add(val, val); }(dx::multiply(upper, lower)));
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

template <canonical_vector T>
struct fast<T> {
    using element_type = T;
    T const& value;

    __DPL_HIDE_FROM_ABI explicit constexpr fast(
        T const& val DPL_LIFETIMEBOUND) noexcept
        : value(val) {}

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    constexpr operator element_type() const noexcept {
        return value;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, PURE, NODISCARD)
    constexpr auto DPL_VECTORCALL operator+(
        this fast self, element_type right) noexcept {
        // |self.value| > |right|
        auto upper = self.value + right;
        return fmath::make_pair(upper, self.value - upper + right);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, PURE, NODISCARD)
    friend constexpr auto DPL_VECTORCALL operator+(
        element_type left, fast right) noexcept {
        return fast(left) + right.value;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, PURE, NODISCARD)
    constexpr auto DPL_VECTORCALL operator-(
        this fast self, element_type right) noexcept {
        // |self.value| > |right|
        auto upper = self.value - right;
        return fmath::make_pair(upper, self.value - upper - right);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, PURE, NODISCARD)
    constexpr auto DPL_VECTORCALL operator+(
        this fast self, pair_type_t<element_type> right) noexcept {
        // |self.value| < |right.upper|
        auto upper = self.value + right.upper;
        return fmath::make_pair(
            upper, (self.value - upper) + (right.upper + right.lower));
    }
};

template <simd_double T>
struct fast<T> {
    using element_type = T;
    T const& value;

    __DPL_HIDE_FROM_ABI explicit constexpr fast(
        T const& val DPL_LIFETIMEBOUND) noexcept
        : value(val) {}

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    constexpr operator T() const noexcept {
        return value;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, PURE, NODISCARD)
    constexpr T DPL_VECTORCALL operator+(this fast self, T right) noexcept {
        // |left.upper| >= |right.upper|
        auto s = self.value.upper + right.upper;

        return fmath::make_pair(s,
            (self.value.upper - s) + right.upper +
                (self.value.lower + right.lower));
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, PURE, NODISCARD)
    constexpr T DPL_VECTORCALL operator+(
        this fast self, element_type_if_t<T> right) noexcept {
        // |self.value.upper| > |right|
        auto s = self.value.upper + right;
        return fmath::make_pair(
            s, (self.value.upper - s) + (right + self.value.lower));
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, PURE, NODISCARD)
    friend constexpr T DPL_VECTORCALL operator+(
        simd_element_type_t<T> left, fast right) noexcept {
        auto s = left + right.value.upper;
        return fmath::make_pair(
            s, (left - s) + (right.value.upper + right.value.lower));
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, PURE, NODISCARD)
    constexpr T DPL_VECTORCALL operator-(
        this fast self, simd_element_type_t<T> right) noexcept {
        // |x| >= |y|
        auto s = self.value.upper - right;
        return fmath::make_pair(
            s, self.value.upper - s - right + self.value.lower);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, PURE, NODISCARD)
    constexpr T DPL_VECTORCALL operator-(this fast self, T right) noexcept {
        // |x| >= |y|
        auto s = self.value.upper - right.upper;
        auto t = self.value.upper - s;
        t = t - right.upper;
        t = t + self.value.lower;

        return fmath::make_pair(s, t - right.lower);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, PURE, NODISCARD)
    friend constexpr T DPL_VECTORCALL operator-(T left, fast right) noexcept {
        return fast(left) - right.value;
    }
};

template <typename T>
explicit fast(T const&) -> fast<T>;

struct ln2_t : dx::ln2_t {
    __DPL_HIDE_FROM_ABI explicit constexpr ln2_t() noexcept = default;

    template <simd_double T>
    requires same_as<simd_value_type_t<T>, float>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator T(this ln2_t) noexcept {
        return fmath::make_pair<simd_abi_type_t<T>, float>(
            0.693145751953125f, 1.428606765330187045e-06f);
    }

    template <simd_double T>
    requires same_as<simd_value_type_t<T>, double>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator T(this ln2_t) noexcept {
        return fmath::make_pair<simd_abi_type_t<T>, double>(
            0.69314718055966295651160180568695068359375,
            0.28235290563031577122588448175013436025525412068e-12);
    }
};

inline constexpr ln2_t ln2{};

template <typename T>
requires explicitly_convertible_to<ln2_t, T>
inline constexpr auto ln2_v = static_cast<T>(ln2);

template <typename E, typename A>
using make_pair_type_t = pair_type_t<make_canonical_vector_t<E, A>>;

} // namespace datapar::fmath
__DPL_DEFAULT_NAMESPACE_END
