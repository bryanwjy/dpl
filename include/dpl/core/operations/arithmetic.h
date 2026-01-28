// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/operations/operation_base.h"
#include "dpl/core/operations/transform.h"

#if !DPL_MODULES
#  include "dpl/core/basic/reinterpret.h"
#  include "dpl/core/concepts/arithmetic_type.h"
#  include "dpl/core/concepts/common_arithmetic_with.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_equivalence.h"
#  include "dpl/core/constants/msb.h"
#  include "dpl/core/constants/zero.h"
#  include "dpl/core/type_traits/common_arithmetic_type.h"
#  include "dpl/std/concepts/enumeration.h"
#  include "dpl/std/concepts/floating_point.h"
#  include "dpl/std/concepts/integral.h"
#  include "dpl/std/type_traits/enable_if.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void add(...) noexcept = delete;
void sub(...) noexcept = delete;
void mul(...) noexcept = delete;
void div(...) noexcept = delete;
void negate(...) noexcept = delete;

struct add_t : binary_operation_base<add_t> {
private:
    template <simd_abi A, typename L, typename R>
    requires requires(L lhs, R rhs) { add(internal::abi<A>, lhs, rhs); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, L left, R right) noexcept {
        return add(internal::abi<A>, left, right);
    }

    template <arithmetic_type L, arithmetic_type R, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd<L, A> left, basic_simd<R, A> right) noexcept {
        using T = common_arithmetic_type_t<L, R>;
        return internal::transform<basic_simd<T, A>>(left, right,
            [](auto lhs, auto rhs) { return static_cast<T>(lhs + rhs); });
    }

    template <simd_type L, simd_type R>
    using result_for =
        common_arithmetic_simd_t<basic_type_t<L>, basic_type_t<R>>;

public:
    template <basic_simd_type L, common_arithmetic_simd_with<L> R>
    requires simd_same_abi_as<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(L left, R right) noexcept {
        using T = common_arithmetic_simd_t<L, R>;
        if constexpr (requires { add(internal::abi<T>, left, right); }) {
            if consteval {
                return fallback(left, right);
            } else {
                return add(internal::abi<T>, left, right);
            }
        } else {
            return fallback(left, right);
        }
    }

    template <basic_simd_type L, common_arithmetic_simd_with<L> R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(L left, R right) noexcept
        -> equivalent_simd_as<result_for<L, R>> auto {
        using A = common_abi_t<L, R>;
        return add(internal::abi<A>, left, right);
    }

    template <simd_type L, common_arithmetic_simd_with<L> R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(L left, R right) noexcept
        -> equivalent_simd_as<result_for<L, R>> auto {
        using A = common_abi_t<L, R>;
        if constexpr (requires { add(internal::abi<A>, left, right); }) {
            return add(internal::abi<A>, left, right);
        } else {
            return operator()(
                dx::to_basic_type(left), dx::to_basic_type(right));
        }
    }

    using binary_operation_base<add_t>::operator();
};

struct sub_t : binary_operation_base<sub_t> {
private:
    template <simd_abi A, typename L, typename R>
    requires requires(L lhs, R rhs) { sub(internal::abi<A>, lhs, rhs); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, L left, R right) noexcept {
        return sub(internal::abi<A>, left, right);
    }

    template <arithmetic_type L, arithmetic_type R, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd<L, A> left, basic_simd<R, A> right) noexcept {
        using T = common_arithmetic_type_t<L, R>;
        return internal::transform<basic_simd<T, A>>(left, right,
            [](auto lhs, auto rhs) { return static_cast<T>(lhs - rhs); });
    }

    template <simd_type L, simd_type R>
    using result_for =
        common_arithmetic_simd_t<basic_type_t<L>, basic_type_t<R>>;

public:
    template <basic_simd_type L, common_arithmetic_simd_with<L> R>
    requires simd_same_abi_as<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(L left, R right) noexcept {
        using T = common_arithmetic_simd_t<L, R>;
        if constexpr (requires { sub(internal::abi<T>, left, right); }) {
            if consteval {
                return fallback(left, right);
            } else {
                return sub(internal::abi<T>, left, right);
            }
        } else {
            return fallback(left, right);
        }
    }

    template <basic_simd_type L, common_arithmetic_simd_with<L> R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(L left, R right) noexcept
        -> equivalent_simd_as<result_for<L, R>> auto {
        using A = common_abi_t<L, R>;
        return sub(internal::abi<A>, left, right);
    }

    template <simd_type L, common_arithmetic_simd_with<L> R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(L left, R right) noexcept
        -> equivalent_simd_as<result_for<L, R>> auto {
        using A = common_abi_t<L, R>;
        if constexpr (requires { sub(internal::abi<A>, left, right); }) {
            return sub(internal::abi<A>, left, right);
        } else {
            return operator()(
                dx::to_basic_type(left), dx::to_basic_type(right));
        }
    }

    using binary_operation_base<sub_t>::operator();
};

struct mul_t : binary_operation_base<mul_t> {
private:
    template <simd_abi A, typename L, typename R>
    requires requires(L lhs, R rhs) { mul(internal::abi<A>, lhs, rhs); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, L left, R right) noexcept {
        return mul(internal::abi<A>, left, right);
    }

    template <arithmetic_type L, arithmetic_type R, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd<L, A> left, basic_simd<R, A> right) noexcept {
        using T = common_arithmetic_type_t<L, R>;
        return internal::transform<basic_simd<T, A>>(left, right,
            [](auto lhs, auto rhs) { return static_cast<T>(lhs * rhs); });
    }

    template <simd_type L, simd_type R>
    using result_for =
        common_arithmetic_simd_t<basic_type_t<L>, basic_type_t<R>>;

public:
    template <basic_simd_type L, common_arithmetic_simd_with<L> R>
    requires simd_same_abi_as<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(L left, R right) noexcept {
        using T = common_arithmetic_simd_t<L, R>;
        if constexpr (requires { mul(internal::abi<T>, left, right); }) {
            if consteval {
                return fallback(left, right);
            } else {
                return mul(internal::abi<T>, left, right);
            }
        } else {
            return fallback(left, right);
        }
    }

    template <basic_simd_type L, common_arithmetic_simd_with<L> R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(L left, R right) noexcept
        -> equivalent_simd_as<result_for<L, R>> auto {
        using A = common_abi_t<L, R>;
        return mul(internal::abi<A>, left, right);
    }

    template <simd_type L, common_arithmetic_simd_with<L> R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(L left, R right) noexcept
        -> equivalent_simd_as<result_for<L, R>> auto {
        using A = common_abi_t<L, R>;
        if constexpr (requires { mul(internal::abi<A>, left, right); }) {
            return mul(internal::abi<A>, left, right);
        } else {
            return operator()(
                dx::to_basic_type(left), dx::to_basic_type(right));
        }
    }

    using binary_operation_base<mul_t>::operator();
};

/**
 * Divisions are only supported for floating point elements
 */
struct div_t : binary_operation_base<div_t> {
private:
    template <simd_abi A, typename L, typename R>
    requires requires(L lhs, R rhs) { div(internal::abi<A>, lhs, rhs); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, L left, R right) noexcept {
        return div(internal::abi<A>, left, right);
    }

    template <floating_point E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_simd<E, A> DPL_VECTORCALL
        fallback(basic_simd<E, A> left, basic_simd<E, A> right) noexcept {
        return dx::internal::transform<basic_simd<E, A>>(left, right,
            [](auto lhs, auto rhs) { return static_cast<E>(lhs / rhs); });
    }

public:
    template <floating_point E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_simd<E, A> DPL_VECTORCALL operator()(
        basic_simd<E, A> left, basic_simd<E, A> right) noexcept {
        if constexpr (requires { div(internal::abi<A>, left, right); }) {
            if consteval {
                return fallback(left, right);
            } else {
                return div(internal::abi<A>, left, right);
            }
        } else {
            return fallback(left, right);
        }
    }

    template <floating_point E, simd_abi A, simd_abi B>
    requires (!same_as<A, B>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        basic_simd<E, A> left, basic_simd<E, B> right) noexcept {
        return div(internal::abi<common_abi_t<A, B>>, left, right);
    }

    template <floating_point_simd L, common_float_simd_with<L> R>
    requires (!basic_simd_type<L> || !basic_simd_type<R>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(L left, R right) noexcept
        -> common_float_simd_with<common_float_simd_t<L, R>> auto {
        using abi_type = common_abi_t<L, R>;
        if constexpr (div(internal::abi<abi_type>, left, right)) {
            return div(internal::abi<abi_type>, left, right);
        } else {
            return operator()(
                dx::to_basic_type(left), dx::to_basic_type(right));
        }
    }

    using binary_operation_base<div_t>::operator();
};

struct simple_negate_t {
private:
    template <arithmetic_type E>
    using Result DPL_NODEBUG = decltype(-__DPL declval<E>());

    template <arithmetic_simd T>
    using ResultSimd DPL_NODEBUG =
        rebind_simd_t<T, Result<typename T::value_type>>;

    template <arithmetic_type E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd<E, A> val) noexcept {
        if constexpr (enumeration<E>) {
            return dx::reinterpret<Result<E>>(val);
        } else if constexpr (integral<E>) {
            return dx::reinterpret<E>(sub_t::operator()(dx::zero, val));
        } else {
            static_assert(floating_point<E>);
            return dx::reinterpret<E>(dx::msb ^ val);
        }
    }

public:
    template <arithmetic_type E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        basic_simd<E, A> val) noexcept -> simd_with<Result<E>, A> auto {
        if constexpr (requires { negate(internal::abi<A>, val); }) {
            if consteval {
                return fallback(val);
            } else {
                return negate(internal::abi<A>, val);
            }
        } else {
            return fallback(val);
        }
    }

    template <arithmetic_simd T>
    requires (!basic_simd_type<T>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(T val) noexcept
        -> equivalent_simd_as<ResultSimd<T>> auto {
        if constexpr (requires { negate(internal::abi<T>, val); }) {
            return negate(internal::abi<T>, val);
        } else {
            return operator()(dx::to_basic_type(val));
        }
    }
};

// TODO

/**
 * Multiplies integral elements in the upper half of the vector
 * to produce a result that is twice the width of the argument
 * integral.
 */
struct wmulhi_t : binary_operation_base<wmulhi_t> {};

/**
 * Multiplies integral elements in the lower half of the vector
 * to produce a result that is twice the width of the argument
 * integral.
 */
struct wmullo_t : binary_operation_base<wmullo_t> {};

/**
 * Multiplies integral elements in the upper half of the vector
 * to produce an intermediate integral that is twice the width
 * of the argument integral. Stores the upper bits of the
 * intermediate result in the destination
 */
struct mulhi_t : binary_operation_base<mulhi_t> {};

/**
 * Multiplies integral elements in the lower half of the vector
 * to produce an intermediate integral that is twice the width
 * of the argument integral. Stores the lower bits of the
 * intermediate result in the destination
 */
struct mullo_t : binary_operation_base<mullo_t> {};

/**
 * Multiplies integral elements the vector to produce
 * a resulting vector twice the size of the argument vector,
 * containing integral elements twice the width of the argument's
 * integral element. The ABI of the resulting vector will differ
 * from the inputs. Supported only on extendable ABIs.
 */
struct wmul_t : binary_operation_base<wmul_t> {};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::add_t add{};
DPL_EXPORT inline constexpr internal::sub_t sub{};
DPL_EXPORT inline constexpr internal::mul_t mul{};
DPL_EXPORT inline constexpr internal::div_t div{};
} // namespace cpo

DPL_EXPORT template <typename D>
class arithmetic_simd_interface {
public:
    template <typename R>
    requires regular_invocable<internal::add_t, D, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr invoke_result_t<internal::add_t, D, R> operator+(
        this D lhs, R rhs) noexcept
    requires simd_type<D>
    {
        return datapar::add(lhs, rhs);
    }

    template <typename R>
    requires regular_invocable<internal::sub_t, D, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr invoke_result_t<internal::sub_t, D, R> operator-(
        this D lhs, R rhs) noexcept
    requires simd_type<D>
    {
        return datapar::sub(lhs, rhs);
    }

    template <typename R>
    requires regular_invocable<internal::mul_t, D, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr invoke_result_t<internal::mul_t, D, R> operator*(
        this D lhs, R rhs) noexcept
    requires simd_type<D>
    {
        return datapar::mul(lhs, rhs);
    }

    template <typename R>
    requires regular_invocable<internal::div_t, D, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr invoke_result_t<internal::div_t, D, R> operator/(
        this D lhs, R rhs) noexcept
    requires simd_type<D>
    {
        return datapar::div(lhs, rhs);
    }

    template <typename R>
    requires regular_invocable<internal::add_t, D, R> &&
        regular_invocable<internal::reinterpret_t<D>,
            invoke_result_t<internal::add_t, D, R>>
        DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE)
        constexpr D& operator+=(this D& lhs, R rhs) noexcept
    requires simd_type<D>
    {
        return lhs = datapar::reinterpret<D>(datapar::add(lhs, rhs));
    }

    template <typename R>
    requires regular_invocable<internal::sub_t, D, R> &&
        regular_invocable<internal::reinterpret_t<D>,
            invoke_result_t<internal::sub_t, D, R>>
        DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE)
        constexpr D& operator-=(this D& lhs, R rhs) noexcept
    requires simd_type<D>
    {
        return lhs = datapar::reinterpret<D>(datapar::sub(lhs, rhs));
    }

    template <typename R>
    requires regular_invocable<internal::mul_t, D, R> &&
        regular_invocable<internal::reinterpret_t<D>,
            invoke_result_t<internal::mul_t, D, R>>
        DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE)
        constexpr D& operator*=(this D& lhs, R rhs) noexcept
    requires simd_type<D>
    {
        return lhs = datapar::reinterpret<D>(datapar::mul(lhs, rhs));
    }

    template <typename R>
    requires regular_invocable<internal::div_t, D, R> &&
        regular_invocable<internal::reinterpret_t<D>,
            invoke_result_t<internal::div_t, D, R>>
        DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE)
        constexpr D& operator/=(this D& lhs, R rhs) noexcept
    requires simd_type<D>
    {
        return lhs = datapar::reinterpret<D>(datapar::div(lhs, rhs));
    }

    template <typename L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr invoke_result_t<internal::add_t, L, D> operator+(
        L lhs, D rhs) noexcept
    requires simd_type<D>
    {
        return datapar::add(lhs, rhs);
    }

    template <typename L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr invoke_result_t<internal::sub_t, L, D> operator-(
        L lhs, D rhs) noexcept
    requires simd_type<D>
    {
        return datapar::sub(lhs, rhs);
    }

    template <typename L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr invoke_result_t<internal::mul_t, L, D> operator*(
        L lhs, D rhs) noexcept
    requires simd_type<D>
    {
        return datapar::mul(lhs, rhs);
    }

    template <typename L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr invoke_result_t<internal::div_t, L, D> operator/(
        L lhs, D rhs) noexcept
    requires simd_type<D>
    {
        return datapar::div(lhs, rhs);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr D operator-(this D self) noexcept
    requires simd_type<D> && arithmetic_type<typename D::value_type> &&
        regular_invocable<internal::simple_negate_t, D>
    {
        return internal::simple_negate_t::operator()(self);
    }
};

/**
 * The following are not exported by design
 */
template <typename L, typename R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr invoke_result_t<internal::add_t, L, R> operator+(
    L lhs, R rhs) noexcept {
    return datapar::add(lhs, rhs);
}

template <typename L, typename R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr invoke_result_t<internal::sub_t, L, R> operator-(
    L lhs, R rhs) noexcept {
    return datapar::sub(lhs, rhs);
}

template <typename L, typename R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr invoke_result_t<internal::mul_t, L, R> operator*(
    L lhs, R rhs) noexcept {
    return datapar::mul(lhs, rhs);
}

template <typename L, typename R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr invoke_result_t<internal::div_t, L, R> operator/(
    L lhs, R rhs) noexcept {
    return datapar::div(lhs, rhs);
}

template <typename L, typename R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE)
constexpr auto operator+=(L& lhs, R rhs) noexcept
    -> enable_if_t<equivalent_simd_as<L, decltype(lhs + rhs)>, L&> {
    return lhs = datapar::reinterpret<L>(lhs + rhs);
}

template <typename L, typename R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE)
constexpr auto operator-=(L& lhs, R rhs) noexcept
    -> enable_if_t<equivalent_simd_as<L, decltype(lhs - rhs)>, L&> {
    return lhs = datapar::reinterpret<L>(lhs - rhs);
}

template <typename L, typename R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE)
constexpr auto operator*=(L& lhs, R rhs) noexcept
    -> enable_if_t<equivalent_simd_as<L, decltype(lhs * rhs)>, L&> {
    return lhs = datapar::reinterpret<L>(lhs * rhs);
}

template <typename L, typename R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE)
constexpr auto operator/=(L& lhs, R rhs) noexcept
    -> enable_if_t<equivalent_simd_as<L, decltype(lhs / rhs)>, L&> {
    return lhs = datapar::reinterpret<L>(lhs / rhs);
}

template <typename T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr auto operator-(T val) noexcept
    -> invoke_result_t<internal::simple_negate_t, T> {
    return internal::simple_negate_t::operator()(val);
}

} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
