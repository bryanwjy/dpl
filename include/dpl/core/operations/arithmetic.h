// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

// IWYU pragma: always_keep

#include "dpl/core/operations/operation_base.h"
#include "dpl/core/operations/transform.h"

#if !DPL_MODULES
#  include "dpl/core/basic/reinterpret.h"
#  include "dpl/core/concepts/arithmetic_type.h"
#  include "dpl/core/concepts/common_arithmetic_with.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_equivalence.h"
#  include "dpl/core/constants/zero.h"
#  include "dpl/core/type_traits/common_arithmetic_type.h"
#  include "dpl/core/type_traits/rebind_simd.h"
#  include "dpl/std/concepts/enumeration.h"
#  include "dpl/std/concepts/floating_point.h"
#  include "dpl/std/type_traits/enable_if.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void add(...) noexcept = delete;
void subtract(...) noexcept = delete;
void multiply(...) noexcept = delete;
void divide(...) noexcept = delete;
void negate(...) noexcept = delete;

template <typename T, typename L, typename R>
concept arithmetic_result =
    common_arithmetic_simd_with<T, common_arithmetic_simd_t<L, R>> &&
    same_abi_as<common_abi_t<L, R>, typename T::abi_type>;

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_add = requires(L lhs, R rhs) {
    { add(internal::abi<A>, lhs, rhs) } -> arithmetic_result<L, R>;
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_subtract = requires(L lhs, R rhs) {
    { subtract(internal::abi<A>, lhs, rhs) } -> arithmetic_result<L, R>;
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_multiply = requires(L lhs, R rhs) {
    { multiply(internal::abi<A>, lhs, rhs) } -> arithmetic_result<L, R>;
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_divide = requires(L lhs, R rhs) {
    { divide(internal::abi<A>, lhs, rhs) } -> arithmetic_result<L, R>;
};

struct add_t : binary_operation_base<add_t> {
private:
    friend binary_operation_base<add_t>;

    template <simd_abi A, typename L, typename R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, L lhs, R rhs) noexcept
    requires requires {
        { add(internal::abi<A>, lhs, rhs) } -> simd_with_abi<A>;
    }
    {
        return add(internal::abi<A>, lhs, rhs);
    }

    template <arithmetic_type L, arithmetic_type R, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd<L, A> lhs, basic_simd<R, A> rhs) noexcept {
        using T = common_arithmetic_type_t<L, R>;
        return internal::transform<basic_simd<T, A>>(
            [](L lhs, R rhs) { return static_cast<T>(lhs + rhs); }, lhs, rhs);
    }

    template <simd_type L, simd_type R>
    using result_for =
        common_arithmetic_simd_t<basic_type_t<L>, basic_type_t<R>>;

public:
    template <simd_type L, common_arithmetic_simd_with<L> R>
    requires same_abi_simd_as<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = typename L::abi_type; // Same ABI, just pick one
        if constexpr (unqualified_add<L, R, A>) {
            if constexpr (basic_simd_type<L> && basic_simd_type<R>) {
                if consteval {
                    return fallback(lhs, rhs);
                } else {
                    return add(internal::abi<A>, lhs, rhs);
                }
            } else {
                return add(internal::abi<A>, lhs, rhs);
            }
        } else if constexpr (basic_simd_type<L> && basic_simd_type<R>) {
            return fallback(lhs, rhs);
        } else {
            return operator()(dx::to_basic_type(lhs), dx::to_basic_type(rhs));
        }
    }

    template <simd_type L, common_arithmetic_simd_with<L> R>
    requires (!same_abi_simd_as<L, R>) &&
        (unqualified_add<L, R> ||
            unqualified_add<basic_type_t<L>, basic_type_t<R>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = common_abi_t<L, R>;
        if constexpr (unqualified_add<L, R>) {
            return add(internal::abi<A>, lhs, rhs);
        } else {
            return add(internal::abi<A>, dx::to_basic_type(lhs),
                dx::to_basic_type(rhs));
        }
    }

    using binary_operation_base<add_t>::operator();
};

struct subtract_t : binary_operation_base<subtract_t> {
private:
    friend binary_operation_base<subtract_t>;

    template <simd_abi A, typename L, typename R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, L lhs, R rhs) noexcept
    requires requires {
        { subtract(internal::abi<A>, lhs, rhs) } -> simd_with_abi<A>;
    }
    {
        return subtract(internal::abi<A>, lhs, rhs);
    }

    template <arithmetic_type L, arithmetic_type R, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd<L, A> lhs, basic_simd<R, A> rhs) noexcept {
        using T = common_arithmetic_type_t<L, R>;
        return internal::transform<basic_simd<T, A>>(
            [](L lhs, R rhs) { return static_cast<T>(lhs - rhs); }, lhs, rhs);
    }

    template <simd_type L, simd_type R>
    using result_for =
        common_arithmetic_simd_t<basic_type_t<L>, basic_type_t<R>>;

public:
    template <simd_type L, common_arithmetic_simd_with<L> R>
    requires same_abi_simd_as<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = typename L::abi_type; // Same ABI, just pick one
        if constexpr (unqualified_subtract<L, R, A>) {
            if constexpr (basic_simd_type<L> && basic_simd_type<R>) {
                if consteval {
                    return fallback(lhs, rhs);
                } else {
                    return subtract(internal::abi<A>, lhs, rhs);
                }
            } else {
                return subtract(internal::abi<A>, lhs, rhs);
            }
        } else if constexpr (basic_simd_type<L> && basic_simd_type<R>) {
            return fallback(lhs, rhs);
        } else {
            return operator()(dx::to_basic_type(lhs), dx::to_basic_type(rhs));
        }
    }

    template <simd_type L, common_arithmetic_simd_with<L> R>
    requires (!same_abi_simd_as<L, R>) &&
        (unqualified_subtract<L, R> ||
            unqualified_subtract<basic_type_t<L>, basic_type_t<R>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = common_abi_t<L, R>;
        if constexpr (unqualified_subtract<L, R>) {
            return subtract(internal::abi<A>, lhs, rhs);
        } else {
            return subtract(internal::abi<A>, dx::to_basic_type(lhs),
                dx::to_basic_type(rhs));
        }
    }

    using binary_operation_base<subtract_t>::operator();
};

struct multiply_t : binary_operation_base<multiply_t> {
private:
    friend binary_operation_base<multiply_t>;

    template <simd_abi A, typename L, typename R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, L lhs, R rhs) noexcept
    requires requires {
        { multiply(internal::abi<A>, lhs, rhs) } -> simd_with_abi<A>;
    }
    {
        return multiply(internal::abi<A>, lhs, rhs);
    }

    template <arithmetic_type L, arithmetic_type R, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd<L, A> lhs, basic_simd<R, A> rhs) noexcept {
        using T = common_arithmetic_type_t<L, R>;
        return internal::transform<basic_simd<T, A>>(
            [](L lhs, R rhs) { return static_cast<T>(lhs - rhs); }, lhs, rhs);
    }

    template <simd_type L, simd_type R>
    using result_for =
        common_arithmetic_simd_t<basic_type_t<L>, basic_type_t<R>>;

public:
    template <simd_type L, common_arithmetic_simd_with<L> R>
    requires same_abi_simd_as<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = typename L::abi_type; // Same ABI, just pick one
        if constexpr (unqualified_multiply<L, R, A>) {
            if constexpr (basic_simd_type<L> && basic_simd_type<R>) {
                if consteval {
                    return fallback(lhs, rhs);
                } else {
                    return multiply(internal::abi<A>, lhs, rhs);
                }
            } else {
                return multiply(internal::abi<A>, lhs, rhs);
            }
        } else if constexpr (basic_simd_type<L> && basic_simd_type<R>) {
            return fallback(lhs, rhs);
        } else {
            return operator()(dx::to_basic_type(lhs), dx::to_basic_type(rhs));
        }
    }

    template <simd_type L, common_arithmetic_simd_with<L> R>
    requires (!same_abi_simd_as<L, R>) &&
        (unqualified_multiply<L, R> ||
            unqualified_multiply<basic_type_t<L>, basic_type_t<R>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = common_abi_t<L, R>;
        if constexpr (unqualified_multiply<L, R>) {
            return multiply(internal::abi<A>, lhs, rhs);
        } else {
            return multiply(internal::abi<A>, dx::to_basic_type(lhs),
                dx::to_basic_type(rhs));
        }
    }

    using binary_operation_base<multiply_t>::operator();
};

/**
 * Divisions are only supported for floating point elements
 */
struct divide_t : binary_operation_base<divide_t> {
private:
    friend binary_operation_base<divide_t>;

    template <simd_abi A, typename L, typename R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, L lhs, R rhs) noexcept
    requires requires {
        { divide(internal::abi<A>, lhs, rhs) } -> simd_with_abi<A>;
    }
    {
        return divide(internal::abi<A>, lhs, rhs);
    }

    template <floating_point E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_simd<E, A> DPL_VECTORCALL
        fallback(basic_simd<E, A> lhs, basic_simd<E, A> rhs) noexcept {
        return dx::internal::transform<basic_simd<E, A>>(lhs, rhs,
            [](auto lhs, auto rhs) { return static_cast<E>(lhs / rhs); });
    }

    template <simd_type L, simd_type R>
    using result_for =
        common_arithmetic_simd_t<basic_type_t<L>, basic_type_t<R>>;

public:
    template <simd_type L, common_float_simd_with<L> R>
    requires same_abi_simd_as<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = typename L::abi_type; // Same ABI, just pick one
        if constexpr (unqualified_divide<L, R, A>) {
            if constexpr (basic_simd_type<L> && basic_simd_type<R>) {
                if consteval {
                    return fallback(lhs, rhs);
                } else {
                    return divide(internal::abi<A>, lhs, rhs);
                }
            } else {
                return divide(internal::abi<A>, lhs, rhs);
            }
        } else if constexpr (basic_simd_type<L> && basic_simd_type<R>) {
            return fallback(lhs, rhs);
        } else {
            return operator()(dx::to_basic_type(lhs), dx::to_basic_type(rhs));
        }
    }

    template <simd_type L, common_float_simd_with<L> R>
    requires (!same_abi_simd_as<L, R>) &&
        (unqualified_divide<L, R> ||
            unqualified_divide<basic_type_t<L>, basic_type_t<R>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = common_abi_t<L, R>;
        if constexpr (unqualified_divide<L, R>) {
            return divide(internal::abi<A>, lhs, rhs);
        } else {
            return divide(internal::abi<A>, dx::to_basic_type(lhs),
                dx::to_basic_type(rhs));
        }
    }

    using binary_operation_base<divide_t>::operator();
};

struct simple_negate_t {
private:
    template <arithmetic_type E>
    using result DPL_NODEBUG = decltype(-__DPL declval<E>());

    template <arithmetic_simd T>
    using result_simd DPL_NODEBUG =
        rebind_simd_t<T, result<typename T::value_type>>;

    template <arithmetic_type E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallback(basic_simd<E, A> val) noexcept {
        if constexpr (enumeration<E>) {
            return dx::reinterpret<result<E>>(val);
        } else {
            // do not assume ieee for floats
            return dx::reinterpret<E>(subtract_t::operator()(dx::zero, val));
        }
    }

public:
    template <arithmetic_type E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(basic_simd<E, A> val) noexcept
        -> simd_with<result<E>, A> auto {
        if constexpr (requires {
                          {
                              negate(internal::abi<A>, val)
                          } -> simd_with<result<E>, A>;
                      }) {
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
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val) noexcept {
        if constexpr (requires {
                          {
                              negate(internal::abi<T>, val)
                          } -> equivalent_simd_as<result_simd<T>>;
                      }) {
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
struct wmultiply_t : binary_operation_base<wmultiply_t> {};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::add_t add{};
DPL_EXPORT inline constexpr internal::subtract_t subtract{};
DPL_EXPORT inline constexpr internal::multiply_t multiply{};
DPL_EXPORT inline constexpr internal::divide_t divide{};
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
    requires regular_invocable<internal::subtract_t, D, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr invoke_result_t<internal::subtract_t, D, R> operator-(
        this D lhs, R rhs) noexcept
    requires simd_type<D>
    {
        return datapar::subtract(lhs, rhs);
    }

    template <typename R>
    requires regular_invocable<internal::multiply_t, D, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr invoke_result_t<internal::multiply_t, D, R> operator*(
        this D lhs, R rhs) noexcept
    requires simd_type<D>
    {
        return datapar::multiply(lhs, rhs);
    }

    template <typename R>
    requires regular_invocable<internal::divide_t, D, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr invoke_result_t<internal::divide_t, D, R> operator/(
        this D lhs, R rhs) noexcept
    requires simd_type<D>
    {
        return datapar::divide(lhs, rhs);
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
    requires regular_invocable<internal::subtract_t, D, R> &&
        regular_invocable<internal::reinterpret_t<D>,
            invoke_result_t<internal::subtract_t, D, R>>
        DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE)
        constexpr D& operator-=(this D& lhs, R rhs) noexcept
    requires simd_type<D>
    {
        return lhs = datapar::reinterpret<D>(datapar::subtract(lhs, rhs));
    }

    template <typename R>
    requires regular_invocable<internal::multiply_t, D, R> &&
        regular_invocable<internal::reinterpret_t<D>,
            invoke_result_t<internal::multiply_t, D, R>>
        DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE)
        constexpr D& operator*=(this D& lhs, R rhs) noexcept
    requires simd_type<D>
    {
        return lhs = datapar::reinterpret<D>(datapar::multiply(lhs, rhs));
    }

    template <typename R>
    requires regular_invocable<internal::divide_t, D, R> &&
        regular_invocable<internal::reinterpret_t<D>,
            invoke_result_t<internal::divide_t, D, R>>
        DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE)
        constexpr D& operator/=(this D& lhs, R rhs) noexcept
    requires simd_type<D>
    {
        return lhs = datapar::reinterpret<D>(datapar::divide(lhs, rhs));
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
    friend constexpr invoke_result_t<internal::subtract_t, L, D> operator-(
        L lhs, D rhs) noexcept
    requires simd_type<D>
    {
        return datapar::subtract(lhs, rhs);
    }

    template <typename L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr invoke_result_t<internal::multiply_t, L, D> operator*(
        L lhs, D rhs) noexcept
    requires simd_type<D>
    {
        return datapar::multiply(lhs, rhs);
    }

    template <typename L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr invoke_result_t<internal::divide_t, L, D> operator/(
        L lhs, D rhs) noexcept
    requires simd_type<D>
    {
        return datapar::divide(lhs, rhs);
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
DPL_EXPORT template <typename L, typename R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr invoke_result_t<internal::add_t, L, R> operator+(
    L lhs, R rhs) noexcept {
    return datapar::add(lhs, rhs);
}

DPL_EXPORT template <typename L, typename R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr invoke_result_t<internal::subtract_t, L, R> operator-(
    L lhs, R rhs) noexcept {
    return datapar::subtract(lhs, rhs);
}

DPL_EXPORT template <typename L, typename R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr invoke_result_t<internal::multiply_t, L, R> operator*(
    L lhs, R rhs) noexcept {
    return datapar::multiply(lhs, rhs);
}

DPL_EXPORT template <typename L, typename R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr invoke_result_t<internal::divide_t, L, R> operator/(
    L lhs, R rhs) noexcept {
    return datapar::divide(lhs, rhs);
}

DPL_EXPORT template <typename L, typename R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE)
constexpr auto operator+=(L& lhs, R rhs) noexcept
    -> enable_if_t<equivalent_simd_as<L, decltype(lhs + rhs)>, L&> {
    return lhs = datapar::reinterpret<L>(lhs + rhs);
}

DPL_EXPORT template <typename L, typename R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE)
constexpr auto operator-=(L& lhs, R rhs) noexcept
    -> enable_if_t<equivalent_simd_as<L, decltype(lhs - rhs)>, L&> {
    return lhs = datapar::reinterpret<L>(lhs - rhs);
}

DPL_EXPORT template <typename L, typename R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE)
constexpr auto operator*=(L& lhs, R rhs) noexcept
    -> enable_if_t<equivalent_simd_as<L, decltype(lhs * rhs)>, L&> {
    return lhs = datapar::reinterpret<L>(lhs * rhs);
}

DPL_EXPORT template <typename L, typename R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE)
constexpr auto operator/=(L& lhs, R rhs) noexcept
    -> enable_if_t<equivalent_simd_as<L, decltype(lhs / rhs)>, L&> {
    return lhs = datapar::reinterpret<L>(lhs / rhs);
}

DPL_EXPORT template <typename T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr auto operator-(T val) noexcept
    -> invoke_result_t<internal::simple_negate_t, T> {
    return internal::simple_negate_t::operator()(val);
}

} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
