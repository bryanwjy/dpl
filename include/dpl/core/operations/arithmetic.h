// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

// IWYU pragma: always_keep

#include "dpl/core/operations/operation_base.h"
#include "dpl/core/operations/reinterpret.h"
#include "dpl/core/operations/transform.h"

#if !DPL_MODULES
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

template <typename T>
concept unqualified_negate = requires(T val) {
    {
        negate(internal::abi<T>, val)
    } -> equivalent_simd_as<common_arithmetic_simd_t<T, T>>;
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

    template <typename L, typename R, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_vector<L, A> lhs, basic_vector<R, A> rhs) noexcept {
        using T = common_arithmetic_type_t<L, R>;
        return internal::transform<basic_vector<T, A>>(
            [](L lhs, R rhs) { return static_cast<T>(lhs + rhs); }, lhs, rhs);
    }

    template <simd_vector L, simd_vector R>
    using result_for =
        common_arithmetic_simd_t<canonical_type_t<L>, canonical_type_t<R>>;

public:
    template <simd_vector L, common_arithmetic_simd_with<L> R>
    requires same_abi_simd_as<L, R> && fixed_width_abi<common_abi_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = typename L::abi_type; // Same ABI, just pick one
        if constexpr (unqualified_add<L, R, A>) {
            if constexpr (canonical_vector<L> && canonical_vector<R>) {
                if consteval {
                    using E = typename decltype(add(
                        internal::abi<A>, lhs, rhs))::value_type;
                    return dx::reinterpret<E>(fallback(lhs, rhs));
                } else {
                    return add(internal::abi<A>, lhs, rhs);
                }
            } else {
                return add(internal::abi<A>, lhs, rhs);
            }
        } else if constexpr (canonical_vector<L> && canonical_vector<R>) {
            return fallback(lhs, rhs);
        } else {
            return operator()(dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }

    template <simd_vector L, common_arithmetic_simd_with<L> R>
    requires (!same_abi_simd_as<L, R> || scalable_vector<common_abi_t<L, R>>) &&
        (unqualified_add<L, R> ||
            unqualified_add<canonical_type_t<L>, canonical_type_t<R>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = common_abi_t<L, R>;
        if constexpr (unqualified_add<L, R>) {
            return add(internal::abi<A>, lhs, rhs);
        } else {
            return add(
                internal::abi<A>, dx::to_canonical(lhs), dx::to_canonical(rhs));
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

    template <typename L, typename R, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_vector<L, A> lhs, basic_vector<R, A> rhs) noexcept {
        using T = common_arithmetic_type_t<L, R>;
        return internal::transform<basic_vector<T, A>>(
            [](L lhs, R rhs) { return static_cast<T>(lhs - rhs); }, lhs, rhs);
    }

    template <simd_vector L, simd_vector R>
    using result_for =
        common_arithmetic_simd_t<canonical_type_t<L>, canonical_type_t<R>>;

public:
    template <simd_vector L, common_arithmetic_simd_with<L> R>
    requires same_abi_simd_as<L, R> && fixed_width_abi<common_abi_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = typename L::abi_type; // Same ABI, just pick one
        if constexpr (unqualified_subtract<L, R, A>) {
            if constexpr (canonical_vector<L> && canonical_vector<R>) {
                if consteval {
                    using E = typename decltype(subtract(
                        internal::abi<A>, lhs, rhs))::value_type;
                    return dx::reinterpret<E>(fallback(lhs, rhs));
                } else {
                    return subtract(internal::abi<A>, lhs, rhs);
                }
            } else {
                return subtract(internal::abi<A>, lhs, rhs);
            }
        } else if constexpr (canonical_vector<L> && canonical_vector<R>) {
            return fallback(lhs, rhs);
        } else {
            return operator()(dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }

    template <simd_vector L, common_arithmetic_simd_with<L> R>
    requires (!same_abi_simd_as<L, R> || scalable_abi<common_abi_t<L, R>>) &&
        (unqualified_subtract<L, R> ||
            unqualified_subtract<canonical_type_t<L>, canonical_type_t<R>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = common_abi_t<L, R>;
        if constexpr (unqualified_subtract<L, R>) {
            return subtract(internal::abi<A>, lhs, rhs);
        } else {
            return subtract(
                internal::abi<A>, dx::to_canonical(lhs), dx::to_canonical(rhs));
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

    template <typename L, typename R, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_vector<L, A> lhs, basic_vector<R, A> rhs) noexcept {
        using T = common_arithmetic_type_t<L, R>;
        return internal::transform<basic_vector<T, A>>(
            [](L lhs, R rhs) { return static_cast<T>(lhs - rhs); }, lhs, rhs);
    }

    template <simd_vector L, simd_vector R>
    using result_for =
        common_arithmetic_simd_t<canonical_type_t<L>, canonical_type_t<R>>;

public:
    template <simd_vector L, common_arithmetic_simd_with<L> R>
    requires same_abi_simd_as<L, R> && fixed_width_abi<common_abi_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = typename L::abi_type; // Same ABI, just pick one
        if constexpr (unqualified_multiply<L, R, A>) {
            if constexpr (canonical_vector<L> && canonical_vector<R>) {
                if consteval {
                    using E = typename decltype(multiply(
                        internal::abi<A>, lhs, rhs))::value_type;
                    return dx::reinterpret<E>(fallback(lhs, rhs));
                } else {
                    return multiply(internal::abi<A>, lhs, rhs);
                }
            } else {
                return multiply(internal::abi<A>, lhs, rhs);
            }
        } else if constexpr (canonical_vector<L> && canonical_vector<R>) {
            return fallback(lhs, rhs);
        } else {
            return operator()(dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }

    template <simd_vector L, common_arithmetic_simd_with<L> R>
    requires (!same_abi_simd_as<L, R> || scalable_abi<common_abi_t<L, R>>) &&
        (unqualified_multiply<L, R> ||
            unqualified_multiply<canonical_type_t<L>, canonical_type_t<R>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = common_abi_t<L, R>;
        if constexpr (unqualified_multiply<L, R>) {
            return multiply(internal::abi<A>, lhs, rhs);
        } else {
            return multiply(
                internal::abi<A>, dx::to_canonical(lhs), dx::to_canonical(rhs));
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

    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_vector<E, A>
        DPL_VECTORCALL fallback(
            basic_vector<E, A> lhs, basic_vector<E, A> rhs) noexcept {
        return dx::internal::transform<basic_vector<E, A>>(lhs, rhs,
            [](auto lhs, auto rhs) { return static_cast<E>(lhs / rhs); });
    }

    template <simd_vector L, simd_vector R>
    using result_for =
        common_arithmetic_simd_t<canonical_type_t<L>, canonical_type_t<R>>;

public:
    template <simd_vector L, common_float_simd_with<L> R>
    requires same_abi_simd_as<L, R> && fixed_width_abi<common_abi_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = typename L::abi_type; // Same ABI, just pick one
        if constexpr (unqualified_divide<L, R, A>) {
            if constexpr (canonical_vector<L> && canonical_vector<R>) {
                if consteval {
                    using E = typename decltype(divide(
                        internal::abi<A>, lhs, rhs))::value_type;
                    return dx::reinterpret<E>(fallback(lhs, rhs));
                } else {
                    return divide(internal::abi<A>, lhs, rhs);
                }
            } else {
                return divide(internal::abi<A>, lhs, rhs);
            }
        } else if constexpr (canonical_vector<L> && canonical_vector<R>) {
            return fallback(lhs, rhs);
        } else {
            return operator()(dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }

    template <simd_vector L, common_float_simd_with<L> R>
    requires (!same_abi_simd_as<L, R> || scalable_abi<common_abi_t<L, R>>) &&
        (unqualified_divide<L, R> ||
            unqualified_divide<canonical_type_t<L>, canonical_type_t<R>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        using A = common_abi_t<L, R>;
        if constexpr (unqualified_divide<L, R>) {
            return divide(internal::abi<A>, lhs, rhs);
        } else {
            return divide(
                internal::abi<A>, dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }

    using binary_operation_base<divide_t>::operator();
};

struct basic_negate_t {
private:
    template <arithmetic_type E>
    using result DPL_NODEBUG = common_arithmetic_type_t<E, E>;

    template <arithmetic_simd T>
    using result_simd DPL_NODEBUG = common_arithmetic_simd_t<T, T>;

    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_vector<E, A> val) noexcept {
        if constexpr (enumeration<E>) {
            return dx::reinterpret<result<E>>(val);
        } else {
            // do not assume ieee for floats
            return dx::reinterpret<E>(subtract_t::operator()(dx::zero, val));
        }
    }

public:
    template <arithmetic_simd T>
    requires fixed_width_vector<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_simd<T> operator()(T val) noexcept {
        if constexpr (unqualified_negate<T>) {
            if constexpr (canonical_vector<T>) {
                if consteval {
                    using E = typename decltype(negate(
                        internal::abi<T>, val))::value_type;
                    return dx::reinterpret<E>(fallback(val));
                } else {
                    return negate(internal::abi<T>, val);
                }
            } else {
                return negate(internal::abi<T>, val);
            }
        } else if constexpr (canonical_vector<T>) {
            return fallback(val);
        } else {
            return operator()(dx::to_canonical(val));
        }
    }

    template <arithmetic_simd T>
    requires scalable_vector<T> &&
        (unqualified_negate<T> || unqualified_negate<canonical_type_t<T>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_simd<T> operator()(T val) noexcept {
        if constexpr (unqualified_negate<T>) {
            return negate(internal::abi<T>, val);
        } else {
            return negate(internal::abi<T>, dx::to_canonical(val));
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
    requires simd_vector<D>
    {
        return datapar::add(lhs, rhs);
    }

    template <typename R>
    requires regular_invocable<internal::subtract_t, D, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr invoke_result_t<internal::subtract_t, D, R> operator-(
        this D lhs, R rhs) noexcept
    requires simd_vector<D>
    {
        return datapar::subtract(lhs, rhs);
    }

    template <typename R>
    requires regular_invocable<internal::multiply_t, D, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr invoke_result_t<internal::multiply_t, D, R> operator*(
        this D lhs, R rhs) noexcept
    requires simd_vector<D>
    {
        return datapar::multiply(lhs, rhs);
    }

    template <typename R>
    requires regular_invocable<internal::divide_t, D, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr invoke_result_t<internal::divide_t, D, R> operator/(
        this D lhs, R rhs) noexcept
    requires simd_vector<D>
    {
        return datapar::divide(lhs, rhs);
    }

    template <typename R>
    requires regular_invocable<internal::add_t, D, R> &&
        regular_invocable<internal::reinterpret_t<D>,
            invoke_result_t<internal::add_t, D, R>>
        DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE)
        constexpr D& operator+=(this D& lhs, R rhs) noexcept
    requires simd_vector<D>
    {
        return lhs = datapar::reinterpret<D>(datapar::add(lhs, rhs));
    }

    template <typename R>
    requires regular_invocable<internal::subtract_t, D, R> &&
        regular_invocable<internal::reinterpret_t<D>,
            invoke_result_t<internal::subtract_t, D, R>>
        DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE)
        constexpr D& operator-=(this D& lhs, R rhs) noexcept
    requires simd_vector<D>
    {
        return lhs = datapar::reinterpret<D>(datapar::subtract(lhs, rhs));
    }

    template <typename R>
    requires regular_invocable<internal::multiply_t, D, R> &&
        regular_invocable<internal::reinterpret_t<D>,
            invoke_result_t<internal::multiply_t, D, R>>
        DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE)
        constexpr D& operator*=(this D& lhs, R rhs) noexcept
    requires simd_vector<D>
    {
        return lhs = datapar::reinterpret<D>(datapar::multiply(lhs, rhs));
    }

    template <typename R>
    requires regular_invocable<internal::divide_t, D, R> &&
        regular_invocable<internal::reinterpret_t<D>,
            invoke_result_t<internal::divide_t, D, R>>
        DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE)
        constexpr D& operator/=(this D& lhs, R rhs) noexcept
    requires simd_vector<D>
    {
        return lhs = datapar::reinterpret<D>(datapar::divide(lhs, rhs));
    }

    template <typename L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr invoke_result_t<internal::add_t, L, D> operator+(
        L lhs, D rhs) noexcept
    requires simd_vector<D>
    {
        return datapar::add(lhs, rhs);
    }

    template <typename L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr invoke_result_t<internal::subtract_t, L, D> operator-(
        L lhs, D rhs) noexcept
    requires simd_vector<D>
    {
        return datapar::subtract(lhs, rhs);
    }

    template <typename L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr invoke_result_t<internal::multiply_t, L, D> operator*(
        L lhs, D rhs) noexcept
    requires simd_vector<D>
    {
        return datapar::multiply(lhs, rhs);
    }

    template <typename L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr invoke_result_t<internal::divide_t, L, D> operator/(
        L lhs, D rhs) noexcept
    requires simd_vector<D>
    {
        return datapar::divide(lhs, rhs);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr D operator-(this D self) noexcept
    requires simd_vector<D> && arithmetic_type<typename D::value_type> &&
        regular_invocable<internal::basic_negate_t, D>
    {
        return internal::basic_negate_t::operator()(self);
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
    -> invoke_result_t<internal::basic_negate_t, T> {
    return internal::basic_negate_t::operator()(val);
}

} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
