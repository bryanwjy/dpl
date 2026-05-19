// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

// IWYU pragma: always_keep

#include "dpl/core/operations/masked.h"
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
template <auto>
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
concept unqualified_canonical_negate = requires(T val) {
    {
        negate(internal::abi<T>, val)
    } -> equivalent_simd_as<common_arithmetic_simd_t<T, T>>;
};

template <typename T>
concept unqualified_extended_negate = requires(T val) {
    { negate(val) } -> equivalent_simd_as<common_arithmetic_simd_t<T, T>>;
};

template <typename T>
concept unqualified_negate =
    unqualified_canonical_negate<T> || unqualified_extended_negate<T> ||
    (decayable_vector_for<T, operation_category::lane_agnostic> &&
        unqualified_canonical_negate<canonical_type_t<T>>);

template <typename S, typename C, typename T, typename A = common_abi_t<C, T>>
concept unqualified_canonical_mnegate = requires(S src, C mask, T val) {
    {
        negate(internal::abi<A>, src, mask, val)
    } -> equivalent_simd_as<canonical_if_zero_t<S, T, A>>;
};

template <typename S, typename C, typename T, typename A = common_abi_t<C, T>>
concept unqualified_extended_mnegate = requires(S src, C mask, T val) {
    {
        negate(src, mask, val)
    } -> equivalent_simd_as<canonical_if_zero_t<S, T, A>>;
};

template <typename Abs, typename S, typename C, typename T,
    typename A = common_abi_t<T, C>>
concept decayable_mnegate = decayable_vector_for<canonical_if_zero_t<S, T, A>,
                                operation_category::lane_agnostic> &&
    decayable_vector_for<T, operation_category::lane_agnostic> &&
    decayable_mask_for<C, operation_category::lane_agnostic> &&
    requires(canonical_or_zero_t<S, T, A> s, canonical_type_t<C> c,
        canonical_type_t<T> t) { Abs::operator()(s, c, t); };

template <typename Abs, typename S, typename C, typename T,
    typename A = common_abi_t<T, C>>
concept extended_mnegate = unqualified_extended_mnegate<S, C, T, A> ||
    decayable_mnegate<Abs, S, C, T, A>;

template <typename S, typename M, typename T,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept unqualified_canonical_negatei = requires(S src, T val) {
    {
        negate<const_mask_v<canonical_if_zero_t<S, T>, M>>(
            internal::abi<A>, src, dx::masked_operation, val)
    } -> equivalent_simd_as<canonical_if_zero_t<S, T, A>>;
};

template <typename S, typename M, typename T,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept unqualified_extended_negatei = requires(S src, T val) {
    {
        negate<const_mask_v<canonical_if_zero_t<S, T>, M>>(
            src, dx::masked_operation, val)
    } -> equivalent_simd_as<canonical_if_zero_t<S, T, A>>;
};

template <typename Abs, typename S, typename M, typename T,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept decayable_negatei = decayable_vector_for<canonical_if_zero_t<S, T>,
                                operation_category::lane_agnostic> &&
    decayable_vector_for<T, operation_category::lane_agnostic> &&
    requires(canonical_or_zero_t<S, T, A> s, M mask, canonical_type_t<T> t) {
        Abs::operator()(s, mask, t);
    };

template <typename Abs, typename S, typename M, typename T,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept extended_negatei = unqualified_extended_negatei<S, M, T, A> ||
    decayable_negatei<Abs, S, M, T, A>;

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

struct negate_t {
private:
    template <arithmetic_type E>
    using result DPL_NODEBUG = common_arithmetic_type_t<E, E>;

    template <arithmetic_vector T>
    using result_simd DPL_NODEBUG = common_arithmetic_simd_t<T, T>;

    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_vector<E, A>
        DPL_VECTORCALL fallback(basic_vector<E, A> val) noexcept {
        if constexpr (unsigned_integral<E>) {
            return val;
        } else {
            return subtract_t::operator()(dx::zero, val);
        }
    }

public:
    template <fixed_width_abi A, simd_element_for<A> E>
    requires arithmetic_type<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_negate<basic_vector<E, A>>) {
            if consteval {
                return fallback(val);
            } else {
                return negate(internal::abi<A>, val);
            }
        } else {
            return fallback(val);
        }
    }

    template <simd_abi A, simd_element_for<A> E>
    requires (scalable_abi<A> || !arithmetic_type<E>) &&
        unqualified_canonical_negate<basic_vector<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val) noexcept {
        return negate(internal::abi<A>, val);
    }

    template <extended_vector T>
    requires unqualified_negate<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val) noexcept {
        if constexpr (unqualified_extended_negate<T>) {
            return negate(val);
        } else {
            return operator()(dx::to_canonical(val));
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        common_size_with<E> MaskE>
    requires arithmetic_type<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_vector<E, A> pass,
        basic_mask<MaskE, A> mask, basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_mnegate<basic_vector<E, A>,
                          basic_mask<MaskE, A>, basic_vector<E, A>>) {
            if consteval {
                return internal::masked<negate_t>(pass, mask, val);
            } else {
                return negate(internal::abi<A>, pass, mask, val);
            }
        } else {
            return internal::masked<negate_t>(pass, mask, val);
        }
    }

    template <simd_abi A1, simd_element_for<A1> E, common_size_with<E> MaskE,
        simd_abi A2>
    requires (different_from<A1, A2> || scalable_abi<A1> || scalable_abi<A2> ||
                 !arithmetic_type<E>) &&
        simd_element_for<E, A2> &&
        maskable_args<basic_vector<E, A1>, basic_mask<MaskE, A1>,
            basic_vector<E, A2>> &&
        unqualified_canonical_mnegate<basic_vector<E, A1>,
            basic_mask<MaskE, A1>, basic_vector<E, A2>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A1> operator()(basic_vector<E, A1> pass,
        basic_mask<MaskE, A1> mask, basic_vector<E, A2> val) noexcept {
        return negate(internal::abi<A1>, pass, mask, val);
    }

    template <simd_vector Pass, simd_mask Mask, simd_vector Arg>
    requires (extended_vector<Pass> || extended_mask<Mask> ||
                 extended_vector<Arg>) &&
        maskable_args<Pass, Mask, Arg> &&
        extended_mnegate<negate_t, Pass, Mask, Arg>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(Pass pass, Mask mask, Arg arg) noexcept {
        if constexpr (unqualified_extended_mnegate<Pass, Mask, Arg>) {
            return negate(pass, mask, arg);
        } else {
            return operator()(dx::to_canonical(pass), dx::to_canonical(mask),
                dx::to_canonical(arg));
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        common_size_with<E> MaskE>
    requires arithmetic_type<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_mask<MaskE, A> mask, basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_mnegate<zero_t,
                          basic_mask<MaskE, A>, basic_vector<E, A>>) {
            if consteval {
                return internal::masked<negate_t>(mask, val);
            } else {
                return negate(internal::abi<A>, dx::zero, mask, val);
            }
        } else {
            return operator()(dx::zero_v<basic_vector<E, A>>, mask, val);
        }
    }

    template <simd_abi A1, simd_element_for<A1> E, common_size_with<E> MaskE,
        simd_abi A2>
    requires (different_from<A1, A2> || scalable_abi<A1> || scalable_abi<A2> ||
                 !arithmetic_type<E>) &&
        simd_element_for<E, A2> &&
        zmaskable_args<basic_mask<MaskE, A1>, basic_vector<E, A2>> &&
        unqualified_canonical_mnegate<dx::zero_t, basic_mask<MaskE, A1>,
            basic_vector<E, A2>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        basic_mask<MaskE, A1> mask, basic_vector<E, A2> val) noexcept {
        return negate(internal::abi<A1>, dx::zero, mask, val);
    }

    template <simd_mask Mask, simd_vector Arg>
    requires (extended_mask<Mask> || extended_vector<Arg>) &&
        zmaskable_args<Mask, Arg> &&
        extended_mnegate<negate_t, zero_t, Mask, Arg>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(Mask mask, Arg arg) noexcept {
        if constexpr (unqualified_extended_mnegate<dx::zero_t, Mask, Arg>) {
            return negate(mask, arg);
        } else {
            return operator()(dx::to_canonical(mask), dx::to_canonical(arg));
        }
    }

    template <simd_mask Mask, simd_vector Arg>
    requires requires(Mask mask, Arg arg) { negate_t::operator()(mask, arg); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t, Mask mask, Arg arg) noexcept {
        return operator()(mask, arg);
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        const_mask_for<basic_vector<E, A>> Mask>
    requires arithmetic_type<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> pass, Mask mask, basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_negatei<basic_vector<E, A>, Mask,
                          basic_vector<E, A>>) {
            if consteval {
                return internal::masked<negate_t>(pass, mask, val);
            } else {
                constexpr auto V = const_mask_v<basic_vector<E, A>, Mask>;
                return negate<V>(internal::abi<A>, pass, masked_operation, val);
            }
        } else {
            return internal::masked<negate_t>(pass, mask, val);
        }
    }

    template <simd_abi PassA, simd_element_for<PassA> E,
        const_mask_for<basic_vector<E, PassA>> Mask, simd_abi InA>
    requires (different_from<PassA, InA> || scalable_abi<PassA> ||
                 scalable_abi<InA> || !arithmetic_type<E>) &&
        simd_element_for<E, InA> &&
        imm_maskable_args<basic_vector<E, PassA>, basic_vector<E, InA>> &&
        unqualified_canonical_negatei<basic_vector<E, PassA>, Mask,
            basic_vector<E, InA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, PassA> operator()(
        basic_vector<E, PassA> pass, Mask mask,
        basic_vector<E, InA> val) noexcept {
        constexpr auto V = const_mask_v<basic_vector<E, PassA>, Mask>;
        return negate<V>(internal::abi<PassA>, pass, masked_operation, val);
    }

    template <simd_vector Pass, const_mask_for<Pass> Mask, simd_vector Arg>
    requires (extended_vector<Pass> || extended_vector<Arg>) &&
        imm_maskable_args<Pass, Arg> &&
        extended_negatei<negate_t, Pass, Mask, Arg>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(Pass pass, Mask mask, Arg arg) noexcept {
        if constexpr (unqualified_extended_mnegate<Pass, Mask, Arg>) {
            constexpr auto V = const_mask_v<Pass, Mask>;
            return negate<V>(pass, masked_operation, arg);
        } else {
            return operator()(dx::to_canonical(pass),
                dx::to_compatible_const_mask<Pass>(mask),
                dx::to_canonical(arg));
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        const_mask_for<basic_vector<E, A>> Mask>
    requires arithmetic_type<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        Mask mask, basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_negatei<zero_t, Mask,
                          basic_vector<E, A>>) {
            if consteval {
                return internal::masked<negate_t>(mask, val);
            } else {
                constexpr auto V = const_mask_v<basic_vector<E, A>, Mask>;
                return negate<V>(
                    internal::abi<A>, dx::zero, masked_operation, val);
            }
        } else {
            return operator()(dx::zero_v<basic_vector<E, A>>, mask, val);
        }
    }

    template <simd_abi InA, simd_element_for<InA> E,
        const_mask_for<basic_vector<E, InA>> Mask>
    requires (scalable_abi<InA> || !arithmetic_type<E>) &&
        simd_element_for<E, InA> && imm_zmaskable_args<basic_vector<E, InA>> &&
        unqualified_canonical_negatei<zero_t, Mask, basic_vector<E, InA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, InA> operator()(
        Mask mask, basic_vector<E, InA> val) noexcept {
        constexpr auto V = const_mask_v<basic_vector<E, InA>, Mask>;
        return negate<V>(internal::abi<InA>, dx::zero, masked_operation, val);
    }

    template <extended_vector Arg, const_mask_for<Arg> Mask>
    requires imm_zmaskable_args<Arg> &&
        extended_negatei<negate_t, zero_t, Mask, Arg>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(Mask mask, Arg arg) noexcept {
        if constexpr (unqualified_extended_mnegate<zero_t, Mask, Arg>) {
            constexpr auto V = const_mask_v<Arg, Mask>;
            return negate<V>(dx::zero, masked_operation, arg);
        } else {
            return operator()(
                dx::to_compatible_const_mask<Arg>(mask), dx::to_canonical(arg));
        }
    }

    template <simd_vector Arg, const_mask_for<Arg> Mask>
    requires requires(Mask mask, Arg arg) { negate_t::operator()(mask, arg); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t, Mask mask, Arg arg) noexcept {
        return operator()(mask, arg);
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
DPL_EXPORT inline constexpr internal::negate_t negate{};
} // namespace cpo

DPL_EXPORT template <typename D>
class arithmetic_vector_interface {
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
        regular_invocable<internal::negate_t, D>
    {
        return internal::negate_t::operator()(self);
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
    -> invoke_result_t<internal::negate_t, T> {
    return internal::negate_t::operator()(val);
}

} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
