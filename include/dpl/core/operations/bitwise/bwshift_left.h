// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

// IWYU pragma: always_keep

#include "dpl/core/operations/bitwise/result.h"
#include "dpl/core/operations/internal/masked.h"
#include "dpl/core/operations/internal/transform.h"

#if !DPL_MODULES
#  include "dpl/core/basic/initialize.h"
#  include "dpl/core/concepts/decayable.h"
#  include "dpl/core/concepts/operation_category.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_equivalence.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void bwshift_left(...) noexcept = delete;

struct bwshift_left_t;

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_canonical_bwslv = requires(L lhs, R rhs) {
    {
        bwshift_left(internal::abi<A>, lhs, rhs)
    } -> canonical_bitshift_result<L, A>;
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_extended_bwslv = requires(L lhs, R rhs) {
    { bwshift_left(lhs, rhs) } -> extended_operation_vector<A>;
};

template <typename L, typename R>
concept expression_bwslv = (simd_expression<L> || simd_expression<R>) &&
    invocable<abs_t, simd_expression_result_t<L>, simd_expression_result_t<R>>;

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept decayable_bwslv =
    decayable_vector_for<L, operation_category::lane_agnostic> &&
    decayable_vector_for<R, operation_category::lane_agnostic> &&
    regular_invocable<bwshift_left_t, canonical_type_t<L>, canonical_type_t<R>>;

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept extended_bwslv = unqualified_extended_bwslv<L, R, A> ||
    expression_bwslv<L, R> || decayable_bwslv<L, R, A>;

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<L, R, M>>
concept unqualified_canonical_mbwslv = requires(S src, M mask, L lhs, R rhs) {
    {
        bwshift_left(internal::abi<A>, src, mask, lhs, rhs)
    } -> equivalent_simd_as<
        canonical_if_zero_t<S, operation_result_t<bwshift_left_t, L, R>, A>>;
};

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<L, R, M>>
concept unqualified_extended_mbwslv = requires(S src, M mask, L lhs, R rhs) {
    { bwshift_left(src, mask, lhs, rhs) } -> extended_operation_vector<A>;
};

template <typename S, typename M, typename L, typename R>
concept expression_mbwslv = (simd_expression<S> || simd_expression<M> ||
                                simd_expression<L> || simd_expression<R>) &&
    invocable<bwshift_left_t, expression_result_or_zero_t<S>,
        simd_expression_result_t<M>, simd_expression_result_t<L>,
        simd_expression_result_t<R>>;

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<L, R, M>>
concept decayable_mbwslv =
    decayable_vector_for<
        canonical_if_zero_t<S, operation_result_t<bwshift_left_t, L, R>, A>,
        operation_category::lane_agnostic> &&
    decayable_mask_for<M, operation_category::lane_agnostic> &&
    decayable_vector_for<L, operation_category::lane_agnostic> &&
    decayable_vector_for<R, operation_category::lane_agnostic> &&
    regular_invocable<bwshift_left_t,
        canonical_or_zero_t<S, operation_result_t<bwshift_left_t, L, R>, A>,
        canonical_type_t<M>, canonical_type_t<L>, canonical_type_t<R>>;

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<L, R, M>>
concept extended_mbwslv = unqualified_extended_mbwslv<S, M, L, R, A> ||
    expression_mbwslv<S, M, L, R> || decayable_mbwslv<S, M, L, R, A>;

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<
        canonical_if_zero_t<S, operation_result_t<bwshift_left_t, L, R>>,
        operation_result_t<bwshift_left_t, L, R>>>
concept unqualified_canonical_imbwslv = requires(S src, M mask, L lhs, R rhs) {
    {
        bwshift_left(internal::abi<A>, src,
            internal::to_const_mask<A, bwshift_left_t, S, L, R>(mask), lhs, rhs)
    } -> equivalent_simd_as<
        canonical_if_zero_t<S, operation_result_t<bwshift_left_t, L, R>, A>>;
};

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<
        canonical_if_zero_t<S, operation_result_t<bwshift_left_t, L, R>>,
        operation_result_t<bwshift_left_t, L, R>>>
concept unqualified_extended_imbwslv = requires(S src, M mask, L lhs, R rhs) {
    {
        bwshift_left(src,
            internal::to_const_mask<A, bwshift_left_t, S, L, R>(mask), lhs, rhs)
    } -> extended_operation_vector<A>;
};

template <typename S, typename M, typename L, typename R>
concept expression_imbwslv =
    (simd_expression<S> || simd_expression<L> || simd_expression<R>) &&
    invocable<bwshift_left_t, expression_result_or_zero_t<S>, M,
        simd_expression_result_t<L>, simd_expression_result_t<R>>;

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<
        canonical_if_zero_t<S, operation_result_t<bwshift_left_t, L, R>>,
        operation_result_t<bwshift_left_t, L, R>>>
concept decayable_imbwslv =
    decayable_vector_for<
        canonical_if_zero_t<S, operation_result_t<bwshift_left_t, L, R>, A>,
        operation_category::lane_agnostic> &&
    decayable_vector_for<L, operation_category::lane_agnostic> &&
    decayable_vector_for<R, operation_category::lane_agnostic> &&
    regular_invocable<bwshift_left_t,
        canonical_or_zero_t<S, operation_result_t<bwshift_left_t, L, R>, A>, M,
        canonical_type_t<L>, canonical_type_t<R>>;

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<
        canonical_if_zero_t<S, operation_result_t<bwshift_left_t, L, R>>,
        operation_result_t<bwshift_left_t, L, R>>>
concept extended_imbwslv = unqualified_extended_imbwslv<S, M, L, R, A> ||
    expression_imbwslv<S, M, L, R> || decayable_imbwslv<S, M, L, R, A>;

///

template <typename T>
concept unqualified_canonical_bwsl = requires(T val, size_t shift) {
    {
        bwshift_left(internal::abi<T>, val, shift)
    } -> canonical_bitshift_result<T>;
};

template <typename T>
concept unqualified_extended_bwsl = requires(T val, size_t shift) {
    {
        bwshift_left(val, shift)
    } -> extended_operation_vector<typename T::abi_type>;
};

template <typename T>
concept expression_bwsl = simd_expression<T> &&
    regular_invocable<bwshift_left_t, simd_expression_result_t<T>, size_t>;

template <typename T>
concept decayable_bwsl =
    decayable_vector_for<T, operation_category::lane_agnostic> &&
    regular_invocable<bwshift_left_t, canonical_type_t<T>, size_t>;

template <typename T>
concept extended_bwsl =
    unqualified_extended_bwsl<T> || expression_bwsl<T> || decayable_bwsl<T>;

template <typename S, typename M, typename T, typename A = common_abi_t<M, T>>
concept unqualified_canonical_mbwsl =
    requires(S src, M mask, T val, size_t shift) {
        {
            bwshift_left(internal::abi<A>, src, mask, val, shift)
        } -> equivalent_simd_as<canonical_if_zero_t<S, T, A>>;
    };

template <typename S, typename M, typename T, typename A = common_abi_t<M, T>>
concept unqualified_extended_mbwsl =
    requires(S src, M mask, T val, size_t shift) {
        { bwshift_left(src, mask, val, shift) } -> extended_operation_vector<A>;
    };

template <typename S, typename M, typename T>
concept expression_mbwsl =
    (simd_expression<S> || simd_expression<M> || simd_expression<T>) &&
    regular_invocable<bwshift_left_t, expression_result_or_zero_t<S>,
        simd_expression_result_t<M>, simd_expression_result_t<T>, size_t>;

template <typename S, typename M, typename T, typename A = common_abi_t<T, M>>
concept decayable_mbwsl = decayable_vector_for<canonical_if_zero_t<S, T, A>,
                              operation_category::lane_agnostic> &&
    decayable_vector_for<T, operation_category::lane_agnostic> &&
    decayable_mask_for<M, operation_category::lane_agnostic> &&
    regular_invocable<bwshift_left_t, canonical_or_zero_t<S, T, A>,
        canonical_type_t<M>, canonical_type_t<T>, size_t>;

template <typename S, typename M, typename T, typename A = common_abi_t<T, M>>
concept extended_mbwsl = unqualified_extended_mbwsl<S, M, T, A> ||
    expression_mbwsl<S, M, T> || decayable_mbwsl<S, M, T, A>;

template <typename S, typename M, typename T,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept unqualified_canonical_imbwsl =
    requires(S src, M mask, T val, size_t shift) {
        {
            bwshift_left(internal::abi<A>, src,
                internal::to_const_mask<A, bwshift_left_t, S, T, size_t>(mask),
                val, shift)
        } -> equivalent_simd_as<canonical_if_zero_t<S, T, A>>;
    };

template <typename S, typename M, typename T,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept unqualified_extended_imbwsl =
    requires(S src, M mask, T val, size_t shift) {
        {
            bwshift_left(src,
                internal::to_const_mask<A, bwshift_left_t, S, T, size_t>(mask),
                val, shift)
        } -> extended_operation_vector<A>;
    };

template <typename S, typename M, typename T>
concept expression_imbwsl = (simd_expression<S> || simd_expression<T>) &&
    regular_invocable<bwshift_left_t, expression_result_or_zero_t<S>, M,
        simd_expression_result_t<T>, size_t>;

template <typename S, typename M, typename T,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept decayable_imbwsl = decayable_vector_for<canonical_if_zero_t<S, T>,
                               operation_category::lane_agnostic> &&
    decayable_vector_for<T, operation_category::lane_agnostic> &&
    regular_invocable<bwshift_left_t, canonical_or_zero_t<S, T, A>, M,
        canonical_type_t<T>, size_t>;

template <typename S, typename M, typename T,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept extended_imbwsl = unqualified_extended_imbwsl<S, M, T, A> ||
    expression_imbwsl<S, M, T> || decayable_imbwsl<S, M, T, A>;
///

template <typename L, typename R, typename A = typename L::abi_type>
concept unqualified_canonical_bwsli = requires(L lhs, R rhs) {
    {
        bwshift_left(internal::abi<A>, lhs, rhs)
    } -> canonical_bitshift_result<L>;
};

template <typename L, typename R, typename A = typename L::abi_type>
concept unqualified_extended_bwsli = requires(L lhs, R rhs) {
    { bwshift_left(lhs, rhs) } -> extended_operation_vector<A>;
};

template <typename L, typename R>
concept expression_bwsli =
    simd_expression<L> && invocable<abs_t, simd_expression_result_t<L>, R>;

template <typename L, typename R, typename A = typename L::abi_type>
concept decayable_bwsli =
    decayable_vector_for<L, operation_category::lane_agnostic> &&
    regular_invocable<bwshift_left_t, canonical_type_t<L>, R>;

template <typename L, typename R, typename A = typename L::abi_type>
concept extended_bwsli = unqualified_extended_bwsli<L, R, A> ||
    expression_bwsli<L, R> || decayable_bwsli<L, R, A>;

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<L, M>>
concept unqualified_canonical_mbwsli = requires(S src, M mask, L lhs, R rhs) {
    {
        bwshift_left(internal::abi<A>, src, mask, lhs, rhs)
    } -> equivalent_simd_as<
        canonical_if_zero_t<S, operation_result_t<bwshift_left_t, L, R>, A>>;
};

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<L, M>>
concept unqualified_extended_mbwsli = requires(S src, M mask, L lhs, R rhs) {
    { bwshift_left(src, mask, lhs, rhs) } -> extended_operation_vector<A>;
};

template <typename S, typename M, typename L, typename R>
concept expression_mbwsli =
    (simd_expression<S> || simd_expression<M> || simd_expression<L>) &&
    invocable<bwshift_left_t, expression_result_or_zero_t<S>,
        simd_expression_result_t<M>, simd_expression_result_t<L>, R>;

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<L, M>>
concept decayable_mbwsli =
    decayable_vector_for<
        canonical_if_zero_t<S, operation_result_t<bwshift_left_t, L, R>, A>,
        operation_category::lane_agnostic> &&
    decayable_mask_for<M, operation_category::lane_agnostic> &&
    decayable_vector_for<L, operation_category::lane_agnostic> &&
    regular_invocable<bwshift_left_t,
        canonical_if_zero_t<S, operation_result_t<bwshift_left_t, L, R>, A>,
        canonical_type_t<M>, canonical_type_t<L>, R>;

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<L, M>>
concept extended_mbwsli = unqualified_extended_mbwsli<S, M, L, R, A> ||
    expression_mbwsli<S, M, L, R> || decayable_mbwsli<S, M, L, R, A>;

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<
        canonical_if_zero_t<S, operation_result_t<bwshift_left_t, L, R>>,
        operation_result_t<bwshift_left_t, L, R>>>
concept unqualified_canonical_imbwsli = requires(S src, M mask, L lhs, R rhs) {
    {
        bwshift_left(internal::abi<A>, src,
            internal::to_const_mask<A, bwshift_left_t, S, L, R>(mask), lhs, rhs)
    } -> equivalent_simd_as<
        canonical_if_zero_t<S, operation_result_t<bwshift_left_t, L, R>, A>>;
};

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<
        canonical_if_zero_t<S, operation_result_t<bwshift_left_t, L, R>>,
        operation_result_t<bwshift_left_t, L, R>>>
concept unqualified_extended_imbwsli = requires(S src, M mask, L lhs, R rhs) {
    {
        bwshift_left(src,
            internal::to_const_mask<A, bwshift_left_t, S, L, R>(mask), lhs, rhs)
    } -> extended_operation_vector<A>;
};

template <typename S, typename M, typename L, typename R>
concept expression_imbwsli = (simd_expression<S> || simd_expression<L>) &&
    invocable<bwshift_left_t, expression_result_or_zero_t<S>, M,
        simd_expression_result_t<L>, R>;

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<
        canonical_if_zero_t<S, operation_result_t<bwshift_left_t, L, R>>,
        operation_result_t<bwshift_left_t, L, R>>>
concept decayable_imbwsli =
    decayable_vector_for<
        canonical_if_zero_t<S, operation_result_t<bwshift_left_t, L, R>, A>,
        operation_category::lane_agnostic> &&
    decayable_vector_for<L, operation_category::lane_agnostic> &&
    regular_invocable<bwshift_left_t,
        canonical_or_zero_t<S, operation_result_t<bwshift_left_t, L, R>, A>, M,
        canonical_type_t<L>, R>;

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<
        canonical_if_zero_t<S, operation_result_t<bwshift_left_t, L, R>>,
        operation_result_t<bwshift_left_t, L, R>>>
concept extended_imbwsli = unqualified_extended_imbwsli<S, M, L, R, A> ||
    expression_imbwsli<S, M, L, R> || decayable_imbwsli<S, M, L, R, A>;

///

template <typename T>
concept unqualified_canonical_mask_bwsl = unqualified_canonical_bwsl<T>;

template <typename T>
concept unqualified_extended_mask_bwsl = unqualified_extended_bwsl<T>;

template <typename T>
concept expression_mask_bwsl = simd_expression<T> &&
    regular_invocable<bwshift_left_t, simd_expression_result_t<T>, size_t>;

template <typename T>
concept decayable_mask_bwsl =
    decayable_mask_for<T, operation_category::lane_permutation> &&
    regular_invocable<bwshift_left_t, canonical_type_t<T>, size_t>;

template <typename T>
concept extended_mask_bwsl = unqualified_extended_mask_bwsl<T> ||
    expression_mask_bwsl<T> || decayable_mask_bwsl<T>;

///

template <typename T, typename N>
concept unqualified_canonical_mask_bwsli = unqualified_canonical_bwsli<T, N>;

template <typename T, typename N>
concept unqualified_extended_mask_bwsli = unqualified_extended_bwsli<T, N>;

template <typename T, typename N>
concept expression_mask_bwsli = simd_expression<T> &&
    regular_invocable<bwshift_left_t, simd_expression_result_t<T>, N>;

template <typename T, typename N>
concept decayable_mask_bwsli =
    decayable_mask_for<T, operation_category::lane_permutation> &&
    regular_invocable<bwshift_left_t, canonical_type_t<T>, N>;

template <typename T, typename N>
concept extended_mask_bwsli = unqualified_extended_bwsli<T, N> ||
    expression_mask_bwsli<T, N> || decayable_mask_bwsli<T, N>;

struct bwshift_left_t {
private:
    template <typename LE, integral RE, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_vector<LE, A> val, basic_vector<RE, A> shift) noexcept {
        using bit_type = bit_type_t<sizeof(LE) * char_bit_v>;
        return internal::transform<basic_vector<LE, A>>(
            [](auto lhs, auto rhs) {
                if constexpr (integral<LE> && sizeof(LE) < sizeof(int)) {
                    return static_cast<LE>(
                        __DPL bit_cast<bit_type>(lhs) << rhs);
                } else {
                    return __DPL bit_cast<LE>(
                        __DPL bit_cast<bit_type>(lhs) << rhs);
                }
            },
            val, shift);
    }

    template <typename LE, simd_abi LA, integral_constant_like R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_vector<LE, LA> val, R shift) noexcept {
        using bit_type = bit_type_t<sizeof(LE) * char_bit_v>;
        return internal::transform<basic_vector<LE, LA>>(
            [](auto lhs) {
                if constexpr (integral<LE> && sizeof(LE) < sizeof(int)) {
                    return static_cast<LE>(
                        __DPL bit_cast<bit_type>(lhs) << R::value);
                } else {
                    return __DPL bit_cast<LE>(
                        __DPL bit_cast<bit_type>(lhs) << R::value);
                }
            },
            val);
    }

    template <typename L, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_vector<L, A> val, size_t shift) noexcept {
        using bit_type = bit_type_t<sizeof(L) * char_bit_v>;
        return internal::transform<basic_vector<L, A>>(
            [shift](auto lhs) {
                if constexpr (integral<L> && sizeof(L) < sizeof(int)) {
                    return static_cast<L>(
                        __DPL bit_cast<bit_type>(lhs) << shift);
                } else {
                    return __DPL bit_cast<L>(
                        __DPL bit_cast<bit_type>(lhs) << shift);
                }
            },
            val);
    }

    template <typename LE, simd_abi LA, integral_constant_like R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_mask<LE, LA> val, R shift) noexcept {
        using bit_type = bit_type_t<sizeof(LE) * char_bit_v>;
        using mask_type = basic_mask<LE, LA>;
        if constexpr (R::value >= mask_type::size()) {
            return dx::broadcast<LE, LA>(false_type{});
        } else {
            return []<size_t... Is>(mask_type val, index_sequence<Is...>) {
                return dx::initialize<mask_type>(
                    bitset((Is >= R::value ? val[Is - R::value] : false)...));
            }(val, iota_sequence<LE, LA>);
        }
    }

    template <typename L, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_mask<L, A> val, size_t shift) noexcept {
        using bit_type = bit_type_t<sizeof(L) * char_bit_v>;
        using mask_type = basic_mask<L, A>;
        return []<size_t... Is>(
                   mask_type val, unsigned shift, index_sequence<Is...>) {
            return dx::initialize<mask_type>(
                bitset((Is >= shift ? val[Is - shift] : false)...));
        }(val, static_cast<unsigned>(shift), iota_sequence<L, A>);
    }

public:
    template <fixed_width_abi A, simd_element_for<A> LE, simd_element_for<A> RE>
    requires common_size_with<LE, RE> && integral<RE>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<LE, A> operator()(
        basic_vector<LE, A> val, basic_vector<RE, A> shift) noexcept {
        if constexpr (unqualified_canonical_bwslv<basic_vector<LE, A>,
                          basic_vector<RE, A>>) {
            if consteval {
                return fallback(val, shift);
            } else {
                return bwshift_left(internal::abi<A>, val, shift);
            }
        } else {
            return fallback(val, shift);
        }
    }

    template <simd_abi LA, simd_element_for<LA> LE, common_abi_with<LA> RA,
        simd_element_for<RA> RE>
    requires (scalable_abi<LA> || scalable_abi<RA> || different_from<LA, RA>) &&
        integral<RE> &&
        unqualified_canonical_bwslv<basic_vector<LE, LA>, basic_vector<RE, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<LE, common_abi_t<LA, RA>> operator()(
        basic_vector<LE, LA> val, basic_vector<RE, RA> shift) noexcept {
        using A = common_abi_t<LA, RA>;
        return bwshift_left(internal::abi<A>, val, shift);
    }

    template <simd_vector L, simd_vector R>
    requires (extended_vector<L> || extended_vector<R>) &&
        common_abi_with<typename L::abi_type, typename R::abi_type> &&
        common_size_with<typename L::value_type, typename R::value_type> &&
        integral<typename R::value_type> && extended_bwslv<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L val, R shift) noexcept {
        if constexpr (unqualified_extended_bwslv<L, R>) {
            return bwshift_left(val, shift);
        } else if constexpr (expression_bwslv<L, R>) {
            return operator()(dx::evaluate(val), dx::evaluate(shift));
        } else {
            return operator()(dx::to_canonical(val), shift);
        }
    }

    template <fixed_width_abi A, simd_element_for<A> LE, simd_element_for<A> RE,
        common_size_with<LE> ME>
    requires integral<RE> && common_size_with<LE, RE>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<LE, A> operator()(basic_vector<LE, A> src,
        basic_mask<ME, A> mask, basic_vector<LE, A> lhs,
        basic_vector<RE, A> rhs) noexcept {
        if constexpr (unqualified_canonical_mbwslv<basic_vector<LE, A>,
                          basic_mask<ME, A>, basic_vector<LE, A>,
                          basic_vector<RE, A>>) {
            if consteval {
                return internal::masked<bwshift_left_t>(src, mask, lhs, rhs);
            } else {
                return bwshift_left(internal::abi<A>, src, mask, lhs, rhs);
            }
        } else {
            return internal::masked<bwshift_left_t>(src, mask, lhs, rhs);
        }
    }

    template <simd_abi SA, simd_element_for<SA> LE, common_size_with<LE> ME,
        simd_abi LA, common_abi_with<LA> RA, simd_element_for<RA> RE>
    requires integral<RE> && common_size_with<LE, RE> &&
        (different_from<LA, RA> || scalable_abi<SA> || scalable_abi<LA> ||
            scalable_abi<RA>) &&
        simd_element_for<LE, LA> &&
        maskable_args<basic_vector<LE, SA>, basic_mask<ME, SA>,
            basic_vector<LE, LA>, basic_vector<RE, RA>> &&
        unqualified_canonical_mbwslv<basic_vector<LE, SA>, basic_mask<ME, SA>,
            basic_vector<LE, LA>, basic_vector<RE, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<LE, SA> operator()(basic_vector<LE, SA> src,
        basic_mask<ME, SA> mask, basic_vector<LE, LA> lhs,
        basic_vector<RE, RA> rhs) noexcept {
        return bwshift_left(internal::abi<SA>, src, mask, lhs, rhs);
    }

    template <simd_vector S, simd_mask M, simd_vector L, simd_vector R>
    requires integral<typename R::value_type> &&
        common_size_with<typename L::value_type, typename R::value_type> &&
        (extended_vector<S> || extended_mask<M> || extended_vector<L> ||
            extended_vector<R>) &&
        maskable_args<S, M, L, R> && extended_mbwslv<S, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, L lhs, R rhs) noexcept {
        if constexpr (unqualified_extended_mbwslv<S, M, L, R>) {
            return bwshift_left(src, mask, lhs, rhs);
        } else if constexpr (expression_mbwslv<S, M, L, R>) {
            return operator()(dx::evaluate(src), dx::evaluate(mask),
                dx::evaluate(lhs), dx::evaluate(rhs));
        } else {
            return operator()(dx::to_canonical(src), dx::to_canonical(mask),
                dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }

    template <fixed_width_abi A, simd_element_for<A> LE, simd_element_for<A> RE,
        common_size_with<LE> ME>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<LE, A> operator()(basic_mask<ME, A> mask,
        basic_vector<LE, A> lhs, basic_vector<RE, A> rhs) noexcept {
        if constexpr (unqualified_canonical_mbwslv<dx::zero_t,
                          basic_mask<ME, A>, basic_vector<LE, A>,
                          basic_vector<RE, A>>) {
            if consteval {
                return internal::masked<bwshift_left_t>(mask, lhs, rhs);
            } else {
                return bwshift_left(internal::abi<A>, dx::zero, mask, lhs, rhs);
            }
        } else {
            return operator()(dx::broadcast<LE, A>(dx::zero), mask, lhs, rhs);
        }
    }

    template <simd_abi SA, simd_element_for<SA> E, common_size_with<E> ME,
        simd_abi LA, common_abi_with<LA> RA>
    requires (different_from<SA, common_abi_t<LA, RA>> ||
                 different_from<LA, RA> || scalable_abi<SA> ||
                 scalable_abi<LA> || scalable_abi<RA>) &&
        simd_element_for<E, LA> && simd_element_for<E, RA> &&
        zmaskable_args<basic_mask<ME, SA>, basic_vector<E, LA>,
            basic_vector<E, RA>> &&
        unqualified_canonical_mbwslv<dx::zero_t, basic_mask<ME, SA>,
            basic_vector<E, LA>, basic_vector<E, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(basic_mask<ME, SA> mask,
        basic_vector<E, LA> lhs, basic_vector<E, RA> rhs) noexcept {
        return bwshift_left(
            internal::abi<common_abi_t<LA, RA>>, dx::zero, mask, lhs, rhs);
    }

    template <simd_mask M, simd_vector L, simd_vector R>
    requires (extended_mask<M> || extended_vector<L> || extended_vector<R>) &&
        zmaskable_args<M, L, R> && extended_mbwslv<dx::zero_t, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, L lhs, R rhs) noexcept {
        if constexpr (unqualified_extended_mbwslv<dx::zero_t, M, L, R>) {
            return bwshift_left(dx::zero, mask, lhs, rhs);
        } else if constexpr (expression_mbwslv<dx::zero_t, M, L, R>) {
            return operator()(
                dx::evaluate(mask), dx::evaluate(lhs), dx::evaluate(rhs));
        } else {
            return operator()(dx::to_canonical(mask), dx::to_canonical(lhs),
                dx::to_canonical(rhs));
        }
    }

    template <simd_mask M, simd_vector L, simd_vector R>
    requires invocable<bwshift_left_t, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t, M mask, L lhs, R rhs) noexcept {
        return operator()(mask, lhs, rhs);
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        const_mask_for<basic_vector<E, A>> M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_vector<E, A> src,
        M mask, basic_vector<E, A> lhs, basic_vector<E, A> rhs) noexcept {
        if constexpr (unqualified_canonical_imbwslv<basic_vector<E, A>, M,
                          basic_vector<E, A>, basic_vector<E, A>>) {
            if consteval {
                return internal::masked<bwshift_left_t>(src, mask, lhs, rhs);
            } else {
                return bwshift_left(internal::abi<A>, src,
                    dx::to_compatible_const_mask<basic_vector<E, A>>(mask), lhs,
                    rhs);
            }
        } else {
            return internal::masked<bwshift_left_t>(src, mask, lhs, rhs);
        }
    }

    template <simd_abi SA, simd_element_for<SA> E,
        const_mask_for<basic_vector<E, SA>> M, simd_abi LA,
        common_abi_with<LA> RA>
    requires (different_from<SA, common_abi_t<LA, RA>> ||
                 different_from<LA, RA> || scalable_abi<SA> ||
                 scalable_abi<LA> || scalable_abi<RA>) &&
        simd_element_for<E, LA> && simd_element_for<E, RA> &&
        imm_maskable_args<basic_vector<E, SA>, basic_vector<E, LA>,
            basic_vector<E, RA>> &&
        unqualified_canonical_imbwslv<basic_vector<E, SA>, M,
            basic_vector<E, LA>, basic_vector<E, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, SA> operator()(basic_vector<E, SA> src,
        M mask, basic_vector<E, LA> lhs, basic_vector<E, RA> rhs) noexcept {
        return bwshift_left(internal::abi<SA>, src,
            dx::to_compatible_const_mask<basic_vector<E, SA>>(mask), lhs, rhs);
    }

    template <simd_vector S, const_mask_for<S> M, simd_vector L, simd_vector R>
    requires (extended_vector<S> || extended_vector<L> || extended_vector<R>) &&
        imm_maskable_args<S, L, R> && extended_imbwslv<S, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, L lhs, R rhs) noexcept {
        if constexpr (unqualified_extended_imbwslv<S, M, L, R>) {
            return bwshift_left(
                src, dx::to_compatible_const_mask<S>(mask), lhs, rhs);
        } else if constexpr (expression_imbwslv<S, M, L, R>) {
            return operator()(
                dx::evaluate(src), mask, dx::evaluate(lhs), dx::evaluate(rhs));
        } else {
            return operator()(dx::to_canonical(src), mask,
                dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        const_mask_for<basic_vector<E, A>> M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        M mask, basic_vector<E, A> lhs, basic_vector<E, A> rhs) noexcept {
        if constexpr (unqualified_canonical_imbwslv<dx::zero_t, M,
                          basic_vector<E, A>, basic_vector<E, A>>) {
            if consteval {
                return internal::masked<bwshift_left_t>(mask, lhs, rhs);
            } else {
                return bwshift_left(internal::abi<A>, dx::zero,
                    dx::to_compatible_const_mask<basic_vector<E, A>>(mask), lhs,
                    rhs);
            }
        } else {
            return operator()(dx::zero_v<basic_vector<E, A>>, mask, lhs, rhs);
        }
    }

    template <simd_abi LA, common_abi_with<LA> RA,
        simd_element_for<common_abi_t<LA, RA>> E,
        const_mask_for<basic_vector<E, common_abi_t<LA, RA>>> M>
    requires (different_from<LA, RA> || scalable_abi<LA> || scalable_abi<RA>) &&
        simd_element_for<E, LA> && simd_element_for<E, RA> &&
        imm_zmaskable_args<basic_vector<E, LA>, basic_vector<E, RA>> &&
        unqualified_canonical_imbwslv<dx::zero_t, M, basic_vector<E, LA>,
            basic_vector<E, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, common_abi_t<LA, RA>> operator()(
        M mask, basic_vector<E, LA> lhs, basic_vector<E, RA> rhs) noexcept {
        using A = common_abi_t<LA, RA>;
        return bwshift_left(internal::abi<A>, dx::zero,
            dx::to_compatible_const_mask<basic_vector<E, A>>(mask), lhs, rhs);
    }

    template <simd_vector L, simd_vector R,
        const_mask_for<operation_result_t<bwshift_left_t, L, R>> M>
    requires (extended_vector<L> || extended_vector<R>) &&
        imm_zmaskable_args<L, R> && extended_imbwslv<dx::zero_t, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, L lhs, R rhs) noexcept {
        using S = operation_result_t<bwshift_left_t, L, R>;
        if constexpr (unqualified_extended_imbwslv<dx::zero_t, M, L, R>) {
            return bwshift_left(
                dx::zero, dx::to_compatible_const_mask<S>(mask), lhs, rhs);
        } else if constexpr (expression_imbwslv<dx::zero_t, M, L, R>) {
            return operator()(mask, dx::evaluate(lhs), dx::evaluate(rhs));
        } else {
            return operator()(
                mask, dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }

    template <simd_vector L, simd_vector R,
        const_mask_for<operation_result_t<bwshift_left_t, L, R>> M>
    requires invocable<bwshift_left_t, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t, M mask, L lhs, R rhs) noexcept {
        return operator()(mask, lhs, rhs);
    }

    ///
    template <fixed_width_abi A, simd_element_for<A> E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val, size_t shift) noexcept {
        if constexpr (unqualified_canonical_bwsl<basic_vector<E, A>>) {
            if consteval {
                return fallback(val, shift);
            } else {
                return bwshift_left(internal::abi<A>, val, shift);
            }
        } else {
            return fallback(val, shift);
        }
    }

    template <scalable_abi A, simd_element_for<A> E>
    requires unqualified_canonical_bwsl<basic_vector<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val, size_t shift) noexcept {
        return bwshift_left(internal::abi<A>, val, shift);
    }

    template <extended_vector T>
    requires extended_bwsl<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val, size_t shift) noexcept {
        if constexpr (unqualified_extended_bwsl<T>) {
            return bwshift_left(val, shift);
        } else if constexpr (expression_bwsl<T>) {
            return operator()(dx::evaluate(val), shift);
        } else {
            return operator()(dx::to_canonical(val), shift);
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E, simd_element_for<A> ME>
    requires common_size_with<E, ME>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_vector<E, A> src,
        basic_mask<ME, A> mask, basic_vector<E, A> val, size_t shift) noexcept {
        if constexpr (unqualified_canonical_mbwsl<basic_vector<E, A>,
                          basic_mask<ME, A>, basic_vector<E, A>>) {
            if consteval {
                return internal::masked<bwshift_left_t>(src, mask, val, shift);
            } else {
                return bwshift_left(internal::abi<A>, src, mask, val, shift);
            }
        } else {
            return internal::masked<bwshift_left_t>(src, mask, val, shift);
        }
    }

    template <simd_abi MA, simd_element_for<MA> ME, simd_abi TA,
        simd_element_for<TA> E>
    requires common_size_with<E, ME> &&
        (different_from<MA, TA> || scalable_abi<MA> || scalable_abi<TA>) &&
        maskable_args<basic_vector<E, MA>, basic_mask<ME, MA>,
            basic_vector<E, TA>> &&
        unqualified_canonical_mbwsl<basic_vector<E, MA>, basic_mask<ME, MA>,
            basic_vector<E, TA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, MA> operator()(basic_vector<E, MA> src,
        basic_mask<ME, MA> mask, basic_vector<E, TA> val,
        size_t shift) noexcept {
        return bwshift_left(internal::abi<MA>, src, mask, val, shift);
    }

    template <simd_vector S, simd_mask M, simd_vector T>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        maskable_args<S, M, T> && extended_mbwsl<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        S src, M mask, T val, size_t shift) noexcept {
        if constexpr (unqualified_extended_mbwsl<S, M, T>) {
            return bwshift_left(src, mask, val, shift);
        } else if constexpr (expression_mbwsl<S, M, T>) {
            return operator()(dx::evaluate(src), dx::evaluate(mask),
                dx::evaluate(val), shift);
        } else {
            return operator()(dx::to_canonical(src), dx::to_canonical(mask),
                dx::to_canonical(val), shift);
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E, simd_element_for<A> ME>
    requires common_size_with<E, ME>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_mask<ME, A> mask, basic_vector<E, A> val, size_t shift) noexcept {
        if constexpr (unqualified_canonical_mbwsl<dx::zero_t, basic_mask<ME, A>,
                          basic_vector<E, A>>) {
            if consteval {
                return internal::masked<bwshift_left_t>(mask, val, shift);
            } else {
                return bwshift_left(
                    internal::abi<A>, dx::zero, mask, val, shift);
            }
        } else {
            return operator()(dx::zero_v<basic_vector<E, A>>, mask, val, shift);
        }
    }

    template <simd_abi MA, simd_element_for<MA> ME, simd_abi TA,
        simd_element_for<TA> E>
    requires common_size_with<E, ME> &&
        (different_from<MA, TA> || scalable_abi<MA> || scalable_abi<TA>) &&
        zmaskable_args<basic_mask<ME, MA>, basic_vector<E, TA>> &&
        unqualified_canonical_mbwsl<dx::zero_t, basic_mask<ME, MA>,
            basic_vector<E, TA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(basic_mask<ME, MA> mask,
        basic_vector<E, TA> val, size_t shift) noexcept {
        return bwshift_left(internal::abi<MA>, dx::zero, mask, val, shift);
    }

    template <simd_mask M, simd_vector T>
    requires (extended_mask<M> || extended_vector<T>) && zmaskable_args<M, T> &&
        extended_mbwsl<dx::zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val, size_t shift) noexcept {
        if constexpr (unqualified_extended_mbwsl<dx::zero_t, M, T>) {
            return bwshift_left(dx::zero, mask, val, shift);
        } else if constexpr (expression_mbwsl<dx::zero_t, M, T>) {
            return operator()(dx::evaluate(mask), dx::evaluate(val), shift);
        } else {
            return operator()(
                dx::to_canonical(mask), dx::to_canonical(val), shift);
        }
    }

    template <simd_mask M, simd_vector T>
    requires invocable<bwshift_left_t, M, T, size_t>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t, M mask, T val, size_t shift) noexcept {
        return operator()(mask, val, shift);
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        const_mask_for<basic_vector<E, A>> M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_vector<E, A> src,
        M mask, basic_vector<E, A> val, size_t shift) noexcept {
        if constexpr (unqualified_canonical_imbwsl<basic_vector<E, A>, M,
                          basic_vector<E, A>>) {
            if consteval {
                return internal::masked<bwshift_left_t>(src, mask, val, shift);
            } else {
                return bwshift_left(internal::abi<A>, src,
                    dx::to_compatible_const_mask<basic_vector<E, A>>(mask), val,
                    shift);
            }
        } else {
            return internal::masked<bwshift_left_t>(src, mask, val, shift);
        }
    }

    template <simd_abi SA, simd_element_for<SA> E,
        const_mask_for<basic_vector<E, SA>> M, simd_abi TA>
    requires (different_from<SA, TA> || scalable_abi<SA> || scalable_abi<TA>) &&
        simd_element_for<E, TA> &&
        imm_maskable_args<basic_vector<E, SA>, basic_vector<E, TA>> &&
        unqualified_canonical_imbwsl<basic_vector<E, SA>, M,
            basic_vector<E, TA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, SA> operator()(basic_vector<E, SA> src,
        M mask, basic_vector<E, TA> val, size_t shift) noexcept {
        return bwshift_left(internal::abi<SA>, src,
            dx::to_compatible_const_mask<basic_vector<E, SA>>(mask), val,
            shift);
    }

    template <simd_vector S, const_mask_for<S> M, simd_vector T>
    requires (extended_vector<S> || extended_vector<T>) &&
        imm_maskable_args<S, T> && extended_imbwsl<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        S src, M mask, T val, size_t shift) noexcept {
        if constexpr (unqualified_extended_mbwsl<S, M, T>) {
            return bwshift_left(
                src, dx::to_compatible_const_mask<S>(mask), val, shift);
        } else if constexpr (expression_imbwsl<S, M, T>) {
            return operator()(
                dx::evaluate(src), mask, dx::evaluate(val), shift);
        } else {
            return operator()(
                dx::to_canonical(src), mask, dx::to_canonical(val), shift);
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        const_mask_for<basic_vector<E, A>> M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        M mask, basic_vector<E, A> val, size_t shift) noexcept {
        if constexpr (unqualified_canonical_imbwsl<dx::zero_t, M,
                          basic_vector<E, A>>) {
            if consteval {
                return internal::masked<bwshift_left_t>(mask, val, shift);
            } else {
                return bwshift_left(internal::abi<A>, dx::zero,
                    dx::to_compatible_const_mask<basic_vector<E, A>>(mask), val,
                    shift);
            }
        } else {
            return operator()(dx::zero_v<basic_vector<E, A>>, mask, val, shift);
        }
    }

    template <simd_abi TA, simd_element_for<TA> E,
        const_mask_for<basic_vector<E, TA>> M>
    requires scalable_abi<TA> && simd_element_for<E, TA> &&
        imm_zmaskable_args<basic_vector<E, TA>> &&
        unqualified_canonical_imbwsl<dx::zero_t, M, basic_vector<E, TA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, TA> operator()(
        M mask, basic_vector<E, TA> val, size_t shift) noexcept {
        return bwshift_left(internal::abi<TA>, dx::zero,
            dx::to_compatible_const_mask<basic_vector<E, TA>>(mask), val,
            shift);
    }

    template <extended_vector T, const_mask_for<T> M>
    requires imm_zmaskable_args<T> && extended_imbwsl<dx::zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val, size_t shift) noexcept {
        if constexpr (unqualified_extended_imbwsl<dx::zero_t, M, T>) {
            return bwshift_left(
                dx::zero, dx::to_compatible_const_mask<T>(mask), val, shift);
        } else if constexpr (expression_imbwsl<dx::zero_t, M, T>) {
            return operator()(mask, dx::evaluate(val), shift);
        } else {
            return operator()(dx::to_compatible_const_mask<T>(mask),
                dx::to_canonical(val), shift);
        }
    }

    template <simd_vector T, const_mask_for<T> M>
    requires invocable<bwshift_left_t, M, T, size_t>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t, M mask, T val, size_t shift) noexcept {
        return operator()(mask, val, shift);
    }
    //

    template <fixed_width_abi A, simd_element_for<A> E,
        integral_constant_like N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val, N shift) noexcept {
        if constexpr (unqualified_canonical_bwsli<basic_vector<E, A>, N>) {
            if consteval {
                return fallback(val, shift);
            } else {
                return bwshift_left(internal::abi<A>, val, shift);
            }
        } else {
            return fallback(val, shift);
        }
    }

    template <scalable_abi A, simd_element_for<A> E, integral_constant_like N>
    requires unqualified_canonical_bwsli<basic_vector<E, A>, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val, N shift) noexcept {
        return bwshift_left(internal::abi<A>, val, shift);
    }

    template <extended_vector T, integral_constant_like N>
    requires extended_bwsli<T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val, N shift) noexcept {
        if constexpr (unqualified_extended_bwsli<T, N>) {
            return bwshift_left(val, shift);
        } else if constexpr (expression_bwsli<T, N>) {
            return operator()(dx::evaluate(val), shift);
        } else {
            return operator()(dx::to_canonical(val), shift);
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E, common_size_with<E> ME,
        integral_constant_like N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_vector<E, A> src,
        basic_mask<ME, A> mask, basic_vector<E, A> val, N shift) noexcept {
        if constexpr (unqualified_canonical_mbwsli<basic_vector<E, A>,
                          basic_mask<ME, A>, basic_vector<E, A>, N>) {
            if consteval {
                return internal::masked<bwshift_left_t>(src, mask, val, shift);
            } else {
                return bwshift_left(internal::abi<A>, src, mask, val, shift);
            }
        } else {
            return internal::masked<bwshift_left_t>(src, mask, val, shift);
        }
    }

    template <simd_abi MA, simd_element_for<MA> ME, simd_abi TA,
        simd_element_for<TA> E, integral_constant_like N>
    requires common_size_with<E, ME> &&
        (different_from<MA, TA> || scalable_abi<MA> || scalable_abi<TA>) &&
        maskable_args<basic_vector<E, MA>, basic_mask<ME, MA>,
            basic_vector<E, TA>> &&
        unqualified_canonical_mbwsli<basic_vector<E, MA>, basic_mask<ME, MA>,
            basic_vector<E, TA>, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, MA> operator()(basic_vector<E, MA> src,
        basic_mask<ME, MA> mask, basic_vector<E, TA> val, N shift) noexcept {
        return bwshift_left(internal::abi<MA>, src, mask, val, shift);
    }

    template <simd_vector S, simd_mask M, simd_vector T,
        integral_constant_like N>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        maskable_args<S, M, T> && extended_mbwsli<S, M, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, T val, N shift) noexcept {
        if constexpr (unqualified_extended_mbwsli<S, M, T, N>) {
            return bwshift_left(src, mask, val, shift);
        } else if constexpr (expression_mbwsli<S, M, T, N>) {
            return operator()(dx::evaluate(src), dx::evaluate(mask),
                dx::evaluate(val), shift);
        } else {
            return operator()(dx::to_canonical(src), dx::to_canonical(mask),
                dx::to_canonical(val), shift);
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E, common_size_with<E> ME,
        integral_constant_like N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_mask<ME, A> mask, basic_vector<E, A> val, N shift) noexcept {
        if constexpr (unqualified_canonical_mbwsli<dx::zero_t,
                          basic_mask<ME, A>, basic_vector<E, A>, N>) {
            if consteval {
                return internal::masked<bwshift_left_t>(mask, val, shift);
            } else {
                return bwshift_left(
                    internal::abi<A>, dx::zero, mask, val, shift);
            }
        } else {
            return internal::masked<bwshift_left_t>(mask, val, shift);
        }
    }

    template <simd_abi MA, simd_element_for<MA> ME, simd_abi TA,
        simd_element_for<TA> E, integral_constant_like N>
    requires common_size_with<E, ME> &&
        (different_from<MA, TA> || scalable_abi<MA> || scalable_abi<TA>) &&
        zmaskable_args<basic_mask<ME, MA>, basic_vector<E, TA>> &&
        unqualified_canonical_mbwsli<dx::zero_t, basic_mask<ME, MA>,
            basic_vector<E, TA>, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        basic_mask<ME, MA> mask, basic_vector<E, TA> val, N shift) noexcept {
        return bwshift_left(internal::abi<MA>, dx::zero, mask, val, shift);
    }

    template <simd_mask M, simd_vector T, integral_constant_like N>
    requires (extended_mask<M> || extended_vector<T>) && zmaskable_args<M, T> &&
        extended_mbwsli<dx::zero_t, M, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val, N shift) noexcept {
        if constexpr (unqualified_extended_mbwsli<dx::zero_t, M, T, N>) {
            return bwshift_left(dx::zero, mask, val, shift);
        } else if constexpr (expression_mbwsli<dx::zero_t, M, T, N>) {
            return operator()(dx::evaluate(mask), dx::evaluate(val), shift);
        } else {
            return operator()(
                dx::to_canonical(mask), dx::to_canonical(val), shift);
        }
    }

    template <simd_mask M, simd_vector T, integral_constant_like N>
    requires invocable<bwshift_left_t, M, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t, M mask, T val, N shift) noexcept {
        return operator()(mask, val, shift);
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        const_mask_for<basic_vector<E, A>> M, integral_constant_like N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_vector<E, A> src,
        M mask, basic_vector<E, A> val, N shift) noexcept {
        if constexpr (unqualified_canonical_imbwsli<basic_vector<E, A>, M,
                          basic_vector<E, A>, N>) {
            if consteval {
                return internal::masked<bwshift_left_t>(src, mask, val, shift);
            } else {
                return bwshift_left(internal::abi<A>, src,
                    dx::to_compatible_const_mask<basic_vector<E, A>>(mask), val,
                    shift);
            }
        } else {
            return internal::masked<bwshift_left_t>(src, mask, val, shift);
        }
    }

    template <simd_abi SA, simd_element_for<SA> E,
        const_mask_for<basic_vector<E, SA>> M, simd_abi TA,
        integral_constant_like N>
    requires (different_from<SA, TA> || scalable_abi<SA> || scalable_abi<TA>) &&
        simd_element_for<E, TA> &&
        imm_maskable_args<basic_vector<E, SA>, basic_vector<E, TA>> &&
        unqualified_canonical_imbwsli<basic_vector<E, SA>, M,
            basic_vector<E, TA>, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, SA> operator()(basic_vector<E, SA> src,
        M mask, basic_vector<E, TA> val, N shift) noexcept {
        return bwshift_left(internal::abi<SA>, src,
            dx::to_compatible_const_mask<basic_vector<E, SA>>(mask), val,
            shift);
    }

    template <simd_vector S, const_mask_for<S> M, simd_vector T,
        integral_constant_like N>
    requires (extended_vector<S> || extended_vector<T>) &&
        imm_maskable_args<S, T> && extended_imbwsli<S, M, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, T val, N shift) noexcept {
        if constexpr (unqualified_extended_imbwsli<S, M, T, N>) {
            return bwshift_left(
                src, dx::to_compatible_const_mask<S>(mask), val, shift);
        } else if constexpr (expression_imbwsli<S, M, T, N>) {
            return operator()(
                dx::evaluate(src), mask, dx::evaluate(val), shift);
        } else {
            return operator()(
                dx::to_canonical(src), mask, dx::to_canonical(val), shift);
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        const_mask_for<basic_vector<E, A>> M, integral_constant_like N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        M mask, basic_vector<E, A> val, N shift) noexcept {
        if constexpr (unqualified_canonical_imbwsli<dx::zero_t, M,
                          basic_vector<E, A>, N>) {
            if consteval {
                return internal::masked<bwshift_left_t>(mask, val, shift);
            } else {
                return bwshift_left(internal::abi<A>, dx::zero,
                    dx::to_compatible_const_mask<basic_vector<E, A>>(mask), val,
                    shift);
            }
        } else {
            return operator()(dx::zero_v<basic_vector<E, A>>, mask, val, shift);
        }
    }

    template <simd_abi TA, simd_element_for<TA> E,
        const_mask_for<basic_vector<E, TA>> M, integral_constant_like N>
    requires scalable_abi<TA> && simd_element_for<E, TA> &&
        imm_zmaskable_args<basic_vector<E, TA>> &&
        unqualified_canonical_imbwsli<dx::zero_t, M, basic_vector<E, TA>, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, TA> operator()(
        M mask, basic_vector<E, TA> val, N shift) noexcept {
        return bwshift_left(internal::abi<TA>, dx::zero,
            dx::to_compatible_const_mask<basic_vector<E, TA>>(mask), val,
            shift);
    }

    template <extended_vector T, const_mask_for<T> M, integral_constant_like N>
    requires imm_zmaskable_args<T> && extended_imbwsli<dx::zero_t, M, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val, N shift) noexcept {
        if constexpr (unqualified_extended_mbwsli<dx::zero_t, M, T, N>) {
            return bwshift_left(
                dx::zero, dx::to_compatible_const_mask<T>(mask), val, shift);
        } else if constexpr (expression_imbwsli<dx::zero_t, M, T, N>) {
            return operator()(mask, dx::evaluate(val), shift);
        } else {
            return operator()(dx::to_compatible_const_mask<T>(mask),
                dx::to_canonical(val), shift);
        }
    }

    template <simd_vector T, const_mask_for<T> M, integral_constant_like N>
    requires invocable<bwshift_left_t, M, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t, M mask, T val, N shift) noexcept {
        return operator()(mask, val, shift);
    }
    ///

    template <fixed_width_abi A, simd_element_for<A> E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<E, A> operator()(
        basic_mask<E, A> val, size_t shift) noexcept {
        if constexpr (unqualified_canonical_mask_bwsl<basic_mask<E, A>>) {
            if consteval {
                return fallback(val, shift);
            } else {
                return bwshift_left(internal::abi<A>, val, shift);
            }
        } else {
            return fallback(val, shift);
        }
    }

    template <scalable_abi A, simd_element_for<A> E>
    requires unqualified_canonical_mask_bwsl<basic_mask<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<E, A> operator()(
        basic_mask<E, A> val, size_t shift) noexcept {
        return bwshift_left(internal::abi<A>, val, shift);
    }

    template <extended_mask T>
    requires extended_mask_bwsl<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val, size_t shift) noexcept {
        if constexpr (unqualified_extended_mask_bwsl<T>) {
            return bwshift_left(val, shift);
        } else if constexpr (expression_mask_bwsl<T>) {
            return operator()(dx::evaluate(val), shift);
        } else {
            return operator()(dx::to_canonical(val), shift);
        }
    }
    ///

    template <fixed_width_abi A, simd_element_for<A> E,
        integral_constant_like N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<E, A> operator()(
        basic_mask<E, A> val, N shift) noexcept {
        if constexpr (unqualified_canonical_mask_bwsli<basic_mask<E, A>, N>) {
            if consteval {
                return operator()(val, N::value);
            } else {
                return bwshift_left(internal::abi<A>, val, shift);
            }
        } else {
            return operator()(val, N::value);
        }
    }

    template <scalable_abi A, simd_element_for<A> E, integral_constant_like N>
    requires unqualified_canonical_mask_bwsli<basic_mask<E, A>, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<E, A> operator()(
        basic_mask<E, A> val, N shift) noexcept {
        return bwshift_left(internal::abi<A>, val, shift);
    }

    template <extended_mask T, integral_constant_like N>
    requires extended_mask_bwsli<T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val, N shift) noexcept {
        if constexpr (unqualified_extended_bwsli<T, N>) {
            return bwshift_left(val, shift);
        } else {
            return operator()(dx::to_canonical(val), shift);
        }
    }
};

template <size_t N>
struct bwshift_lefti_t {
    template <typename... Args>
    requires invocable<bwshift_left_t, Args..., size_constant<N>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(Args... val) noexcept {
        constexpr size_constant<N> shift;
        return bwshift_left_t::operator()(val..., shift);
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::bwshift_left_t bwshift_left{};
DPL_EXPORT template <size_t N>
inline constexpr internal::bwshift_lefti_t<N> bwshift_lefti{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
