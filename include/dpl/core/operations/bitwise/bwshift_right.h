// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

// IWYU pragma: always_keep

#include "dpl/core/operations/bitwise/result.h"
#include "dpl/core/operations/internal/masked.h"
#include "dpl/core/operations/internal/transform.h"
#if !DPL_MODULES
#  include "dpl/core/basic/initialize.h"
#  include "dpl/core/basic/internal/abi.h"
#  include "dpl/core/concepts/decayable.h"
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/concepts/simd_abi.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void bwshift_right(...) noexcept = delete;

struct bwshift_right_t;

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_canonical_bwsrv = requires(L lhs, R rhs) {
    {
        bwshift_right(internal::abi<A>, lhs, rhs)
    } -> canonical_bitshift_result<L, A>;
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_extended_bwsrv = requires(L lhs, R rhs) {
    { bwshift_right(lhs, rhs) } -> vector_with_common_abi<A>;
};

template <typename L, typename R>
concept expression_bwsrv = (simd_expression<L> || simd_expression<R>) &&
    invocable<bwshift_right_t, simd_expression_result_t<L>,
        simd_expression_result_t<R>>;

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept decayable_bwsrv =
    decayable_vector_for<L, operation_category::lane_agnostic> &&
    decayable_vector_for<R, operation_category::lane_agnostic> &&
    regular_invocable<bwshift_right_t, canonical_type_t<L>,
        canonical_type_t<R>>;

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept extended_bwsrv = unqualified_extended_bwsrv<L, R, A> ||
    expression_bwsrv<L, R> || decayable_bwsrv<L, R, A>;

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<L, R, M>>
concept unqualified_canonical_mbwsrv = requires(S src, M mask, L lhs, R rhs) {
    {
        bwshift_right(internal::abi<A>, src, mask, lhs, rhs)
    } -> equivalent_simd_type_with<
        canonical_if_zero_t<S, operation_result_t<bwshift_right_t, L, R>, A>>;
};

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<L, R, M>>
concept unqualified_extended_mbwsrv = requires(S src, M mask, L lhs, R rhs) {
    { bwshift_right(src, mask, lhs, rhs) } -> vector_with_common_abi<A>;
};

template <typename S, typename M, typename L, typename R>
concept expression_mbwsrv = (simd_expression<S> || simd_expression<M> ||
                                simd_expression<L> || simd_expression<R>) &&
    invocable<bwshift_right_t, expression_result_or_zero_t<S>,
        simd_expression_result_t<M>, simd_expression_result_t<L>,
        simd_expression_result_t<R>>;

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<L, R, M>>
concept decayable_mbwsrv =
    decayable_vector_for<
        canonical_if_zero_t<S, operation_result_t<bwshift_right_t, L, R>, A>,
        operation_category::lane_agnostic> &&
    decayable_mask_for<M, operation_category::lane_agnostic> &&
    decayable_vector_for<L, operation_category::lane_agnostic> &&
    decayable_vector_for<R, operation_category::lane_agnostic> &&
    regular_invocable<bwshift_right_t,
        canonical_or_zero_t<S, operation_result_t<bwshift_right_t, L, R>, A>,
        canonical_type_t<M>, canonical_type_t<L>, canonical_type_t<R>>;

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<L, R, M>>
concept extended_mbwsrv = unqualified_extended_mbwsrv<S, M, L, R, A> ||
    expression_mbwsrv<S, M, L, R> || decayable_mbwsrv<S, M, L, R, A>;

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<
        canonical_if_zero_t<S, operation_result_t<bwshift_right_t, L, R>>,
        operation_result_t<bwshift_right_t, L, R>>>
concept unqualified_canonical_imbwsrv = requires(S src, M mask, L lhs, R rhs) {
    {
        bwshift_right(internal::abi<A>, src,
            internal::to_const_mask<A, bwshift_right_t, S, L, R>(mask), lhs,
            rhs)
    } -> equivalent_simd_type_with<
        canonical_if_zero_t<S, operation_result_t<bwshift_right_t, L, R>, A>>;
};

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<
        canonical_if_zero_t<S, operation_result_t<bwshift_right_t, L, R>>,
        operation_result_t<bwshift_right_t, L, R>>>
concept unqualified_extended_imbwsrv = requires(S src, M mask, L lhs, R rhs) {
    {
        bwshift_right(src,
            internal::to_const_mask<A, bwshift_right_t, S, L, R>(mask), lhs,
            rhs)
    } -> vector_with_common_abi<A>;
};

template <typename S, typename M, typename L, typename R>
concept expression_imbwsrv =
    (simd_expression<S> || simd_expression<L> || simd_expression<R>) &&
    invocable<bwshift_right_t, expression_result_or_zero_t<S>, M,
        simd_expression_result_t<L>, simd_expression_result_t<R>>;

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<
        canonical_if_zero_t<S, operation_result_t<bwshift_right_t, L, R>>,
        operation_result_t<bwshift_right_t, L, R>>>
concept decayable_imbwsrv =
    decayable_vector_for<
        canonical_if_zero_t<S, operation_result_t<bwshift_right_t, L, R>, A>,
        operation_category::lane_agnostic> &&
    decayable_vector_for<L, operation_category::lane_agnostic> &&
    decayable_vector_for<R, operation_category::lane_agnostic> &&
    regular_invocable<bwshift_right_t,
        canonical_or_zero_t<S, operation_result_t<bwshift_right_t, L, R>, A>, M,
        canonical_type_t<L>, canonical_type_t<R>>;

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<
        canonical_if_zero_t<S, operation_result_t<bwshift_right_t, L, R>>,
        operation_result_t<bwshift_right_t, L, R>>>
concept extended_imbwsrv = unqualified_extended_imbwsrv<S, M, L, R, A> ||
    expression_imbwsrv<S, M, L, R> || decayable_imbwsrv<S, M, L, R, A>;

///

template <typename T>
concept unqualified_canonical_bwsr = requires(T val, size_t shift) {
    {
        bwshift_right(internal::abi<T>, val, shift)
    } -> canonical_bitshift_result<T>;
};

template <typename T>
concept unqualified_extended_bwsr = requires(T val, size_t shift) {
    {
        bwshift_right(val, shift)
    } -> vector_with_common_abi<typename T::abi_type>;
};

template <typename T>
concept expression_bwsr = simd_expression<T> &&
    regular_invocable<bwshift_right_t, simd_expression_result_t<T>, size_t>;

template <typename T>
concept decayable_bwsr =
    decayable_vector_for<T, operation_category::lane_agnostic> &&
    regular_invocable<bwshift_right_t, canonical_type_t<T>, size_t>;

template <typename T>
concept extended_bwsr =
    unqualified_extended_bwsr<T> || expression_bwsr<T> || decayable_bwsr<T>;

template <typename S, typename M, typename T, typename A = common_abi_t<M, T>>
concept unqualified_canonical_mbwsr =
    requires(S src, M mask, T val, size_t shift) {
        {
            bwshift_right(internal::abi<A>, src, mask, val, shift)
        } -> equivalent_simd_type_with<canonical_if_zero_t<S, T, A>>;
    };

template <typename S, typename M, typename T, typename A = common_abi_t<M, T>>
concept unqualified_extended_mbwsr =
    requires(S src, M mask, T val, size_t shift) {
        { bwshift_right(src, mask, val, shift) } -> vector_with_common_abi<A>;
    };

template <typename S, typename M, typename T>
concept expression_mbwsr =
    (simd_expression<S> || simd_expression<M> || simd_expression<T>) &&
    regular_invocable<bwshift_right_t, expression_result_or_zero_t<S>,
        simd_expression_result_t<M>, simd_expression_result_t<T>, size_t>;

template <typename S, typename M, typename T, typename A = common_abi_t<T, M>>
concept decayable_mbwsr = decayable_vector_for<canonical_if_zero_t<S, T, A>,
                              operation_category::lane_agnostic> &&
    decayable_vector_for<T, operation_category::lane_agnostic> &&
    decayable_mask_for<M, operation_category::lane_agnostic> &&
    regular_invocable<bwshift_right_t, canonical_or_zero_t<S, T, A>,
        canonical_type_t<M>, canonical_type_t<T>, size_t>;

template <typename S, typename M, typename T, typename A = common_abi_t<T, M>>
concept extended_mbwsr = unqualified_extended_mbwsr<S, M, T, A> ||
    expression_mbwsr<S, M, T> || decayable_mbwsr<S, M, T, A>;

template <typename S, typename M, typename T,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept unqualified_canonical_imbwsr =
    requires(S src, M mask, T val, size_t shift) {
        {
            bwshift_right(internal::abi<A>, src,
                internal::to_const_mask<A, bwshift_right_t, S, T, size_t>(mask),
                val, shift)
        } -> equivalent_simd_type_with<canonical_if_zero_t<S, T, A>>;
    };

template <typename S, typename M, typename T,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept unqualified_extended_imbwsr =
    requires(S src, M mask, T val, size_t shift) {
        {
            bwshift_right(src,
                internal::to_const_mask<A, bwshift_right_t, S, T, size_t>(mask),
                val, shift)
        } -> vector_with_common_abi<A>;
    };

template <typename S, typename M, typename T>
concept expression_imbwsr = (simd_expression<S> || simd_expression<T>) &&
    regular_invocable<bwshift_right_t, expression_result_or_zero_t<S>, M,
        simd_expression_result_t<T>, size_t>;

template <typename S, typename M, typename T,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept decayable_imbwsr = decayable_vector_for<canonical_if_zero_t<S, T>,
                               operation_category::lane_agnostic> &&
    decayable_vector_for<T, operation_category::lane_agnostic> &&
    regular_invocable<bwshift_right_t, canonical_or_zero_t<S, T, A>, M,
        canonical_type_t<T>, size_t>;

template <typename S, typename M, typename T,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept extended_imbwsr = unqualified_extended_imbwsr<S, M, T, A> ||
    expression_imbwsr<S, M, T> || decayable_imbwsr<S, M, T, A>;
///

template <typename L, typename R, typename A = typename L::abi_type>
concept unqualified_canonical_bwsri = requires(L lhs, R rhs) {
    {
        bwshift_right(internal::abi<A>, lhs, rhs)
    } -> canonical_bitshift_result<L>;
};

template <typename L, typename R, typename A = typename L::abi_type>
concept unqualified_extended_bwsri = requires(L lhs, R rhs) {
    { bwshift_right(lhs, rhs) } -> vector_with_common_abi<A>;
};

template <typename L, typename R>
concept expression_bwsri = simd_expression<L> &&
    invocable<bwshift_right_t, simd_expression_result_t<L>, R>;

template <typename L, typename R, typename A = typename L::abi_type>
concept decayable_bwsri =
    decayable_vector_for<L, operation_category::lane_agnostic> &&
    regular_invocable<bwshift_right_t, canonical_type_t<L>, R>;

template <typename L, typename R, typename A = typename L::abi_type>
concept extended_bwsri = unqualified_extended_bwsri<L, R, A> ||
    expression_bwsri<L, R> || decayable_bwsri<L, R, A>;

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<L, M>>
concept unqualified_canonical_mbwsri = requires(S src, M mask, L lhs, R rhs) {
    {
        bwshift_right(internal::abi<A>, src, mask, lhs, rhs)
    } -> equivalent_simd_type_with<
        canonical_if_zero_t<S, operation_result_t<bwshift_right_t, L, R>, A>>;
};

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<L, M>>
concept unqualified_extended_mbwsri = requires(S src, M mask, L lhs, R rhs) {
    { bwshift_right(src, mask, lhs, rhs) } -> vector_with_common_abi<A>;
};

template <typename S, typename M, typename L, typename R>
concept expression_mbwsri =
    (simd_expression<S> || simd_expression<M> || simd_expression<L>) &&
    invocable<bwshift_right_t, expression_result_or_zero_t<S>,
        simd_expression_result_t<M>, simd_expression_result_t<L>, R>;

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<L, M>>
concept decayable_mbwsri =
    decayable_vector_for<
        canonical_if_zero_t<S, operation_result_t<bwshift_right_t, L, R>, A>,
        operation_category::lane_agnostic> &&
    decayable_mask_for<M, operation_category::lane_agnostic> &&
    decayable_vector_for<L, operation_category::lane_agnostic> &&
    regular_invocable<bwshift_right_t,
        canonical_if_zero_t<S, operation_result_t<bwshift_right_t, L, R>, A>,
        canonical_type_t<M>, canonical_type_t<L>, R>;

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<L, M>>
concept extended_mbwsri = unqualified_extended_mbwsri<S, M, L, R, A> ||
    expression_mbwsri<S, M, L, R> || decayable_mbwsri<S, M, L, R, A>;

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<
        canonical_if_zero_t<S, operation_result_t<bwshift_right_t, L, R>>,
        operation_result_t<bwshift_right_t, L, R>>>
concept unqualified_canonical_imbwsri = requires(S src, M mask, L lhs, R rhs) {
    {
        bwshift_right(internal::abi<A>, src,
            internal::to_const_mask<A, bwshift_right_t, S, L, R>(mask), lhs,
            rhs)
    } -> equivalent_simd_type_with<
        canonical_if_zero_t<S, operation_result_t<bwshift_right_t, L, R>, A>>;
};

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<
        canonical_if_zero_t<S, operation_result_t<bwshift_right_t, L, R>>,
        operation_result_t<bwshift_right_t, L, R>>>
concept unqualified_extended_imbwsri = requires(S src, M mask, L lhs, R rhs) {
    {
        bwshift_right(src,
            internal::to_const_mask<A, bwshift_right_t, S, L, R>(mask), lhs,
            rhs)
    } -> vector_with_common_abi<A>;
};

template <typename S, typename M, typename L, typename R>
concept expression_imbwsri = (simd_expression<S> || simd_expression<L>) &&
    invocable<bwshift_right_t, expression_result_or_zero_t<S>, M,
        simd_expression_result_t<L>, R>;

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<
        canonical_if_zero_t<S, operation_result_t<bwshift_right_t, L, R>>,
        operation_result_t<bwshift_right_t, L, R>>>
concept decayable_imbwsri =
    decayable_vector_for<
        canonical_if_zero_t<S, operation_result_t<bwshift_right_t, L, R>, A>,
        operation_category::lane_agnostic> &&
    decayable_vector_for<L, operation_category::lane_agnostic> &&
    regular_invocable<bwshift_right_t,
        canonical_or_zero_t<S, operation_result_t<bwshift_right_t, L, R>, A>, M,
        canonical_type_t<L>, R>;

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<
        canonical_if_zero_t<S, operation_result_t<bwshift_right_t, L, R>>,
        operation_result_t<bwshift_right_t, L, R>>>
concept extended_imbwsri = unqualified_extended_imbwsri<S, M, L, R, A> ||
    expression_imbwsri<S, M, L, R> || decayable_imbwsri<S, M, L, R, A>;

///

template <typename T>
concept unqualified_canonical_mask_bwsr = unqualified_canonical_bwsr<T>;

template <typename T>
concept unqualified_extended_mask_bwsr = unqualified_extended_bwsr<T>;

template <typename T>
concept expression_mask_bwsr = simd_expression<T> &&
    regular_invocable<bwshift_right_t, simd_expression_result_t<T>, size_t>;

template <typename T>
concept decayable_mask_bwsr =
    decayable_mask_for<T, operation_category::lane_permutation> &&
    regular_invocable<bwshift_right_t, canonical_type_t<T>, size_t>;

template <typename T>
concept extended_mask_bwsr = unqualified_extended_mask_bwsr<T> ||
    expression_mask_bwsr<T> || decayable_mask_bwsr<T>;

///

template <typename T, typename N>
concept unqualified_canonical_mask_bwsri = unqualified_canonical_bwsri<T, N>;

template <typename T, typename N>
concept unqualified_extended_mask_bwsri = unqualified_extended_bwsri<T, N>;

template <typename T, typename N>
concept expression_mask_bwsri = simd_expression<T> &&
    regular_invocable<bwshift_right_t, simd_expression_result_t<T>, N>;

template <typename T, typename N>
concept decayable_mask_bwsri =
    decayable_mask_for<T, operation_category::lane_permutation> &&
    regular_invocable<bwshift_right_t, canonical_type_t<T>, N>;

template <typename T, typename N>
concept extended_mask_bwsri = unqualified_extended_bwsri<T, N> ||
    expression_mask_bwsri<T, N> || decayable_mask_bwsri<T, N>;

struct bwshift_right_t {
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
                        __DPL bit_cast<bit_type>(lhs) >> rhs);
                } else {
                    return __DPL bit_cast<LE>(
                        __DPL bit_cast<bit_type>(lhs) >> rhs);
                }
            },
            val, shift);
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
                        __DPL bit_cast<bit_type>(lhs) >> shift);
                } else {
                    return __DPL bit_cast<L>(
                        __DPL bit_cast<bit_type>(lhs) >> shift);
                }
            },
            val);
    }

    template <typename L, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_mask<L, A> val, size_t shift) noexcept {
        using bit_type = bit_type_t<sizeof(L) * char_bit_v>;
        using mask_type = basic_mask<L, A>;
        return []<size_t... Is>(
                   mask_type val, unsigned shift, index_sequence<Is...>) {
            return dx::initialize<mask_type>(bitset(
                (Is + shift < simd_abi_traits<L, A>::size ? val[Is + shift]
                                                          : false)...));
        }(val, static_cast<unsigned>(shift), iota_sequence<L, A>);
    }

public:
    template <fixed_width_abi A, simd_element_for<A> LE, simd_element_for<A> RE>
    requires common_size_with<LE, RE> && integral<RE>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<LE, A> operator()(
        basic_vector<LE, A> val, basic_vector<RE, A> shift) noexcept {
        if constexpr (unqualified_canonical_bwsrv<basic_vector<LE, A>,
                          basic_vector<RE, A>>) {
            if consteval {
                return fallback(val, shift);
            } else {
                return bwshift_right(internal::abi<A>, val, shift);
            }
        } else {
            return fallback(val, shift);
        }
    }

    template <simd_abi LA, simd_element_for<LA> LE, common_abi_with<LA> RA,
        simd_element_for<RA> RE>
    requires (scalable_abi<LA> || scalable_abi<RA> || different_from<LA, RA>) &&
        integral<RE> &&
        unqualified_canonical_bwsrv<basic_vector<LE, LA>, basic_vector<RE, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<LE, common_abi_t<LA, RA>> operator()(
        basic_vector<LE, LA> val, basic_vector<RE, RA> shift) noexcept {
        using A = common_abi_t<LA, RA>;
        return bwshift_right(internal::abi<A>, val, shift);
    }

    template <simd_vector L, simd_vector R>
    requires (extended_vector<L> || extended_vector<R>) &&
        common_abi_with<typename L::abi_type, typename R::abi_type> &&
        common_size_with<typename L::value_type, typename R::value_type> &&
        integral<typename R::value_type> && extended_bwsrv<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L val, R shift) noexcept {
        if constexpr (unqualified_extended_bwsrv<L, R>) {
            return bwshift_right(val, shift);
        } else if constexpr (expression_bwsrv<L, R>) {
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
        if constexpr (unqualified_canonical_mbwsrv<basic_vector<LE, A>,
                          basic_mask<ME, A>, basic_vector<LE, A>,
                          basic_vector<RE, A>>) {
            if consteval {
                return internal::masked<bwshift_right_t>(src, mask, lhs, rhs);
            } else {
                return bwshift_right(internal::abi<A>, src, mask, lhs, rhs);
            }
        } else {
            return internal::masked<bwshift_right_t>(src, mask, lhs, rhs);
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
        unqualified_canonical_mbwsrv<basic_vector<LE, SA>, basic_mask<ME, SA>,
            basic_vector<LE, LA>, basic_vector<RE, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<LE, SA> operator()(basic_vector<LE, SA> src,
        basic_mask<ME, SA> mask, basic_vector<LE, LA> lhs,
        basic_vector<RE, RA> rhs) noexcept {
        return bwshift_right(internal::abi<SA>, src, mask, lhs, rhs);
    }

    template <simd_vector S, simd_mask M, simd_vector L, simd_vector R>
    requires integral<typename R::value_type> &&
        common_size_with<typename L::value_type, typename R::value_type> &&
        (extended_vector<S> || extended_mask<M> || extended_vector<L> ||
            extended_vector<R>) &&
        maskable_args<S, M, L, R> && extended_mbwsrv<S, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, L lhs, R rhs) noexcept {
        if constexpr (unqualified_extended_mbwsrv<S, M, L, R>) {
            return bwshift_right(src, mask, lhs, rhs);
        } else if constexpr (expression_mbwsrv<S, M, L, R>) {
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
        if constexpr (unqualified_canonical_mbwsrv<dx::zero_t,
                          basic_mask<ME, A>, basic_vector<LE, A>,
                          basic_vector<RE, A>>) {
            if consteval {
                return internal::masked<bwshift_right_t>(mask, lhs, rhs);
            } else {
                return bwshift_right(
                    internal::abi<A>, dx::zero, mask, lhs, rhs);
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
        unqualified_canonical_mbwsrv<dx::zero_t, basic_mask<ME, SA>,
            basic_vector<E, LA>, basic_vector<E, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(basic_mask<ME, SA> mask,
        basic_vector<E, LA> lhs, basic_vector<E, RA> rhs) noexcept {
        return bwshift_right(
            internal::abi<common_abi_t<LA, RA>>, dx::zero, mask, lhs, rhs);
    }

    template <simd_mask M, simd_vector L, simd_vector R>
    requires (extended_mask<M> || extended_vector<L> || extended_vector<R>) &&
        zmaskable_args<M, L, R> && extended_mbwsrv<dx::zero_t, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, L lhs, R rhs) noexcept {
        if constexpr (unqualified_extended_mbwsrv<dx::zero_t, M, L, R>) {
            return bwshift_right(dx::zero, mask, lhs, rhs);
        } else if constexpr (expression_mbwsrv<dx::zero_t, M, L, R>) {
            return operator()(
                dx::evaluate(mask), dx::evaluate(lhs), dx::evaluate(rhs));
        } else {
            return operator()(dx::to_canonical(mask), dx::to_canonical(lhs),
                dx::to_canonical(rhs));
        }
    }

    template <simd_mask M, simd_vector L, simd_vector R>
    requires invocable<bwshift_right_t, M, L, R>
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
        if constexpr (unqualified_canonical_imbwsrv<basic_vector<E, A>, M,
                          basic_vector<E, A>, basic_vector<E, A>>) {
            if consteval {
                return internal::masked<bwshift_right_t>(src, mask, lhs, rhs);
            } else {
                return bwshift_right(internal::abi<A>, src,
                    dx::to_compatible_const_mask<basic_vector<E, A>>(mask), lhs,
                    rhs);
            }
        } else {
            return internal::masked<bwshift_right_t>(src, mask, lhs, rhs);
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
        unqualified_canonical_imbwsrv<basic_vector<E, SA>, M,
            basic_vector<E, LA>, basic_vector<E, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, SA> operator()(basic_vector<E, SA> src,
        M mask, basic_vector<E, LA> lhs, basic_vector<E, RA> rhs) noexcept {
        return bwshift_right(internal::abi<SA>, src,
            dx::to_compatible_const_mask<basic_vector<E, SA>>(mask), lhs, rhs);
    }

    template <simd_vector S, const_mask_for<S> M, simd_vector L, simd_vector R>
    requires (extended_vector<S> || extended_vector<L> || extended_vector<R>) &&
        imm_maskable_args<S, L, R> && extended_imbwsrv<S, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, L lhs, R rhs) noexcept {
        if constexpr (unqualified_extended_imbwsrv<S, M, L, R>) {
            return bwshift_right(
                src, dx::to_compatible_const_mask<S>(mask), lhs, rhs);
        } else if constexpr (expression_imbwsrv<S, M, L, R>) {
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
        if constexpr (unqualified_canonical_imbwsrv<dx::zero_t, M,
                          basic_vector<E, A>, basic_vector<E, A>>) {
            if consteval {
                return internal::masked<bwshift_right_t>(mask, lhs, rhs);
            } else {
                return bwshift_right(internal::abi<A>, dx::zero,
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
        unqualified_canonical_imbwsrv<dx::zero_t, M, basic_vector<E, LA>,
            basic_vector<E, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, common_abi_t<LA, RA>> operator()(
        M mask, basic_vector<E, LA> lhs, basic_vector<E, RA> rhs) noexcept {
        using A = common_abi_t<LA, RA>;
        return bwshift_right(internal::abi<A>, dx::zero,
            dx::to_compatible_const_mask<basic_vector<E, A>>(mask), lhs, rhs);
    }

    template <simd_vector L, simd_vector R,
        const_mask_for<operation_result_t<bwshift_right_t, L, R>> M>
    requires (extended_vector<L> || extended_vector<R>) &&
        imm_zmaskable_args<L, R> && extended_imbwsrv<dx::zero_t, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, L lhs, R rhs) noexcept {
        using S = operation_result_t<bwshift_right_t, L, R>;
        if constexpr (unqualified_extended_imbwsrv<dx::zero_t, M, L, R>) {
            return bwshift_right(
                dx::zero, dx::to_compatible_const_mask<S>(mask), lhs, rhs);
        } else if constexpr (expression_imbwsrv<dx::zero_t, M, L, R>) {
            return operator()(mask, dx::evaluate(lhs), dx::evaluate(rhs));
        } else {
            return operator()(
                mask, dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }

    template <simd_vector L, simd_vector R,
        const_mask_for<operation_result_t<bwshift_right_t, L, R>> M>
    requires invocable<bwshift_right_t, M, L, R>
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
        if constexpr (unqualified_canonical_bwsr<basic_vector<E, A>>) {
            if consteval {
                return fallback(val, shift);
            } else {
                return bwshift_right(internal::abi<A>, val, shift);
            }
        } else {
            return fallback(val, shift);
        }
    }

    template <scalable_abi A, simd_element_for<A> E>
    requires unqualified_canonical_bwsr<basic_vector<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val, size_t shift) noexcept {
        return bwshift_right(internal::abi<A>, val, shift);
    }

    template <extended_vector T>
    requires extended_bwsr<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val, size_t shift) noexcept {
        if constexpr (unqualified_extended_bwsr<T>) {
            return bwshift_right(val, shift);
        } else if constexpr (expression_bwsr<T>) {
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
        if constexpr (unqualified_canonical_mbwsr<basic_vector<E, A>,
                          basic_mask<ME, A>, basic_vector<E, A>>) {
            if consteval {
                return internal::masked<bwshift_right_t>(src, mask, val, shift);
            } else {
                return bwshift_right(internal::abi<A>, src, mask, val, shift);
            }
        } else {
            return internal::masked<bwshift_right_t>(src, mask, val, shift);
        }
    }

    template <simd_abi MA, simd_element_for<MA> ME, simd_abi TA,
        simd_element_for<TA> E>
    requires common_size_with<E, ME> &&
        (different_from<MA, TA> || scalable_abi<MA> || scalable_abi<TA>) &&
        maskable_args<basic_vector<E, MA>, basic_mask<ME, MA>,
            basic_vector<E, TA>> &&
        unqualified_canonical_mbwsr<basic_vector<E, MA>, basic_mask<ME, MA>,
            basic_vector<E, TA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, MA> operator()(basic_vector<E, MA> src,
        basic_mask<ME, MA> mask, basic_vector<E, TA> val,
        size_t shift) noexcept {
        return bwshift_right(internal::abi<MA>, src, mask, val, shift);
    }

    template <simd_vector S, simd_mask M, simd_vector T>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        maskable_args<S, M, T> && extended_mbwsr<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        S src, M mask, T val, size_t shift) noexcept {
        if constexpr (unqualified_extended_mbwsr<S, M, T>) {
            return bwshift_right(src, mask, val, shift);
        } else if constexpr (expression_mbwsr<S, M, T>) {
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
        if constexpr (unqualified_canonical_mbwsr<dx::zero_t, basic_mask<ME, A>,
                          basic_vector<E, A>>) {
            if consteval {
                return internal::masked<bwshift_right_t>(mask, val, shift);
            } else {
                return bwshift_right(
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
        unqualified_canonical_mbwsr<dx::zero_t, basic_mask<ME, MA>,
            basic_vector<E, TA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(basic_mask<ME, MA> mask,
        basic_vector<E, TA> val, size_t shift) noexcept {
        return bwshift_right(internal::abi<MA>, dx::zero, mask, val, shift);
    }

    template <simd_mask M, simd_vector T>
    requires (extended_mask<M> || extended_vector<T>) && zmaskable_args<M, T> &&
        extended_mbwsr<dx::zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val, size_t shift) noexcept {
        if constexpr (unqualified_extended_mbwsr<dx::zero_t, M, T>) {
            return bwshift_right(dx::zero, mask, val, shift);
        } else if constexpr (expression_mbwsr<dx::zero_t, M, T>) {
            return operator()(dx::evaluate(mask), dx::evaluate(val), shift);
        } else {
            return operator()(
                dx::to_canonical(mask), dx::to_canonical(val), shift);
        }
    }

    template <simd_mask M, simd_vector T>
    requires invocable<bwshift_right_t, M, T, size_t>
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
        if constexpr (unqualified_canonical_imbwsr<basic_vector<E, A>, M,
                          basic_vector<E, A>>) {
            if consteval {
                return internal::masked<bwshift_right_t>(src, mask, val, shift);
            } else {
                return bwshift_right(internal::abi<A>, src,
                    dx::to_compatible_const_mask<basic_vector<E, A>>(mask), val,
                    shift);
            }
        } else {
            return internal::masked<bwshift_right_t>(src, mask, val, shift);
        }
    }

    template <simd_abi SA, simd_element_for<SA> E,
        const_mask_for<basic_vector<E, SA>> M, simd_abi TA>
    requires (different_from<SA, TA> || scalable_abi<SA> || scalable_abi<TA>) &&
        simd_element_for<E, TA> &&
        imm_maskable_args<basic_vector<E, SA>, basic_vector<E, TA>> &&
        unqualified_canonical_imbwsr<basic_vector<E, SA>, M,
            basic_vector<E, TA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, SA> operator()(basic_vector<E, SA> src,
        M mask, basic_vector<E, TA> val, size_t shift) noexcept {
        return bwshift_right(internal::abi<SA>, src,
            dx::to_compatible_const_mask<basic_vector<E, SA>>(mask), val,
            shift);
    }

    template <simd_vector S, const_mask_for<S> M, simd_vector T>
    requires (extended_vector<S> || extended_vector<T>) &&
        imm_maskable_args<S, T> && extended_imbwsr<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        S src, M mask, T val, size_t shift) noexcept {
        if constexpr (unqualified_extended_mbwsr<S, M, T>) {
            return bwshift_right(
                src, dx::to_compatible_const_mask<S>(mask), val, shift);
        } else if constexpr (expression_imbwsr<S, M, T>) {
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
        if constexpr (unqualified_canonical_imbwsr<dx::zero_t, M,
                          basic_vector<E, A>>) {
            if consteval {
                return internal::masked<bwshift_right_t>(mask, val, shift);
            } else {
                return bwshift_right(internal::abi<A>, dx::zero,
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
        unqualified_canonical_imbwsr<dx::zero_t, M, basic_vector<E, TA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, TA> operator()(
        M mask, basic_vector<E, TA> val, size_t shift) noexcept {
        return bwshift_right(internal::abi<TA>, dx::zero,
            dx::to_compatible_const_mask<basic_vector<E, TA>>(mask), val,
            shift);
    }

    template <extended_vector T, const_mask_for<T> M>
    requires imm_zmaskable_args<T> && extended_imbwsr<dx::zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val, size_t shift) noexcept {
        if constexpr (unqualified_extended_imbwsr<dx::zero_t, M, T>) {
            return bwshift_right(
                dx::zero, dx::to_compatible_const_mask<T>(mask), val, shift);
        } else if constexpr (expression_imbwsr<dx::zero_t, M, T>) {
            return operator()(mask, dx::evaluate(val), shift);
        } else {
            return operator()(dx::to_compatible_const_mask<T>(mask),
                dx::to_canonical(val), shift);
        }
    }

    template <simd_vector T, const_mask_for<T> M>
    requires invocable<bwshift_right_t, M, T, size_t>
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
        if constexpr (unqualified_canonical_bwsri<basic_vector<E, A>, N>) {
            if consteval {
                return fallback(val, shift);
            } else {
                return bwshift_right(internal::abi<A>, val, shift);
            }
        } else {
            return fallback(val, shift);
        }
    }

    template <scalable_abi A, simd_element_for<A> E, integral_constant_like N>
    requires unqualified_canonical_bwsri<basic_vector<E, A>, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val, N shift) noexcept {
        return bwshift_right(internal::abi<A>, val, shift);
    }

    template <extended_vector T, integral_constant_like N>
    requires extended_bwsri<T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val, N shift) noexcept {
        if constexpr (unqualified_extended_bwsri<T, N>) {
            return bwshift_right(val, shift);
        } else if constexpr (expression_bwsri<T, N>) {
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
        if constexpr (unqualified_canonical_mbwsri<basic_vector<E, A>,
                          basic_mask<ME, A>, basic_vector<E, A>, N>) {
            if consteval {
                return internal::masked<bwshift_right_t>(src, mask, val, shift);
            } else {
                return bwshift_right(internal::abi<A>, src, mask, val, shift);
            }
        } else {
            return internal::masked<bwshift_right_t>(src, mask, val, shift);
        }
    }

    template <simd_abi MA, simd_element_for<MA> ME, simd_abi TA,
        simd_element_for<TA> E, integral_constant_like N>
    requires common_size_with<E, ME> &&
        (different_from<MA, TA> || scalable_abi<MA> || scalable_abi<TA>) &&
        maskable_args<basic_vector<E, MA>, basic_mask<ME, MA>,
            basic_vector<E, TA>> &&
        unqualified_canonical_mbwsri<basic_vector<E, MA>, basic_mask<ME, MA>,
            basic_vector<E, TA>, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, MA> operator()(basic_vector<E, MA> src,
        basic_mask<ME, MA> mask, basic_vector<E, TA> val, N shift) noexcept {
        return bwshift_right(internal::abi<MA>, src, mask, val, shift);
    }

    template <simd_vector S, simd_mask M, simd_vector T,
        integral_constant_like N>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        maskable_args<S, M, T> && extended_mbwsri<S, M, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, T val, N shift) noexcept {
        if constexpr (unqualified_extended_mbwsri<S, M, T, N>) {
            return bwshift_right(src, mask, val, shift);
        } else if constexpr (expression_mbwsri<S, M, T, N>) {
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
        if constexpr (unqualified_canonical_mbwsri<dx::zero_t,
                          basic_mask<ME, A>, basic_vector<E, A>, N>) {
            if consteval {
                return internal::masked<bwshift_right_t>(mask, val, shift);
            } else {
                return bwshift_right(
                    internal::abi<A>, dx::zero, mask, val, shift);
            }
        } else {
            return internal::masked<bwshift_right_t>(mask, val, shift);
        }
    }

    template <simd_abi MA, simd_element_for<MA> ME, simd_abi TA,
        simd_element_for<TA> E, integral_constant_like N>
    requires common_size_with<E, ME> &&
        (different_from<MA, TA> || scalable_abi<MA> || scalable_abi<TA>) &&
        zmaskable_args<basic_mask<ME, MA>, basic_vector<E, TA>> &&
        unqualified_canonical_mbwsri<dx::zero_t, basic_mask<ME, MA>,
            basic_vector<E, TA>, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        basic_mask<ME, MA> mask, basic_vector<E, TA> val, N shift) noexcept {
        return bwshift_right(internal::abi<MA>, dx::zero, mask, val, shift);
    }

    template <simd_mask M, simd_vector T, integral_constant_like N>
    requires (extended_mask<M> || extended_vector<T>) && zmaskable_args<M, T> &&
        extended_mbwsri<dx::zero_t, M, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val, N shift) noexcept {
        if constexpr (unqualified_extended_mbwsri<dx::zero_t, M, T, N>) {
            return bwshift_right(dx::zero, mask, val, shift);
        } else if constexpr (expression_mbwsri<dx::zero_t, M, T, N>) {
            return operator()(dx::evaluate(mask), dx::evaluate(val), shift);
        } else {
            return operator()(
                dx::to_canonical(mask), dx::to_canonical(val), shift);
        }
    }

    template <simd_mask M, simd_vector T, integral_constant_like N>
    requires invocable<bwshift_right_t, M, T, N>
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
        if constexpr (unqualified_canonical_imbwsri<basic_vector<E, A>, M,
                          basic_vector<E, A>, N>) {
            if consteval {
                return internal::masked<bwshift_right_t>(src, mask, val, shift);
            } else {
                return bwshift_right(internal::abi<A>, src,
                    dx::to_compatible_const_mask<basic_vector<E, A>>(mask), val,
                    shift);
            }
        } else {
            return internal::masked<bwshift_right_t>(src, mask, val, shift);
        }
    }

    template <simd_abi SA, simd_element_for<SA> E,
        const_mask_for<basic_vector<E, SA>> M, simd_abi TA,
        integral_constant_like N>
    requires (different_from<SA, TA> || scalable_abi<SA> || scalable_abi<TA>) &&
        simd_element_for<E, TA> &&
        imm_maskable_args<basic_vector<E, SA>, basic_vector<E, TA>> &&
        unqualified_canonical_imbwsri<basic_vector<E, SA>, M,
            basic_vector<E, TA>, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, SA> operator()(basic_vector<E, SA> src,
        M mask, basic_vector<E, TA> val, N shift) noexcept {
        return bwshift_right(internal::abi<SA>, src,
            dx::to_compatible_const_mask<basic_vector<E, SA>>(mask), val,
            shift);
    }

    template <simd_vector S, const_mask_for<S> M, simd_vector T,
        integral_constant_like N>
    requires (extended_vector<S> || extended_vector<T>) &&
        imm_maskable_args<S, T> && extended_imbwsri<S, M, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, T val, N shift) noexcept {
        if constexpr (unqualified_extended_imbwsri<S, M, T, N>) {
            return bwshift_right(
                src, dx::to_compatible_const_mask<S>(mask), val, shift);
        } else if constexpr (expression_imbwsri<S, M, T, N>) {
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
        if constexpr (unqualified_canonical_imbwsri<dx::zero_t, M,
                          basic_vector<E, A>, N>) {
            if consteval {
                return internal::masked<bwshift_right_t>(mask, val, shift);
            } else {
                return bwshift_right(internal::abi<A>, dx::zero,
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
        unqualified_canonical_imbwsri<dx::zero_t, M, basic_vector<E, TA>, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, TA> operator()(
        M mask, basic_vector<E, TA> val, N shift) noexcept {
        return bwshift_right(internal::abi<TA>, dx::zero,
            dx::to_compatible_const_mask<basic_vector<E, TA>>(mask), val,
            shift);
    }

    template <extended_vector T, const_mask_for<T> M, integral_constant_like N>
    requires imm_zmaskable_args<T> && extended_imbwsri<dx::zero_t, M, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val, N shift) noexcept {
        if constexpr (unqualified_extended_mbwsri<dx::zero_t, M, T, N>) {
            return bwshift_right(
                dx::zero, dx::to_compatible_const_mask<T>(mask), val, shift);
        } else if constexpr (expression_imbwsri<dx::zero_t, M, T, N>) {
            return operator()(mask, dx::evaluate(val), shift);
        } else {
            return operator()(dx::to_compatible_const_mask<T>(mask),
                dx::to_canonical(val), shift);
        }
    }

    template <simd_vector T, const_mask_for<T> M, integral_constant_like N>
    requires invocable<bwshift_right_t, M, T, N>
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
        if constexpr (unqualified_canonical_mask_bwsr<basic_mask<E, A>>) {
            if consteval {
                return fallback(val, shift);
            } else {
                return bwshift_right(internal::abi<A>, val, shift);
            }
        } else {
            return fallback(val, shift);
        }
    }

    template <scalable_abi A, simd_element_for<A> E>
    requires unqualified_canonical_mask_bwsr<basic_mask<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<E, A> operator()(
        basic_mask<E, A> val, size_t shift) noexcept {
        return bwshift_right(internal::abi<A>, val, shift);
    }

    template <extended_mask T>
    requires extended_mask_bwsr<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val, size_t shift) noexcept {
        if constexpr (unqualified_extended_mask_bwsr<T>) {
            return bwshift_right(val, shift);
        } else if constexpr (expression_mask_bwsr<T>) {
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
        if constexpr (unqualified_canonical_mask_bwsri<basic_mask<E, A>, N>) {
            if consteval {
                return operator()(val, N::value);
            } else {
                return bwshift_right(internal::abi<A>, val, shift);
            }
        } else {
            return operator()(val, N::value);
        }
    }

    template <scalable_abi A, simd_element_for<A> E, integral_constant_like N>
    requires unqualified_canonical_mask_bwsri<basic_mask<E, A>, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<E, A> operator()(
        basic_mask<E, A> val, N shift) noexcept {
        return bwshift_right(internal::abi<A>, val, shift);
    }

    template <extended_mask T, integral_constant_like N>
    requires extended_mask_bwsri<T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val, N shift) noexcept {
        if constexpr (unqualified_extended_bwsri<T, N>) {
            return bwshift_right(val, shift);
        } else {
            return operator()(dx::to_canonical(val), shift);
        }
    }
};

template <size_t N>
struct bwshift_righti_t {
    template <typename... Args>
    requires invocable<bwshift_right_t, Args..., size_constant<N>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(Args... val) noexcept {
        constexpr size_constant<N> shift;
        return bwshift_right_t::operator()(val..., shift);
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::bwshift_right_t bwshift_right{};
DPL_EXPORT template <size_t N>
inline constexpr internal::bwshift_righti_t<N> bwshift_righti{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
