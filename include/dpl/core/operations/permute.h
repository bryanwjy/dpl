// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/operations/evaluate.h"
#include "dpl/core/operations/internal/masked.h"
#if !DPL_MODULES
#  include "dpl/core/basic/broadcast.h"
#  include "dpl/core/basic/immediate.h"
#  include "dpl/core/basic/initialize.h"
#  include "dpl/core/basic/internal/abi.h"
#  include "dpl/core/basic/internal/iota_sequence.h"
#  include "dpl/core/basic/to_canonical.h"
#  include "dpl/core/concepts/common_size_with.h"
#  include "dpl/core/concepts/decayable.h"
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/type_traits/simd_abi_type.h"
#  include "dpl/core/type_traits/simd_element_type.h"
#  include "dpl/std/utility/sequence.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void permute(...) noexcept = delete;

struct permute_t;

template <typename T, typename I>
concept unqualified_canonical_permutei = requires(T val, I idx) {
    {
        permute(internal::abi<T>, val, __DPL to_index_sequence(idx))
    } -> equivalent_vector_with<T>;
};

template <typename T, typename I>
concept unqualified_extended_permutei = requires(T val, I idx) {
    {
        permute(val, __DPL to_index_sequence(idx))
    } -> vector_with_common_abi<simd_abi_type_t<T>>;
};

template <typename T, typename I>
concept expression_permutei =
    simd_expression<T> && invocable<permute_t, simd_expression_result_t<T>, I>;

template <typename T, typename I>
concept decayable_permutei =
    decayable_vector_for<T, operation_category::lane_permutation> &&
    regular_invocable<permute_t, canonical_type_t<T>, I>;

template <typename T, typename I>
concept extended_permutei = unqualified_extended_permutei<T, I> ||
    expression_permutei<T, I> || decayable_permutei<T, I>;

template <typename S, typename M, typename T, typename I,
    typename A = common_abi_t<M, T>>
concept unqualified_canonical_mpermutei = requires(
    S src, M mask, T val, I idx) {
    {
        permute(internal::abi<A>, src, mask, val, __DPL to_index_sequence(idx))
    } -> equivalent_vector_with<canonical_if_zero_t<S, T, A>>;
};

template <typename S, typename M, typename T, typename I,
    typename A = common_abi_t<M, T>>
concept unqualified_extended_mpermutei = requires(S src, M mask, T val, I idx) {
    {
        permute(src, mask, val, __DPL to_index_sequence(idx))
    } -> vector_with_common_abi<A>;
};

template <typename S, typename M, typename T, typename I>
concept expression_mpermutei =
    (simd_expression<S> || simd_expression<M> || simd_expression<T>) &&
    regular_invocable<permute_t, expression_result_or_zero_t<S>,
        simd_expression_result_t<M>, simd_expression_result_t<T>, I>;

template <typename S, typename M, typename T, typename I,
    typename A = common_abi_t<T, M>>
concept decayable_mpermutei = decayable_vector_for<canonical_if_zero_t<S, T, A>,
                                  operation_category::lane_permutation> &&
    decayable_vector_for<T, operation_category::lane_permutation> &&
    decayable_mask_for<M, operation_category::lane_permutation> &&
    regular_invocable<permute_t, canonical_or_zero_t<S, T, A>,
        canonical_type_t<M>, canonical_type_t<T>, I>;

template <typename S, typename M, typename T, typename I,
    typename A = common_abi_t<T, M>>
concept extended_mpermutei = unqualified_extended_mpermutei<S, M, T, I, A> ||
    expression_mpermutei<S, M, T, I> || decayable_mpermutei<S, M, T, I, A>;

template <typename S, typename M, typename T, typename I,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept unqualified_canonical_impermutei =
    requires(S src, M mask, T val, I idx) {
        {
            permute(internal::abi<A>, src,
                internal::to_const_mask<A, permute_t, S, T>(mask), val,
                __DPL to_index_sequence(idx))
        } -> equivalent_vector_with<canonical_if_zero_t<S, T, A>>;
    };

template <typename S, typename M, typename T, typename I,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept unqualified_extended_impermutei =
    requires(S src, M mask, T val, I idx) {
        {
            permute(src, internal::to_const_mask<A, permute_t, S, T>(mask), val,
                __DPL to_index_sequence(idx))
        } -> vector_with_common_abi<A>;
    };

template <typename S, typename M, typename T, typename I>
concept expression_impermutei = (simd_expression<S> || simd_expression<T>) &&
    regular_invocable<permute_t, expression_result_or_zero_t<S>, M,
        simd_expression_result_t<T>, I>;

template <typename S, typename M, typename T, typename I,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept decayable_impermutei = decayable_vector_for<canonical_if_zero_t<S, T>,
                                   operation_category::lane_permutation> &&
    decayable_vector_for<T, operation_category::lane_permutation> &&
    regular_invocable<permute_t, canonical_or_zero_t<S, T, A>, M,
        canonical_type_t<T>, I>;

template <typename S, typename M, typename T, typename I,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept extended_impermutei = unqualified_extended_impermutei<S, M, T, I, A> ||
    expression_impermutei<S, M, T, I> || decayable_impermutei<S, M, T, I, A>;

///

template <typename T, typename I, typename A = common_abi_t<T, I>>
concept unqualified_canonical_permute = requires(T val, I idx) {
    { permute(internal::abi<A>, val, idx) } -> equivalent_vector_with<T>;
};

template <typename T, typename I, typename A = common_abi_t<T, I>>
concept unqualified_extended_permute = requires(T val, I idx) {
    { permute(val, idx) } -> vector_with_common_abi<A>;
};

template <typename T, typename I>
concept expression_permute =
    simd_expression<T> && invocable<permute_t, simd_expression_result_t<T>, I>;

template <typename T, typename I>
concept decayable_permute =
    decayable_vector_for<T, operation_category::lane_permutation> &&
    regular_invocable<permute_t, canonical_type_t<T>, I>;

template <typename T, typename I, typename A = common_abi_t<T, I>>
concept extended_permute = unqualified_extended_permute<T, I, A> ||
    expression_permute<T, I> || decayable_permute<T, I>;

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<L, R, M>>
concept unqualified_canonical_mpermute = requires(S src, M mask, L lhs, R rhs) {
    {
        permute(internal::abi<A>, src, mask, lhs, rhs)
    } -> equivalent_vector_with<
        canonical_if_zero_t<S, operation_result_t<permute_t, L, R>, A>>;
};

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<L, R, M>>
concept unqualified_extended_mpermute = requires(S src, M mask, L lhs, R rhs) {
    { permute(src, mask, lhs, rhs) } -> vector_with_common_abi<A>;
};

template <typename S, typename M, typename L, typename R>
concept expression_mpermute =
    (simd_expression<S> || simd_expression<M> || simd_expression<L>) &&
    invocable<permute_t, expression_result_or_zero_t<S>,
        simd_expression_result_t<M>, simd_expression_result_t<L>, R>;

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<L, R, M>>
concept decayable_mpermute =
    decayable_vector_for<
        canonical_if_zero_t<S, operation_result_t<permute_t, L, R>, A>,
        operation_category::lane_permutation> &&
    decayable_mask_for<M, operation_category::lane_permutation> &&
    decayable_vector_for<L, operation_category::lane_permutation> &&
    regular_invocable<permute_t,
        canonical_or_zero_t<S, operation_result_t<permute_t, L, R>, A>,
        canonical_type_t<M>, canonical_type_t<L>, R>;

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<L, R, M>>
concept extended_mpermute = unqualified_extended_mpermute<S, M, L, R, A> ||
    expression_mpermute<S, M, L, R> || decayable_mpermute<S, M, L, R, A>;

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<
        canonical_if_zero_t<S, operation_result_t<permute_t, L, R>>,
        operation_result_t<permute_t, L, R>>>
concept unqualified_canonical_impermute =
    requires(S src, M mask, L lhs, R rhs) {
        {
            permute(internal::abi<A>, src,
                internal::to_const_mask<A, permute_t, S, L, R>(mask), lhs, rhs)
        } -> equivalent_vector_with<
            canonical_if_zero_t<S, operation_result_t<permute_t, L, R>, A>>;
    };

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<
        canonical_if_zero_t<S, operation_result_t<permute_t, L, R>>,
        operation_result_t<permute_t, L, R>>>
concept unqualified_extended_impermute = requires(S src, M mask, L lhs, R rhs) {
    {
        permute(
            src, internal::to_const_mask<A, permute_t, S, L, R>(mask), lhs, rhs)
    } -> vector_with_common_abi<A>;
};

template <typename S, typename M, typename L, typename R>
concept expression_impermute =
    (simd_expression<S> || simd_expression<M> || simd_expression<L>) &&
    regular_invocable<permute_t, expression_result_or_zero_t<S>, M,
        simd_expression_result_t<L>, R>;

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<
        canonical_if_zero_t<S, operation_result_t<permute_t, L, R>>,
        operation_result_t<permute_t, L, R>>>
concept decayable_impermute =
    decayable_vector_for<
        canonical_if_zero_t<S, operation_result_t<permute_t, L, R>, A>,
        operation_category::lane_permutation> &&
    decayable_vector_for<L, operation_category::lane_permutation> &&
    decayable_vector_for<R, operation_category::lane_permutation> &&
    regular_invocable<permute_t,
        canonical_or_zero_t<S, operation_result_t<permute_t, L, R>, A>, M,
        canonical_type_t<L>, R>;

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<
        canonical_if_zero_t<S, operation_result_t<permute_t, L, R>>,
        operation_result_t<permute_t, L, R>>>
concept extended_impermute = unqualified_extended_impermute<S, M, L, R, A> ||
    expression_impermute<S, M, L, R> || decayable_impermute<S, M, L, R, A>;

struct permute_t {
private:
    template <canonical_simd_type T, common_size_with<simd_element_type_t<T>> E,
        same_as<simd_abi_type_t<T>> A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL fallback(
        T arg, basic_vector<E, A> idx) noexcept {
        return []<size_t... Is>(
                   T arg, basic_vector<E, A> idx, index_sequence<Is...>) {
            constexpr auto simd_size = simd_abi_traits<T>::size();
            using TE = simd_element_type_t<T>;
            auto const zero = TE();
            return dx::initialize<T>(
                (idx[imm<Is>] < simd_size ? arg[idx[imm<Is>]] : zero)...);
        }(arg, idx, iota_sequence<T>);
    }

    template <canonical_simd_type T, index_sequence_like I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL fallbacki(T arg, I idx) noexcept {
        static_assert(I::size() <= simd_abi_traits<T>::size());
        if constexpr (I::size() < simd_abi_traits<T>::size()) {
            constexpr auto D = simd_abi_traits<T>::size() - I::size();
            constexpr auto seq = []<size_t... Is, size_t... Js>(
                                     index_sequence<Is...>,
                                     index_sequence<Js...>) {
                return index_sequence<Is..., (sizeof...(Is) + Js)...>{};
            }(idx, make_index_sequence<D>{});
            return operator()(arg, seq);
        } else {
            return __DPL apply(
                [&](auto... i) {
                    constexpr auto simd_size = simd_abi_traits<T>::size();
                    using TE = simd_element_type_t<T>;
                    auto const zero = TE();
                    return dx::initialize<T>(
                        (i < simd_size ? arg[i] : zero)...);
                },
                idx);
        }
    }

public:
    template <fixed_width_abi A, simd_element_for<A> E, index_sequence_like I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(basic_vector<E, A> val, I idx) noexcept {
        constexpr auto seq = __DPL to_index_sequence(idx);
        if constexpr (unqualified_canonical_permutei<basic_vector<E, A>, I>) {
            if consteval {
                return fallback(val, seq);
            } else {
                return permute(internal::abi<A>, val, seq);
            }
        } else {
            return fallback(val, seq);
        }
    }

    template <scalable_abi A, simd_element_for<A> E, index_sequence_like I>
    requires unqualified_canonical_permutei<basic_vector<E, A>, I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(basic_vector<E, A> val, I idx) noexcept {
        constexpr auto seq = __DPL to_index_sequence(idx);
        return permute(internal::abi<A>, val, seq);
    }

    template <extended_vector T, index_sequence_like I>
    requires extended_permutei<T, I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val, I idx) {
        if constexpr (unqualified_extended_permutei<T, I>) {
            constexpr auto seq = __DPL to_index_sequence(idx);
            return permute(val, seq);
        } else if constexpr (expression_permutei<T, I>) {
            return operator()(dx::evaluate(val), idx);
        } else {
            return operator()(dx::to_canonical(val), idx);
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E, common_size_with<E> ME,
        index_sequence_like I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_vector<E, A> src,
        basic_mask<ME, A> mask, basic_vector<E, A> lhs, I rhs) noexcept {
        if constexpr (unqualified_canonical_mpermutei<basic_vector<E, A>,
                          basic_mask<ME, A>, basic_vector<E, A>, I>) {
            if consteval {
                return internal::masked<permute_t>(src, mask, lhs, rhs);
            } else {
                constexpr auto seq = __DPL to_index_sequence(rhs);
                return permute(internal::abi<A>, src, mask, lhs, seq);
            }
        } else {
            return internal::masked<permute_t>(src, mask, lhs, rhs);
        }
    }

    template <simd_abi SA, simd_element_for<SA> LE, common_size_with<LE> ME,
        simd_abi LA, index_sequence_like I>
    requires (different_from<SA, LA> || scalable_abi<SA> || scalable_abi<LA>) &&
        simd_element_for<LE, LA> &&
        maskable_args<basic_vector<LE, SA>, basic_mask<ME, SA>,
            basic_vector<LE, LA>> &&
        unqualified_canonical_mpermutei<basic_vector<LE, SA>,
            basic_mask<ME, SA>, basic_vector<LE, LA>, I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<LE, SA> operator()(basic_vector<LE, SA> src,
        basic_mask<ME, SA> mask, basic_vector<LE, LA> lhs, I rhs) noexcept {
        constexpr auto seq = __DPL to_index_sequence(rhs);
        return permute(internal::abi<SA>, src, mask, lhs, seq);
    }

    template <simd_vector S, simd_mask M, simd_vector L, index_sequence_like I>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<L>) &&
        maskable_args<S, M, L> && extended_mpermutei<S, M, L, I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, L lhs, I rhs) {
        if constexpr (unqualified_extended_mpermutei<S, M, L, I>) {
            constexpr auto seq = __DPL to_index_sequence(rhs);
            return permute(src, mask, lhs, seq);
        } else if constexpr (expression_mpermutei<S, M, L, I>) {
            return operator()(
                dx::evaluate(src), dx::evaluate(mask), dx::evaluate(lhs), rhs);
        } else {
            return operator()(dx::to_canonical(src), dx::to_canonical(mask),
                dx::to_canonical(lhs), rhs);
        }
    }

    template <fixed_width_abi A, simd_element_for<A> LE,
        common_size_with<LE> ME, index_sequence_like I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<LE, A> operator()(
        basic_mask<ME, A> mask, basic_vector<LE, A> lhs, I rhs) noexcept {
        if constexpr (unqualified_canonical_mpermutei<zero_t, basic_mask<ME, A>,
                          basic_vector<LE, A>, I>) {
            if consteval {
                return internal::masked<permute_t>(mask, lhs, rhs);
            } else {
                constexpr auto seq = __DPL to_index_sequence(rhs);
                return permute(internal::abi<A>, dx::zero, mask, lhs, seq);
            }
        } else {
            return internal::masked<permute_t>(mask, lhs, rhs);
        }
    }

    template <simd_abi SA, simd_element_for<SA> E, common_size_with<E> ME,
        simd_abi LA, index_sequence_like I>
    requires (different_from<SA, LA> || scalable_abi<SA> || scalable_abi<LA>) &&
        simd_element_for<E, LA> &&
        zmaskable_args<basic_mask<ME, SA>, basic_vector<E, LA>> &&
        unqualified_canonical_mpermutei<zero_t, basic_mask<ME, SA>,
            basic_vector<E, LA>, I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        basic_mask<ME, SA> mask, basic_vector<E, LA> lhs, I rhs) noexcept {
        constexpr auto seq = __DPL to_index_sequence(rhs);
        return permute(
            internal::abi<common_abi_t<LA, SA>>, dx::zero, mask, lhs, seq);
    }

    template <simd_mask M, simd_vector L, index_sequence_like I>
    requires (extended_mask<M> || extended_vector<L>) && zmaskable_args<M, L> &&
        extended_mpermutei<zero_t, M, L, I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, L lhs, I rhs) {
        if constexpr (unqualified_extended_mpermutei<zero_t, M, L, I>) {
            constexpr auto seq = __DPL to_index_sequence(rhs);
            return permute(dx::zero, mask, lhs, seq);
        } else if constexpr (expression_mpermutei<zero_t, M, L, I>) {
            return operator()(dx::evaluate(mask), dx::evaluate(lhs), rhs);
        } else {
            return operator()(
                dx::to_canonical(mask), dx::to_canonical(lhs), rhs);
        }
    }

    template <simd_mask M, simd_vector L, index_sequence_like I>
    requires invocable<permute_t, M, L, I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t, M mask, L lhs, I rhs) noexcept {
        return operator()(mask, lhs, rhs);
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        const_mask_for<basic_vector<E, A>> M, index_sequence_like I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_vector<E, A> src,
        M mask, basic_vector<E, A> lhs, I rhs) noexcept {
        if constexpr (unqualified_canonical_impermutei<basic_vector<E, A>, M,
                          basic_vector<E, A>, I>) {
            if consteval {
                return internal::masked<permute_t>(src, mask, lhs, rhs);
            } else {
                constexpr auto cmask =
                    dx::to_compatible_const_mask<basic_vector<E, A>>(mask);
                constexpr auto seq = __DPL to_index_sequence(rhs);
                return permute(internal::abi<A>, src, cmask, lhs, seq);
            }
        } else {
            return internal::masked<permute_t>(src, mask, lhs, rhs);
        }
    }

    template <fixed_width_abi SA, simd_element_for<SA> E,
        const_mask_for<basic_vector<E, SA>> M, fixed_width_abi LA,
        index_sequence_like I>
    requires different_from<SA, LA> && simd_element_for<E, LA> &&
        imm_maskable_args<basic_vector<E, SA>, basic_vector<E, LA>> &&
        unqualified_canonical_impermutei<basic_vector<E, SA>, M,
            basic_vector<E, LA>, I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, SA> operator()(basic_vector<E, SA> src,
        M mask, basic_vector<E, LA> lhs, I rhs) noexcept {
        constexpr auto cmask =
            dx::to_compatible_const_mask<basic_vector<E, SA>>(mask);
        constexpr auto seq = __DPL to_index_sequence(rhs);
        return permute(internal::abi<SA>, src, cmask, lhs, seq);
    }

    template <simd_vector S, const_mask_for<S> M, simd_vector L,
        index_sequence_like I>
    requires (extended_vector<S> || extended_vector<L>) &&
        integral<simd_element_type_t<I>> && imm_maskable_args<S, L> &&
        extended_impermutei<S, M, L, I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, L lhs, I rhs) {
        if constexpr (unqualified_extended_impermutei<S, M, L, I>) {
            constexpr auto seq = __DPL to_index_sequence(rhs);
            return permute(
                src, dx::to_compatible_const_mask<S>(mask), lhs, seq);
        } else if constexpr (expression_impermutei<S, M, L, I>) {
            return operator()(dx::evaluate(src), mask, dx::evaluate(lhs), rhs);
        } else {
            return operator()(
                dx::to_canonical(src), mask, dx::to_canonical(lhs), rhs);
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        const_mask_for<basic_vector<E, A>> M, index_sequence_like I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        M mask, basic_vector<E, A> lhs, I rhs) noexcept {
        if constexpr (unqualified_canonical_impermutei<zero_t, M,
                          basic_vector<E, A>, I>) {
            if consteval {
                return internal::masked<permute_t>(mask, lhs, rhs);
            } else {
                constexpr auto cmask =
                    dx::to_compatible_const_mask<basic_vector<E, A>>(mask);
                constexpr auto seq = __DPL to_index_sequence(rhs);
                return permute(internal::abi<A>, dx::zero, cmask, lhs, seq);
            }
        } else {
            return internal::masked<permute_t>(mask, lhs, rhs);
        }
    }

    template <extended_vector L, const_mask_for<L> M, index_sequence_like I>
    requires extended_impermutei<zero_t, M, L, I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, L lhs, I rhs) {
        if constexpr (unqualified_extended_impermutei<zero_t, M, L, I>) {
            constexpr auto cmask = dx::to_compatible_const_mask<L>(mask);
            constexpr auto seq = __DPL to_index_sequence(rhs);
            return permute(dx::zero, cmask, lhs, seq);
        } else if constexpr (expression_impermutei<zero_t, M, L, I>) {
            return operator()(mask, dx::evaluate(lhs), rhs);
        } else {
            return operator()(mask, dx::to_canonical(lhs), rhs);
        }
    }

    template <simd_vector L, index_sequence_like I, const_mask_for<L> M>
    requires invocable<permute_t, M, L, I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t, M mask, L lhs, I rhs) noexcept {
        return operator()(mask, lhs, rhs);
    }

    ///

    template <fixed_width_abi A, simd_element_for<A> E, simd_element_for<A> I>
    requires common_size_with<E, I> && integral<I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        basic_vector<E, A> val, basic_vector<I, A> idx) noexcept {
        if constexpr (unqualified_canonical_permute<basic_vector<E, A>,
                          basic_vector<I, A>>) {
            if consteval {
                return fallback(val, idx);
            } else {
                return permute(internal::abi<A>, val, idx);
            }
        } else {
            return fallback(val, idx);
        }
    }

    template <simd_abi LA, simd_abi RA, simd_element_for<LA> E,
        simd_element_for<RA> I>
    requires common_size_with<E, I> && integral<I> &&
        (different_from<LA, RA> || scalable_abi<LA> || scalable_abi<RA>) &&
        unqualified_canonical_permute<basic_vector<E, LA>, basic_vector<I, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        basic_vector<E, LA> val, basic_vector<I, RA> idx) noexcept {
        return permute(internal::abi<common_abi_t<LA, RA>>, val, idx);
    }

    template <extended_vector T, canonical_vector I>
    requires integral<simd_element_type_t<I>> && extended_permute<T, I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val, I idx) {
        if constexpr (unqualified_extended_permute<T, I>) {
            return permute(val, idx);
        } else if constexpr (expression_permute<T, I>) {
            return operator()(dx::evaluate(val), idx);
        } else {
            return operator()(dx::to_canonical(val), idx);
        }
    }

    template <fixed_width_abi A, simd_element_for<A> LE, simd_element_for<A> RE,
        common_size_with<LE> ME>
    requires integral<RE> && common_size_with<LE, RE>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<LE, A> operator()(basic_vector<LE, A> src,
        basic_mask<ME, A> mask, basic_vector<LE, A> lhs,
        basic_vector<RE, A> rhs) noexcept {
        if constexpr (unqualified_canonical_mpermute<basic_vector<LE, A>,
                          basic_mask<ME, A>, basic_vector<LE, A>,
                          basic_vector<RE, A>>) {
            if consteval {
                return internal::masked<permute_t>(src, mask, lhs, rhs);
            } else {
                return permute(internal::abi<A>, src, mask, lhs, rhs);
            }
        } else {
            return internal::masked<permute_t>(src, mask, lhs, rhs);
        }
    }

    template <simd_abi SA, simd_element_for<SA> LE, common_size_with<LE> ME,
        simd_abi LA, common_abi_with<LA> RA, simd_element_for<RA> RE>
    requires integral<RE> &&
        (common_size_with<LE, RE> || different_from<LA, RA> ||
            scalable_abi<SA> || scalable_abi<LA> || scalable_abi<RA>) &&
        simd_element_for<LE, LA> &&
        maskable_args<basic_vector<LE, SA>, basic_mask<ME, SA>,
            basic_vector<LE, LA>, basic_vector<RE, RA>> &&
        unqualified_canonical_mpermute<basic_vector<LE, SA>, basic_mask<ME, SA>,
            basic_vector<LE, LA>, basic_vector<RE, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<LE, SA> operator()(basic_vector<LE, SA> src,
        basic_mask<ME, SA> mask, basic_vector<LE, LA> lhs,
        basic_vector<RE, RA> rhs) noexcept {
        return permute(internal::abi<SA>, src, mask, lhs, rhs);
    }

    template <simd_vector S, simd_mask M, simd_vector L, canonical_vector R>
    requires integral<simd_element_type_t<R>> &&
        (extended_vector<S> || extended_mask<M> || extended_vector<L>) &&
        maskable_args<S, M, L, R> && extended_mpermute<S, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, L lhs, R rhs) {
        if constexpr (unqualified_extended_mpermute<S, M, L, R>) {
            return permute(src, mask, lhs, rhs);
        } else if constexpr (expression_mpermute<S, M, L, R>) {
            return operator()(
                dx::evaluate(src), dx::evaluate(mask), dx::evaluate(lhs), rhs);
        } else {
            return operator()(dx::to_canonical(src), dx::to_canonical(mask),
                dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }

    template <fixed_width_abi A, simd_element_for<A> LE, simd_element_for<A> RE,
        common_size_with<LE> ME>
    requires integral<RE> && common_size_with<LE, RE>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<LE, A> operator()(basic_mask<ME, A> mask,
        basic_vector<LE, A> lhs, basic_vector<RE, A> rhs) noexcept {
        if constexpr (unqualified_canonical_mpermute<zero_t, basic_mask<ME, A>,
                          basic_vector<LE, A>, basic_vector<RE, A>>) {
            if consteval {
                return internal::masked<permute_t>(mask, lhs, rhs);
            } else {
                return permute(internal::abi<A>, dx::zero, mask, lhs, rhs);
            }
        } else {
            return operator()(dx::broadcast<LE, A>(dx::zero), mask, lhs, rhs);
        }
    }

    template <simd_abi SA, simd_element_for<SA> E, common_size_with<E> ME,
        simd_abi LA, common_abi_with<LA> RA, simd_element_for<RA> RE>
    requires integral<RE> &&
        (different_from<SA, common_abi_t<LA, RA>> || different_from<LA, RA> ||
            scalable_abi<SA> || scalable_abi<LA> || scalable_abi<RA> ||
            !common_size_with<E, RE>) &&
        simd_element_for<E, LA> &&
        zmaskable_args<basic_mask<ME, SA>, basic_vector<E, LA>,
            basic_vector<RE, RA>> &&
        unqualified_canonical_mpermute<zero_t, basic_mask<ME, SA>,
            basic_vector<E, LA>, basic_vector<RE, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(basic_mask<ME, SA> mask,
        basic_vector<E, LA> lhs, basic_vector<RE, RA> rhs) noexcept {
        return permute(
            internal::abi<common_abi_t<LA, RA>>, dx::zero, mask, lhs, rhs);
    }

    template <simd_mask M, simd_vector L, canonical_vector R>
    requires (extended_mask<M> || extended_vector<L>) &&
        integral<simd_element_type_t<R>> && zmaskable_args<M, L, R> &&
        extended_mpermute<zero_t, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, L lhs, R rhs) {
        if constexpr (unqualified_extended_mpermute<zero_t, M, L, R>) {
            return permute(dx::zero, mask, lhs, rhs);
        } else if constexpr (expression_mpermute<zero_t, M, L, R>) {
            return operator()(dx::evaluate(mask), dx::evaluate(lhs), rhs);
        } else {
            return operator()(
                dx::to_canonical(mask), dx::to_canonical(lhs), rhs);
        }
    }

    template <simd_mask M, simd_vector L, simd_vector R>
    requires invocable<permute_t, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t, M mask, L lhs, R rhs) noexcept {
        return operator()(mask, lhs, rhs);
    }

    template <fixed_width_abi A, simd_element_for<A> E, simd_element_for<A> RE,
        const_mask_for<basic_vector<E, A>> M>
    requires integral<RE> && common_size_with<E, RE>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_vector<E, A> src,
        M mask, basic_vector<E, A> lhs, basic_vector<RE, A> rhs) noexcept {
        if constexpr (unqualified_canonical_impermute<basic_vector<E, A>, M,
                          basic_vector<E, A>, basic_vector<RE, A>>) {
            if consteval {
                return internal::masked<permute_t>(src, mask, lhs, rhs);
            } else {
                return permute(internal::abi<A>, src,
                    dx::to_compatible_const_mask<basic_vector<E, A>>(mask), lhs,
                    rhs);
            }
        } else {
            return internal::masked<permute_t>(src, mask, lhs, rhs);
        }
    }

    template <simd_abi SA, simd_element_for<SA> E,
        const_mask_for<basic_vector<E, SA>> M, simd_abi LA,
        common_abi_with<LA> RA, simd_element_for<RA> RE>
    requires integral<RE> &&
        (different_from<SA, common_abi_t<LA, RA>> || different_from<LA, RA> ||
            scalable_abi<SA> || scalable_abi<LA> || scalable_abi<RA> ||
            !common_size_with<RE, E>) &&
        simd_element_for<E, LA> &&
        imm_maskable_args<basic_vector<E, SA>, basic_vector<E, LA>,
            basic_vector<E, RA>> &&
        unqualified_canonical_impermute<basic_vector<E, SA>, M,
            basic_vector<E, LA>, basic_vector<E, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, SA> operator()(basic_vector<E, SA> src,
        M mask, basic_vector<E, LA> lhs, basic_vector<RE, RA> rhs) noexcept {
        return permute(internal::abi<SA>, src,
            dx::to_compatible_const_mask<basic_vector<E, SA>>(mask), lhs, rhs);
    }

    template <simd_vector S, const_mask_for<S> M, simd_vector L,
        canonical_vector R>
    requires (extended_vector<S> || extended_vector<L>) &&
        integral<simd_element_type_t<R>> && imm_maskable_args<S, L, R> &&
        extended_impermute<S, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, L lhs, R rhs) {
        if constexpr (unqualified_extended_impermute<S, M, L, R>) {
            return permute(
                src, dx::to_compatible_const_mask<S>(mask), lhs, rhs);
        } else if constexpr (expression_impermute<S, M, L, R>) {
            return operator()(
                dx::evaluate(src), mask, dx::evaluate(lhs), dx::evaluate(rhs));
        } else {
            return operator()(dx::to_canonical(src), mask,
                dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E, simd_element_for<A> RE,
        const_mask_for<basic_vector<E, A>> M>
    requires integral<RE> && common_size_with<RE, E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        M mask, basic_vector<E, A> lhs, basic_vector<RE, A> rhs) noexcept {
        if constexpr (unqualified_canonical_impermute<zero_t, M,
                          basic_vector<E, A>, basic_vector<RE, A>>) {
            if consteval {
                return internal::masked<permute_t>(mask, lhs, rhs);
            } else {
                return permute(internal::abi<A>, dx::zero,
                    dx::to_compatible_const_mask<basic_vector<E, A>>(mask), lhs,
                    rhs);
            }
        } else {
            return internal::masked<permute_t>(mask, lhs, rhs);
        }
    }

    template <simd_abi LA, common_abi_with<LA> RA,
        simd_element_for<common_abi_t<LA, RA>> E, simd_element_for<RA> RE,
        const_mask_for<basic_vector<E, common_abi_t<LA, RA>>> M>
    requires integral<RE> &&
        (different_from<LA, RA> || scalable_abi<LA> || scalable_abi<RA> ||
            !common_size_with<RE, E>) &&
        simd_element_for<E, LA> &&
        imm_zmaskable_args<basic_vector<E, LA>, basic_vector<RE, RA>> &&
        unqualified_canonical_impermute<zero_t, M, basic_vector<E, LA>,
            basic_vector<RE, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, common_abi_t<LA, RA>> operator()(
        M mask, basic_vector<E, LA> lhs, basic_vector<RE, RA> rhs) noexcept {
        using A = common_abi_t<LA, RA>;
        return permute(internal::abi<A>, dx::zero,
            dx::to_compatible_const_mask<basic_vector<E, A>>(mask), lhs, rhs);
    }

    template <extended_vector L, canonical_vector R,
        const_mask_for<operation_result_t<permute_t, L, R>> M>
    requires imm_zmaskable_args<L, R> && extended_impermute<zero_t, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, L lhs, R rhs) {
        using S = operation_result_t<permute_t, L, R>;
        if constexpr (unqualified_extended_impermute<zero_t, M, L, R>) {
            return permute(
                dx::zero, dx::to_compatible_const_mask<S>(mask), lhs, rhs);
        } else if constexpr (expression_impermute<zero_t, M, L, R>) {
            return operator()(mask, dx::evaluate(lhs), dx::evaluate(rhs));
        } else {
            return operator()(
                mask, dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }

    template <simd_vector L, simd_vector R,
        const_mask_for<operation_result_t<permute_t, L, R>> M>
    requires invocable<permute_t, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t, M mask, L lhs, R rhs) noexcept {
        return operator()(mask, lhs, rhs);
    }
};

template <size_t... Is>
struct permutei_t {
private:
    using seq_t DPL_NODEBUG = index_sequence<Is...>;
    static constexpr seq_t seq{};

public:
    template <simd_vector T>
    requires invocable<permute_t, T, seq_t>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val) noexcept(canonical_vector<T>) {
        return permute_t::operator()(val, seq);
    }

    template <typename S, typename M, simd_vector T>
    requires invocable<permute_t, S, M, T, seq_t>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, T val) noexcept(
        (!simd_mask<S> || canonical_mask<S>) &&
        (!simd_mask<M> || canonical_mask<M>) && canonical_vector<T>) {
        return permute_t::operator()(src, mask, val, seq);
    }

    template <typename M, simd_vector T>
    requires invocable<permute_t, M, T, seq_t>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val) noexcept(
        (!simd_mask<M> || canonical_mask<M>) && canonical_vector<T>) {
        return permute_t::operator()(mask, val, seq);
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::permute_t permute{};
DPL_EXPORT template <size_t... Is>
inline constexpr internal::permutei_t<Is...> permutei{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
