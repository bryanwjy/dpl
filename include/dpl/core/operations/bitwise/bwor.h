// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

// IWYU pragma: always_keep

#include "dpl/core/operations/bitwise/result.h"
#include "dpl/core/operations/masked.h"
#include "dpl/core/operations/operation_base.h"
#include "dpl/core/operations/transform.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/decayable.h"
#  include "dpl/core/concepts/operation_category.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_equivalence.h"
#  include "dpl/core/type_traits/common_bits_type.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void bwor(...) noexcept = delete;

struct bwor_t;

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_canonical_bwor = requires(L lhs, R rhs) {
    { bwor(internal::abi<A>, lhs, rhs) } -> canonical_bitwise_result<L, R, A>;
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_extended_bwor = requires(L lhs, R rhs) {
    { bwor(lhs, rhs) } -> extended_operation_vector<A>;
};

template <typename L, typename R>
concept expression_bwor = (simd_expression<L> || simd_expression<R>) &&
    invocable<abs_t, simd_expression_result_t<L>, simd_expression_result_t<R>>;

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept decayable_bwor =
    (decayable_vector_for<L, operation_category::lane_agnostic> &&
        decayable_vector_for<R, operation_category::lane_agnostic> &&
        regular_invocable<bwor_t, canonical_type_t<L>, canonical_type_t<R>>);

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept extended_bwor = unqualified_extended_bwor<L, R, A> ||
    expression_bwor<L, R> || decayable_bwor<L, R, A>;

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<L, R, M>>
concept unqualified_canonical_mbwor = requires(S src, M mask, L lhs, R rhs) {
    {
        bwor(internal::abi<A>, src, mask, lhs, rhs)
    } -> equivalent_simd_as<
        canonical_if_zero_t<S, operation_result_t<bwor_t, L, R>, A>>;
};

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<L, R, M>>
concept unqualified_extended_mbwor = requires(S src, M mask, L lhs, R rhs) {
    { bwor(src, mask, lhs, rhs) } -> extended_operation_vector<A>;
};

template <typename S, typename M, typename L, typename R>
concept expression_mbwor = (simd_expression<S> || simd_expression<M> ||
                               simd_expression<L> || simd_expression<R>) &&
    invocable<bwor_t, expression_result_or_zero_t<S>,
        simd_expression_result_t<M>, simd_expression_result_t<L>,
        simd_expression_result_t<R>>;

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<L, R, M>>
concept decayable_mbwor =
    decayable_vector_for<
        canonical_if_zero_t<S, operation_result_t<bwor_t, L, R>, A>,
        operation_category::lane_agnostic> &&
    decayable_mask_for<M, operation_category::lane_agnostic> &&
    decayable_vector_for<L, operation_category::lane_agnostic> &&
    decayable_vector_for<R, operation_category::lane_agnostic> &&
    regular_invocable<bwor_t,
        canonical_if_zero_t<S, operation_result_t<bwor_t, L, R>, A>,
        canonical_type_t<M>, canonical_type_t<L>, canonical_type_t<R>>;

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<L, R, M>>
concept extended_mbwor = unqualified_extended_mbwor<S, M, L, R, A> ||
    expression_mbwor<S, M, L, R> || decayable_mbwor<S, M, L, R, A>;

template <typename S, typename M, typename L, typename R,
    typename A =
        common_abi_t<canonical_if_zero_t<S, operation_result_t<bwor_t, L, R>>,
            operation_result_t<bwor_t, L, R>>>
concept unqualified_canonical_imbwor = requires(S src, M mask, L lhs, R rhs) {
    {
        bwor(internal::abi<A>, src,
            internal::to_const_mask<A, bwor_t, S, L, R>(mask), lhs, rhs)
    } -> equivalent_simd_as<
        canonical_if_zero_t<S, operation_result_t<bwor_t, L, R>, A>>;
};

template <typename S, typename M, typename L, typename R,
    typename A =
        common_abi_t<canonical_if_zero_t<S, operation_result_t<bwor_t, L, R>>,
            operation_result_t<bwor_t, L, R>>>
concept unqualified_extended_imbwor = requires(S src, M mask, L lhs, R rhs) {
    {
        bwor(src, internal::to_const_mask<A, bwor_t, S, L, R>(mask), lhs, rhs)
    } -> extended_operation_vector<A>;
};

template <typename S, typename M, typename L, typename R>
concept expression_imbwor =
    (simd_expression<S> || simd_expression<L> || simd_expression<R>) &&
    invocable<bwor_t, expression_result_or_zero_t<S>, M,
        simd_expression_result_t<L>, simd_expression_result_t<R>>;

template <typename S, typename M, typename L, typename R,
    typename A =
        common_abi_t<canonical_if_zero_t<S, operation_result_t<bwor_t, L, R>>,
            operation_result_t<bwor_t, L, R>>>
concept decayable_imbwor =
    decayable_vector_for<
        canonical_if_zero_t<S, operation_result_t<bwor_t, L, R>, A>,
        operation_category::lane_agnostic> &&
    decayable_vector_for<L, operation_category::lane_agnostic> &&
    decayable_vector_for<R, operation_category::lane_agnostic> &&
    requires(bwor_t op,
        canonical_if_zero_t<S, operation_result_t<bwor_t, L, R>, A> s, M mask,
        canonical_type_t<L> l, canonical_type_t<R> r) { op(s, mask, l, r); };

template <typename S, typename M, typename L, typename R,
    typename A =
        common_abi_t<canonical_if_zero_t<S, operation_result_t<bwor_t, L, R>>,
            operation_result_t<bwor_t, L, R>>>
concept extended_imbwor = unqualified_extended_imbwor<S, M, L, R, A> ||
    expression_imbwor<S, M, L, R> || decayable_imbwor<S, M, L, R, A>;

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_canonical_mask_bwor = requires(L lhs, R rhs) {
    { bwor(internal::abi<A>, lhs, rhs) } -> canonical_bitwise_mask<L, R, A>;
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_extended_mask_bwor = requires(L lhs, R rhs) {
    { bwor(lhs, rhs) } -> extended_operation_vector<A>;
};

template <typename L, typename R>
concept expression_mask_bwor = (mask_expression<L> || mask_expression<R>) &&
    invocable<bwor_t, simd_expression_result_t<L>, simd_expression_result_t<R>>;

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept decayable_mask_bwor =
    decayable_mask_for<L, operation_category::lane_agnostic> &&
    decayable_mask_for<R, operation_category::lane_agnostic> &&
    regular_invocable<bwor_t, canonical_type_t<L>, canonical_type_t<R>>;

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept extended_mask_bwor = unqualified_extended_mask_bwor<L, R, A> ||
    decayable_mask_bwor<L, R> || decayable_mask_bwor<L, R, A>;

struct bwor_t : private binary_operation_base<bwor_t> {
private:
    friend binary_operation_base<bwor_t>;

    template <simd_abi A, typename L, typename R>
    requires (!simd_class<L> || canonical_vector<L>) &&
        (!simd_class<R> || canonical_vector<R>) && requires(L lhs, R rhs) {
            { bwor(internal::abi<A>, lhs, rhs) } -> vector_with_abi<A>;
        }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, L lhs, R rhs) noexcept {
        return bwor(internal::abi<A>, lhs, rhs);
    }

    template <simd_abi A, typename L, typename R>
    requires (!simd_class<L> || extended_vector<L>) &&
        (!simd_class<R> || extended_vector<R>) &&
        unqualified_extended_bwor<L, R, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A, L lhs, R rhs) noexcept(
        noexcept(bwor(lhs, rhs))) {
        return bwor(lhs, rhs);
    }

    template <simd_abi A, typename L, typename R>
    requires (!simd_class<L> || canonical_mask<L>) &&
        (!simd_class<R> || canonical_mask<R>) && requires(L lhs, R rhs) {
            { bwor(internal::abi<A>, lhs, rhs) } -> mask_with_abi<A>;
        }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, L lhs, R rhs) noexcept {
        return bwor(internal::abi<A>, lhs, rhs);
    }

    template <simd_abi A, typename L, typename R>
    requires (!simd_class<L> || extended_mask<L>) &&
        (!simd_class<R> || extended_mask<R>) &&
        unqualified_extended_mask_bwor<L, R, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A, L lhs, R rhs) noexcept(
        noexcept(bwor(lhs, rhs))) {
        return bwor(lhs, rhs);
    }

    template <typename L, typename R, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_vector<L, A> lhs, basic_vector<R, A> rhs) noexcept {
        using T = common_bits_type_t<L, R>;
        return internal::transform<basic_vector<T, A>>(
            [](auto lhs, auto rhs) {
                using bit_type = bitset<sizeof(T) * char_bit_v>;
                auto promoted = __DPL bit_cast<bit_type>(lhs) |
                    __DPL bit_cast<bit_type>(rhs);
                return __DPL bit_cast<T>(promoted.reinitialize());
            },
            lhs, rhs);
    }

    template <typename LE, typename RE, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_mask<LE, A> lhs, basic_mask<RE, A> rhs) noexcept {
        return internal::transform<basic_mask<common_size_type_t<LE, RE>, A>>(
            [](auto lhs, auto rhs) { return lhs || rhs; }, lhs, rhs);
    }

public:
    template <fixed_width_abi A, simd_element_for<A> E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> lhs, basic_vector<E, A> rhs) noexcept {
        if constexpr (unqualified_canonical_bwor<basic_vector<E, A>,
                          basic_vector<E, A>>) {
            if consteval {
                return fallback(lhs, rhs);
            } else {
                return bwor(internal::abi<A>, lhs, rhs);
            }
        } else {
            return fallback(lhs, rhs);
        }
    }

    template <simd_abi LA, common_abi_with<LA> RA, typename E>
    requires simd_element_for<E, LA> && simd_element_for<E, RA> &&
        (scalable_abi<LA> || scalable_abi<RA> || different_from<LA, RA>) &&
        unqualified_canonical_bwor<basic_vector<E, LA>, basic_vector<E, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, common_abi_t<LA, RA>> operator()(
        basic_vector<E, LA> lhs, basic_vector<E, RA> rhs) noexcept {
        return bwor(internal::abi<common_abi_t<LA, RA>>, lhs, rhs);
    }

    template <simd_vector L, simd_vector R>
    requires (extended_vector<L> || extended_vector<R>) && extended_bwor<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        if constexpr (unqualified_extended_bwor<L, R>) {
            return bwor(lhs, rhs);
        } else if constexpr (expression_bwor<L, R>) {
            return operator()(dx::evaluate(lhs), dx::evaluate(rhs));
        } else {
            return operator()(dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }

    using binary_operation_base<bwor_t>::operator();

    template <fixed_width_abi A, simd_element_for<A> E, common_size_with<E> ME>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_vector<E, A> src,
        basic_mask<ME, A> mask, basic_vector<E, A> lhs,
        basic_vector<E, A> rhs) noexcept {
        if constexpr (unqualified_canonical_mbwor<basic_vector<E, A>,
                          basic_mask<ME, A>, basic_vector<E, A>,
                          basic_vector<E, A>>) {
            if consteval {
                return internal::masked<bwor_t>(src, mask, lhs, rhs);
            } else {
                return bwor(internal::abi<A>, src, mask, lhs, rhs);
            }
        } else {
            return internal::masked<bwor_t>(src, mask, lhs, rhs);
        }
    }

    template <simd_abi SA, simd_element_for<SA> E, common_size_with<E> ME,
        simd_abi LA, common_abi_with<LA> RA>
    requires (different_from<LA, RA> || scalable_abi<SA> || scalable_abi<LA> ||
                 scalable_abi<RA>) &&
        simd_element_for<E, LA> && simd_element_for<E, RA> &&
        maskable_args<basic_vector<E, SA>, basic_mask<ME, SA>,
            basic_vector<E, LA>, basic_vector<E, RA>> &&
        unqualified_canonical_mbwor<basic_vector<E, SA>, basic_mask<ME, SA>,
            basic_vector<E, LA>, basic_vector<E, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, SA> operator()(basic_vector<E, SA> src,
        basic_mask<ME, SA> mask, basic_vector<E, LA> lhs,
        basic_vector<E, RA> rhs) noexcept {
        return bwor(internal::abi<SA>, src, mask, lhs, rhs);
    }

    template <simd_vector S, simd_mask M, simd_vector L, simd_vector R>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<L> ||
                 extended_vector<R>) &&
        maskable_args<S, M, L, R> && extended_mbwor<S, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, L lhs, R rhs) noexcept {
        if constexpr (unqualified_extended_mbwor<S, M, L, R>) {
            return bwor(src, mask, lhs, rhs);
        } else if constexpr (expression_mbwor<S, M, L, R>) {
            return operator()(dx::evaluate(src), dx::evaluate(mask),
                dx::evaluate(lhs), dx::evaluate(rhs));
        } else {
            return operator()(dx::to_canonical(src), dx::to_canonical(mask),
                dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E, common_size_with<E> ME>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_mask<ME, A> mask,
        basic_vector<E, A> lhs, basic_vector<E, A> rhs) noexcept {
        if constexpr (unqualified_canonical_mbwor<zero_t, basic_mask<ME, A>,
                          basic_vector<E, A>, basic_vector<E, A>>) {
            if consteval {
                return internal::masked<bwor_t>(mask, lhs, rhs);
            } else {
                return bwor(internal::abi<A>, dx::zero, mask, lhs, rhs);
            }
        } else {
            return operator()(dx::zero_v<basic_vector<E, A>>, mask, lhs, rhs);
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
        unqualified_canonical_mbwor<zero_t, basic_mask<ME, SA>,
            basic_vector<E, LA>, basic_vector<E, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(basic_mask<ME, SA> mask,
        basic_vector<E, LA> lhs, basic_vector<E, RA> rhs) noexcept {
        return bwor(
            internal::abi<common_abi_t<LA, RA>>, dx::zero, mask, lhs, rhs);
    }

    template <simd_mask M, simd_vector L, simd_vector R>
    requires (extended_mask<M> || extended_vector<L> || extended_vector<R>) &&
        zmaskable_args<M, L, R> && extended_mbwor<zero_t, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, L lhs, R rhs) noexcept {
        if constexpr (unqualified_extended_mbwor<zero_t, M, L, R>) {
            return bwor(dx::zero, mask, lhs, rhs);
        } else if constexpr (expression_mbwor<zero_t, M, L, R>) {
            return operator()(
                dx::evaluate(mask), dx::evaluate(lhs), dx::evaluate(rhs));
        } else {
            return operator()(dx::to_canonical(mask), dx::to_canonical(lhs),
                dx::to_canonical(rhs));
        }
    }

    template <simd_mask M, simd_vector L, simd_vector R>
    requires invocable<bwor_t, M, L, R>
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
        if constexpr (unqualified_canonical_imbwor<basic_vector<E, A>, M,
                          basic_vector<E, A>, basic_vector<E, A>>) {
            if consteval {
                return internal::masked<bwor_t>(src, mask, lhs, rhs);
            } else {
                return bwor(internal::abi<A>, src,
                    dx::to_compatible_const_mask<basic_vector<E, A>>(mask), lhs,
                    rhs);
            }
        } else {
            return internal::masked<bwor_t>(src, mask, lhs, rhs);
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
        unqualified_canonical_imbwor<basic_vector<E, SA>, M,
            basic_vector<E, LA>, basic_vector<E, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, SA> operator()(basic_vector<E, SA> src,
        M mask, basic_vector<E, LA> lhs, basic_vector<E, RA> rhs) noexcept {
        return bwor(internal::abi<SA>, src,
            dx::to_compatible_const_mask<basic_vector<E, SA>>(mask), lhs, rhs);
    }

    template <simd_vector S, const_mask_for<S> M, simd_vector L, simd_vector R>
    requires (extended_vector<S> || extended_vector<L> || extended_vector<R>) &&
        imm_maskable_args<S, L, R> && extended_imbwor<S, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, L lhs, R rhs) noexcept {
        if constexpr (unqualified_extended_imbwor<S, M, L, R>) {
            return bwor(src, dx::to_compatible_const_mask<S>(mask), lhs, rhs);
        } else if constexpr (expression_imbwor<S, M, L, R>) {
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
        if constexpr (unqualified_canonical_imbwor<zero_t, M,
                          basic_vector<E, A>, basic_vector<E, A>>) {
            if consteval {
                return internal::masked<bwor_t>(mask, lhs, rhs);
            } else {
                return bwor(internal::abi<A>, dx::zero,
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
        unqualified_canonical_imbwor<zero_t, M, basic_vector<E, LA>,
            basic_vector<E, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, common_abi_t<LA, RA>> operator()(
        M mask, basic_vector<E, LA> lhs, basic_vector<E, RA> rhs) noexcept {
        using A = common_abi_t<LA, RA>;
        return bwor(internal::abi<A>, dx::zero,
            dx::to_compatible_const_mask<basic_vector<E, A>>(mask), lhs, rhs);
    }

    template <simd_vector L, simd_vector R,
        const_mask_for<operation_result_t<bwor_t, L, R>> M>
    requires (extended_vector<L> || extended_vector<R>) &&
        imm_zmaskable_args<L, R> && extended_imbwor<zero_t, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, L lhs, R rhs) noexcept {
        using S = operation_result_t<bwor_t, L, R>;
        constexpr auto cmask = dx::to_compatible_const_mask<S>(mask);
        if constexpr (unqualified_extended_imbwor<zero_t, M, L, R>) {
            return bwor(dx::zero, cmask, lhs, rhs);
        } else if constexpr (expression_imbwor<zero_t, M, L, R>) {
            return operator()(cmask, dx::evaluate(lhs), dx::evaluate(rhs));
        } else {
            return operator()(
                cmask, dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }

    template <simd_vector L, simd_vector R,
        const_mask_for<operation_result_t<bwor_t, L, R>> M>
    requires invocable<bwor_t, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t, M mask, L lhs, R rhs) noexcept {
        return operator()(mask, lhs, rhs);
    }

    template <fixed_width_abi A, simd_element_for<A> LE, simd_element_for<A> RE>
    requires common_size_with<LE, RE>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<common_size_type_t<LE, RE>, A> operator()(
        basic_mask<LE, A> lhs, basic_mask<RE, A> rhs) noexcept {
        if constexpr (unqualified_canonical_mask_bwor<basic_mask<LE, A>,
                          basic_mask<RE, A>>) {
            if consteval {
                return fallback(lhs, rhs);
            } else {
                return bwor(internal::abi<A>, lhs, rhs);
            }
        } else {
            return fallback(lhs, rhs);
        }
    }

    template <simd_abi LA, common_abi_with<LA> RA, typename LE,
        common_size_with<LE> RE>
    requires simd_element_for<LE, LA> && simd_element_for<RE, RA> &&
        (scalable_abi<LA> || scalable_abi<RA> || different_from<LA, RA>) &&
        unqualified_canonical_mask_bwor<basic_mask<LE, LA>, basic_mask<RE, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<common_size_type_t<LE, RE>,
        common_abi_t<LA, RA>> operator()(basic_mask<LE, LA> lhs,
        basic_mask<RE, RA> rhs) noexcept {
        return bwor(internal::abi<common_abi_t<LA, RA>>, lhs, rhs);
    }

    template <simd_mask L, simd_mask R>
    requires (extended_mask<L> || extended_mask<R>) && extended_mask_bwor<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        if constexpr (unqualified_extended_mask_bwor<L, R>) {
            return bwor(lhs, rhs);
        } else if constexpr (expression_mask_bwor<L, R>) {
            return operator()(dx::evaluate(lhs), dx::evaluate(rhs));
        } else {
            return operator()(dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::bwor_t bwor{};
} // namespace cpo
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
