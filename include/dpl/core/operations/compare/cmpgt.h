// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

// IWYU pragma: always_keep

#include "dpl/core/operations/compare/cmplt.h"
#include "dpl/core/operations/compare/result.h"
#include "dpl/core/operations/internal/masked.h"
#include "dpl/core/operations/internal/operation_base.h"
#if !DPL_MODULES
#  include "dpl/core/basic/internal/abi.h"
#  include "dpl/core/concepts/decayable.h"
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/std/concepts/invocable.h"
#  include "dpl/std/concepts/totally_ordered.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void cmpgt(...) noexcept = delete;

struct cmpgt_t;

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_canonical_cmpgt = requires(L lhs, R rhs) {
    { cmpgt(internal::abi<A>, lhs, rhs) } -> canonical_compare_result<L, R, A>;
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_extended_cmpgt = requires(L lhs, R rhs) {
    { cmpgt(lhs, rhs) } -> mask_with_common_abi<A>;
};

template <typename L, typename R>
concept expression_cmpgt = (simd_expression<L> || simd_expression<R>) &&
    invocable<cmpgt_t, simd_expression_result_t<L>,
        simd_expression_result_t<R>>;

template <typename L, typename R>
concept decayable_cmpgt =
    decayable_vector_for<L, operation_category::lane_agnostic> &&
    decayable_vector_for<R, operation_category::lane_agnostic> &&
    regular_invocable<cmpgt_t, canonical_type_t<L>, canonical_type_t<R>>;

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept extended_cmpgt = unqualified_extended_cmpgt<L, R> ||
    expression_cmpgt<L, R> || decayable_cmpgt<L, R>;

template <typename M, typename L, typename R,
    typename A = common_abi_t<L, R, M>>
concept unqualified_canonical_mcmpgt = requires(M mask, L lhs, R rhs) {
    {
        cmpgt(internal::abi<A>, mask, lhs, rhs)
    } -> canonical_compare_result<L, R>;
};

template <typename M, typename L, typename R,
    typename A = common_abi_t<L, R, M>>
concept unqualified_extended_mcmpgt = requires(M mask, L lhs, R rhs) {
    { cmpgt(mask, lhs, rhs) } -> mask_with_common_abi<A>;
};

template <typename M, typename L, typename R>
concept expression_mcmpgt =
    (simd_expression<M> || simd_expression<L> || simd_expression<R>) &&
    invocable<cmpgt_t, simd_expression_result_t<M>, simd_expression_result_t<L>,
        simd_expression_result_t<R>>;

template <typename M, typename L, typename R,
    typename A = common_abi_t<L, R, M>>
concept decayable_mcmpgt =
    decayable_mask_for<M, operation_category::lane_agnostic> &&
    decayable_vector_for<L, operation_category::lane_agnostic> &&
    decayable_vector_for<R, operation_category::lane_agnostic> &&
    regular_invocable<cmpgt_t, canonical_type_t<M>, canonical_type_t<L>,
        canonical_type_t<R>>;

template <typename M, typename L, typename R,
    typename A = common_abi_t<L, R, M>>
concept extended_mcmpgt = unqualified_extended_mcmpgt<M, L, R, A> ||
    expression_mcmpgt<M, L, R> || decayable_mcmpgt<M, L, R>;

template <typename M, typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_canonical_imcmpgt = requires(M mask, L lhs, R rhs) {
    {
        cmpgt(internal::abi<A>,
            dx::to_compatible_const_mask<operation_result_t<cmpgt_t, L, R>>(
                mask),
            lhs, rhs)
    } -> canonical_compare_result<L, R>;
};

template <typename M, typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_extended_imcmpgt = requires(M mask, L lhs, R rhs) {
    {
        cmpgt(dx::to_compatible_const_mask<operation_result_t<cmpgt_t, L, R>>(
                  mask),
            lhs, rhs)
    } -> mask_with_common_abi<A>;
};

template <typename M, typename L, typename R>
concept expression_imcmpgt = (simd_expression<L> || simd_expression<R>) &&
    invocable<cmpgt_t, M, simd_expression_result_t<L>,
        simd_expression_result_t<R>>;

template <typename M, typename L, typename R>
concept decayable_imcmpgt =
    decayable_vector_for<L, operation_category::lane_agnostic> &&
    decayable_vector_for<R, operation_category::lane_agnostic> &&
    regular_invocable<cmpgt_t, M, canonical_type_t<L>, canonical_type_t<R>>;

template <typename M, typename L, typename R, typename A = common_abi_t<L, R>>
concept extended_imcmpgt = unqualified_extended_imcmpgt<M, L, R, A> ||
    expression_imcmpgt<M, L, R> || decayable_imcmpgt<M, L, R>;

struct cmpgt_t : binary_operation_base<cmpgt_t> {
private:
    friend binary_operation_base<cmpgt_t>;

    template <simd_abi A, typename L, typename R>
    requires (!simd_type<L> || canonical_vector<L>) &&
        (!simd_type<R> || canonical_vector<R>) && requires(L lhs, R rhs) {
            { cmpgt(internal::abi<A>, lhs, rhs) } -> mask_with_abi<A>;
        }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, L lhs, R rhs) noexcept {
        return cmpgt(internal::abi<A>, lhs, rhs);
    }

    template <simd_abi A, typename L, typename R>
    requires (!simd_type<L> || extended_vector<L>) &&
        (!simd_type<R> || extended_vector<R>) &&
        unqualified_extended_cmpgt<L, R, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A, L lhs, R rhs) noexcept(
        noexcept(cmpgt(lhs, rhs))) {
        return cmpgt(lhs, rhs);
    }

    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_vector<E, A> lhs, basic_vector<E, A> rhs) noexcept {
        return dx::cmplt(rhs, lhs);
    }

public:
    template <fixed_width_abi A, simd_element_for<A> E>
    requires totally_ordered<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<E, A> operator()(
        basic_vector<E, A> lhs, basic_vector<E, A> rhs) noexcept {
        if constexpr (unqualified_canonical_cmpgt<basic_vector<E, A>,
                          basic_vector<E, A>>) {
            if consteval {
                return fallback(lhs, rhs);
            } else {
                return cmpgt(internal::abi<A>, lhs, rhs);
            }
        } else {
            return fallback(lhs, rhs);
        }
    }

    template <simd_abi LA, common_abi_with<LA> RA, typename E>
    requires simd_element_for<E, LA> && simd_element_for<E, RA> &&
        (scalable_abi<LA> || scalable_abi<RA> || different_from<LA, RA> ||
            !totally_ordered<E>) &&
        unqualified_canonical_cmpgt<basic_vector<E, LA>, basic_vector<E, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<E, common_abi_t<LA, RA>> operator()(
        basic_vector<E, LA> lhs, basic_vector<E, RA> rhs) noexcept {
        return cmpgt(internal::abi<common_abi_t<LA, RA>>, lhs, rhs);
    }

    template <simd_vector L, simd_vector R>
    requires (extended_vector<L> || extended_vector<R>) && extended_cmpgt<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        if constexpr (unqualified_extended_cmpgt<L, R>) {
            return cmpgt(lhs, rhs);
        } else if constexpr (expression_cmpgt<L, R>) {
            return operator()(dx::evaluate(lhs), dx::evaluate(rhs));
        } else {
            return operator()(dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }

    using binary_operation_base<cmpgt_t>::operator();

    template <fixed_width_abi A, simd_element_for<A> E, common_size_with<E> ME>
    requires totally_ordered<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<E, A> operator()(basic_mask<ME, A> mask,
        basic_vector<E, A> lhs, basic_vector<E, A> rhs) noexcept {
        if constexpr (unqualified_canonical_mcmpgt<basic_mask<ME, A>,
                          basic_vector<E, A>, basic_vector<E, A>>) {
            if consteval {
                return internal::masked<cmpgt_t>(mask, lhs, rhs);
            } else {
                return cmpgt(internal::abi<A>, mask, lhs, rhs);
            }
        } else {
            return internal::masked<cmpgt_t>(mask, lhs, rhs);
        }
    }

    template <simd_abi SA, simd_element_for<SA> E, common_size_with<E> ME,
        simd_abi LA, common_abi_with<LA> RA>
    requires (different_from<SA, common_abi_t<LA, RA>> ||
                 different_from<LA, RA> || scalable_abi<SA> ||
                 scalable_abi<LA> || scalable_abi<RA> || !totally_ordered<E>) &&
        simd_element_for<E, LA> && simd_element_for<E, RA> &&
        zmaskable_args<basic_mask<ME, SA>, basic_vector<E, LA>,
            basic_vector<E, RA>> &&
        unqualified_canonical_mcmpgt<basic_mask<ME, SA>, basic_vector<E, LA>,
            basic_vector<E, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(basic_mask<ME, SA> mask,
        basic_vector<E, LA> lhs, basic_vector<E, RA> rhs) noexcept {
        return cmpgt(internal::abi<common_abi_t<LA, RA>>, mask, lhs, rhs);
    }

    template <simd_mask M, simd_vector L, simd_vector R>
    requires (extended_mask<M> || extended_vector<L> || extended_vector<R>) &&
        zmaskable_args<M, L, R> && extended_mcmpgt<M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, L lhs, R rhs) noexcept {
        if constexpr (unqualified_extended_mcmpgt<M, L, R>) {
            return cmpgt(mask, lhs, rhs);
        } else if constexpr (expression_mcmpgt<M, L, R>) {
            return operator()(
                dx::evaluate(mask), dx::evaluate(lhs), dx::evaluate(rhs));
        } else {
            return operator()(dx::to_canonical(mask), dx::to_canonical(lhs),
                dx::to_canonical(rhs));
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        const_mask_for<basic_vector<E, A>> M>
    requires totally_ordered<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<E, A> operator()(
        M mask, basic_vector<E, A> lhs, basic_vector<E, A> rhs) noexcept {
        if constexpr (unqualified_canonical_imcmpgt<M, basic_vector<E, A>,
                          basic_vector<E, A>>) {
            if consteval {
                return internal::masked<cmpgt_t>(mask, lhs, rhs);
            } else {
                return cmpgt(internal::abi<A>,
                    dx::to_compatible_const_mask<basic_mask<E, A>>(mask), lhs,
                    rhs);
            }
        } else {
            return internal::masked<cmpgt_t>(mask, lhs, rhs);
        }
    }

    template <simd_abi LA, common_abi_with<LA> RA,
        simd_element_for<common_abi_t<LA, RA>> E,
        const_mask_for<basic_vector<E, common_abi_t<LA, RA>>> M>
    requires (different_from<LA, RA> || scalable_abi<LA> || scalable_abi<RA> ||
                 !totally_ordered<E>) &&
        simd_element_for<E, LA> && simd_element_for<E, RA> &&
        imm_zmaskable_args<basic_vector<E, LA>, basic_vector<E, RA>> &&
        unqualified_canonical_imcmpgt<M, basic_vector<E, LA>,
            basic_vector<E, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<E, common_abi_t<LA, RA>> operator()(
        M mask, basic_vector<E, LA> lhs, basic_vector<E, RA> rhs) noexcept {
        using A = common_abi_t<LA, RA>;

        return cmpgt(internal::abi<A>,
            dx::to_compatible_const_mask<basic_mask<E, A>>(mask), lhs, rhs);
    }

    template <simd_vector L, simd_vector R,
        const_mask_for<operation_result_t<cmpgt_t, L, R>> M>
    requires (extended_vector<L> || extended_vector<R>) &&
        imm_zmaskable_args<L, R> && extended_imcmpgt<M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, L lhs, R rhs) noexcept {
        using S = operation_result_t<cmpgt_t, L, R>;
        if constexpr (unqualified_extended_imcmpgt<M, L, R>) {
            return cmpgt(dx::to_compatible_const_mask<S>(mask), lhs, rhs);
        } else if constexpr (expression_imcmpgt<M, L, R>) {
            return operator()(mask, dx::evaluate(lhs), dx::evaluate(rhs));
        } else {
            return operator()(
                mask, dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::cmpgt_t cmpgt{};
}
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
