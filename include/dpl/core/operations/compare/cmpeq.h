// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

// IWYU pragma: always_keep

#include "dpl/core/operations/compare/result.h"
#include "dpl/core/operations/internal/masked.h"
#include "dpl/core/operations/internal/operation_base.h"
#include "dpl/core/operations/internal/transform.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/decayable.h"
#  include "dpl/core/concepts/operation_category.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_equivalence.h"
#  include "dpl/core/type_traits/common_size_type.h"
#  include "dpl/std/concepts/equality_comparable.h"
#  include "dpl/std/concepts/invocable.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void cmpeq(...) noexcept = delete;

struct cmpeq_t;

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_canonical_cmpeq = requires(L lhs, R rhs) {
    { cmpeq(internal::abi<A>, lhs, rhs) } -> canonical_compare_result<L, R, A>;
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_extended_cmpeq = requires(L lhs, R rhs) {
    { cmpeq(lhs, rhs) } -> extended_operation_mask<A>;
};

template <typename L, typename R>
concept expression_cmpeq = (simd_expression<L> || simd_expression<R>) &&
    invocable<cmpeq_t, simd_expression_result_t<L>,
        simd_expression_result_t<R>>;

template <typename L, typename R>
concept decayable_cmpeq =
    decayable_vector_for<L, operation_category::lane_agnostic> &&
    decayable_vector_for<R, operation_category::lane_agnostic> &&
    regular_invocable<cmpeq_t, canonical_type_t<L>, canonical_type_t<R>>;

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept extended_cmpeq = unqualified_extended_cmpeq<L, R> ||
    expression_cmpeq<L, R> || decayable_cmpeq<L, R>;

template <typename M, typename L, typename R,
    typename A = common_abi_t<L, R, M>>
concept unqualified_canonical_mcmpeq = requires(M mask, L lhs, R rhs) {
    {
        cmpeq(internal::abi<A>, mask, lhs, rhs)
    } -> canonical_compare_result<L, R>;
};

template <typename M, typename L, typename R,
    typename A = common_abi_t<L, R, M>>
concept unqualified_extended_mcmpeq = requires(M mask, L lhs, R rhs) {
    { cmpeq(mask, lhs, rhs) } -> extended_operation_mask<A>;
};

template <typename M, typename L, typename R>
concept expression_mcmpeq =
    (simd_expression<M> || simd_expression<L> || simd_expression<R>) &&
    invocable<cmpeq_t, simd_expression_result_t<M>, simd_expression_result_t<L>,
        simd_expression_result_t<R>>;

template <typename M, typename L, typename R,
    typename A = common_abi_t<L, R, M>>
concept decayable_mcmpeq =
    decayable_mask_for<M, operation_category::lane_agnostic> &&
    decayable_vector_for<L, operation_category::lane_agnostic> &&
    decayable_vector_for<R, operation_category::lane_agnostic> &&
    regular_invocable<cmpeq_t, canonical_type_t<M>, canonical_type_t<L>,
        canonical_type_t<R>>;

template <typename M, typename L, typename R,
    typename A = common_abi_t<L, R, M>>
concept extended_mcmpeq = unqualified_extended_mcmpeq<M, L, R, A> ||
    expression_mcmpeq<M, L, R> || decayable_mcmpeq<M, L, R>;

template <typename M, typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_canonical_imcmpeq = requires(M mask, L lhs, R rhs) {
    {
        cmpeq(internal::abi<A>,
            dx::to_compatible_const_mask<operation_result_t<cmpeq_t, L, R>>(
                mask),
            lhs, rhs)
    } -> canonical_compare_result<L, R>;
};

template <typename M, typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_extended_imcmpeq = requires(M mask, L lhs, R rhs) {
    {
        cmpeq(dx::to_compatible_const_mask<operation_result_t<cmpeq_t, L, R>>(
                  mask),
            lhs, rhs)
    } -> extended_operation_mask<A>;
};

template <typename M, typename L, typename R>
concept expression_imcmpeq = (simd_expression<L> || simd_expression<R>) &&
    invocable<cmpeq_t, M, simd_expression_result_t<L>,
        simd_expression_result_t<R>>;

template <typename M, typename L, typename R>
concept decayable_imcmpeq =
    decayable_vector_for<L, operation_category::lane_agnostic> &&
    decayable_vector_for<R, operation_category::lane_agnostic> &&
    regular_invocable<cmpeq_t, M, canonical_type_t<L>, canonical_type_t<R>>;

template <typename M, typename L, typename R, typename A = common_abi_t<L, R>>
concept extended_imcmpeq = unqualified_extended_imcmpeq<M, L, R, A> ||
    expression_imcmpeq<M, L, R> || decayable_imcmpeq<M, L, R>;

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_canonical_mask_cmpeq = requires(L lhs, R rhs) {
    { cmpeq(internal::abi<A>, lhs, rhs) } -> canonical_compare_mask<L, R, A>;
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_extended_mask_cmpeq = requires(L lhs, R rhs) {
    { cmpeq(lhs, rhs) } -> extended_operation_mask<A>;
};

template <typename L, typename R>
concept expression_mask_cmpeq = (simd_expression<L> || simd_expression<R>) &&
    invocable<cmpeq_t, simd_expression_result_t<L>,
        simd_expression_result_t<R>>;

template <typename L, typename R>
concept decayable_mask_cmpeq =
    decayable_mask_for<L, operation_category::lane_agnostic> &&
    decayable_mask_for<R, operation_category::lane_agnostic> &&
    regular_invocable<cmpeq_t, canonical_type_t<L>, canonical_type_t<R>>;

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept extended_mask_cmpeq = unqualified_extended_mask_cmpeq<L, R> ||
    expression_mask_cmpeq<L, R> || decayable_mask_cmpeq<L, R>;

struct cmpeq_t : binary_operation_base<cmpeq_t> {
private:
    friend binary_operation_base<cmpeq_t>;

    template <simd_abi A, typename L, typename R>
    requires (!simd_class<L> || canonical_vector<L>) &&
        (!simd_class<R> || canonical_vector<R>) && requires(L lhs, R rhs) {
            { cmpeq(internal::abi<A>, lhs, rhs) } -> mask_with_abi<A>;
        }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, L lhs, R rhs) noexcept {
        return cmpeq(internal::abi<A>, lhs, rhs);
    }

    template <simd_abi A, typename L, typename R>
    requires (!simd_class<L> || extended_vector<L>) &&
        (!simd_class<R> || extended_vector<R>) &&
        unqualified_extended_cmpeq<L, R, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A, L lhs, R rhs) noexcept(
        noexcept(cmpeq(lhs, rhs))) {
        return cmpeq(lhs, rhs);
    }

    template <simd_abi A, typename L, typename R>
    requires (!simd_class<L> || canonical_mask<L>) &&
        (!simd_class<R> || canonical_mask<R>) && requires(L lhs, R rhs) {
            { cmpeq(internal::abi<A>, lhs, rhs) } -> mask_with_abi<A>;
        }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, L lhs, R rhs) noexcept {
        return cmpeq(internal::abi<A>, lhs, rhs);
    }

    template <simd_abi A, typename L, typename R>
    requires (!simd_class<L> || extended_mask<L>) &&
        (!simd_class<R> || extended_mask<R>) &&
        unqualified_extended_mask_cmpeq<L, R, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A, L lhs, R rhs) noexcept(
        noexcept(cmpeq(lhs, rhs))) {
        return cmpeq(lhs, rhs);
    }

    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_vector<E, A> lhs, basic_vector<E, A> rhs) noexcept {
        return internal::transform<basic_mask<E, A>>(
            [](auto lhs, auto rhs) -> bool { return lhs == rhs; }, lhs, rhs);
    }

    template <typename L, typename R, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_mask<L, A> lhs, basic_mask<R, A> rhs) noexcept {
        using T = common_size_type_t<L, R>;
        return internal::transform<basic_mask<T, A>>(
            [](auto lhs, auto rhs) -> bool { return lhs == rhs; }, lhs, rhs);
    }

public:
    template <fixed_width_abi A, simd_element_for<A> E>
    requires equality_comparable<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<E, A> operator()(
        basic_vector<E, A> lhs, basic_vector<E, A> rhs) noexcept {
        if constexpr (unqualified_canonical_cmpeq<basic_vector<E, A>,
                          basic_vector<E, A>>) {
            if consteval {
                return fallback(lhs, rhs);
            } else {
                return cmpeq(internal::abi<A>, lhs, rhs);
            }
        } else {
            return fallback(lhs, rhs);
        }
    }

    template <simd_abi LA, common_abi_with<LA> RA, typename E>
    requires simd_element_for<E, LA> && simd_element_for<E, RA> &&
        (scalable_abi<LA> || scalable_abi<RA> || different_from<LA, RA> ||
            !equality_comparable<E>) &&
        unqualified_canonical_cmpeq<basic_vector<E, LA>, basic_vector<E, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<E, common_abi_t<LA, RA>> operator()(
        basic_vector<E, LA> lhs, basic_vector<E, RA> rhs) noexcept {
        return cmpeq(internal::abi<common_abi_t<LA, RA>>, lhs, rhs);
    }

    template <simd_vector L, simd_vector R>
    requires (extended_vector<L> || extended_vector<R>) && extended_cmpeq<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        if constexpr (unqualified_extended_cmpeq<L, R>) {
            return cmpeq(lhs, rhs);
        } else if constexpr (expression_cmpeq<L, R>) {
            return operator()(dx::evaluate(lhs), dx::evaluate(rhs));
        } else {
            return operator()(dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }

    using binary_operation_base<cmpeq_t>::operator();

    template <fixed_width_abi A, simd_element_for<A> E, common_size_with<E> ME>
    requires equality_comparable<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<E, A> operator()(basic_mask<ME, A> mask,
        basic_vector<E, A> lhs, basic_vector<E, A> rhs) noexcept {
        if constexpr (unqualified_canonical_mcmpeq<basic_mask<ME, A>,
                          basic_vector<E, A>, basic_vector<E, A>>) {
            if consteval {
                return internal::masked<cmpeq_t>(mask, lhs, rhs);
            } else {
                return cmpeq(internal::abi<A>, mask, lhs, rhs);
            }
        } else {
            return internal::masked<cmpeq_t>(mask, lhs, rhs);
        }
    }

    template <simd_abi SA, simd_element_for<SA> E, common_size_with<E> ME,
        simd_abi LA, common_abi_with<LA> RA>
    requires (different_from<SA, common_abi_t<LA, RA>> ||
                 different_from<LA, RA> || scalable_abi<SA> ||
                 scalable_abi<LA> || scalable_abi<RA> ||
                 !equality_comparable<E>) &&
        simd_element_for<E, LA> && simd_element_for<E, RA> &&
        zmaskable_args<basic_mask<ME, SA>, basic_vector<E, LA>,
            basic_vector<E, RA>> &&
        unqualified_canonical_mcmpeq<basic_mask<ME, SA>, basic_vector<E, LA>,
            basic_vector<E, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(basic_mask<ME, SA> mask,
        basic_vector<E, LA> lhs, basic_vector<E, RA> rhs) noexcept {
        return cmpeq(internal::abi<common_abi_t<LA, RA>>, mask, lhs, rhs);
    }

    template <simd_mask M, simd_vector L, simd_vector R>
    requires (extended_mask<M> || extended_vector<L> || extended_vector<R>) &&
        zmaskable_args<M, L, R> && extended_mcmpeq<M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, L lhs, R rhs) noexcept {
        if constexpr (unqualified_extended_mcmpeq<M, L, R>) {
            return cmpeq(mask, lhs, rhs);
        } else if constexpr (expression_mcmpeq<M, L, R>) {
            return operator()(
                dx::evaluate(mask), dx::evaluate(lhs), dx::evaluate(rhs));
        } else {
            return operator()(dx::to_canonical(mask), dx::to_canonical(lhs),
                dx::to_canonical(rhs));
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        const_mask_for<basic_vector<E, A>> M>
    requires equality_comparable<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<E, A> operator()(
        M mask, basic_vector<E, A> lhs, basic_vector<E, A> rhs) noexcept {
        if constexpr (unqualified_canonical_imcmpeq<M, basic_vector<E, A>,
                          basic_vector<E, A>>) {
            if consteval {
                return internal::masked<cmpeq_t>(mask, lhs, rhs);
            } else {
                return cmpeq(internal::abi<A>,
                    dx::to_compatible_const_mask<basic_mask<E, A>>(mask), lhs,
                    rhs);
            }
        } else {
            return internal::masked<cmpeq_t>(mask, lhs, rhs);
        }
    }

    template <simd_abi LA, common_abi_with<LA> RA,
        simd_element_for<common_abi_t<LA, RA>> E,
        const_mask_for<basic_vector<E, common_abi_t<LA, RA>>> M>
    requires (different_from<LA, RA> || scalable_abi<LA> || scalable_abi<RA> ||
                 !equality_comparable<E>) &&
        simd_element_for<E, LA> && simd_element_for<E, RA> &&
        imm_zmaskable_args<basic_vector<E, LA>, basic_vector<E, RA>> &&
        unqualified_canonical_imcmpeq<M, basic_vector<E, LA>,
            basic_vector<E, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<E, common_abi_t<LA, RA>> operator()(
        M mask, basic_vector<E, LA> lhs, basic_vector<E, RA> rhs) noexcept {
        using A = common_abi_t<LA, RA>;

        return cmpeq(internal::abi<A>,
            dx::to_compatible_const_mask<basic_mask<E, A>>(mask), lhs, rhs);
    }

    template <simd_vector L, simd_vector R,
        const_mask_for<operation_result_t<cmpeq_t, L, R>> M>
    requires (extended_vector<L> || extended_vector<R>) &&
        imm_zmaskable_args<L, R> && extended_imcmpeq<M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, L lhs, R rhs) noexcept {
        using S = operation_result_t<cmpeq_t, L, R>;
        if constexpr (unqualified_extended_imcmpeq<M, L, R>) {
            return cmpeq(dx::to_compatible_const_mask<S>(mask), lhs, rhs);
        } else if constexpr (expression_imcmpeq<M, L, R>) {
            return operator()(mask, dx::evaluate(lhs), dx::evaluate(rhs));
        } else {
            return operator()(
                mask, dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }
    ///

    template <fixed_width_abi A, simd_element_for<A> E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<E, A> operator()(
        basic_mask<E, A> lhs, basic_mask<E, A> rhs) noexcept {
        if constexpr (unqualified_canonical_mask_cmpeq<basic_mask<E, A>,
                          basic_mask<E, A>>) {
            if consteval {
                return fallback(lhs, rhs);
            } else {
                return cmpeq(internal::abi<A>, lhs, rhs);
            }
        } else {
            return fallback(lhs, rhs);
        }
    }

    template <simd_abi LA, common_abi_with<LA> RA, typename E>
    requires simd_element_for<E, LA> && simd_element_for<E, RA> &&
        (scalable_abi<LA> || scalable_abi<RA> || different_from<LA, RA>) &&
        unqualified_canonical_mask_cmpeq<basic_mask<E, LA>, basic_mask<E, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<E, common_abi_t<LA, RA>> operator()(
        basic_mask<E, LA> lhs, basic_mask<E, RA> rhs) noexcept {
        return cmpeq(internal::abi<common_abi_t<LA, RA>>, lhs, rhs);
    }

    template <simd_mask L, simd_mask R>
    requires (extended_mask<L> || extended_mask<R>) && extended_mask_cmpeq<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        if constexpr (unqualified_extended_mask_cmpeq<L, R>) {
            return cmpeq(lhs, rhs);
        } else if constexpr (expression_mask_cmpeq<L, R>) {
            return operator()(dx::evaluate(lhs), dx::evaluate(rhs));
        } else {
            return operator()(dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::cmpeq_t cmpeq{};
}
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
