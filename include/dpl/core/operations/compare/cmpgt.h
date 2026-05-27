// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

// IWYU pragma: always_keep

#include "dpl/core/operations/compare/cmplt.h"
#include "dpl/core/operations/compare/result.h"
#include "dpl/core/operations/masked.h"
#include "dpl/core/operations/operation_base.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/decayable.h"
#  include "dpl/core/concepts/operation_category.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_equivalence.h"
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
    { cmpgt(lhs, rhs) } -> extended_compare_result<L, R, A>;
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_cmpgt = unqualified_extended_cmpgt<L, R> ||
    (decayable_vector_for<L, operation_category::lane_agnostic> &&
        decayable_vector_for<R, operation_category::lane_agnostic> &&
        regular_invocable<cmpgt_t, canonical_type_t<L>, canonical_type_t<R>>);

template <typename C, typename L, typename R,
    typename A = common_abi_t<L, R, C>>
concept unqualified_canonical_mcmpgt = requires(C mask, L lhs, R rhs) {
    {
        cmpgt(internal::abi<A>, mask, lhs, rhs)
    } -> canonical_compare_result<L, R>;
};

template <typename C, typename L, typename R,
    typename A = common_abi_t<L, R, C>>
concept unqualified_extended_mcmpgt = requires(C mask, L lhs, R rhs) {
    { cmpgt(mask, lhs, rhs) } -> extended_compare_result<L, R, A>;
};

template <typename C, typename L, typename R,
    typename A = common_abi_t<L, R, C>>
concept decayable_mcmpgt =
    decayable_mask_for<C, operation_category::lane_agnostic> &&
    decayable_vector_for<L, operation_category::lane_agnostic> &&
    decayable_vector_for<R, operation_category::lane_agnostic> &&
    requires(cmpgt_t op, canonical_type_t<C> c, canonical_type_t<L> l,
        canonical_type_t<R> r) { op(c, l, r); };

template <typename C, typename L, typename R,
    typename A = common_abi_t<L, R, C>>
concept extended_mcmpgt =
    unqualified_extended_mcmpgt<C, L, R, A> || decayable_mcmpgt<C, L, R, A>;

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
        cmpgt<const_mask_v<operation_result_t<cmpgt_t, L, R>, M>>(
            dx::to_compatible_const_mask<operation_result_t<cmpgt_t, L, R>>(
                mask),
            lhs, rhs)
    } -> extended_compare_result<L, R>;
};

template <typename M, typename L, typename R, typename A = common_abi_t<L, R>>
concept decayable_imcmpgt =
    decayable_vector_for<L, operation_category::lane_agnostic> &&
    decayable_vector_for<R, operation_category::lane_agnostic> &&
    requires(cmpgt_t op, M mask, canonical_type_t<L> l, canonical_type_t<R> r) {
        op(mask, l, r);
    };

template <typename M, typename L, typename R, typename A = common_abi_t<L, R>>
concept extended_imcmpgt =
    unqualified_extended_imcmpgt<M, L, R, A> || decayable_imcmpgt<M, L, R, A>;

struct cmpgt_t : binary_operation_base<cmpgt_t> {
private:
    friend binary_operation_base<cmpgt_t>;

    template <simd_abi A, typename L, typename R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, L left, R right) noexcept
    requires requires {
        { cmpgt(internal::abi<A>, left, right) } -> simd_with_abi<A>;
    }
    {
        return cmpgt(internal::abi<A>, left, right);
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
    requires (extended_vector<L> || extended_vector<R>) &&
        unqualified_cmpgt<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        if constexpr (unqualified_extended_cmpgt<L, R>) {
            return cmpgt(lhs, rhs);
        } else {
            return operator()(dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }

    using binary_operation_base<cmpgt_t>::operator();

    template <fixed_width_abi A, simd_element_for<A> E,
        common_size_with<E> MaskE>
    requires totally_ordered<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<E, A> operator()(basic_mask<MaskE, A> mask,
        basic_vector<E, A> lhs, basic_vector<E, A> rhs) noexcept {
        if constexpr (unqualified_canonical_mcmpgt<basic_mask<MaskE, A>,
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

    template <simd_abi SA, simd_element_for<SA> E, common_size_with<E> MaskE,
        simd_abi LA, common_abi_with<LA> RA>
    requires (different_from<SA, common_abi_t<LA, RA>> ||
                 different_from<LA, RA> || scalable_abi<SA> ||
                 scalable_abi<LA> || scalable_abi<RA> || !totally_ordered<E>) &&
        simd_element_for<E, LA> && simd_element_for<E, RA> &&
        zmaskable_args<basic_mask<MaskE, SA>, basic_vector<E, LA>,
            basic_vector<E, RA>> &&
        unqualified_canonical_mcmpgt<basic_mask<MaskE, SA>, basic_vector<E, LA>,
            basic_vector<E, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(basic_mask<MaskE, SA> mask,
        basic_vector<E, LA> lhs, basic_vector<E, RA> rhs) noexcept {
        return cmpgt(internal::abi<common_abi_t<LA, RA>>, mask, lhs, rhs);
    }

    template <simd_mask Mask, simd_vector L, simd_vector R>
    requires (extended_mask<Mask> || extended_vector<L> ||
                 extended_vector<R>) &&
        zmaskable_args<Mask, L, R> && extended_mcmpgt<Mask, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(Mask mask, L lhs, R rhs) noexcept {
        if constexpr (unqualified_extended_mcmpgt<Mask, L, R>) {
            return cmpgt(mask, lhs, rhs);
        } else {
            return operator()(dx::to_canonical(mask), dx::to_canonical(lhs),
                dx::to_canonical(rhs));
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        const_mask_for<basic_vector<E, A>> Mask>
    requires totally_ordered<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<E, A> operator()(
        Mask mask, basic_vector<E, A> lhs, basic_vector<E, A> rhs) noexcept {
        if constexpr (unqualified_canonical_imcmpgt<Mask, basic_vector<E, A>,
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
        const_mask_for<basic_vector<E, common_abi_t<LA, RA>>> Mask>
    requires (different_from<LA, RA> || scalable_abi<LA> || scalable_abi<RA> ||
                 !totally_ordered<E>) &&
        simd_element_for<E, LA> && simd_element_for<E, RA> &&
        imm_zmaskable_args<basic_vector<E, LA>, basic_vector<E, RA>> &&
        unqualified_canonical_imcmpgt<Mask, basic_vector<E, LA>,
            basic_vector<E, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<E, common_abi_t<LA, RA>> operator()(
        Mask mask, basic_vector<E, LA> lhs, basic_vector<E, RA> rhs) noexcept {
        using A = common_abi_t<LA, RA>;
        return cmpgt(internal::abi<A>,
            dx::to_compatible_const_mask<basic_mask<E, A>>(mask), lhs, rhs);
    }

    template <simd_vector L, simd_vector R,
        const_mask_for<operation_result_t<cmpgt_t, L, R>> Mask>
    requires (extended_vector<L> || extended_vector<R>) &&
        imm_zmaskable_args<L, R> && extended_imcmpgt<Mask, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(Mask mask, L lhs, R rhs) noexcept {
        using S = operation_result_t<cmpgt_t, L, R>;
        if constexpr (unqualified_extended_imcmpgt<Mask, L, R>) {
            return cmpgt(dx::to_compatible_const_mask<S>(mask), lhs, rhs);
        } else {
            return operator()(dx::to_compatible_const_mask<S>(mask),
                dx::to_canonical(lhs), dx::to_canonical(rhs));
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
