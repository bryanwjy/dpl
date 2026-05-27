// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

// IWYU pragma: always_keep
#include "dpl/core/operations/arithmetic/result.h"
#include "dpl/core/operations/masked.h"
#include "dpl/core/operations/operation_base.h"
#include "dpl/core/operations/transform.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/decayable.h"
#  include "dpl/core/concepts/operation_category.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_equivalence.h"
#  include "dpl/core/constants/zero.h"
#  include "dpl/std/concepts/totally_ordered.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void min(...) noexcept = delete;
template <auto>
void min(...) noexcept = delete;

struct min_t;

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_canonical_min = requires(L lhs, R rhs) {
    { min(internal::abi<A>, lhs, rhs) } -> canonical_arithmetic_result<L, R, A>;
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_extended_min = requires(L lhs, R rhs) {
    { min(lhs, rhs) } -> extended_arithmetic_result<L, R, A>;
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_min = unqualified_extended_min<L, R, A> ||
    (decayable_vector_for<L, operation_category::lane_agnostic> &&
        decayable_vector_for<R, operation_category::lane_agnostic> &&
        regular_invocable<min_t, canonical_type_t<L>, canonical_type_t<R>>);

template <typename S, typename C, typename L, typename R,
    typename A = common_abi_t<L, R, C>>
concept unqualified_canonical_mmin = requires(S src, C mask, L lhs, R rhs) {
    {
        min(internal::abi<A>, src, mask, lhs, rhs)
    } -> equivalent_simd_as<
        canonical_if_zero_t<S, operation_result_t<min_t, L, R>, A>>;
};

template <typename S, typename C, typename L, typename R,
    typename A = common_abi_t<L, R, C>>
concept unqualified_extended_mmin = requires(S src, C mask, L lhs, R rhs) {
    {
        min(src, mask, lhs, rhs)
    } -> equivalent_simd_as<
        canonical_if_zero_t<S, operation_result_t<min_t, L, R>, A>>;
};

template <typename S, typename C, typename L, typename R,
    typename A = common_abi_t<L, R, C>>
concept decayable_mmin =
    decayable_vector_for<
        canonical_if_zero_t<S, operation_result_t<min_t, L, R>, A>,
        operation_category::lane_agnostic> &&
    decayable_mask_for<C, operation_category::lane_agnostic> &&
    decayable_vector_for<L, operation_category::lane_agnostic> &&
    decayable_vector_for<R, operation_category::lane_agnostic> &&
    requires(min_t op,
        canonical_if_zero_t<S, operation_result_t<min_t, L, R>, A> s,
        canonical_type_t<C> c, canonical_type_t<L> l,
        canonical_type_t<R> r) { op(s, c, l, r); };

template <typename S, typename C, typename L, typename R,
    typename A = common_abi_t<L, R, C>>
concept extended_mmin =
    unqualified_extended_mmin<S, C, L, R, A> || decayable_mmin<S, C, L, R, A>;

template <typename S, typename M, typename L, typename R,
    typename A =
        common_abi_t<canonical_if_zero_t<S, operation_result_t<min_t, L, R>>,
            operation_result_t<min_t, L, R>>>
concept unqualified_canonical_immin = requires(S src, M mask, L lhs, R rhs) {
    {
        min(internal::abi<A>, src,
            internal::to_const_mask<A, min_t, S, L, R>(mask), lhs, rhs)
    } -> equivalent_simd_as<
        canonical_if_zero_t<S, operation_result_t<min_t, L, R>, A>>;
};

template <typename S, typename M, typename L, typename R,
    typename A =
        common_abi_t<canonical_if_zero_t<S, operation_result_t<min_t, L, R>>,
            operation_result_t<min_t, L, R>>>
concept unqualified_extended_immin = requires(S src, M mask, L lhs, R rhs) {
    {
        min(src, internal::to_const_mask<A, min_t, S, L, R>(mask), lhs, rhs)
    } -> equivalent_simd_as<
        canonical_if_zero_t<S, operation_result_t<min_t, L, R>, A>>;
};

template <typename S, typename M, typename L, typename R,
    typename A =
        common_abi_t<canonical_if_zero_t<S, operation_result_t<min_t, L, R>>,
            operation_result_t<min_t, L, R>>>
concept decayable_immin =
    decayable_vector_for<
        canonical_if_zero_t<S, operation_result_t<min_t, L, R>, A>,
        operation_category::lane_agnostic> &&
    decayable_vector_for<L, operation_category::lane_agnostic> &&
    decayable_vector_for<R, operation_category::lane_agnostic> &&
    requires(min_t op,
        canonical_if_zero_t<S, operation_result_t<min_t, L, R>, A> s, M mask,
        canonical_type_t<L> l, canonical_type_t<R> r) { op(s, mask, l, r); };

template <typename S, typename M, typename L, typename R,
    typename A =
        common_abi_t<canonical_if_zero_t<S, operation_result_t<min_t, L, R>>,
            operation_result_t<min_t, L, R>>>
concept extended_immin =
    unqualified_extended_immin<S, M, L, R, A> || decayable_immin<S, M, L, R, A>;

struct min_t : binary_operation_base<min_t> {
private:
    friend binary_operation_base<min_t>;

    template <simd_abi A, typename L, typename R>
    requires (canonical_vector<L> || canonical_vector<R>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, L lhs, R rhs) noexcept
    requires requires {
        {
            min(internal::abi<A>, lhs, rhs)
        } -> broadcasting_arithmetic_result<A>;
    }
    {
        return min(internal::abi<A>, lhs, rhs);
    }

    template <simd_abi A, typename L, typename R>
    requires (extended_vector<L> || extended_vector<R>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, L lhs, R rhs) noexcept
    requires requires {
        { min(lhs, rhs) } -> broadcasting_arithmetic_result<A>;
    }
    {
        return min(lhs, rhs);
    }

    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_vector<E, A> lhs, basic_vector<E, A> rhs) noexcept {
        return internal::transform<basic_vector<E, A>>(
            [](E lhs, E rhs) { return lhs < rhs ? lhs : rhs; }, lhs, rhs);
    }

public:
    template <fixed_width_abi A, simd_element_for<A> E>
    requires totally_ordered<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> lhs, basic_vector<E, A> rhs) noexcept {
        if constexpr (unqualified_canonical_min<basic_vector<E, A>,
                          basic_vector<E, A>>) {
            if consteval {
                return fallback(lhs, rhs);
            } else {
                return min(internal::abi<A>, lhs, rhs);
            }
        } else {
            return fallback(lhs, rhs);
        }
    }

    template <simd_abi LA, common_abi_with<LA> RA, typename E>
    requires simd_element_for<E, LA> && simd_element_for<E, RA> &&
        (scalable_abi<LA> || scalable_abi<RA> || different_from<LA, RA> ||
            !totally_ordered<E>) &&
        unqualified_canonical_min<basic_vector<E, LA>, basic_vector<E, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, common_abi_t<LA, RA>> operator()(
        basic_vector<E, LA> lhs, basic_vector<E, RA> rhs) noexcept {
        return min(internal::abi<common_abi_t<LA, RA>>, lhs, rhs);
    }

    template <simd_vector L, simd_vector R>
    requires (extended_vector<L> || extended_vector<R>) && unqualified_min<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        if constexpr (unqualified_extended_min<L, R>) {
            return min(lhs, rhs);
        } else {
            return operator()(dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }

    using binary_operation_base<min_t>::operator();

    template <fixed_width_abi A, simd_element_for<A> E,
        common_size_with<E> MaskE>
    requires totally_ordered<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_vector<E, A> src,
        basic_mask<MaskE, A> mask, basic_vector<E, A> lhs,
        basic_vector<E, A> rhs) noexcept {
        if constexpr (unqualified_canonical_mmin<basic_vector<E, A>,
                          basic_mask<MaskE, A>, basic_vector<E, A>,
                          basic_vector<E, A>>) {
            if consteval {
                return internal::masked<min_t>(src, mask, lhs, rhs);
            } else {
                return min(internal::abi<A>, src, mask, lhs, rhs);
            }
        } else {
            return internal::masked<min_t>(src, mask, lhs, rhs);
        }
    }

    template <simd_abi SA, simd_element_for<SA> E, common_size_with<E> MaskE,
        simd_abi LA, common_abi_with<LA> RA>
    requires (different_from<LA, RA> || scalable_abi<SA> || scalable_abi<LA> ||
                 scalable_abi<RA> || !totally_ordered<E>) &&
        simd_element_for<E, LA> && simd_element_for<E, RA> &&
        maskable_args<basic_vector<E, SA>, basic_mask<MaskE, SA>,
            basic_vector<E, LA>, basic_vector<E, RA>> &&
        unqualified_canonical_mmin<basic_vector<E, SA>, basic_mask<MaskE, SA>,
            basic_vector<E, LA>, basic_vector<E, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, SA> operator()(basic_vector<E, SA> src,
        basic_mask<MaskE, SA> mask, basic_vector<E, LA> lhs,
        basic_vector<E, RA> rhs) noexcept {
        return min(internal::abi<SA>, src, mask, lhs, rhs);
    }

    template <simd_vector S, simd_mask Mask, simd_vector L, simd_vector R>
    requires (extended_vector<S> || extended_mask<Mask> || extended_vector<L> ||
                 extended_vector<R>) &&
        maskable_args<S, Mask, L, R> && extended_mmin<S, Mask, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, Mask mask, L lhs, R rhs) noexcept {
        if constexpr (unqualified_extended_mmin<S, Mask, L, R>) {
            return min(src, mask, lhs, rhs);
        } else {
            return operator()(dx::to_canonical(src), dx::to_canonical(mask),
                dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        common_size_with<E> MaskE>
    requires totally_ordered<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_mask<MaskE, A> mask,
        basic_vector<E, A> lhs, basic_vector<E, A> rhs) noexcept {
        if constexpr (unqualified_canonical_mmin<zero_t, basic_mask<MaskE, A>,
                          basic_vector<E, A>, basic_vector<E, A>>) {
            if consteval {
                return internal::masked<min_t>(mask, lhs, rhs);
            } else {
                return min(internal::abi<A>, dx::zero, mask, lhs, rhs);
            }
        } else {
            return operator()(dx::zero_v<basic_vector<E, A>>, mask, lhs, rhs);
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
        unqualified_canonical_mmin<zero_t, basic_mask<MaskE, SA>,
            basic_vector<E, LA>, basic_vector<E, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(basic_mask<MaskE, SA> mask,
        basic_vector<E, LA> lhs, basic_vector<E, RA> rhs) noexcept {
        return min(
            internal::abi<common_abi_t<LA, RA>>, dx::zero, mask, lhs, rhs);
    }

    template <simd_mask Mask, simd_vector L, simd_vector R>
    requires (extended_mask<Mask> || extended_vector<L> ||
                 extended_vector<R>) &&
        zmaskable_args<Mask, L, R> && extended_mmin<zero_t, Mask, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(Mask mask, L lhs, R rhs) noexcept {
        if constexpr (unqualified_extended_mmin<zero_t, Mask, L, R>) {
            return min(mask, lhs, rhs);
        } else {
            return operator()(dx::to_canonical(mask), dx::to_canonical(lhs),
                dx::to_canonical(rhs));
        }
    }

    template <simd_mask Mask, simd_vector L, simd_vector R>
    requires requires(
        Mask mask, L lhs, R rhs) { min_t::operator()(mask, lhs, rhs); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t, Mask mask, L lhs, R rhs) noexcept {
        return operator()(mask, lhs, rhs);
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        const_mask_for<basic_vector<E, A>> Mask>
    requires totally_ordered<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_vector<E, A> src,
        Mask mask, basic_vector<E, A> lhs, basic_vector<E, A> rhs) noexcept {
        if constexpr (unqualified_canonical_immin<basic_vector<E, A>, Mask,
                          basic_vector<E, A>, basic_vector<E, A>>) {
            if consteval {
                return internal::masked<min_t>(src, mask, lhs, rhs);
            } else {
                return min(internal::abi<A>, src,
                    dx::to_compatible_const_mask<basic_vector<E, A>>(mask), lhs,
                    rhs);
            }
        } else {
            return internal::masked<min_t>(src, mask, lhs, rhs);
        }
    }

    template <simd_abi SA, simd_element_for<SA> E,
        const_mask_for<basic_vector<E, SA>> Mask, simd_abi LA,
        common_abi_with<LA> RA>
    requires (different_from<SA, common_abi_t<LA, RA>> ||
                 different_from<LA, RA> || scalable_abi<SA> ||
                 scalable_abi<LA> || scalable_abi<RA> || !totally_ordered<E>) &&
        simd_element_for<E, LA> && simd_element_for<E, RA> &&
        imm_maskable_args<basic_vector<E, SA>, basic_vector<E, LA>,
            basic_vector<E, RA>> &&
        unqualified_canonical_immin<basic_vector<E, SA>, Mask,
            basic_vector<E, LA>, basic_vector<E, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, SA> operator()(basic_vector<E, SA> src,
        Mask mask, basic_vector<E, LA> lhs, basic_vector<E, RA> rhs) noexcept {
        return min(internal::abi<SA>, src,
            dx::to_compatible_const_mask<basic_vector<E, SA>>(mask), lhs, rhs);
    }

    template <simd_vector S, const_mask_for<S> Mask, simd_vector L,
        simd_vector R>
    requires (extended_vector<S> || extended_vector<L> || extended_vector<R>) &&
        imm_maskable_args<S, L, R> && extended_immin<S, Mask, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, Mask mask, L lhs, R rhs) noexcept {
        if constexpr (unqualified_extended_immin<S, Mask, L, R>) {
            return min(src, dx::to_compatible_const_mask<S>(mask), lhs, rhs);
        } else {
            return operator()(dx::to_canonical(src), mask,
                dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        const_mask_for<basic_vector<E, A>> Mask>
    requires totally_ordered<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        Mask mask, basic_vector<E, A> lhs, basic_vector<E, A> rhs) noexcept {
        if constexpr (unqualified_canonical_immin<zero_t, Mask,
                          basic_vector<E, A>, basic_vector<E, A>>) {
            if consteval {
                return internal::masked<min_t>(mask, lhs, rhs);
            } else {
                return min(internal::abi<A>, dx::zero,
                    dx::to_compatible_const_mask<basic_vector<E, A>>(mask), lhs,
                    rhs);
            }
        } else {
            return operator()(dx::zero_v<basic_vector<E, A>>, mask, lhs, rhs);
        }
    }

    template <simd_abi LA, common_abi_with<LA> RA,
        simd_element_for<common_abi_t<LA, RA>> E,
        const_mask_for<basic_vector<E, common_abi_t<LA, RA>>> Mask>
    requires (different_from<LA, RA> || scalable_abi<LA> || scalable_abi<RA> ||
                 !totally_ordered<E>) &&
        simd_element_for<E, LA> && simd_element_for<E, RA> &&
        imm_zmaskable_args<basic_vector<E, LA>, basic_vector<E, RA>> &&
        unqualified_canonical_immin<zero_t, Mask, basic_vector<E, LA>,
            basic_vector<E, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, common_abi_t<LA, RA>> operator()(
        Mask mask, basic_vector<E, LA> lhs, basic_vector<E, RA> rhs) noexcept {
        using A = common_abi_t<LA, RA>;
        return min(internal::abi<A>, dx::zero,
            dx::to_compatible_const_mask<basic_vector<E, A>>(mask), lhs, rhs);
    }

    template <simd_vector L, simd_vector R,
        const_mask_for<operation_result_t<min_t, L, R>> Mask>
    requires (extended_vector<L> || extended_vector<R>) &&
        imm_zmaskable_args<L, R> && extended_immin<zero_t, Mask, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(Mask mask, L lhs, R rhs) noexcept {
        using S = operation_result_t<min_t, L, R>;
        if constexpr (unqualified_extended_immin<zero_t, Mask, L, R>) {
            return min(
                dx::zero, dx::to_compatible_const_mask<S>(mask), lhs, rhs);
        } else {
            return operator()(dx::to_compatible_const_mask<S>(mask),
                dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }

    template <simd_vector L, simd_vector R, const_mask_like Mask>
    requires requires(
        Mask mask, L lhs, R rhs) { min_t::operator()(mask, lhs, rhs); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t, Mask mask, L lhs, R rhs) noexcept {
        return operator()(mask, lhs, rhs);
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::min_t min{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
