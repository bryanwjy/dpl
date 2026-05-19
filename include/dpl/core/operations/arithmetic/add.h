// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

// IWYU pragma: always_keep

#include "dpl/core/operations/arithmetic/result.h"
#include "dpl/core/operations/masked.h"
#include "dpl/core/operations/operation_base.h"
#include "dpl/core/operations/transform.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/arithmetic_type.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_equivalence.h"
#  include "dpl/core/constants/zero.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void add(...) noexcept = delete;
template <auto>
void add(...) noexcept = delete;

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_canonical_add = requires(L lhs, R rhs) {
    { add(internal::abi<A>, lhs, rhs) } -> canonical_arithmetic_result<L, R, A>;
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_extended_add = requires(L lhs, R rhs) {
    { add(lhs, rhs) } -> extended_arithmetic_result<L, R, A>;
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_add =
    unqualified_canonical_add<L, R> || unqualified_extended_add<L, R> ||
    (decayable_vector_for<L, operation_category::lane_agnostic> &&
        decayable_vector_for<R, operation_category::lane_agnostic> &&
        unqualified_canonical_add<canonical_type_t<L>, canonical_type_t<R>>);

template <typename Op, typename S, typename C, typename L, typename R,
    typename A = common_abi_t<L, R, C>>
concept unqualified_canonical_madd = requires(S src, C mask, L lhs, R rhs) {
    {
        add(internal::abi<A>, src, mask, lhs, rhs)
    } -> equivalent_simd_as<
        canonical_if_zero_t<S, operation_result_t<Op, L, R>, A>>;
};

template <typename Op, typename S, typename C, typename L, typename R,
    typename A = common_abi_t<L, R, C>>
concept unqualified_extended_madd = requires(S src, C mask, L lhs, R rhs) {
    {
        add(src, mask, lhs, rhs)
    } -> equivalent_simd_as<
        canonical_if_zero_t<S, operation_result_t<Op, L, R>, A>>;
};

template <typename Op, typename S, typename C, typename L, typename R,
    typename A = common_abi_t<L, R, C>>
concept decayable_madd =
    decayable_vector_for<
        canonical_if_zero_t<S, operation_result_t<Op, L, R>, A>,
        operation_category::lane_agnostic> &&
    decayable_mask_for<C, operation_category::lane_agnostic> &&
    decayable_vector_for<L, operation_category::lane_agnostic> &&
    decayable_vector_for<R, operation_category::lane_agnostic> &&
    requires(canonical_if_zero_t<S, operation_result_t<Op, L, R>, A> s,
        canonical_type_t<C> c, canonical_type_t<L> l,
        canonical_type_t<R> r) { Op::operator()(s, c, l, r); };

template <typename Op, typename S, typename C, typename L, typename R,
    typename A = common_abi_t<L, R, C>>
concept extended_madd = unqualified_extended_madd<Op, S, C, L, R, A> ||
    decayable_madd<Op, S, C, L, R, A>;

template <typename Op, typename S, typename M, typename L, typename R,
    typename A =
        common_abi_t<canonical_if_zero_t<S, operation_result_t<Op, L, R>>,
            operation_result_t<Op, L, R>>>
concept unqualified_canonical_addi = requires(S src, L lhs, R rhs) {
    {
        add<const_mask_v<
            canonical_if_zero_t<S, operation_result_t<Op, L, R>, A>, M>>(
            internal::abi<A>, src, dx::masked_operation, lhs, rhs)
    } -> equivalent_simd_as<
        canonical_if_zero_t<S, operation_result_t<Op, L, R>, A>>;
};

template <typename Op, typename S, typename M, typename L, typename R,
    typename A =
        common_abi_t<canonical_if_zero_t<S, operation_result_t<Op, L, R>>,
            operation_result_t<Op, L, R>>>
concept unqualified_extended_addi = requires(S src, L lhs, R rhs) {
    {
        add<const_mask_v<
            canonical_if_zero_t<S, operation_result_t<Op, L, R>, A>, M>>(
            src, dx::masked_operation, lhs, rhs)
    } -> equivalent_simd_as<
        canonical_if_zero_t<S, operation_result_t<Op, L, R>, A>>;
};

template <typename Op, typename S, typename M, typename L, typename R,
    typename A =
        common_abi_t<canonical_if_zero_t<S, operation_result_t<Op, L, R>>,
            operation_result_t<Op, L, R>>>
concept decayable_addi =
    decayable_vector_for<
        canonical_if_zero_t<S, operation_result_t<Op, L, R>, A>,
        operation_category::lane_agnostic> &&
    decayable_vector_for<L, operation_category::lane_agnostic> &&
    decayable_vector_for<R, operation_category::lane_agnostic> &&
    requires(canonical_if_zero_t<S, operation_result_t<Op, L, R>, A> s, M mask,
        canonical_type_t<L> l,
        canonical_type_t<R> r) { Op::operator()(s, mask, l, r); };

template <typename Op, typename S, typename M, typename L, typename R,
    typename A =
        common_abi_t<canonical_if_zero_t<S, operation_result_t<Op, L, R>>,
            operation_result_t<Op, L, R>>>
concept extended_addi = unqualified_extended_addi<Op, S, M, L, R, A> ||
    decayable_addi<Op, S, M, L, R, A>;

struct add_t : binary_operation_base<add_t> {
private:
    friend binary_operation_base<add_t>;

    template <simd_abi A, typename L, typename R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, L lhs, R rhs) noexcept
    requires requires {
        {
            add(internal::abi<A>, lhs, rhs)
        } -> broadcasting_arithmetic_result<A>;
    }
    {
        return add(internal::abi<A>, lhs, rhs);
    }

    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_vector<E, A> lhs, basic_vector<E, A> rhs) noexcept {
        return internal::transform<basic_vector<E, A>>(
            [](E lhs, E rhs) { return static_cast<E>(lhs + rhs); }, lhs, rhs);
    }

public:
    template <fixed_width_abi A, simd_element_for<A> E>
    requires arithmetic_type<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> lhs, basic_vector<E, A> rhs) noexcept {
        if constexpr (unqualified_canonical_add<basic_vector<E, A>,
                          basic_vector<E, A>>) {
            if consteval {
                return fallback(lhs, rhs);
            } else {
                return add(internal::abi<A>, lhs, rhs);
            }
        } else {
            return fallback(lhs, rhs);
        }
    }

    template <simd_abi LA, common_abi_with<LA> RA, typename E>
    requires simd_element_for<E, LA> && simd_element_for<E, RA> &&
        (scalable_abi<LA> || scalable_abi<RA> || different_from<LA, RA> ||
            !arithmetic_type<E>) &&
        unqualified_canonical_add<basic_vector<E, LA>, basic_vector<E, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, common_abi_t<LA, RA>> operator()(
        basic_vector<E, LA> lhs, basic_vector<E, RA> rhs) noexcept {
        return add(internal::abi<common_abi_t<LA, RA>>, lhs, rhs);
    }

    template <simd_vector L, simd_vector R>
    requires (extended_vector<L> || extended_vector<R>) && unqualified_add<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        if constexpr (unqualified_extended_add<L, R>) {
            return add(lhs, rhs);
        } else {
            return operator()(dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }

    using binary_operation_base<add_t>::operator();

    template <fixed_width_abi A, simd_element_for<A> E,
        common_size_with<E> MaskE>
    requires arithmetic_type<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_vector<E, A> src,
        basic_mask<MaskE, A> mask, basic_vector<E, A> lhs,
        basic_vector<E, A> rhs) noexcept {
        if constexpr (unqualified_canonical_madd<add_t, basic_vector<E, A>,
                          basic_mask<MaskE, A>, basic_vector<E, A>,
                          basic_vector<E, A>>) {
            if consteval {
                return internal::masked<add_t>(src, mask, lhs, rhs);
            } else {
                return add(internal::abi<A>, src, mask, lhs, rhs);
            }
        } else {
            return internal::masked<add_t>(src, mask, lhs, rhs);
        }
    }

    template <simd_abi SA, simd_element_for<SA> E, common_size_with<E> MaskE,
        simd_abi LA, common_abi_with<LA> RA>
    requires (different_from<LA, RA> || scalable_abi<SA> || scalable_abi<LA> ||
                 scalable_abi<RA> || !arithmetic_type<E>) &&
        simd_element_for<E, LA> && simd_element_for<E, RA> &&
        maskable_args<basic_vector<E, SA>, basic_mask<MaskE, SA>,
            basic_vector<E, LA>, basic_vector<E, RA>> &&
        unqualified_canonical_madd<add_t, basic_vector<E, SA>,
            basic_mask<MaskE, SA>, basic_vector<E, LA>, basic_vector<E, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, SA> operator()(basic_vector<E, SA> src,
        basic_mask<MaskE, SA> mask, basic_vector<E, LA> lhs,
        basic_vector<E, RA> rhs) noexcept {
        return add(internal::abi<SA>, src, mask, lhs, rhs);
    }

    template <simd_vector S, simd_mask Mask, simd_vector L, simd_vector R>
    requires (extended_vector<S> || extended_mask<Mask> || extended_vector<L> ||
                 extended_vector<R>) &&
        maskable_args<S, Mask, L, R> && extended_madd<add_t, S, Mask, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, Mask mask, L lhs, R rhs) noexcept {
        if constexpr (unqualified_extended_madd<add_t, S, Mask, L, R>) {
            return add(src, mask, lhs, rhs);
        } else {
            return operator()(dx::to_canonical(src), dx::to_canonical(mask),
                dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        common_size_with<E> MaskE>
    requires arithmetic_type<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_mask<MaskE, A> mask,
        basic_vector<E, A> lhs, basic_vector<E, A> rhs) noexcept {
        if constexpr (unqualified_canonical_madd<add_t, zero_t,
                          basic_mask<MaskE, A>, basic_vector<E, A>,
                          basic_vector<E, A>>) {
            if consteval {
                return internal::masked<add_t>(mask, lhs, rhs);
            } else {
                return add(internal::abi<A>, dx::zero, mask, lhs, rhs);
            }
        } else {
            return operator()(dx::zero_v<basic_vector<E, A>>, mask, lhs, rhs);
        }
    }

    template <simd_abi SA, simd_element_for<SA> E, common_size_with<E> MaskE,
        simd_abi LA, common_abi_with<LA> RA>
    requires (different_from<SA, common_abi_t<LA, RA>> ||
                 different_from<LA, RA> || scalable_abi<SA> ||
                 scalable_abi<LA> || scalable_abi<RA> || !arithmetic_type<E>) &&
        simd_element_for<E, LA> && simd_element_for<E, RA> &&
        zmaskable_args<basic_mask<MaskE, SA>, basic_vector<E, LA>,
            basic_vector<E, RA>> &&
        unqualified_canonical_madd<add_t, zero_t, basic_mask<MaskE, SA>,
            basic_vector<E, LA>, basic_vector<E, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(basic_mask<MaskE, SA> mask,
        basic_vector<E, LA> lhs, basic_vector<E, RA> rhs) noexcept {
        return add(
            internal::abi<common_abi_t<LA, RA>>, dx::zero, mask, lhs, rhs);
    }

    template <simd_vector S, simd_mask Mask, simd_vector L, simd_vector R>
    requires (extended_vector<S> || extended_mask<Mask> || extended_vector<L> ||
                 extended_vector<R>) &&
        zmaskable_args<Mask, L, R> && extended_madd<add_t, zero_t, Mask, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(Mask mask, L lhs, R rhs) noexcept {
        if constexpr (unqualified_extended_madd<add_t, zero_t, Mask, L, R>) {
            return add(mask, lhs, rhs);
        } else {
            return operator()(dx::to_canonical(mask), dx::to_canonical(lhs),
                dx::to_canonical(rhs));
        }
    }

    template <simd_mask Mask, simd_vector L, simd_vector R>
    requires requires(
        Mask mask, L lhs, R rhs) { add_t::operator()(mask, lhs, rhs); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t, Mask mask, L lhs, R rhs) noexcept {
        return operator()(mask, lhs, rhs);
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        const_mask_for<basic_vector<E, A>> Mask>
    requires arithmetic_type<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_vector<E, A> src,
        Mask mask, basic_vector<E, A> lhs, basic_vector<E, A> rhs) noexcept {
        if constexpr (unqualified_canonical_addi<add_t, basic_vector<E, A>,
                          Mask, basic_vector<E, A>, basic_vector<E, A>>) {
            if consteval {
                return internal::masked<add_t>(src, mask, lhs, rhs);
            } else {
                constexpr auto V = const_mask_v<basic_vector<E, A>, Mask>;
                return add<V>(
                    internal::abi<A>, src, masked_operation, lhs, rhs);
            }
        } else {
            return internal::masked<add_t>(src, mask, lhs, rhs);
        }
    }

    template <simd_abi SA, simd_element_for<SA> E,
        const_mask_for<basic_vector<E, SA>> Mask, simd_abi LA,
        common_abi_with<LA> RA>
    requires (different_from<SA, common_abi_t<LA, RA>> ||
                 different_from<LA, RA> || scalable_abi<SA> ||
                 scalable_abi<LA> || scalable_abi<RA> || !arithmetic_type<E>) &&
        simd_element_for<E, LA> && simd_element_for<E, RA> &&
        imm_maskable_args<basic_vector<E, SA>, basic_vector<E, LA>,
            basic_vector<E, RA>> &&
        unqualified_canonical_addi<add_t, basic_vector<E, SA>, Mask,
            basic_vector<E, LA>, basic_vector<E, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, SA> operator()(basic_vector<E, SA> src,
        Mask mask, basic_vector<E, LA> lhs, basic_vector<E, RA> rhs) noexcept {
        constexpr auto V = const_mask_v<basic_vector<E, SA>, Mask>;
        return add<V>(internal::abi<SA>, src, masked_operation, lhs, rhs);
    }

    template <simd_vector S, const_mask_for<S> Mask, simd_vector L,
        simd_vector R>
    requires (extended_vector<S> || extended_vector<L> || extended_vector<R>) &&
        imm_maskable_args<S, L, R> && extended_addi<add_t, S, Mask, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, Mask mask, L lhs, R rhs) noexcept {
        if constexpr (unqualified_extended_addi<add_t, S, Mask, L, R>) {
            constexpr auto V = const_mask_v<S, Mask>;
            return add<V>(src, masked_operation, lhs, rhs);
        } else {
            return operator()(dx::to_canonical(src), mask,
                dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        const_mask_for<basic_vector<E, A>> Mask>
    requires arithmetic_type<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        Mask mask, basic_vector<E, A> lhs, basic_vector<E, A> rhs) noexcept {
        if constexpr (unqualified_canonical_addi<add_t, zero_t, Mask,
                          basic_vector<E, A>, basic_vector<E, A>>) {
            if consteval {
                return internal::masked<add_t>(mask, lhs, rhs);
            } else {
                constexpr auto V = const_mask_v<basic_vector<E, A>, Mask>;
                return add<V>(
                    internal::abi<A>, dx::zero, masked_operation, lhs, rhs);
            }
        } else {
            return operator()(dx::zero_v<basic_vector<E, A>>, mask, lhs, rhs);
        }
    }

    template <simd_abi LA, common_abi_with<LA> RA,
        simd_element_for<common_abi_t<LA, RA>> E,
        const_mask_for<basic_vector<E, common_abi_t<LA, RA>>> Mask>
    requires (different_from<LA, RA> || scalable_abi<LA> || scalable_abi<RA> ||
                 !arithmetic_type<E>) &&
        simd_element_for<E, LA> && simd_element_for<E, RA> &&
        imm_zmaskable_args<basic_vector<E, LA>, basic_vector<E, RA>> &&
        unqualified_canonical_addi<add_t, zero_t, Mask, basic_vector<E, LA>,
            basic_vector<E, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, common_abi_t<LA, RA>> operator()(
        Mask mask, basic_vector<E, LA> lhs, basic_vector<E, RA> rhs) noexcept {
        using A = common_abi_t<LA, RA>;
        constexpr auto V = const_mask_v<basic_vector<E, A>, Mask>;
        return add<V>(internal::abi<A>, dx::zero, masked_operation, lhs, rhs);
    }

    template <simd_vector L, simd_vector R,
        const_mask_for<operation_result_t<add_t, L, R>> Mask>
    requires (extended_vector<L> || extended_vector<R>) &&
        imm_zmaskable_args<L, R> && extended_addi<add_t, zero_t, Mask, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(Mask mask, L lhs, R rhs) noexcept {
        using S = operation_result_t<add_t, L, R>;
        if constexpr (unqualified_extended_madd<add_t, zero_t, Mask, L, R>) {
            constexpr auto V = const_mask_v<S, Mask>;
            return add<V>(dx::zero, masked_operation, lhs, rhs);
        } else {
            return operator()(dx::to_compatible_const_mask<S>(mask),
                dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }

    template <simd_vector L, simd_vector R, const_mask_like Mask>
    requires requires(
        Mask mask, L lhs, R rhs) { add_t::operator()(mask, lhs, rhs); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t, Mask mask, L lhs, R rhs) noexcept {
        return operator()(mask, lhs, rhs);
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::add_t add{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
