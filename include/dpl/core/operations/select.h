// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/operations/internal/transform.h"

#if !DPL_MODULES
#  include "dpl/core/basic/internal/abi.h"
#  include "dpl/core/concepts/common_size_with.h"
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/concepts/mask_compatibility.h"
#  include "dpl/core/concepts/simd_type.h"
#  include "dpl/core/dispatch/broadcastable/selection.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/operation/primitive.h"
#  include "dpl/core/immediate/const_mask.h"
#  include "dpl/core/immediate/constants/all_bits.h"
#  include "dpl/core/immediate/constants/zero.h"
#  include "dpl/core/type_traits/canonical_type.h"
#  include "dpl/core/type_traits/common_size_type.h"
#  include "dpl/std/concepts/integral.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void select(...) noexcept = delete;

struct select_t :
    public primitive_operation_base<select_t>,
    public selection_broadcastable_operation<select_t> {
    using operation_base<select_t>::operator();
    using selection_broadcastable_operation<select_t>::operator();
};

template <>
struct operation_signature<select_t> {
    template <simd_mask M, typename L, typename R>
    requires simd_type<L> || simd_type<R>
    static consteval void operator()(M&&, L&&, R&&) noexcept {}

    template <integral_constant_like M, typename L, typename R>
    requires (!simd_mask<M>) && (simd_type<L> || simd_type<R>)
    static consteval void operator()(M&&, L&&, R&&) noexcept {}
};

struct bwand_t;
struct bwandnot_t;
struct bwor_t;
struct bwornot_t;

template <>
struct fallback_impl<select_t> : selection_broadcasting_fallback<select_t> {
private:
    struct sbwand : operation_base<bwand_t> {
        using operation_base<bwand_t>::operator();
    };
    struct sbwandnot : operation_base<bwandnot_t> {
        using operation_base<bwandnot_t>::operator();
    };
    struct sbwor : operation_base<bwor_t> {
        using operation_base<bwor_t>::operator();
    };
    struct sbwornot : operation_base<bwornot_t> {
        using operation_base<bwornot_t>::operator();
    };

    template <typename L, typename R>
    using common_mask_t DPL_NODEBUG = make_canonical_mask_t<
        common_size_type_t<simd_element_type_t<L>, simd_element_type_t<R>>,
        common_abi_t<L, R>>;

public:
    using selection_broadcasting_fallback<select_t>::operator();

    template <canonical_vector T>
    requires fixed_width_abi<simd_abi_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        simd_mask_type_t<T> mask, T tval, T fval) noexcept {
        using E = simd_element_type_t<T>;
        return internal::transform<T>(
            [](bool cond, E tval, E fval) {
                return static_cast<E>(cond ? tval : fval);
            },
            mask, tval, fval);
    }

    template <canonical_mask L, common_mask_with<L> R>
    requires cpo_invocable<bwand_t, common_mask_t<L, R>, L> &&
        cpo_invocable<bwandnot_t, R, common_mask_t<L, R>> &&
        cpo_invocable<bwor_t, cpo_result_t<bwand_t, common_mask_t<L, R>, L>,
            cpo_result_t<bwandnot_t, R, common_mask_t<L, R>>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr common_mask_t<L, R>
        DPL_VECTORCALL operator()(
            common_mask_t<L, R> mask, L lhs, R rhs) noexcept {
        constexpr auto bwand = sbwand{};
        constexpr auto bwandnot = sbwandnot{};
        constexpr auto bwor = sbwor{};
        return bwor(bwand(mask, lhs), bwandnot(rhs, mask));
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires fixed_width_abi<simd_abi_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        M mask, T lhs, T rhs) noexcept {
        constexpr auto cmask = dx::to_const_mask<T>(mask);
        using E = simd_element_type_t<T>;
        return internal::itransform<T>(
            [cmask](auto idx, E lhs, E rhs) {
                return static_cast<E>(cmask[idx] ? lhs : rhs);
            },
            lhs, rhs);
    }

    template <canonical_mask L, equivalent_mask_with<L> R, const_mask_for<L> M>
    requires fixed_width_abi<simd_abi_type_t<L>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        M mask, L lhs, R rhs) noexcept {
        constexpr auto cmask = dx::to_const_mask<L>(mask);
        return internal::itransform<common_mask_t<L, R>>(
            [cmask](auto idx, bool lhs, bool rhs) {
                return cmask[idx] ? lhs : rhs;
            },
            lhs, rhs);
    }

    template <simd_mask L, equivalent_mask_with<L> M>
    requires cpo_invocable<bwand_t, M, L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M&& mask, L&& lhs, dx::zero_t) noexcept(
        canonical_mask<M> && canonical_mask<L>) {
        constexpr sbwand bwand;
        return bwand( __DPL forward<M>(mask), __DPL forward<L>(lhs));
    }

    template <simd_mask R, equivalent_mask_with<R> M>
    requires cpo_invocable<bwandnot_t, R, M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M&& mask, dx::zero_t, R&& rhs) noexcept(
        canonical_mask<M> && canonical_mask<R>) {
        constexpr sbwandnot bwandnot;
        return bwandnot( __DPL forward<R>(rhs), __DPL forward<M>(mask));
    }

    template <simd_mask L, equivalent_mask_with<L> M>
    requires cpo_invocable<bwornot_t, L, M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M&& mask, L&& lhs,
        dx::all_bits_t) noexcept(canonical_mask<M> && canonical_mask<L>) {
        constexpr sbwornot bwornot;
        return bwornot( __DPL forward<L>(lhs), __DPL forward<M>(mask));
    }

    template <simd_mask R, equivalent_mask_with<R> M>
    requires cpo_invocable<bwor_t, M, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M&& mask, dx::all_bits_t, R rhs) noexcept(
        canonical_mask<M> && canonical_mask<R>) {
        constexpr sbwor bwor;
        return bwor( __DPL forward<M>(mask), __DPL forward<R>(rhs));
    }

    template <simd_mask L, const_mask_for<L> M>
    requires cpo_invocable<bwand_t, canonical_type_t<L>, L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, L&& lhs, dx::zero_t) noexcept(
        canonical_mask<L>) {
        constexpr sbwand bwand;
        constexpr auto cmask = dx::to_const_mask<L>(mask);
        return bwand(
            static_cast<canonical_type_t<L>>(cmask), __DPL forward<L>(lhs));
    }

    template <simd_mask R, const_mask_for<R> M>
    requires cpo_invocable<bwandnot_t, R, canonical_type_t<R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, dx::zero_t, R&& rhs) noexcept(
        canonical_mask<R>) {
        constexpr sbwandnot bwandnot;
        constexpr auto cmask = dx::to_const_mask<R>(mask);
        return bwandnot(
            __DPL forward<R>(rhs), static_cast<canonical_type_t<R>>(cmask));
    }

    template <simd_mask L, const_mask_for<L> M>
    requires cpo_invocable<bwornot_t, L, canonical_type_t<L>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, L&& lhs, dx::all_bits_t) noexcept(
        canonical_mask<L>) {
        constexpr sbwornot bwornot;
        constexpr auto cmask = dx::to_const_mask<L>(mask);
        return bwornot(
            __DPL forward<L>(lhs), static_cast<canonical_type_t<L>>(cmask));
    }

    template <simd_mask R, const_mask_for<R> M>
    requires cpo_invocable<bwor_t, canonical_type_t<R>, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, dx::all_bits_t, R rhs) noexcept(
        canonical_mask<R>) {
        constexpr sbwor bwor;
        constexpr auto cmask = dx::to_const_mask<R>(mask);
        return bwor(
            static_cast<canonical_type_t<R>>(cmask), __DPL forward<R>(rhs));
    }
};

template <>
struct canonical_impl<select_t> {
private:
    template <typename L, typename R>
    using vresult_t DPL_NODEBUG =
        make_canonical_vector_t<simd_element_type_t<L>, common_abi_t<L, R>>;

    template <typename L, typename R>
    using vmask_t DPL_NODEBUG = simd_mask_type_t<vresult_t<L, R>>;

    template <typename L, typename R>
    using common_mask_t DPL_NODEBUG = make_canonical_mask_t<
        common_size_type_t<simd_element_type_t<L>, simd_element_type_t<R>>,
        common_abi_t<L, R>>;

public:
    template <canonical_vector L, common_vector_with<L> R>
    requires canonical_vector<R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr vresult_t<L, R> operator()(
        vmask_t<L, R> mask, L lhs, R rhs) noexcept
    requires requires {
        select(internal::abi<common_abi_t<L, R>>, mask, lhs, rhs);
    }
    {
        return select(internal::abi<common_abi_t<L, R>>, mask, lhs, rhs);
    }

    template <canonical_vector L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr L operator()(
        simd_mask_type_t<L> mask, L lhs, dx::zero_t zero) noexcept
    requires requires { select(internal::abi<L>, mask, lhs, zero); }
    {
        return select(internal::abi<L>, mask, lhs, zero);
    }

    template <canonical_vector R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr R operator()(
        simd_mask_type_t<R> mask, dx::zero_t zero, R rhs) noexcept
    requires requires { select(internal::abi<R>, mask, zero, rhs); }
    {
        return select(internal::abi<R>, mask, zero, rhs);
    }

    template <canonical_vector L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr L operator()(
        simd_mask_type_t<L> mask, L lhs, dx::all_bits_t all_bits) noexcept
    requires requires { select(internal::abi<L>, mask, lhs, all_bits); }
    {
        return select(internal::abi<L>, mask, lhs, all_bits);
    }

    template <canonical_vector R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr R operator()(
        simd_mask_type_t<R> mask, dx::all_bits_t all_bits, R rhs) noexcept
    requires requires { select(internal::abi<R>, mask, all_bits, rhs); }
    {
        return select(internal::abi<R>, mask, all_bits, rhs);
    }

    template <canonical_mask L, common_mask_with<L> R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr common_mask_t<L, R> operator()(
        common_mask_t<L, R> mask, L lhs, R rhs) noexcept
    requires requires {
        select(internal::abi<common_abi_t<L, R>>, mask, lhs, rhs);
    }
    {
        return select(internal::abi<common_abi_t<L, R>>, mask, lhs, rhs);
    }

    template <canonical_vector L, broadcastable_to<L> R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr L operator()(
        simd_mask_type_t<L> mask, L lhs, R&& rhs) noexcept
    requires requires {
        select(internal::abi<L>, mask, lhs, internal::declarg<R>());
    }
    {
        return select(internal::abi<L>, mask, lhs, __DPL forward<R>(rhs));
    }

    template <canonical_vector R, broadcastable_to<R> L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr R operator()(
        simd_mask_type_t<R> mask, L&& lhs, R rhs) noexcept
    requires requires {
        select(internal::abi<R>, mask, internal::declarg<L>(), rhs);
    }
    {
        return select(internal::abi<R>, mask, __DPL forward<L>(lhs), rhs);
    }
    ///

    template <canonical_vector L, common_vector_with<L> R,
        const_mask_for<vresult_t<L, R>> M>
    requires canonical_vector<R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr vresult_t<L, R> operator()(M mask, L lhs, R rhs) noexcept
    requires requires(launder_cmask_t<vresult_t<L, R>, M> cmask) {
        select(internal::abi<common_abi_t<L, R>>, cmask, lhs, rhs);
    }
    {
        constexpr auto cmask = dx::to_const_mask<vresult_t<L, R>>(mask);
        return select(internal::abi<common_abi_t<L, R>>, cmask, lhs, rhs);
    }

    template <canonical_mask L, common_mask_with<L> R,
        const_mask_for<common_mask_t<L, R>> M>
    requires canonical_mask<R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr common_mask_t<L, R> operator()(
        M mask, L lhs, R rhs) noexcept
    requires requires(launder_cmask_t<common_mask_t<L, R>, M> cmask) {
        select(internal::abi<common_abi_t<L, R>>, cmask, lhs, rhs);
    }
    {
        constexpr auto cmask = dx::to_const_mask<common_mask_t<L, R>>(mask);
        return select(internal::abi<common_abi_t<L, R>>, cmask, lhs, rhs);
    }

    template <canonical_vector L, const_mask_for<L> M, broadcastable_to<L> R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr L operator()(M mask, L lhs, R&& rhs) noexcept
    requires requires(launder_cmask_t<L, M> cmask) {
        select(internal::abi<L>, cmask, lhs, internal::declarg<R>());
    }
    {
        constexpr auto cmask = dx::to_const_mask<L>(mask);
        return select(internal::abi<L>, cmask, lhs, __DPL forward<R>(rhs));
    }

    template <canonical_vector R, const_mask_for<R> M, broadcastable_to<R> L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr R operator()(M mask, L&& lhs, R rhs) noexcept
    requires requires(launder_cmask_t<R, M> cmask) {
        select(internal::abi<R>, cmask, internal::declarg<L>(), rhs);
    }
    {
        constexpr auto cmask = dx::to_const_mask<R>(mask);
        return select(internal::abi<R>, cmask, __DPL forward<L>(lhs), rhs);
    }

    template <canonical_simd_type L, const_mask_for<L> M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr L operator()(M mask, L lhs, dx::zero_t zero) noexcept
    requires requires(launder_cmask_t<L, M> cmask) {
        select(internal::abi<L>, cmask, lhs, zero);
    }
    {
        constexpr auto cmask = dx::to_const_mask<L>(mask);
        return select(internal::abi<L>, cmask, lhs, zero);
    }

    template <canonical_simd_type R, const_mask_for<R> M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr R operator()(M mask, dx::zero_t zero, R rhs) noexcept
    requires requires(launder_cmask_t<R, M> cmask) {
        select(internal::abi<R>, cmask, zero, rhs);
    }
    {
        constexpr auto cmask = dx::to_const_mask<R>(mask);
        return select(internal::abi<R>, cmask, zero, rhs);
    }

    template <canonical_simd_type L, const_mask_for<L> M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr L operator()(
        M mask, L lhs, dx::all_bits_t all_bits) noexcept
    requires requires(launder_cmask_t<L, M> cmask) {
        select(internal::abi<L>, cmask, lhs, all_bits);
    }
    {
        constexpr auto cmask = dx::to_const_mask<L>(mask);
        return select(internal::abi<L>, cmask, lhs, all_bits);
    }

    template <canonical_simd_type R, const_mask_for<R> M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr R operator()(
        M mask, dx::all_bits_t all_bits, R rhs) noexcept
    requires requires(launder_cmask_t<R, M> cmask) {
        select(internal::abi<R>, cmask, all_bits, rhs);
    }
    {
        constexpr auto cmask = dx::to_const_mask<R>(mask);
        return select(internal::abi<R>, cmask, all_bits, rhs);
    }
};

template <typename T, typename L, typename R, typename A = common_abi_t<L, R>>
concept selection_result = (!simd_type<L> || common_simd_type_with<T, L>) &&
    (!simd_type<R> || common_simd_type_with<T, R>) &&
    simd_type_with_common_abi<T, A>;

template <typename M, typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_extended_select =
    (!simd_type<M> || (simd_mask<M> && same_abi_as<simd_abi_type_t<M>, A>)) &&
    requires {
        {
            select(internal::declarg<M>(), internal::declarg<R>(),
                internal::declarg<L>())
        } -> selection_result<L, R, A>;
    };

template <>
struct extended_impl<select_t> {
private:
    template <typename L, typename R>
    using common_mask_t DPL_NODEBUG = make_canonical_mask_t<
        common_size_type_t<simd_element_type_t<L>, simd_element_type_t<R>>,
        common_abi_t<L, R>>;

    template <typename L, typename R>
    using vresult_t DPL_NODEBUG =
        make_canonical_vector_t<simd_element_type_t<L>, common_abi_t<L, R>>;

public:
    template <simd_vector L, common_vector_with<L> R,
        exact_mask_for<vresult_t<L, R>> M>
    requires (extended_mask<M> || extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_select<M, L, R>
        DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M&& mask, L&& lhs, R&& rhs)
        -> common_vector_with<vresult_t<L, R>> auto {
        return select(__DPL forward<M>(mask), __DPL forward<L>(lhs),
            __DPL forward<R>(rhs));
    }

    template <simd_vector L, exact_mask_for<L> M>
    requires (extended_mask<M> || extended_vector<L>) &&
        unqualified_extended_select<M, L, dx::zero_t, common_abi_t<M, L>>
        DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M&& mask, L&& lhs, dx::zero_t zero)
        -> common_vector_with<vresult_t<L, M>> auto {
        return select(__DPL forward<M>(mask), __DPL forward<L>(lhs), zero);
    }

    template <simd_vector R, exact_mask_for<R> M>
    requires (extended_mask<M> || extended_vector<R>) &&
        unqualified_extended_select<M, dx::zero_t, R, common_abi_t<M, R>>
        DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M&& mask, dx::zero_t zero, R&& rhs)
        -> common_vector_with<vresult_t<R, M>> auto {
        return select(__DPL forward<M>(mask), zero, __DPL forward<R>(rhs));
    }

    template <simd_vector L, exact_mask_for<L> M>
    requires (extended_mask<M> || extended_vector<L>) &&
        unqualified_extended_select<M, L, dx::all_bits_t, common_abi_t<M, L>>
        DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M&& mask, L&& lhs, dx::all_bits_t all_bits)
        -> common_vector_with<vresult_t<L, M>> auto {
        return select(__DPL forward<M>(mask), __DPL forward<L>(lhs), all_bits);
    }

    template <simd_vector R, exact_mask_for<R> M>
    requires (extended_mask<M> || extended_vector<R>) &&
        unqualified_extended_select<M, dx::all_bits_t, R, common_abi_t<M, R>>
        DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M&& mask, dx::all_bits_t all_bits, R&& rhs)
        -> common_vector_with<vresult_t<R, M>> auto {
        return select(__DPL forward<M>(mask), all_bits, __DPL forward<R>(rhs));
    }

    template <simd_mask L, common_mask_with<L> R,
        exact_mask_for<common_mask_t<L, R>> M>
    requires (extended_mask<M> || extended_mask<L> || extended_mask<R>) &&
        unqualified_extended_select<M, L, R>
        DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M&& mask, L&& lhs, R&& rhs)
        -> common_mask_with<common_mask_t<L, R>> auto {
        return select(__DPL forward<M>(mask), __DPL forward<L>(lhs),
            __DPL forward<R>(rhs));
    }

    template <simd_mask L, exact_mask_for<L> M>
    requires (extended_mask<M> || extended_mask<L>) &&
        unqualified_extended_select<M, L, dx::zero_t, common_abi_t<M, L>>
        DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M&& mask, L&& lhs, dx::zero_t zero)
        -> common_mask_with<L> auto {
        return select(__DPL forward<M>(mask), __DPL forward<L>(lhs), zero);
    }

    template <simd_mask R, exact_mask_for<R> M>
    requires (extended_mask<M> || extended_mask<R>) &&
        unqualified_extended_select<M, dx::zero_t, R, common_abi_t<M, R>>
        DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M&& mask, dx::zero_t zero, R&& rhs)
        -> common_mask_with<R> auto {
        return select(__DPL forward<M>(mask), zero, __DPL forward<R>(rhs));
    }

    template <simd_mask L, exact_mask_for<L> M>
    requires (extended_mask<M> || extended_mask<L>) &&
        unqualified_extended_select<M, L, dx::all_bits_t, common_abi_t<M, L>>
        DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M&& mask, L&& lhs, dx::all_bits_t all_bits)
        -> common_mask_with<L> auto {
        return select(__DPL forward<M>(mask), __DPL forward<L>(lhs), all_bits);
    }

    template <simd_mask R, exact_mask_for<R> M>
    requires (extended_mask<M> || extended_mask<R>) &&
        unqualified_extended_select<M, dx::all_bits_t, R, common_abi_t<M, R>>
        DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M&& mask, dx::all_bits_t all_bits, R&& rhs)
        -> common_mask_with<R> auto {
        return select(__DPL forward<M>(mask), all_bits, __DPL forward<R>(rhs));
    }

    template <simd_vector L, exact_mask_for<L> M, broadcastable_to<L> R>
    requires (extended_mask<M> || extended_vector<L>) &&
        unqualified_extended_select<M, L, R, common_abi_t<M, L>>
        DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M&& mask, L&& lhs, R&& rhs)
        -> common_vector_with<L> auto {
        return select(__DPL forward<M>(mask), __DPL forward<L>(lhs),
            __DPL forward<R>(rhs));
    }

    template <simd_vector R, exact_mask_for<R> M, broadcastable_to<R> L>
    requires (extended_mask<M> || extended_vector<R>) &&
        unqualified_extended_select<M, L, R, common_abi_t<M, R>>
        DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M&& mask, L&& lhs, R&& rhs)
        -> common_vector_with<R> auto {
        return select(__DPL forward<M>(mask), __DPL forward<L>(lhs),
            __DPL forward<R>(rhs));
    }
    ///

    template <simd_vector L, common_vector_with<L> R,
        const_mask_for<vresult_t<L, R>> M>
    requires (extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_select<launder_cmask_t<vresult_t<L, R>, M>, L, R>
        DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, L&& lhs, R&& rhs)
        -> common_vector_with<vresult_t<L, R>> auto {
        constexpr auto cmask = dx::to_const_mask<vresult_t<L, R>>(mask);
        return select(cmask, __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <extended_vector L, const_mask_for<L> M>
    requires unqualified_extended_select<launder_cmask_t<L, M>, L, dx::zero_t,
        simd_abi_type_t<L>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, L&& lhs, dx::zero_t zero)
        -> common_vector_with<L> auto {
        constexpr auto cmask = dx::to_const_mask<L>(mask);
        return select(cmask, __DPL forward<L>(lhs), zero);
    }

    template <extended_vector R, const_mask_for<R> M>
    requires unqualified_extended_select<launder_cmask_t<R, M>, dx::zero_t, R,
        simd_abi_type_t<R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, dx::zero_t zero, R&& rhs)
        -> common_vector_with<R> auto {
        constexpr auto cmask = dx::to_const_mask<R>(mask);
        return select(cmask, zero, __DPL forward<R>(rhs));
    }

    template <extended_vector L, const_mask_for<L> M>
    requires unqualified_extended_select<launder_cmask_t<L, M>, L,
        dx::all_bits_t, simd_abi_type_t<L>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, L&& lhs, dx::all_bits_t all_bits)
        -> common_vector_with<L> auto {
        constexpr auto cmask = dx::to_const_mask<L>(mask);
        return select(cmask, __DPL forward<L>(lhs), all_bits);
    }

    template <extended_vector R, const_mask_for<R> M>
    requires unqualified_extended_select<launder_cmask_t<R, M>, dx::all_bits_t,
        R, simd_abi_type_t<R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, dx::all_bits_t all_bits, R&& rhs)
        -> common_vector_with<R> auto {
        constexpr auto cmask = dx::to_const_mask<R>(mask);
        return select(cmask, all_bits, __DPL forward<R>(rhs));
    }

    template <extended_mask L, const_mask_for<L> M>
    requires unqualified_extended_select<launder_cmask_t<L, M>, L, dx::zero_t,
        simd_abi_type_t<L>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, L&& lhs, dx::zero_t zero)
        -> common_mask_with<L> auto {
        constexpr auto cmask = dx::to_const_mask<L>(mask);
        return select(cmask, __DPL forward<L>(lhs), zero);
    }

    template <extended_mask R, const_mask_for<R> M>
    requires unqualified_extended_select<launder_cmask_t<R, M>, dx::zero_t, R,
        simd_abi_type_t<R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, dx::zero_t zero, R&& rhs)
        -> common_mask_with<R> auto {
        constexpr auto cmask = dx::to_const_mask<R>(mask);
        return select(cmask, zero, __DPL forward<R>(rhs));
    }

    template <extended_mask L, const_mask_for<L> M>
    requires unqualified_extended_select<launder_cmask_t<L, M>, L,
        dx::all_bits_t, simd_abi_type_t<L>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, L&& lhs, dx::all_bits_t all_bits)
        -> common_mask_with<L> auto {
        constexpr auto cmask = dx::to_const_mask<L>(mask);
        return select(cmask, __DPL forward<L>(lhs), all_bits);
    }

    template <extended_mask R, const_mask_for<R> M>
    requires unqualified_extended_select<launder_cmask_t<R, M>, dx::all_bits_t,
        R, simd_abi_type_t<R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, dx::all_bits_t all_bits, R&& rhs)
        -> common_mask_with<R> auto {
        constexpr auto cmask = dx::to_const_mask<R>(mask);
        return select(cmask, all_bits, __DPL forward<R>(rhs));
    }

    template <simd_mask L, common_mask_with<L> R,
        const_mask_for<common_mask_t<L, R>> M>
    requires (extended_mask<L> || extended_mask<R>) &&
        unqualified_extended_select<launder_cmask_t<common_mask_t<L, R>, M>, L,
            R>
        DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, L&& lhs, R&& rhs)
        -> common_mask_with<common_mask_t<L, R>> auto {
        constexpr auto cmask = dx::to_const_mask<common_mask_t<L, R>>(mask);
        return select(cmask, __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <extended_vector L, const_mask_for<L> M, broadcastable_to<L> R>
    requires unqualified_extended_select<launder_cmask_t<L, M>, L, R,
        simd_abi_type_t<L>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, L&& lhs, R&& rhs)
        -> common_vector_with<L> auto {
        constexpr auto cmask = dx::to_const_mask<L>(mask);
        return select(internal::abi<L>, cmask, __DPL forward<L>(lhs),
            __DPL forward<R>(rhs));
    }

    template <extended_vector R, const_mask_for<R> M, broadcastable_to<R> L>
    requires unqualified_extended_select<launder_cmask_t<R, M>, L, R,
        simd_abi_type_t<R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, L&& lhs, R&& rhs)
        -> common_vector_with<R> auto {
        constexpr auto cmask = dx::to_const_mask<R>(mask);
        return select(internal::abi<R>, cmask, __DPL forward<L>(lhs),
            __DPL forward<R>(rhs));
    }
};

template <auto V>
struct selecti_t {
    template <typename L, typename R>
    requires requires { typename cmask_t<V>; } &&
        cpo_invocable<select_t, cmask_t<V>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R&& rhs) noexcept {
        return select_t::operator()(
            cmask_v<V>, __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
template <integral auto V>
inline constexpr internal::selecti_t<V> selecti{};
inline constexpr internal::select_t select{};
} // namespace cpo
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
