// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

// IWYU pragma: always_keep

#include "dpl/core/operations/internal/transform.h"
#include "dpl/core/operations/pack_mask.h"

#if !DPL_MODULES
#  include "dpl/core/basic/initialize.h"
#  include "dpl/core/basic/internal/abi.h"
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/concepts/mask_compatibility.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/dispatch/broadcastable/binary.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/maskable/transform.h"
#  include "dpl/core/dispatch/operation/primitive.h"
#  include "dpl/std/bit/char_bit.h"
#  include "dpl/std/utility/bitset.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void bwand(...) noexcept = delete;

struct DPL_EMPTY_BASES bwand_t :
    private bitwise_base<bwand_t>,
    private maskable_transform_base<bwand_t>,
    private binary_broadcastable_operation<bwand_t> {
    using bitwise_base<bwand_t>::operator();
    using maskable_transform_base<bwand_t>::operator();
};

template <>
struct operation_signature<bwand_t> {
    template <typename L, typename R>
    requires simd_type<L> || simd_type<R>
    static consteval void operator()(L&&, R&&) noexcept {}
};

template <>
struct fallback_impl<bwand_t> : binary_broadcasting_fallback<bwand_t> {

    template <typename E>
    using bitset_t DPL_NODEBUG = bitset<sizeof(E) * char_bit_v>;

    template <fixed_width_abi A, simd_element_for<A> E>
    requires (sizeof(bitset_t<E>) == sizeof(E))
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_vector<E, A>
        DPL_VECTORCALL operator()(
            basic_vector<E, A> lhs, basic_vector<E, A> rhs) noexcept {
        return internal::transform<basic_vector<E, A>>(
            [](auto lhs, auto rhs) {
                auto promoted = __DPL bit_cast<bitset_t<E>>(lhs) &
                    __DPL bit_cast<bitset_t<E>>(rhs);
                return __DPL bit_cast<E>(promoted.reinitialize());
            },
            lhs, rhs);
    }

    template <fixed_width_abi A, simd_element_for<A> LE, simd_element_for<A> RE>
    requires common_size_with<LE, RE> &&
        invocable<pack_mask_t, basic_mask<LE, A>> &&
        invocable<pack_mask_t, basic_mask<RE, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        basic_mask<LE, A> lhs, basic_mask<RE, A> rhs) noexcept {
        using T = common_size_type_t<LE, RE>;
        return dx::initialize<T, A>(dx::pack_mask(lhs) & dx::pack_mask(rhs));
    }

    using binary_broadcasting_fallback<bwand_t>::operator();
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_canonical_bwand = requires {
    {
        bwand(internal::abi<A>, internal::declarg<L>(), internal::declarg<R>())
    } -> canonical_vector;
};

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<L, R>>
concept unqualified_canonical_mbwand =
    (!simd_type<S> || equivalent_vector_with<S, cpo_result_t<bwand_t, L, R>>) &&
    requires {
        {
            bwand(internal::abi<A>, internal::declarg<S>(),
                internal::declarg<M>(), internal::declarg<L>(),
                internal::declarg<R>())
        } -> equivalent_vector_with<cpo_result_t<bwand_t, L, R>>;
    };

template <>
struct canonical_impl<bwand_t> {
private:
    template <typename L, typename R>
    using source_t DPL_NODEBUG =
        basic_vector<simd_element_type_t<L>, common_abi_t<L, R>>;

    template <typename L, typename R>
    using mask_t DPL_NODEBUG =
        basic_mask<simd_element_type_t<L>, common_abi_t<L, R>>;

    template <typename L, typename R>
    using imask_t DPL_NODEBUG = mask_value_t<
        simd_abi_traits<simd_element_type_t<L>, common_abi_t<L, R>>::size>;

    template <typename L, typename R, imask_t<L, R> M>
    using cmask_t DPL_NODEBUG = const_mask<
        simd_abi_traits<simd_element_type_t<L>, common_abi_t<L, R>>::size, M>;

public:
    template <simd_abi LA, common_abi_with<LA> RA, simd_element_for<LA> E,
        simd_abi A = common_abi_t<LA, RA>>
    requires simd_element_for<E, RA>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, LA> lhs, basic_vector<E, RA> rhs) noexcept
    requires requires { bwand(internal::abi<A>, lhs, rhs); }
    {
        return bwand(internal::abi<A>, lhs, rhs);
    }

    template <simd_abi LA, common_abi_with<LA> RA, simd_element_for<LA> LE,
        simd_element_for<RA> RE, simd_abi A = common_abi_t<LA, RA>>
    requires common_size_with<LE, RE>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<common_size_type_t<LE, RE>, A> operator()(
        basic_mask<LE, LA> lhs, basic_mask<RE, RA> rhs) noexcept
    requires requires { bwand(internal::abi<A>, lhs, rhs); }
    {
        return bwand(internal::abi<A>, lhs, rhs);
    }

    template <canonical_vector L, broadcastable_to<L> R>
    requires unqualified_canonical_bwand<L, R, simd_abi_type_t<L>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R&& rhs) noexcept {
        return bwand(internal::abi<L>, lhs, __DPL forward<R>(rhs));
    }

    template <canonical_vector R, broadcastable_to<R> L>
    requires unqualified_canonical_bwand<L, R, simd_abi_type_t<R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R rhs) noexcept {
        return bwand(internal::abi<R>, __DPL forward<L>(lhs), rhs);
    }

    template <canonical_vector L, common_vector_with<L> R>
    requires canonical_vector<R> &&
        unqualified_canonical_mbwand<source_t<L, R>, mask_t<L, R>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        source_t<L, R> src, mask_t<L, R> mask, L lhs, R rhs) noexcept {
        return bwand(internal::abi<common_abi_t<L, R>>, src, mask, lhs, rhs);
    }

    template <canonical_vector L, common_vector_with<L> R, imask_t<L, R> M>
    requires canonical_vector<R> &&
        unqualified_canonical_mbwand<source_t<L, R>, cmask_t<L, R, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        source_t<L, R> src, cmask_t<L, R, M> cmask, L lhs, R rhs) noexcept {
        return bwand(internal::abi<common_abi_t<L, R>>, src, cmask, lhs, rhs);
    }

    template <canonical_vector L, common_vector_with<L> R>
    requires canonical_vector<R> &&
        unqualified_canonical_mbwand<dx::zero_t, mask_t<L, R>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, mask_t<L, R> mask, L lhs, R rhs) noexcept {
        return bwand(internal::abi<common_abi_t<L, R>>, zero, mask, lhs, rhs);
    }

    template <canonical_vector L, common_vector_with<L> R, imask_t<L, R> M>
    requires canonical_vector<R> &&
        unqualified_canonical_mbwand<dx::zero_t, cmask_t<L, R, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, cmask_t<L, R, M> cmask, L lhs, R rhs) noexcept {
        return bwand(internal::abi<common_abi_t<L, R>>, zero, cmask, lhs, rhs);
    }
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_extended_mask_bwand = requires {
    {
        bwand(internal::declarg<L>(), internal::declarg<R>())
    } -> mask_with_common_abi<A>;
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_extended_bwand = requires {
    {
        bwand(internal::declarg<L>(), internal::declarg<R>())
    } -> vector_with_common_abi<A>;
};

template <typename S, typename M, typename L, typename R>
concept unqualified_extended_mbwand =
    (!simd_type<S> || equivalent_vector_with<S, cpo_result_t<bwand_t, L, R>>) &&
    requires {
        {
            bwand(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<L>(), internal::declarg<R>())
        } -> equivalent_vector_with<cpo_result_t<bwand_t, L, R>>;
    };

template <>
struct extended_impl<bwand_t> {
private:
    template <typename S>
    using simask_t DPL_NODEBUG = mask_value_t<simd_abi_traits<S>::size>;

    template <typename S, simask_t<S> V>
    using scmask_t DPL_NODEBUG = const_mask<simd_abi_traits<S>::size, V>;

    template <typename L, typename R>
    using imask_t DPL_NODEBUG = simask_t<cpo_result_t<bwand_t, L, R>>;

    template <typename L, typename R, imask_t<L, R> M>
    using cmask_t DPL_NODEBUG = scmask_t<cpo_result_t<bwand_t, L, R>, M>;

    template <typename L, typename R>
    using mask_t DPL_NODEBUG =
        basic_mask<simd_element_type_t<L>, common_abi_t<L, R>>;

public:
    template <simd_mask L, simd_mask R>
    requires (extended_mask<L> || extended_mask<R>) &&
        unqualified_extended_mask_bwand<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R&& rhs) {
        return bwand(__DPL forward<L>(lhs), __DPL forward<L>(rhs));
    }

    template <simd_vector L, simd_vector R>
    requires (extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_bwand<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R&& rhs) {
        return bwand(__DPL forward<L>(lhs), __DPL forward<L>(rhs));
    }

    template <typename L, typename R>
    requires (extended_vector<L> &&
                 broadcastable_to<R, result_or_decayed_t<L>> &&
                 unqualified_extended_bwand<L, R, simd_abi_type_t<L>>) ||
        (extended_vector<R> && broadcastable_to<L, result_or_decayed_t<R>> &&
            unqualified_extended_bwand<L, R, simd_abi_type_t<R>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R&& rhs) {
        return bwand(__DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <simd_vector S, exact_mask_for<S> M, common_vector_with<S> L,
        common_vector_with<L> R>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<L> ||
                 extended_vector<R>) &&
        unqualified_extended_mbwand<S, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, L&& lhs, R&& rhs) {
        return bwand(__DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <fixed_width_vector S, simask_t<S> M, common_vector_with<S> L,
        common_vector_with<L> R>
    requires (extended_vector<S> || extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_mbwand<S, scmask_t<S, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        S&& src, scmask_t<S, M> cmask, L&& lhs, R&& rhs) {
        return bwand(
            src, cmask, __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <simd_vector L, common_vector_with<L> R,
        common_mask_with<mask_t<L, R>> M>
    requires (extended_mask<M> || extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_mbwand<dx::zero_t, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, L&& lhs, R&& rhs) {
        return bwand(zero, __DPL forward<M>(mask), __DPL forward<L>(lhs),
            __DPL forward<R>(rhs));
    }

    template <simd_vector L, common_vector_with<L> R, imask_t<L, R> M>
    requires (extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_mbwand<dx::zero_t, cmask_t<L, R, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, cmask_t<L, R, M> cmask, L&& lhs, R&& rhs) {
        return bwand(
            zero, cmask, __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::bwand_t bwand{};
} // namespace cpo
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
