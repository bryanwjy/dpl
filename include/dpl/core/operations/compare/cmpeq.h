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
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/dispatch/broadcastable/binary.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/maskable/predicate.h"
#  include "dpl/core/dispatch/operation/primitive.h"
#  include "dpl/core/type_traits/common_size_type.h"
#  include "dpl/std/concepts/equality_comparable.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void cmpeq(...) noexcept = delete;

struct DPL_EMPTY_BASES cmpeq_t :
    private comparison_base<cmpeq_t>,
    private maskable_predicate_base<cmpeq_t>,
    private binary_broadcastable_operation<cmpeq_t> {
    using operation_base<cmpeq_t>::operator();
    using maskable_predicate_base<cmpeq_t>::operator();
    using binary_broadcastable_operation<cmpeq_t>::operator();
};

template <>
struct operation_signature<cmpeq_t> {
    template <typename L, typename R>
    requires simd_type<L> || simd_type<R>
    static consteval void operator()(L&&, R&&) noexcept {}
};

template <>
struct fallback_impl<cmpeq_t> : binary_broadcasting_fallback<cmpeq_t> {

    template <fixed_width_abi A, simd_element_for<A> E>
    requires equality_comparable<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_mask<E, A>
        DPL_VECTORCALL operator()(
            basic_vector<E, A> lhs, basic_vector<E, A> rhs) noexcept {
        return internal::transform<basic_mask<E, A>>(
            [](auto lhs, auto rhs) -> bool { return lhs == rhs; }, lhs, rhs);
    }

    template <fixed_width_abi A, simd_element_for<A> LE, simd_element_for<A> RE>
    requires common_size_with<LE, RE> &&
        cpo_invocable<pack_mask_t, basic_mask<LE, A>> &&
        cpo_invocable<pack_mask_t, basic_mask<RE, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        basic_mask<LE, A> lhs, basic_mask<RE, A> rhs) noexcept {
        using T = common_size_type_t<LE, RE>;
        return dx::initialize<T, A>(
            (dx::pack_mask(lhs) ^ dx::pack_mask(rhs)).invert());
    }

    using binary_broadcasting_fallback<cmpeq_t>::operator();
};

template <typename L, typename R, typename A = common_abi_t<L, R>,
    typename E = simd_element_type_t<L>>
concept unqualified_canonical_cmpeq = requires {
    {
        cmpeq(internal::abi<A>, internal::declarg<L>(), internal::declarg<R>())
    } -> equivalent_mask_with<basic_mask<simd_element_type_t<L>, A>>;
};

template <typename S, typename L, typename R>
concept unqualified_canonical_mcmpeq = canonical_vector<S> &&
    (!simd_mask<S> || same_as<S, cpo_result_t<cmpeq_t, L, R>>) && requires {
        {
            cmpeq(internal::abi<S>, internal::declarg<S>(),
                internal::declarg<L>(), internal::declarg<R>())
        } -> same_as<cpo_result_t<cmpeq_t, L, R>>;
    };

template <>
struct canonical_impl<cmpeq_t> {
private:
    template <typename L, typename R>
    using result_t DPL_NODEBUG =
        basic_mask<simd_element_type_t<L>, common_abi_t<L, R>>;

    template <typename L, typename R>
    using mresult_t DPL_NODEBUG = basic_mask<
        common_size_type_t<simd_element_type_t<L>, simd_element_type_t<R>>,
        common_abi_t<L, R>>;

    template <typename L, typename R>
    using mask_t DPL_NODEBUG = cpo_result_t<cmpeq_t, L, R>;

    template <typename L, typename R>
    using imask_t DPL_NODEBUG = mask_value_t<
        simd_abi_traits<simd_element_type_t<L>, common_abi_t<L, R>>::size>;

    template <typename L, typename R, imask_t<L, R> V>
    using cmask_t DPL_NODEBUG = const_mask<
        simd_abi_traits<simd_element_type_t<L>, common_abi_t<L, R>>::size, V>;

public:
    template <canonical_vector L, common_vector_with<L> R>
    requires unqualified_canonical_cmpeq<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<L, R> operator()(L lhs, R rhs) noexcept {
        return cmpeq(internal::abi<common_abi_t<L, R>>, lhs, rhs);
    }

    template <canonical_mask L, common_mask_with<L> R>
    requires unqualified_canonical_cmpeq<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr mresult_t<L, R> operator()(L lhs, R rhs) noexcept {
        return cmpeq(internal::abi<common_abi_t<L, R>>, lhs, rhs);
    }

    template <canonical_vector L, broadcastable_to<L> R>
    requires unqualified_canonical_cmpeq<L, R, simd_abi_type_t<L>,
        simd_element_type_t<L>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<L, L> operator()(L lhs, R&& rhs) noexcept {
        return cmpeq(internal::abi<L>, lhs, __DPL forward<R>(rhs));
    }

    template <canonical_vector R, broadcastable_to<R> L>
    requires unqualified_canonical_cmpeq<L, R, simd_abi_type_t<R>,
        simd_element_type_t<R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<R, R> operator()(L&& lhs, R rhs) noexcept {
        return cmpeq(internal::abi<R>, __DPL forward<L>(lhs), rhs);
    }

    template <canonical_vector L, common_vector_with<L> R>
    requires canonical_vector<R> &&
        unqualified_canonical_mcmpeq<mask_t<L, R>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(mask_t<L, R> src, L lhs, R rhs) noexcept {
        return cmpeq(internal::abi<common_abi_t<L, R>>, src, lhs, rhs);
    }

    template <canonical_vector L, common_vector_with<L> R, imask_t<L, R> M>
    requires canonical_vector<R> &&
        unqualified_canonical_mcmpeq<cmask_t<L, R, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        cmask_t<L, R, M> cmask, L lhs, R rhs) noexcept {
        return cmpeq(internal::abi<common_abi_t<L, R>>, cmask, lhs, rhs);
    }
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_extended_cmpeq = requires(L lhs, R rhs) {
    { cmpeq(lhs, rhs) } -> mask_with_common_abi<A>;
};

template <typename S, typename L, typename R>
concept unqualified_extended_mcmpeq =
    (!simd_mask<S> || equivalent_mask_with<S, cpo_result_t<cmpeq_t, L, R>>) &&
    requires {
        {
            cmpeq(internal::declarg<S>(), internal::declarg<L>(),
                internal::declarg<R>())
        } -> equivalent_mask_with<cpo_result_t<cmpeq_t, L, R>>;
    };

template <>
struct extended_impl<cmpeq_t> {
private:
    template <typename L, typename R>
    using imask_t DPL_NODEBUG = mask_value_t<
        simd_abi_traits<simd_element_type_t<L>, common_abi_t<L, R>>::size>;

    template <typename L, typename R, imask_t<L, R> V>
    using cmask_t DPL_NODEBUG = const_mask<
        simd_abi_traits<simd_element_type_t<L>, common_abi_t<L, R>>::size, V>;

public:
    template <simd_vector L, common_vector_with<L> R>
    requires (extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_cmpeq<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R&& rhs) {
        return cmpeq(__DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <simd_mask L, common_mask_with<L> R>
    requires (extended_mask<L> || extended_mask<R>) &&
        unqualified_extended_cmpeq<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R&& rhs) {
        return cmpeq(__DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <typename L, typename R>
    requires (extended_vector<L> &&
                 broadcastable_to<R, result_or_decayed_t<L>> &&
                 unqualified_extended_cmpeq<L, R, simd_abi_type_t<L>>) ||
        (extended_vector<R> && broadcastable_to<L, result_or_decayed_t<R>> &&
            unqualified_extended_cmpeq<L, R, simd_abi_type_t<R>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R&& rhs) {
        return cmpeq(__DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <simd_mask S, simd_vector L, common_vector_with<L> R>
    requires (extended_mask<S> || extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_mcmpeq<S, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, L&& lhs, R&& rhs) {
        return cmpeq(__DPL forward<S>(src), __DPL forward<L>(lhs),
            __DPL forward<R>(rhs));
    }

    template <simd_vector L, common_vector_with<L> R, imask_t<L, R> M>
    requires (extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_mcmpeq<cmask_t<L, R, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(cmask_t<L, R, M> cmask, L&& lhs, R&& rhs) {
        return cmpeq(cmask, __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::cmpeq_t cmpeq{};
}
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
