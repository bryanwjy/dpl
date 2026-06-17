// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

// IWYU pragma: always_keep

#include "dpl/core/operations/compare/cmplt.h"

#if !DPL_MODULES
#  include "dpl/core/basic/internal/abi.h"
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/dispatch/broadcastable/binary.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/maskable/predicate.h"
#  include "dpl/core/dispatch/operation/primitive.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void cmpgt(...) noexcept = delete;

struct DPL_EMPTY_BASES cmpgt_t :
    private comparison_base<cmpgt_t>,
    private maskable_predicate_base<cmpgt_t>,
    private binary_broadcastable_operation<cmpgt_t> {
    using operation_base<cmpgt_t>::operator();
    using maskable_predicate_base<cmpgt_t>::operator();
    using binary_broadcastable_operation<cmpgt_t>::operator();
};

template <>
struct operation_signature<cmpgt_t> {
    template <typename L, typename R>
    requires simd_vector<L> || simd_vector<R>
    static consteval void operator()(L&&, R&&) noexcept {}
};

template <>
struct fallback_impl<cmpgt_t> : binary_broadcasting_fallback<cmpgt_t> {

    template <simd_abi LA, common_abi_with<LA> RA, simd_element_for<LA> E,
        typename A = common_abi_t<LA, RA>>
    requires simd_element_for<E, RA> &&
        cpo_invocable<cmplt_t, basic_vector<E, LA>, basic_vector<E, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_mask<E, A>
        DPL_VECTORCALL operator()(
            basic_vector<E, LA> lhs, basic_vector<E, RA> rhs) noexcept {
        return dx::cmplt(rhs, lhs);
    }

    using binary_broadcasting_fallback<cmpgt_t>::operator();
};
template <typename L, typename R, typename A = common_abi_t<L, R>,
    typename E = simd_element_type_t<L>>
concept unqualified_canonical_cmpgt = requires {
    {
        cmpgt(internal::abi<A>, internal::declarg<L>(), internal::declarg<R>())
    } -> equivalent_mask_with<basic_mask<simd_element_type_t<L>, A>>;
};

template <typename S, typename L, typename R>
concept unqualified_canonical_mcmpgt = canonical_vector<S> &&
    (!simd_mask<S> || same_as<S, cpo_result_t<cmpgt_t, L, R>>) && requires {
        {
            cmpgt(internal::abi<S>, internal::declarg<S>(),
                internal::declarg<L>(), internal::declarg<R>())
        } -> same_as<cpo_result_t<cmpgt_t, L, R>>;
    };

template <>
struct canonical_impl<cmpgt_t> {
private:
    template <typename L, typename R>
    using result_t DPL_NODEBUG =
        basic_mask<simd_element_type_t<L>, common_abi_t<L, R>>;

    template <typename L, typename R>
    using mresult_t DPL_NODEBUG = basic_mask<
        common_size_type_t<simd_element_type_t<L>, simd_element_type_t<R>>,
        common_abi_t<L, R>>;

    template <typename L, typename R>
    using mask_t DPL_NODEBUG = cpo_result_t<cmpgt_t, L, R>;

    template <typename L, typename R>
    using imask_t DPL_NODEBUG = mask_value_t<
        simd_abi_traits<simd_element_type_t<L>, common_abi_t<L, R>>::size>;

    template <typename L, typename R, imask_t<L, R> V>
    using cmask_t DPL_NODEBUG = const_mask<
        simd_abi_traits<simd_element_type_t<L>, common_abi_t<L, R>>::size, V>;

public:
    template <canonical_vector L, common_vector_with<L> R>
    requires unqualified_canonical_cmpgt<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<L, R> operator()(L lhs, R rhs) noexcept {
        return cmpgt(internal::abi<common_abi_t<L, R>>, lhs, rhs);
    }

    template <canonical_vector L, broadcastable_to<L> R>
    requires unqualified_canonical_cmpgt<L, R, simd_abi_type_t<L>,
        simd_element_type_t<L>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<L, L> operator()(L lhs, R&& rhs) noexcept {
        return cmpgt(internal::abi<L>, lhs, __DPL forward<R>(rhs));
    }

    template <canonical_vector R, broadcastable_to<R> L>
    requires unqualified_canonical_cmpgt<L, R, simd_abi_type_t<R>,
        simd_element_type_t<R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<R, R> operator()(L&& lhs, R rhs) noexcept {
        return cmpgt(internal::abi<R>, __DPL forward<L>(lhs), rhs);
    }

    template <canonical_vector L, common_vector_with<L> R>
    requires canonical_vector<R> &&
        unqualified_canonical_mcmpgt<mask_t<L, R>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(mask_t<L, R> src, L lhs, R rhs) noexcept {
        return cmpgt(internal::abi<common_abi_t<L, R>>, src, lhs, rhs);
    }

    template <canonical_vector L, common_vector_with<L> R, imask_t<L, R> M>
    requires canonical_vector<R> &&
        unqualified_canonical_mcmpgt<cmask_t<L, R, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        cmask_t<L, R, M> cmask, L lhs, R rhs) noexcept {
        return cmpgt(internal::abi<common_abi_t<L, R>>, cmask, lhs, rhs);
    }
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_extended_cmpgt = requires(L lhs, R rhs) {
    { cmpgt(lhs, rhs) } -> mask_with_common_abi<A>;
};

template <typename S, typename L, typename R>
concept unqualified_extended_mcmpgt =
    (!simd_mask<S> || equivalent_mask_with<S, cpo_result_t<cmpgt_t, L, R>>) &&
    requires {
        {
            cmpgt(internal::declarg<S>(), internal::declarg<L>(),
                internal::declarg<R>())
        } -> equivalent_mask_with<cpo_result_t<cmpgt_t, L, R>>;
    };

template <>
struct extended_impl<cmpgt_t> {
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
        unqualified_extended_cmpgt<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R&& rhs) {
        return cmpgt(__DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <typename L, typename R>
    requires (extended_vector<L> &&
                 broadcastable_to<R, result_or_decayed_t<L>> &&
                 unqualified_extended_cmpgt<L, R, simd_abi_type_t<L>>) ||
        (extended_vector<R> && broadcastable_to<L, result_or_decayed_t<R>> &&
            unqualified_extended_cmpgt<L, R, simd_abi_type_t<R>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R&& rhs) {
        return cmpgt(__DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <simd_mask S, simd_vector L, common_vector_with<L> R>
    requires (extended_mask<S> || extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_mcmpgt<S, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, L&& lhs, R&& rhs) {
        return cmpgt(__DPL forward<S>(src), __DPL forward<L>(lhs),
            __DPL forward<R>(rhs));
    }

    template <simd_vector L, common_vector_with<L> R, imask_t<L, R> M>
    requires (extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_mcmpgt<cmask_t<L, R, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(cmask_t<L, R, M> cmask, L&& lhs, R&& rhs) {
        return cmpgt(cmask, __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::cmpgt_t cmpgt{};
}
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
