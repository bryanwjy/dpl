// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

// IWYU pragma: always_keep

#include "dpl/core/operations/internal/transform.h"

#if !DPL_MODULES
#  include "dpl/core/basic/internal/abi.h"
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/dispatch/broadcastable/binary.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/maskable/predicate.h"
#  include "dpl/core/dispatch/operation/primitive.h"
#  include "dpl/std/concepts/totally_ordered.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void cmplt(...) noexcept = delete;

struct DPL_EMPTY_BASES cmplt_t :
    private comparison_base<cmplt_t>,
    private maskable_predicate_base<cmplt_t>,
    private binary_broadcastable_operation<cmplt_t> {
    using operation_base<cmplt_t>::operator();
    using maskable_predicate_base<cmplt_t>::operator();
    using binary_broadcastable_operation<cmplt_t>::operator();
};

template <>
struct operation_signature<cmplt_t> {
    template <typename L, typename R>
    requires simd_vector<L> || simd_vector<R>
    static consteval void operator()(L&&, R&&) noexcept {}
};

template <>
struct fallback_impl<cmplt_t> : binary_broadcasting_fallback<cmplt_t> {
    template <fixed_width_abi A, simd_element_for<A> E>
    requires totally_ordered<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_mask<E, A>
        DPL_VECTORCALL operator()(
            basic_vector<E, A> lhs, basic_vector<E, A> rhs) noexcept {
        return internal::transform<basic_mask<E, A>>(
            [](auto lhs, auto rhs) -> bool { return lhs < rhs; }, lhs, rhs);
    }

    using binary_broadcasting_fallback<cmplt_t>::operator();
};
template <typename L, typename R, typename A = common_abi_t<L, R>,
    typename E = simd_element_type_t<L>>
concept unqualified_canonical_cmplt = requires {
    {
        cmplt(internal::abi<A>, internal::declarg<L>(), internal::declarg<R>())
    } -> equivalent_mask_with<basic_mask<simd_element_type_t<L>, A>>;
};

template <typename S, typename L, typename R>
concept unqualified_canonical_mcmplt = cpo_invocable<cmplt_t, L, R> &&
    (!simd_mask<S> || same_as<S, cpo_result_t<cmplt_t, L, R>>) && requires {
        {
            cmplt(internal::abi<S>, internal::declarg<S>(),
                internal::declarg<L>(), internal::declarg<R>())
        } -> same_as<cpo_result_t<cmplt_t, L, R>>;
    };

template <>
struct canonical_impl<cmplt_t> {
private:
    template <typename L, typename R>
    using result_t DPL_NODEBUG =
        basic_mask<simd_element_type_t<L>, common_abi_t<L, R>>;

    template <typename L, typename R>
    using mresult_t DPL_NODEBUG = basic_mask<
        common_size_type_t<simd_element_type_t<L>, simd_element_type_t<R>>,
        common_abi_t<L, R>>;

    template <typename L, typename R>
    using mask_t DPL_NODEBUG = cpo_result_t<cmplt_t, L, R>;

public:
    template <canonical_vector L, common_vector_with<L> R>
    requires unqualified_canonical_cmplt<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<L, R> operator()(L lhs, R rhs) noexcept {
        return cmplt(internal::abi<common_abi_t<L, R>>, lhs, rhs);
    }

    template <canonical_vector L, broadcastable_to<L> R>
    requires unqualified_canonical_cmplt<L, R, simd_abi_type_t<L>,
        simd_element_type_t<L>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<L, L> operator()(L lhs, R&& rhs) noexcept {
        return cmplt(internal::abi<L>, lhs, __DPL forward<R>(rhs));
    }

    template <canonical_vector R, broadcastable_to<R> L>
    requires unqualified_canonical_cmplt<L, R, simd_abi_type_t<R>,
        simd_element_type_t<R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<R, R> operator()(L&& lhs, R rhs) noexcept {
        return cmplt(internal::abi<R>, __DPL forward<L>(lhs), rhs);
    }

    template <canonical_vector L, common_vector_with<L> R>
    requires canonical_vector<R> &&
        unqualified_canonical_mcmplt<mask_t<L, R>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(mask_t<L, R> src, L lhs, R rhs) noexcept {
        return cmplt(internal::abi<common_abi_t<L, R>>, src, lhs, rhs);
    }

    template <canonical_vector L, common_vector_with<L> R,
        const_mask_for<mask_t<L, R>> M>
    requires canonical_vector<R> &&
        unqualified_canonical_mcmplt<launder_cmask_t<mask_t<L, R>, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M cmask, L lhs, R rhs) noexcept {
        return cmplt(internal::abi<common_abi_t<L, R>>,
            dx::to_const_mask<mask_t<L, R>>(cmask), lhs, rhs);
    }
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_extended_cmplt = requires(L lhs, R rhs) {
    { cmplt(lhs, rhs) } -> mask_with_common_abi<A>;
};

template <typename S, typename L, typename R>
concept unqualified_extended_mcmplt = cpo_invocable<cmplt_t, L, R> &&
    (!simd_mask<S> || equivalent_mask_with<S, cpo_result_t<cmplt_t, L, R>>) &&
    requires {
        {
            cmplt(internal::declarg<S>(), internal::declarg<L>(),
                internal::declarg<R>())
        } -> equivalent_mask_with<cpo_result_t<cmplt_t, L, R>>;
    };

template <>
struct extended_impl<cmplt_t> {
public:
    template <simd_vector L, common_vector_with<L> R>
    requires (extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_cmplt<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R&& rhs) {
        return cmplt(__DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <typename L, typename R>
    requires (extended_vector<L> &&
                 broadcastable_to<R, result_or_decayed_t<L>> &&
                 unqualified_extended_cmplt<L, R, simd_abi_type_t<L>>) ||
        (extended_vector<R> && broadcastable_to<L, result_or_decayed_t<R>> &&
            unqualified_extended_cmplt<L, R, simd_abi_type_t<R>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R&& rhs) {
        return cmplt(__DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <simd_mask S, simd_vector L, common_vector_with<L> R>
    requires (extended_mask<S> || extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_mcmplt<S, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, L&& lhs, R&& rhs) {
        return cmplt(__DPL forward<S>(src), __DPL forward<L>(lhs),
            __DPL forward<R>(rhs));
    }

    template <simd_vector L, common_vector_with<L> R,
        result_cmask_for<cmplt_t, L, R> M>
    requires (extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_mcmplt<
            launder_cmask_t<cpo_result_t<cmplt_t, L, R>, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M cmask, L&& lhs, R&& rhs) {
        return cmplt(dx::to_const_mask<cpo_result_t<cmplt_t, L, R>>(cmask),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::cmplt_t cmplt{};
}
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
