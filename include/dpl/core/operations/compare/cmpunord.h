// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

// IWYU pragma: always_keep

#include "dpl/core/operations/bitwise/bwor.h"
#include "dpl/core/operations/compare/cmpneq.h"

#if !DPL_MODULES
#  include "dpl/core/basic/internal/abi.h"
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/dispatch/broadcastable/binary.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/maskable/predicate.h"
#  include "dpl/core/dispatch/operation/primitive.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void cmpunord(...) noexcept = delete;

struct DPL_EMPTY_BASES cmpunord_t :
    public comparison_base<cmpunord_t>,
    public maskable_predicate_base<cmpunord_t>,
    public binary_broadcastable_operation<cmpunord_t> {
    using operation_base<cmpunord_t>::operator();
    using maskable_predicate_base<cmpunord_t>::operator();
    using binary_broadcastable_operation<cmpunord_t>::operator();
};

template <>
struct operation_signature<cmpunord_t> {
    template <typename L, typename R>
    requires simd_vector<L> || simd_vector<R>
    static consteval void operator()(L&&, R&&) noexcept {}
};

template <>
struct fallback_impl<cmpunord_t> : binary_broadcasting_fallback<cmpunord_t> {
private:
    template <typename L, typename R>
    using vector_t DPL_NODEBUG = common_canonical_simd_t<L, R>;
    template <typename L, typename R>
    using result_t DPL_NODEBUG = simd_mask_type_t<vector_t<L, R>>;

public:
    template <canonical_vector L, common_vector_with<L> R>
    requires floating_point_like<simd_element_type_t<vector_t<L, R>>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr result_t<L, R>
        DPL_VECTORCALL operator()(L lhs, R rhs) noexcept {
        return dx::bwor(dx::cmpneq(lhs, lhs), dx::cmpneq(rhs, rhs));
    }

    using binary_broadcasting_fallback<cmpunord_t>::operator();
};

template <typename L, typename R, typename T = common_canonical_simd_t<L, R>>
concept unqualified_canonical_cmpunord = requires {
    {
        cmpunord(
            internal::abi<T>, internal::declarg<L>(), internal::declarg<R>())
    } -> equivalent_mask_with<simd_mask_type_t<T>>;
};

template <typename S, typename L, typename R>
concept unqualified_canonical_mcmpunord = cpo_invocable<cmpunord_t, L, R> &&
    (!simd_mask<S> || same_as<S, cpo_result_t<cmpunord_t, L, R>>) && requires {
        {
            cmpunord(internal::abi<common_abi_t<L, R>>, internal::declarg<S>(),
                internal::declarg<L>(), internal::declarg<R>())
        } -> same_as<cpo_result_t<cmpunord_t, L, R>>;
    };

template <>
struct canonical_impl<cmpunord_t> {
private:
    template <typename L, typename R>
    using result_t DPL_NODEBUG =
        simd_mask_type_t<common_canonical_simd_t<L, R>>;

    template <typename L, typename R>
    using mask_t DPL_NODEBUG = result_t<L, R>;

public:
    template <canonical_vector L, common_vector_with<L> R>
    requires unqualified_canonical_cmpunord<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<L, R> operator()(L lhs, R rhs) noexcept {
        return cmpunord(internal::abi<common_abi_t<L, R>>, lhs, rhs);
    }

    template <canonical_vector L, broadcastable_to<L> R>
    requires unqualified_canonical_cmpunord<L, R, L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<L, L> operator()(L lhs, R&& rhs) noexcept {
        return cmpunord(internal::abi<L>, lhs, __DPL forward<R>(rhs));
    }

    template <canonical_vector R, broadcastable_to<R> L>
    requires unqualified_canonical_cmpunord<L, R, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<R, R> operator()(L&& lhs, R rhs) noexcept {
        return cmpunord(internal::abi<R>, __DPL forward<L>(lhs), rhs);
    }

    template <canonical_vector L, common_vector_with<L> R>
    requires canonical_vector<R> &&
        unqualified_canonical_mcmpunord<mask_t<L, R>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(mask_t<L, R> src, L lhs, R rhs) noexcept {
        return cmpunord(internal::abi<common_abi_t<L, R>>, src, lhs, rhs);
    }

    template <canonical_vector L, common_vector_with<L> R,
        const_mask_for<mask_t<L, R>> M>
    requires canonical_vector<R> &&
        unqualified_canonical_mcmpunord<launder_cmask_t<mask_t<L, R>, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M cmask, L lhs, R rhs) noexcept {
        return cmpunord(internal::abi<common_abi_t<L, R>>,
            dx::to_const_mask<mask_t<L, R>>(cmask), lhs, rhs);
    }
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_extended_cmpunord = requires(L lhs, R rhs) {
    { cmpunord(lhs, rhs) } -> mask_with_common_abi<A>;
};

template <typename S, typename L, typename R>
concept unqualified_extended_mcmpunord = cpo_invocable<cmpunord_t, L, R> &&
    (!simd_mask<S> ||
        equivalent_mask_with<S, cpo_result_t<cmpunord_t, L, R>>) &&
    requires {
        {
            cmpunord(internal::declarg<S>(), internal::declarg<L>(),
                internal::declarg<R>())
        } -> equivalent_mask_with<cpo_result_t<cmpunord_t, L, R>>;
    };

template <>
struct extended_impl<cmpunord_t> {
public:
    template <simd_vector L, common_vector_with<L> R>
    requires (extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_cmpunord<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R&& rhs) {
        return cmpunord(__DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <typename L, typename R>
    requires (extended_vector<L> &&
                 broadcastable_to<R, result_or_decayed_t<L>> &&
                 unqualified_extended_cmpunord<L, R, simd_abi_type_t<L>>) ||
        (extended_vector<R> && broadcastable_to<L, result_or_decayed_t<R>> &&
            unqualified_extended_cmpunord<L, R, simd_abi_type_t<R>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R&& rhs) {
        return cmpunord(__DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <simd_mask S, simd_vector L, common_vector_with<L> R>
    requires (extended_mask<S> || extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_mcmpunord<S, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, L&& lhs, R&& rhs) {
        return cmpunord(__DPL forward<S>(src), __DPL forward<L>(lhs),
            __DPL forward<R>(rhs));
    }

    template <simd_vector L, common_vector_with<L> R,
        result_cmask_for<cmpunord_t, L, R> M>
    requires (extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_mcmpunord<
            launder_cmask_t<cpo_result_t<cmpunord_t, L, R>, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M cmask, L&& lhs, R&& rhs) {
        return cmpunord(
            dx::to_const_mask<cpo_result_t<cmpunord_t, L, R>>(cmask),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::cmpunord_t cmpunord{};
}
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
