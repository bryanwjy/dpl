// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

// IWYU pragma: always_keep

#include "dpl/core/operations/compare/cmplt.h"

#if !DPL_MODULES
#  include "dpl/core/basic/internal/abi.h"
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/maskable/predicate.h"
#  include "dpl/core/dispatch/operation/primitive.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void cmpgt(...) noexcept = delete;

struct DPL_EMPTY_BASES cmpgt_t :
    public comparison_base<cmpgt_t>,
    public maskable_predicate_base<cmpgt_t> {
    using operation_base<cmpgt_t>::operator();
    using maskable_predicate_base<cmpgt_t>::operator();
};

template <>
struct operation_signature<cmpgt_t> {
    template <typename L, typename R>
    requires simd_vector<L> || simd_vector<R>
    static consteval void operator()(L&&, R&&) noexcept {}
};

template <>
struct fallback_impl<cmpgt_t> : binary_canonical_broadcaster<cmpgt_t> {

    template <canonical_vector L, common_vector_with<L> R>
    requires canonical_vector<R> && cpo_invocable<cmplt_t, R, L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr cpo_result_t<cmplt_t, R, L>
        DPL_VECTORCALL operator()(L lhs, R rhs) noexcept {
        return dx::cmplt(rhs, lhs);
    }

    using binary_canonical_broadcaster<cmpgt_t>::operator();
};

template <typename L, typename R, typename T = common_canonical_simd_t<L, R>>
concept unqualified_canonical_cmpgt = requires {
    {
        cmpgt(internal::abi<T>, internal::declarg<L>(), internal::declarg<R>())
    } -> equivalent_mask_with<simd_mask_type_t<T>>;
};

template <typename M, typename L, typename R,
    typename T = simd_mask_type_t<common_canonical_simd_t<L, R>>>
concept unqualified_canonical_mcmpgt =
    cpo_invocable<cmpgt_t, L, R> && requires {
        {
            cmpgt(internal::abi<T>, internal::declarg<M>(),
                internal::declarg<L>(), internal::declarg<R>())
        } -> equivalent_mask_with<T>;
    };

template <>
struct canonical_impl<cmpgt_t> {
private:
    template <typename L, typename R>
    using result_t DPL_NODEBUG = cpo_result_t<cmpgt_t, L, R>;

    template <typename L, typename R>
    using mask_t DPL_NODEBUG = cpo_result_t<cmpgt_t, L, R>;

    template <typename L, typename R>
    using main_result_t DPL_NODEBUG =
        simd_mask_type_t<common_canonical_simd_t<L, R>>;

public:
    template <canonical_vector L, common_vector_with<L> R>
    requires canonical_vector<R> && unqualified_canonical_cmpgt<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr main_result_t<L, R> operator()(L lhs, R rhs) noexcept {
        return cmpgt(internal::abi<common_abi_t<L, R>>, lhs, rhs);
    }

    template <canonical_vector L, broadcastable_to<L> R>
    requires unqualified_canonical_cmpgt<L, R, L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr simd_mask_type_t<L> operator()(L lhs, R&& rhs) noexcept {
        return cmpgt(internal::abi<L>, lhs, __DPL forward<R>(rhs));
    }

    template <canonical_vector R, broadcastable_to<R> L>
    requires unqualified_canonical_cmpgt<L, R, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr simd_mask_type_t<R> operator()(L&& lhs, R rhs) noexcept {
        return cmpgt(internal::abi<R>, __DPL forward<L>(lhs), rhs);
    }

    template <unextended_type L, unextended_terminal_of<cmpgt_t, L> R>
    requires unqualified_canonical_mcmpgt<mask_t<L, R>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<L, R> operator()(
        mask_t<L, R> src, L lhs, R rhs) noexcept {
        return cmpgt(internal::abi<common_abi_t<L, R>>, src, lhs, rhs);
    }

    template <unextended_type L, unextended_type R,
        result_cmask_for<cmpgt_t, L, R> M>
    requires unqualified_canonical_mcmpgt<launder_cmask_t<result_t<L, R>, M>, L,
        R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<L, R> operator()(M mask, L lhs, R rhs) noexcept {
        return cmpgt(internal::abi<common_abi_t<L, R>>,
            dx::to_const_mask<mask_t<L, R>>(mask), lhs, rhs);
    }
};

template <typename L, typename R, typename T = common_canonical_simd_t<L, R>>
concept unqualified_extended_cmpgt = requires(L lhs, R rhs) {
    { cmpgt(lhs, rhs) } -> equivalent_mask_with<simd_mask_type_t<T>>;
};

template <typename S, typename L, typename R>
concept unqualified_extended_mcmpgt = cpo_invocable<cmpgt_t, L, R> && requires {
    {
        cmpgt(internal::declarg<S>(), internal::declarg<L>(),
            internal::declarg<R>())
    } -> equivalent_mask_with<cpo_result_t<cmpgt_t, L, R>>;
};

template <>
struct extended_impl<cmpgt_t> {
public:
    template <simd_vector L, common_vector_with<L> R>
    requires (extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_cmpgt<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R&& rhs) {
        return cmpgt(__DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <extended_vector L, broadcastable_to<L> R>
    requires unqualified_extended_cmpgt<L, R, L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R&& rhs) {
        return cmpgt(__DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <extended_vector R, broadcastable_to<R> L>
    requires unqualified_extended_cmpgt<L, R, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R&& rhs) {
        return cmpgt(__DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <simd_mask S, typename L, typename R>
    requires (extended_mask<S> || extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_mcmpgt<S, L, R> &&
        equivalent_mask_with<S, cpo_result_t<cmpgt_t, L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, L&& lhs, R&& rhs) {
        return cmpgt(__DPL forward<S>(src), __DPL forward<L>(lhs),
            __DPL forward<R>(rhs));
    }

    template <typename L, typename R, result_cmask_for<cmpgt_t, L, R> M>
    requires (extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_mcmpgt<
            launder_cmask_t<cpo_result_t<cmpgt_t, L, R>, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, L&& lhs, R&& rhs) {
        return cmpgt(dx::to_const_mask<cpo_result_t<cmpgt_t, L, R>>(mask),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::cmpgt_t cmpgt{};
}
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
