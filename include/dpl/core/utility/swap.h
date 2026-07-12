// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/operation/algorithm.h"
#  include "dpl/core/immediate/const_mask.h"
#  include "dpl/core/operations/select.h"
#  include "dpl/std/utility/template_barrier.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {
template <template_barrier_t = __DPL template_barrier, typename M, simd_type L,
    simd_type R>
requires cpo_invocable<internal::select_t, M, L, R> &&
    cpo_invocable<internal::select_t, M, R, L> &&
    assignable_from<L&, cpo_result_t<internal::select_t, M, R, L>> &&
    assignable_from<R&, cpo_result_t<internal::select_t, M, L, R>>
__DPL_HIDE_FROM_ABI constexpr void swap(M cond, L& lhs, R& rhs) noexcept {
    auto left = dx::select(cond, rhs, lhs);
    auto right = dx::select(cond, lhs, rhs);
    lhs = __DPL move(left);
    rhs = __DPL move(right);
}

template <template_barrier_t = __DPL template_barrier, simd_type L,
    simd_type R>
__DPL_HIDE_FROM_ABI constexpr void swap(L& lhs, R& rhs) noexcept {
    auto tmp = __DPL move(lhs);
    lhs = __DPL move(rhs);
    rhs = __DPL move(tmp);
}

template <auto M, simd_type L, simd_type R>
requires cpo_invocable<internal::selecti_t<M>, L, R> &&
    cpo_invocable<internal::selecti_t<M>, R, L> &&
    assignable_from<L&, cpo_result_t<internal::selecti_t<M>, R, L>> &&
    assignable_from<R&, cpo_result_t<internal::selecti_t<M>, L, R>>
__DPL_HIDE_FROM_ABI constexpr void swap(L& lhs, R& rhs) noexcept {
    auto left = dx::selecti<M>(rhs, lhs);
    auto right = dx::selecti<M>(lhs, rhs);
    lhs = __DPL move(left);
    rhs = __DPL move(right);
}
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
