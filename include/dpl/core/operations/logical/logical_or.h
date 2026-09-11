// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

// IWYU pragma: always_keep

#include "dpl/core/operations/bitwise/bwor.h"

#if !DPL_MODULES
#  include "dpl/core/basic/internal/abi.h"
#  include "dpl/core/concepts/canonical.h"
#  include "dpl/core/concepts/cpo_invocable.h"
#  include "dpl/core/concepts/simd_mask.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/operation/primitive.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

void logical_or(...) noexcept = delete;

struct logical_or_t : private logical_base<logical_or_t> {
    using operation_base<logical_or_t>::operator();

    template <simd_mask L, common_mask_with<L> R, common_mask_with<L> T0,
        common_mask_with<L>... Ts>
    requires cpo_invocable<logical_or_t, L, R> &&
        cpo_invocable<logical_or_t, cpo_result_t<logical_or_t, L, R>, T0, Ts...>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(L&& lhs, R&& rhs, T0&& head,
        Ts&&... tail) noexcept(canonical_mask<L> && canonical_mask<R> &&
        (canonical_mask<T0> && ... && canonical_mask<Ts>)) {
        return operator()(operator()(
                              __DPL forward<L>(lhs), __DPL forward<R>(rhs)),
            __DPL forward<T0>(head), __DPL forward<Ts>(tail)...);
    }
};

template <>
struct operation_signature<logical_or_t> {
    template <simd_mask L, simd_mask R>
    static consteval void operator()(L&&, R&&) noexcept {}
};

template <>
struct fallback_impl<logical_or_t> {
    template <simd_mask L, common_mask_with<L> R>
    requires cpo_invocable<bwand_t, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(L&& lhs, R&& rhs) noexcept(
        canonical_mask<L> && canonical_mask<R>) {
        return dx::bwor(__DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::logical_or_t logical_or{};
} // namespace cpo
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
