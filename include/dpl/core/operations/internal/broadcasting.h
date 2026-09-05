// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/basic/broadcast.h"
#  include "dpl/core/concepts/broadcastable_to.h"
#  include "dpl/core/concepts/canonical.h"
#  include "dpl/std/utility/forward.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::internal {

template <typename D>
struct binary_canonical_broadcaster {
    template <canonical_vector L, broadcastable_to<L> R>
    requires cpo_invocable<D, L, L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr cpo_result_t<D, L, L> operator()(L lhs, R&& rhs) noexcept {
        return D::operator()(lhs, fwd::broadcast<L>(__DPL forward<R>(rhs)));
    }

    template <canonical_vector R, broadcastable_to<R> L>
    requires cpo_invocable<D, R, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr cpo_result_t<D, R, R> operator()(L&& lhs, R rhs) noexcept {
        return D::operator()(fwd::broadcast<R>(__DPL forward<L>(lhs)), rhs);
    }
};

} // namespace datapar::internal

__DPL_DEFAULT_NAMESPACE_END
