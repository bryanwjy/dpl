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
struct fm_canonical_broadcaster {
    template <canonical_vector AT, canonical_vector BT, broadcastable_to<AT> CT>
    requires cpo_invocable<D, AT, BT, AT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static consteval AT operator()(AT lhs, BT mid, CT&& rhs) noexcept {
        return D::operator()(
            lhs, mid, dx::broadcast<AT>(__DPL forward<CT>(rhs)));
    }

    template <canonical_vector AT, canonical_vector CT, broadcastable_to<AT> BT>
    requires cpo_invocable<D, AT, AT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static consteval AT operator()(AT lhs, BT&& mid, CT rhs) noexcept {
        return D::operator()(
            lhs, dx::broadcast<AT>(__DPL forward<BT>(mid)), rhs);
    }

    template <canonical_vector BT, canonical_vector CT, broadcastable_to<BT> AT>
    requires cpo_invocable<D, common_canonical_simd_t<BT, CT>, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static consteval common_canonical_simd_t<BT, CT> operator()(
        AT&& lhs, BT mid, CT rhs) noexcept {
        using T DPL_NODEBUG = common_canonical_simd_t<BT, CT>;
        return D::operator()(
            dx::broadcast<T>(__DPL forward<AT>(lhs)), mid, rhs);
    }

    template <canonical_vector AT, broadcastable_to<AT> BT,
        broadcastable_to<AT> CT>
    requires cpo_invocable<D, AT, AT, AT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static consteval AT operator()(AT lhs, BT&& mid, CT&& rhs) noexcept {
        return D::operator()(lhs, dx::broadcast<AT>(__DPL forward<BT>(mid)),
            dx::broadcast<AT>(__DPL forward<CT>(rhs)));
    }

    template <canonical_vector BT, broadcastable_to<BT> AT,
        broadcastable_to<BT> CT>
    requires cpo_invocable<D, BT, BT, BT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static consteval BT operator()(AT&& lhs, BT mid, CT&& rhs) noexcept {
        return D::operator()(dx::broadcast<BT>(__DPL forward<AT>(lhs)), mid,
            dx::broadcast<BT>(__DPL forward<CT>(rhs)));
    }

    template <canonical_vector CT, broadcastable_to<CT> AT,
        broadcastable_to<CT> BT>
    requires cpo_invocable<D, CT, CT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static consteval CT operator()(AT&& lhs, BT&& mid, CT rhs) noexcept {
        return D::operator()(dx::broadcast<CT>(__DPL forward<AT>(lhs)),
            dx::broadcast<CT>(__DPL forward<BT>(mid)), rhs);
    }
};

} // namespace datapar::internal

__DPL_DEFAULT_NAMESPACE_END
