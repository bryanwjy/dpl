// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/basic/immediate_mask.h"
#  include "dpl/core/concepts/compatible_mask_for.h"
#  include "dpl/core/operations/select.h"
#  include "dpl/std/concepts/invocable.h"
#  include "dpl/std/type_traits/is_invocable.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
struct cswap_t {
    template <typename M, typename L, typename R>
    using selection_t DPL_NODEBUG =
        invoke_result_t<internal::select_t, M, L, R>;

public:
    template <simd_class L, selectable_with<L> R, immediate_mask_for<L> M>
    requires immediate_mask_for<M, R> &&
        regular_invocable<internal::select_t, M, L, R> &&
        regular_invocable<internal::select_t, M, R, L> &&
        assignable_from<L&, selection_t<M, R, L> const&> &&
        assignable_from<R&, selection_t<M, L, R> const&>
    __DPL_HIDE_FROM_ABI static constexpr void operator()(
        M cond, L& lhs, R& rhs) noexcept {
        auto const left = dx::select(cond, rhs, lhs);
        auto const right = dx::select(cond, lhs, rhs);
        lhs = left;
        rhs = right;
    }

    template <simd_class L, selectable_with<L> R, compatible_mask_for<L> M>
    requires compatible_mask_for<M, R> &&
        regular_invocable<internal::select_t, M, L, R> &&
        regular_invocable<internal::select_t, M, R, L> &&
        assignable_from<L&, selection_t<M, R, L> const&> &&
        assignable_from<R&, selection_t<M, L, R> const&>
    __DPL_HIDE_FROM_ABI static constexpr void operator()(
        M cond, L& lhs, R& rhs) noexcept {
        auto const left = dx::select(cond, rhs, lhs);
        auto const right = dx::select(cond, lhs, rhs);
        lhs = left;
        rhs = right;
    }
};

template <auto V>
struct cswapi_t {};
template <integral auto V>
struct cswapi_t<V> {
private:
    template <typename T>
    using mask_type DPL_NODEBUG = immediate_mask<element_count<T>, V>;

public:
    template <simd_class L, common_size_simd_with<L> R>
    requires requires {
        typename mask_type<L>;
        requires regular_invocable<cswap_t, mask_type<L>, L, R>;
    }
    __DPL_HIDE_FROM_ABI static constexpr void operator()(
        L& lhs, R& rhs) noexcept {
        constexpr mask_type<L> mask{};
        cswap_t::operator()(mask, lhs, rhs);
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::cswap_t cswap{};
DPL_EXPORT template <auto V>
inline constexpr internal::cswapi_t<V> cswapi{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
