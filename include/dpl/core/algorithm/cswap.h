// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/basic/const_mask.h"
#  include "dpl/core/concepts/compatible_mask_with.h"
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
    template <typename M, simd_class L, simd_class R>
    requires const_mask_for<M, R> && const_mask_for<M, L> &&
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

    template <simd_mask M, simd_class L, simd_class R>
    requires regular_invocable<internal::select_t, M, L, R> &&
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
struct cswapi_t {
private:
    template <typename T>
    using mask_type DPL_NODEBUG = make_const_mask_t<T, V>;
    template <typename L, typename R>
    using selection_t DPL_NODEBUG =
        invoke_result_t<internal::selecti_t<V>, L, R>;

public:
    template <fixed_width_class L, fixed_width_class R>
    requires regular_invocable<internal::selecti_t<V>, L, R> &&
        regular_invocable<internal::selecti_t<V>, R, L> &&
        assignable_from<L&, selection_t<R, L> const&> &&
        assignable_from<R&, selection_t<L, R> const&>
    __DPL_HIDE_FROM_ABI static constexpr void operator()(
        L& lhs, R& rhs) noexcept {
        auto const left = dx::selecti<V>(rhs, lhs);
        auto const right = dx::selecti<V>(lhs, rhs);
        lhs = left;
        rhs = right;
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
