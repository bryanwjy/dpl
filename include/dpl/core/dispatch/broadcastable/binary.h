// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/dispatch/private/fwd.h"

#include "dpl/core/dispatch/broadcastable/base.h"
#include "dpl/core/dispatch/private/concepts.h"
#include "dpl/core/dispatch/private/cpo.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/core/concepts/broadcastable_to.h"
#  include "dpl/core/concepts/canonical.h"
#  include "dpl/std/utility/forward.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::internal {

template <typename D>
struct binary_broadcastable_operation : public broadcastable_operation_base<D> {

    template <canonical_simd_type L, broadcastable_to<L> R>
    requires signature_compatible<D, L, R> &&
        (fallback_cpo_invocable<D, L, R> || canonical_cpo_invocable<D, L, R>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R&& rhs) noexcept {
        if constexpr (fallback_cpo_invocable<D, L, R>) {
            if constexpr (canonical_cpo_invocable<D, L, R>) {
                if consteval {
                    return impl::fallback<D>(lhs, __DPL forward<R>(rhs));
                } else {
                    return impl::canonical<D>(lhs, __DPL forward<R>(rhs));
                }
            } else {
                return impl::fallback<D>(lhs, __DPL forward<R>(rhs));
            }
        } else {
            return impl::canonical<D>(lhs, __DPL forward<R>(rhs));
        }
    }

    template <canonical_simd_type R, broadcastable_to<R> L>
    requires signature_compatible<D, L, R> &&
        (fallback_cpo_invocable<D, L, R> || canonical_cpo_invocable<D, L, R>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R rhs) noexcept {
        if constexpr (fallback_cpo_invocable<D, L, R>) {
            if constexpr (canonical_cpo_invocable<D, L, R>) {
                if consteval {
                    return impl::fallback<D>(__DPL forward<L>(lhs), rhs);
                } else {
                    return impl::canonical<D>(__DPL forward<L>(lhs), rhs);
                }
            } else {
                return impl::fallback<D>(__DPL forward<L>(lhs), rhs);
            }
        } else {
            return impl::canonical<D>(__DPL forward<L>(lhs), rhs);
        }
    }

    template <typename L, typename R>
    requires signature_compatible<D, L, R> &&
        ((extended_simd_type<R> &&
             broadcastable_to<L, result_or_decayed_t<R>>) ||
            (extended_simd_type<L> &&
                broadcastable_to<R, result_or_decayed_t<L>>)) &&
        (extended_cpo_invocable<D, L, R> || fallback_cpo_invocable<D, L, R>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R&& rhs) noexcept {
        if constexpr (extended_cpo_invocable<D, L, R>) {
            return impl::extended<D>(
                __DPL forward<L>(lhs), __DPL forward<R>(rhs));
        } else {
            return impl::fallback<D>(
                __DPL forward<L>(lhs), __DPL forward<R>(rhs));
        }
    }
};

template <typename D>
struct binary_broadcasting_fallback {
    template <typename L, typename R>
    requires (simd_type<R> && broadcastable_to<L, result_or_decayed_t<R>> &&
                 cpo_invocable<D, result_or_decayed_t<R>, R>) ||
        (simd_type<L> && broadcastable_to<R, result_or_decayed_t<L>> &&
            cpo_invocable<D, L, result_or_decayed_t<L>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R&& rhs) noexcept(
        canonical_simd_type<L> && canonical_simd_type<R>) {
        if constexpr (simd_type<L>) {
            return D::operator()(lhs,
                fwd::broadcast<result_or_decayed_t<L>>(__DPL forward<R>(rhs)));
        } else {
            return D::operator()(
                fwd::broadcast<result_or_decayed_t<R>>( __DPL forward<L>(lhs)),
                rhs);
        }
    }
};

} // namespace datapar::internal

DPL_DEFAULT_NAMESPACE_END
