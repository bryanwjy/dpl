// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/dispatch/private/concepts.h"
#include "dpl/core/dispatch/private/cpo.h"
#include "dpl/core/dispatch/private/forward_or_eval.h"

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::internal {

template <typename D>
class operation_base {
protected:
    template <canonical_ornot_simd... Ts>
    requires signature_compatible<D, Ts...> &&
        (fallback_cpo_invocable<D, Ts...> || canonical_cpo_invocable<D, Ts...>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(Ts... args) noexcept {
        if constexpr (fallback_cpo_invocable<D, Ts...>) {
            if constexpr (canonical_cpo_invocable<D, Ts...>) {
                if consteval {
                    return impl::fallback<D>(args...);
                } else {
                    return impl::canonical<D>(args...);
                }
            } else {
                return impl::fallback<D>(args...);
            }
        } else {
            return impl::canonical<D>(args...);
        }
    }

    template <typename... Ts>
    requires signature_compatible<D, Ts...> && extended_arguments<Ts...> &&
        (extendable_operation<D, Ts...> ||
            requires {
                requires (... || simd_expression<Ts>);
                operator()(internal::declarg<result_or_identity_t<Ts>>()...);
            } || fallback_cpo_invocable<D, Ts...>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(Ts&&... args) {
        if constexpr (extended_cpo_invocable<D, Ts...>) {
            return impl::extended<D>(__DPL forward<Ts>(args)...);
        } else if constexpr (extended_nttp_invocable<D, Ts...>) {
            constexpr D Op;
            return extended<Op>(__DPL forward<Ts>(args)...);
        } else if constexpr ((... || simd_expression<Ts>)) {
            return operator()(
                internal::forward_or_eval(__DPL forward<Ts>(args))...);
        } else {
            return impl::fallback<D>(__DPL forward<Ts>(args)...);
        }
    }

    template <typename T>
    requires inherits_from<T, operation_base<T>>
    friend consteval bool operator==(D, T) noexcept {
        return dpl::is_same_v<T, D>;
    }
    template <typename T>
    requires inherits_from<T, operation_base<T>>
    friend consteval bool operator!=(D, D) noexcept {
        return !dpl::is_same_v<T, D>;
    }
};

template <typename D>
class basic_operation_base : public operation_base<D> {};

} // namespace datapar::internal

__DPL_DEFAULT_NAMESPACE_END
