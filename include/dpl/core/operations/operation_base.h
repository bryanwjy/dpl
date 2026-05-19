// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/basic/broadcast.h"
#  include "dpl/core/basic/to_canonical.h"
#  include "dpl/core/concepts/basic_type.h"
#  include "dpl/core/concepts/broadcastable_to.h"
#  include "dpl/core/concepts/common_abi_with.h"
#  include "dpl/std/type_traits/type_identity.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

template <typename L, typename R>
concept only_unqualified =
    !canonical_class<L> || !canonical_class<R> || !same_abi_simd_as<L, R>;

template <typename T, typename A, typename... Args>
concept implements_native = requires(Args... args) {
    internal::abi<A>;
    T::native(internal::abi<A>, args...);
};

template <typename T>
struct binary_operation_base {
private:
    template <typename L, typename R>
    static consteval auto abi_for() noexcept {
        if constexpr (simd_class<L>) {
            return typename L::abi_type{};
        } else {
            return typename R::abi_type{};
        }
    }

    template <simd_class L, typename R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr decltype(auto) fallback(
        type_identity_t<L> const& arg) noexcept {
        return (arg);
    }

    template <typename L, simd_class R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr decltype(auto) fallback(
        type_identity_t<R> const& arg) noexcept {
        return (arg);
    }

    template <typename L, simd_class R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr decltype(auto) fallback(type_identity_t<L> arg) noexcept {
        return dx::broadcast<R>(arg);
    }

    template <simd_class L, typename R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr decltype(auto) fallback(type_identity_t<R> arg) noexcept {
        return dx::broadcast<L>(arg);
    }

public:
    template <typename L, typename R>
    requires (simd_class<L> && !simd_class<R> && broadcastable_to<R, L>) ||
        (simd_class<R> && !simd_class<L> && broadcastable_to<L, R>)
        DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
        static constexpr auto operator()(L lhs, R rhs) noexcept
    requires requires(
        T impl) { impl(fallback<L, R>(lhs), fallback<L, R>(rhs)); }
    {
        using A = decltype(abi_for<L, R>());
        if constexpr (implements_native<T, A, L, R>) {
            if constexpr (canonical_class<L> || canonical_class<R>) {
                if consteval {
                    return T::operator()(
                        fallback<L, R>(lhs), fallback<L, R>(rhs));
                } else {
                    return T::native(internal::abi<L>, lhs, rhs);
                }
            } else {
                return T::native(internal::abi<L>, lhs, rhs);
            }
        } else if constexpr (canonical_class<L> || canonical_class<R>) {
            return T::operator()(fallback<L, R>(lhs), fallback<L, R>(rhs));
        } else if constexpr (simd_class<L>) {
            return operator()(dx::to_canonical(lhs), rhs);
        } else {
            return operator()(rhs, dx::to_canonical(rhs));
        }
    }
};

} // namespace datapar::internal
DPL_DEFAULT_NAMESPACE_END
