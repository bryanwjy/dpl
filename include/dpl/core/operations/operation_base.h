// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/basic/broadcast.h"
#  include "dpl/core/basic/to_basic_type.h"
#  include "dpl/core/concepts/basic_type.h"
#  include "dpl/core/concepts/broadcastable_to.h"
#  include "dpl/core/concepts/common_abi_with.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

template <typename L, typename R>
concept only_unqualified =
    !basic_simd_class<L> || !basic_simd_class<R> || !same_abi_simd_as<L, R>;

template <typename T, typename A, typename... Args>
concept implements_native = requires(Args... args) {
    internal::abi<A>;
    T::native(internal::abi<A>, args...);
};

template <typename T>
struct binary_operation_base {

    template <simd_class L, broadcastable_to<L> R>
    requires implements_native<T, L, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        if constexpr (basic_simd_class<L>) {
            if consteval {
                return T::operator()(lhs, dx::broadcast<L>(rhs));
            } else {
                return T::native(internal::abi<L>, lhs, rhs);
            }
        } else {
            return T::native(internal::abi<L>, lhs, rhs);
        }
    }

    template <simd_class R, broadcastable_to<R> L>
    requires implements_native<T, R, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        if constexpr (basic_simd_class<R>) {
            if consteval {
                return T::operator()(dx::broadcast<R>(lhs), rhs);
            } else {
                return T::native(internal::abi<R>, lhs, rhs);
            }
        } else {
            return T::native(internal::abi<R>, lhs, rhs);
        }
    }

    template <simd_class L, broadcastable_to<L> R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        if constexpr (basic_simd_class<L>) {
            return T::operator()(lhs, dx::broadcast<L>(rhs));
        } else {
            return operator()(dx::to_basic_type(lhs), rhs);
        }
    }

    template <simd_class R, broadcastable_to<R> L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        if constexpr (basic_simd_class<R>) {
            return T::operator()(dx::broadcast<R>(lhs), rhs);
        } else {
            return operator()(lhs, dx::to_basic_type(rhs));
        }
    }
};

} // namespace datapar::internal
DPL_DEFAULT_NAMESPACE_END
