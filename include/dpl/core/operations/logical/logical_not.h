// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

// IWYU pragma: always_keep

#include "dpl/core/operations/bitwise/bwnot.h"

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

void logical_not(...) noexcept = delete;

struct logical_not_t : private logical_base<logical_not_t> {
    using operation_base<logical_not_t>::operator();
};

template <>
struct operation_signature<logical_not_t> {
    template <simd_mask T>
    static consteval void operator()(T&&) noexcept {}
};

template <>
struct fallback_impl<logical_not_t> {

    template <simd_type T>
    requires cpo_invocable<bwnot_t, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(T&& val) noexcept(
        canonical_simd_type<T>) {
        return dx::bwnot(__DPL forward<T>(val));
    }
};

template <>
struct canonical_impl<logical_not_t> {
    template <canonical_mask T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr bool operator()(T val) noexcept
    requires requires { logical_not(internal::abi<T>, val); }
    {
        return logical_not(internal::abi<T>, val);
    }
};

template <typename T, typename A = simd_abi_type_t<T>>
concept unqualified_extended_logical_not = requires(T&& val) {
    { logical_not(__DPL forward<T>(val)) } -> mask_with_common_abi<A>;
};

template <>
struct extended_impl<logical_not_t> {
    template <extended_mask T>
    requires unqualified_extended_logical_not<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val) {
        return logical_not(__DPL forward<T>(val));
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::logical_not_t logical_not{};
} // namespace cpo
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
