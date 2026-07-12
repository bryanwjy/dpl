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
};

template <>
struct operation_signature<logical_or_t> {
    template <simd_mask L, simd_mask R>
    static consteval void operator()(L&&, R&&) noexcept {}
};

template <>
struct fallback_impl<logical_or_t> {
    template <canonical_mask L, canonical_mask R>
    requires common_abi_with<simd_abi_type_t<L>, simd_abi_type_t<R>> &&
        cpo_invocable<bwor_t, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr basic_mask<simd_element_type_t<R>, common_abi_t<L, R>>
        DPL_VECTORCALL operator()(L lhs, R rhs) noexcept {
        return dx::bwor(lhs, rhs);
    }

    template <simd_mask L, simd_mask R>
    requires (extended_mask<L> || extended_mask<R>) &&
        cpo_invocable<bwor_t, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(L&& lhs, R&& rhs) {
        return dx::bwor(__DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }
};

template <>
struct canonical_impl<logical_or_t> {
    template <canonical_mask L, canonical_mask R>
    requires common_abi_with<simd_abi_type_t<L>, simd_abi_type_t<R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<simd_element_type_t<R>, common_abi_t<L, R>>
    operator()(L lhs, R rhs) noexcept
    requires requires {
        logical_or(internal::abi<common_abi_t<L, R>>, lhs, rhs);
    }
    {
        return logical_or(internal::abi<common_abi_t<L, R>>, lhs, rhs);
    }
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_extended_logical_or = requires(L lhs, R rhs) {
    {
        logical_or(internal::declarg<L>(), internal::declarg<R>())
    } -> mask_with_common_abi<A>;
};

template <>
struct extended_impl<logical_or_t> {
    template <simd_mask L, simd_mask R>
    requires (extended_mask<L> || extended_mask<R>) &&
        unqualified_extended_logical_or<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R&& rhs) {
        return logical_or(__DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::logical_or_t logical_or{};
} // namespace cpo
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
