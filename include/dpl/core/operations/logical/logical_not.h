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

struct logical_not_t : public logical_base<logical_not_t> {
    using operation_base<logical_not_t>::operator();
};

template <>
struct operation_signature<logical_not_t> {
    template <simd_mask T>
    static consteval void operator()(T&&) noexcept {}
};

template <>
struct fallback_impl<logical_not_t> {

    template <simd_mask T>
    requires cpo_invocable<bwnot_t, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(T&& val) noexcept(
        canonical_simd_type<T>) {
        return dx::bwnot(__DPL forward<T>(val));
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::logical_not_t logical_not{};
} // namespace cpo
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
