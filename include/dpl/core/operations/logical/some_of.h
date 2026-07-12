// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

// IWYU pragma: always_keep

#include "dpl/core/operations/logical/all_of.h"
#include "dpl/core/operations/logical/any_of.h"

#if !DPL_MODULES
#  include "dpl/core/basic/internal/abi.h"
#  include "dpl/core/concepts/canonical.h"
#  include "dpl/core/concepts/simd_mask.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/operation/primitive.h"
#  include "dpl/core/immediate/const_mask.h"
#  include "dpl/std/concepts/convertible_to.h"
#  include "dpl/std/concepts/invocable.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

void some_of(...) noexcept = delete;

struct some_of_t : private logical_base<some_of_t> {
    using operation_base<some_of_t>::operator();
};

template <>
struct operation_signature<some_of_t> {
    template <simd_mask T>
    static consteval void operator()(T&&) noexcept {}
    template <const_mask_like T>
    static consteval void operator()(T) noexcept {}
};

template <>
struct fallback_impl<some_of_t> {
    template <simd_type T>
    requires regular_invocable<any_of_t, T> && regular_invocable<all_of_t, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr bool DPL_VECTORCALL operator()(T&& val) noexcept(
        canonical_simd_type<T>) {
        return dx::any_of(val) && !dx::all_of(__DPL forward<T>(val));
    }

    template <const_mask_like T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr bool DPL_VECTORCALL operator()(T) noexcept {
        constexpr const_mask<T::width, T::value> cmask;
        return cmask != dx::zero && cmask != dx::all_bits;
    }
};

template <>
struct canonical_impl<some_of_t> {
    template <canonical_mask T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr bool operator()(T val) noexcept
    requires requires { some_of(internal::abi<T>, val); }
    {
        return some_of(internal::abi<T>, val);
    }
};

template <typename T>
concept unqualified_extended_some_of = requires(T&& val) {
    { some_of(__DPL forward<T>(val)) } -> explicitly_convertible_to<bool>;
};

template <>
struct extended_impl<some_of_t> {
    template <extended_mask T>
    requires unqualified_extended_some_of<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val) {
        return some_of(__DPL forward<T>(val));
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::some_of_t some_of{};
} // namespace cpo
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
