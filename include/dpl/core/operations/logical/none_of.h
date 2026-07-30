// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

// IWYU pragma: always_keep

#if !DPL_MODULES
#  include "dpl/core/basic/internal/abi.h"
#  include "dpl/core/basic/to_bitset.h"
#  include "dpl/core/concepts/canonical.h"
#  include "dpl/core/concepts/cpo_invocable.h"
#  include "dpl/core/concepts/simd_mask.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/operation/primitive.h"
#  include "dpl/core/immediate/const_mask.h"
#  include "dpl/core/type_traits/details/cpo_result.h"
#  include "dpl/std/concepts/convertible_to.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

void none_of(...) noexcept = delete;

struct none_of_t : private logical_base<none_of_t> {
    using operation_base<none_of_t>::operator();
};

template <>
struct operation_signature<none_of_t> {
    template <simd_mask T>
    static consteval void operator()(T&&) noexcept {}
    template <const_mask_like T>
    static consteval void operator()(T) noexcept {}
};

template <>
struct fallback_impl<none_of_t> {

    template <canonical_mask T>
    requires fixed_width_mask<T> && cpo_invocable<to_bitset_t, T> &&
        requires(T val) { dx::to_bitset(val); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr bool DPL_VECTORCALL operator()(T val) noexcept {
        using bitset_t = cpo_result_t<to_bitset_t, T>;
        return bitset_t() == dx::to_bitset(val);
    }

    template <const_mask_like T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr bool DPL_VECTORCALL operator()(T) noexcept {
        constexpr const_mask<T::width, T::value> cmask;
        return cmask == dx::zero;
    }
};

template <>
struct canonical_impl<none_of_t> {
    template <canonical_mask T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr bool operator()(T val) noexcept
    requires requires { none_of(internal::abi<T>, val); }
    {
        return none_of(internal::abi<T>, val);
    }
};

template <typename T>
concept unqualified_extended_none_of = requires(T&& val) {
    { none_of(__DPL forward<T>(val)) } -> explicitly_convertible_to<bool>;
};

template <>
struct extended_impl<none_of_t> {
    template <extended_mask T>
    requires unqualified_extended_none_of<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val) {
        return none_of(__DPL forward<T>(val));
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::none_of_t none_of{};
} // namespace cpo
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
