// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/basic/internal/abi.h"
#include "dpl/core/basic/to_canonical.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/core/concepts/canonical.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/operation/basic.h"
#  include "dpl/core/type_traits/canonical_type.h"
#  include "dpl/core/type_traits/simd_value_type.h"
#  include "dpl/std/concepts/integral.h"
#  include "dpl/std/concepts/integral_constant_like.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
template <typename T>
concept extraction_index = integral_constant_like<T> || integral<T>;

void extract(...) noexcept = delete;

struct extract_t : private basic_operation_base<extract_t> {
    using operation_base<extract_t>::operator();
};

template <>
struct operation_signature<extract_t> {
    static consteval void operator()(
        simd_type auto&&, extraction_index auto) noexcept {}
};

template <>
struct canonical_impl<extract_t> {
    template <canonical_simd_type T, extraction_index I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr simd_value_type_t<T> operator()(T src, I idx) noexcept
    requires requires { extract(internal::abi<T>, src, idx); }
    {
        return extract(internal::abi<T>, src, idx);
    }
};

template <>
struct fallback_impl<extract_t> {
    template <extended_simd_type T, extraction_index I>
    requires cpo_invocable<canonical_impl<extract_t>, canonical_type_t<T>, I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE)
    static constexpr simd_value_type_t<T> operator()(T&& src, I idx) noexcept {
        return canonical_impl<extract_t>::operator()(
            dx::to_canonical(__DPL forward<T>(src)), idx);
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::extract_t extract{};
}
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
