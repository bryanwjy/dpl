// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/basic/to_basic_type.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/core/concepts/basic_type.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_class.h"
#  include "dpl/std/concepts/integral.h"
#  include "dpl/std/concepts/integral_constant_like.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
template <typename T>
concept extraction_index = integral_constant_like<T> || integral<T>;

void extract(
    simd_abi auto, simd_class auto, extraction_index auto) noexcept = delete;

struct extract_t {
    /**
     * Prevent basic_simd_class overload from falling back to simd_class
     * overload
     */
    template <basic_simd_class T>
    static constexpr void operator()(T, auto) noexcept = delete;

    template <basic_simd_class T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr typename T::value_type operator()(
        T src, extraction_index auto idx) noexcept
    requires requires { extract(internal::abi<T>, src, idx); }
    {
        return extract(internal::abi<T>, src, idx);
    }

    template <simd_class T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr same_as<typename T::value_type> auto operator()(
        T src, extraction_index auto idx) noexcept {
        if constexpr (requires { extract(internal::abi<T>, src, idx); }) {
            return extract(internal::abi<T>, src, idx);
        } else {
            return operator()(dx::to_basic_type(src), idx);
        }
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::extract_t extract{};
}
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
