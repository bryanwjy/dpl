// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/basic/to_canonical.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/basic_type.h"
#  include "dpl/core/concepts/decayable.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_class.h"
#  include "dpl/std/concepts/integral.h"
#  include "dpl/std/concepts/integral_constant_like.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
template <typename T>
concept extraction_index = integral_constant_like<T> || integral<T>;

void extract(...) noexcept = delete;

struct extract_t {
    /**
     * Prevent canonical_class overload from falling back to simd_class
     * overload
     */
    template <canonical_class T, typename I>
    static constexpr void operator()(T, I) noexcept = delete;

    template <canonical_class T, extraction_index I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr typename T::value_type operator()(T src, I idx) noexcept
    requires requires { extract(internal::abi<T>, src, idx); }
    {
        return extract(internal::abi<T>, src, idx);
    }

    template <extended_class T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr same_as<typename T::value_type> auto operator()(
        T src, extraction_index auto idx) noexcept {
        if constexpr (requires { extract(src, idx); }) {
            return extract(src, idx);
        } else {
            return operator()(dx::to_canonical(src), idx);
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
