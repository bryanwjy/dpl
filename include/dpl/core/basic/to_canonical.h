// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/basic/to_native_type.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/canonical.h"
#  include "dpl/core/concepts/extended.h"
#  include "dpl/core/type_traits/canonical_type.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

DPL_EXPORT template <canonical_simd_type T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr T to_canonical(T src) noexcept {
    return src;
}

DPL_EXPORT template <extended_vector T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr canonical_type_t<T>
    DPL_VECTORCALL to_canonical(T src) noexcept {
    if constexpr (explicitly_convertible_to<T, canonical_type_t<T>>) {
        return static_cast<canonical_type_t<T>>(src);
    } else {
        return datapar::to_native_type(src);
    }
}

} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
