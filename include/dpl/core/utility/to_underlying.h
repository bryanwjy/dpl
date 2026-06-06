// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/simd_type.h"
#  include "dpl/core/operations/reinterpret.h"
#  include "dpl/core/type_traits/simd_abi_type.h"
#  include "dpl/core/type_traits/simd_element_representation.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {
template <simd_type T>
requires different_from<
    simd_element_representation_t<simd_abi_type_t<T>, simd_element_type_t<T>>,
    simd_element_type_t<T>>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr auto to_underlying(T simd) noexcept {
    using To = simd_element_representation_t<simd_abi_type_t<T>,
        simd_element_type_t<T>>;
    return datapar::reinterpret<To>(simd);
}
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
