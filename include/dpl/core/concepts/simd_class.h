// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/concepts/simd_mask.h"
#include "dpl/core/concepts/simd_vector.h"

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar {
DPL_EXPORT template <typename T>
concept simd_class =
    atom::simd_basics<T> && (atom::simd_mask<T> || atom::vector_type<T>);

DPL_EXPORT template <typename T>
concept fixed_width_class =
    simd_class<T> && fixed_width_abi<typename T::abi_type>;

DPL_EXPORT template <typename T>
concept scalable_class = simd_class<T> && scalable_abi<typename T::abi_type>;

namespace internal {
template <fixed_width_abi A>
consteval size_t abi_width() noexcept {
    return A::size;
}

template <fixed_width_class T>
consteval size_t abi_width() noexcept {
    return abi_width<typename T::abi_type>();
}
} // namespace internal

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
