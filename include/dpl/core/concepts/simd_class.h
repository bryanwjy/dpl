// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/concepts/simd_basics.h"

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar {
DPL_EXPORT template <typename T>
concept simd_class = atom::simd_basics<remove_cv_t<T>> &&
    (atom::simd_vector<remove_cv_t<T>> || atom::simd_mask<remove_cv_t<T>>);

DPL_EXPORT template <typename T>
concept fixed_width_class =
    simd_class<T> && fixed_width_abi<typename remove_cv_t<T>::abi_type>;

DPL_EXPORT template <typename T>
concept scalable_class =
    simd_class<T> && scalable_abi<typename remove_cv_t<T>::abi_type>;

namespace internal {
template <fixed_width_abi A>
consteval size_t abi_width() noexcept {
    return remove_cv_t<A>::size;
}

template <fixed_width_class T>
consteval size_t abi_width() noexcept {
    return abi_width<typename remove_cv_t<T>::abi_type>();
}
} // namespace internal

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
