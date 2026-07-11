// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/concepts/simd_abi.h"
#include "dpl/core/concepts/simd_element.h"

#if !DPL_MODULES
#  include "dpl/core/type_traits/enable_simd_mask.h"
#  include "dpl/core/type_traits/enable_simd_vector.h"
#  include "dpl/core/type_traits/simd_abi_type.h"
#  include "dpl/core/type_traits/simd_element_type.h"
#  include "dpl/core/type_traits/simd_mask_type.h"
#  include "dpl/core/type_traits/simd_native_type.h"
#  include "dpl/core/type_traits/simd_value_type.h"
#  include "dpl/core/type_traits/simd_vector_type.h"
#  include "dpl/std/type_traits/is_object.h"
#  include "dpl/std/type_traits/is_trivially_copyable.h"
#  include "dpl/std/type_traits/remove_cvref.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar {
namespace atom {
// Attention: On scalable ABIs, it is currently undefined whether sizeless types
// are trivially copyable
template <typename T>
concept simd_type = is_object_v<T> && is_trivially_copyable_v<T> && requires {
    typename simd_value_type_t<T>;
    typename simd_abi_type_t<T>;
    typename simd_element_type_t<T>;
    typename simd_native_type_t<T>;
    typename simd_mask_type_t<T>;
} && simd_abi<simd_abi_type_t<T>>;

template <typename T>
concept simd_vector = enable_simd_vector<T> &&
    same_as<simd_element_type_t<T>, simd_value_type_t<T>> &&
    simd_element_for<simd_element_type_t<T>, simd_abi_type_t<T>> &&
    explicitly_convertible_to<T, simd_native_type_t<T>>;

template <typename M>
concept simd_mask =
    enable_simd_mask<M> && same_as<bool, simd_value_type_t<M>> &&
    atom::simd_type<simd_vector_type_t<M>> &&
    atom::simd_vector<simd_vector_type_t<M>> &&
    explicitly_convertible_to<M, simd_native_type_t<M>>;
} // namespace atom

template <typename T>
concept simd_type = atom::simd_type<remove_cvref_t<T>> &&
    (atom::simd_vector<remove_cvref_t<T>> ||
        atom::simd_mask<remove_cvref_t<T>>);

template <typename T>
concept fixed_width_simd_type =
    simd_type<T> && fixed_width_abi<simd_abi_type_t<T>>;

template <typename T>
concept scalable_simd_type = simd_type<T> && scalable_abi<simd_abi_type_t<T>>;
} // namespace datapar

__DPL_DEFAULT_NAMESPACE_END
