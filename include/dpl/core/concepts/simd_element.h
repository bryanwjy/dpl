// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/concepts/basic_element.h"
#include "dpl/core/concepts/simd_abi.h"
#include "dpl/core/concepts/simd_class.h"

#if !DPL_MODULES
#  include "dpl/std/bit/has_single_bit.h"
#  include "dpl/std/concepts/semiregular.h"
#  include "dpl/std/type_traits/is_empty.h"
#  include "dpl/std/type_traits/is_object.h"
#  include "dpl/std/type_traits/is_trivially_copyable.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar {
// TODO  Create example, e.g. bfloat16 on MSVC
/**
 * Custom builtins depending on ABI
 */
DPL_EXPORT template <typename T>
inline constexpr bool enable_extension_floating_point = false;
DPL_EXPORT template <typename T>
inline constexpr bool enable_extension_integral = false;

namespace internal {
template <typename T>
concept extension_element =
    (enable_extension_floating_point<T> || enable_extension_integral<T>) &&
    !is_empty_v<T> && is_object_v<T> && is_trivially_copyable_v<T> &&
    semiregular<T> && (__DPL has_single_bit(sizeof(T)));
} // namespace internal

DPL_EXPORT template <typename T>
concept simd_element =
    !same_as<T, bool> && (basic_element<T> || internal::extension_element<T>);

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
