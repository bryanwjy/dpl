// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/concepts/simd_abi.h"
#include "dpl/core/concepts/simd_element_for.h"

#if !DPL_MODULES
#  include "dpl/core/fwd/basic.h"
#  include "dpl/std/type_traits/is_object.h"
#  include "dpl/std/type_traits/is_trivially_copyable.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar {
DPL_EXPORT template <typename T>
inline constexpr bool enable_simd_type = false;
DPL_EXPORT template <simd_abi A, simd_element_for<A> E>
inline constexpr bool enable_simd_type<basic_vector<E, A>> = true;

namespace atom {
template <typename T>
concept simd_basics = is_object_v<T> && is_trivially_copyable_v<T> && requires {
    typename T::value_type;
    typename T::abi_type;
    requires simd_abi<typename T::abi_type>;
} && semiregular<T>;

template <typename T>
concept vector_type = enable_simd_type<T> &&
    requires {
        requires simd_element_for<typename T::value_type, typename T::abi_type>;
        requires sizeof(typename T::value_type) <= T::abi_type::size;
    } &&
    explicitly_convertible_to<T,
        typename T::abi_type::template native_vector<typename T::value_type>>;
} // namespace atom

DPL_EXPORT template <typename T>
concept simd_vector = atom::simd_basics<T> && atom::vector_type<T>;

DPL_EXPORT template <typename T>
concept scalable_vector = simd_vector<T> && scalable_abi<typename T::abi_type>;
DPL_EXPORT template <typename T>
concept fixed_width_vector =
    simd_vector<T> && fixed_width_abi<typename T::abi_type>;
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
