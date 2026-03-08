// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/simd_class.h"
#  include "dpl/core/type_traits/simd_element_type.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

template <simd_class T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr auto DPL_VECTORCALL to_native_vector(T src) noexcept {
    using A = typename T::abi_type;
    using E = simd_element_type_t<T>;
    using native_type = typename A::template native_type<E>;
    return static_cast<native_type>(src);
}

} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
