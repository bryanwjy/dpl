// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/common_abi_with.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_vector.h"
#  include "dpl/core/type_traits/simd_abi_type.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

template <typename T, typename A = simd_abi_type_t<T>>
concept extended_operation_result =
    simd_class<T> && simd_abi<A> && common_abi_with<simd_abi_type_t<T>, A>;

template <typename T, typename A = simd_abi_type_t<T>>
concept extended_operation_vector =
    extended_operation_result<T, A> && simd_vector<T>;

template <typename T, typename A = simd_abi_type_t<T>>
concept extended_operation_mask =
    extended_operation_result<T, A> && simd_mask<T>;

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
