// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/simd_expression.h"
#  include "dpl/std/type_traits/remove_cvref.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::internal {

template <typename T>
struct result_or_decayed : remove_cvref<T> {};
template <typename T>
using result_or_decayed_t DPL_NODEBUG = typename result_or_decayed<T>::type;
template <simd_expression T>
struct result_or_decayed<T> : simd_expression_result<T> {};

template <typename T>
struct result_or_identity {
    using type DPL_NODEBUG = T;
};
template <typename T>
using result_or_identity_t DPL_NODEBUG = typename result_or_identity<T>::type;

template <simd_expression T>
struct result_or_identity<T> : simd_expression_result<T> {};

} // namespace datapar::internal

__DPL_DEFAULT_NAMESPACE_END
