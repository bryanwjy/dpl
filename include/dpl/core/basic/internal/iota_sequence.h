// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/type_traits/simd_abi_traits.h"
#  include "dpl/std/type_traits/sequence.h"
#  include "dpl/std/utility/ignore.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {
template <typename T, typename U = __DPL ignore_t>
requires requires {
    typename integral_constant<size_t, simd_abi_traits<T, U>::size>;
}
using iota_sequence_t = make_index_sequence<simd_abi_traits<T, U>::size>;

template <typename T, typename U = __DPL ignore_t>
requires requires { typename iota_sequence_t<T, U>; }
inline constexpr auto iota_sequence = iota_sequence_t<T, U>{};

} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
