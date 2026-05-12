// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/type_traits/simd_abi_traits.h"

#if !DPL_MODULES
#  include "dpl/std/utility/sequence.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {
DPL_EXPORT template <typename... Ts>
requires requires {
    typename integral_constant<size_t, simd_abi_traits<Ts...>::size>;
}
inline constexpr auto iota_sequence =
    make_index_sequence<simd_abi_traits<Ts...>::size>{};

} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
