// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/type_traits/element_count.h"

#if !DPL_MODULES
#  include "dpl/std/utility/sequence.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {
template <typename... Ts>
inline constexpr auto iota_sequence =
    make_index_sequence<element_count<Ts...>>{};
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
