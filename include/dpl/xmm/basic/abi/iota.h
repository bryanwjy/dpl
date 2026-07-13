// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

// IWYU pragma: private, include "dpl/xmm/basic/abi.h"

#include "dpl/xmm/basic/abi/xmmdef.h"

#if !DPL_MODULES
#  include "dpl/core/type_traits/simd_abi_traits.h"
#  include "dpl/std/type_traits/sequence.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {
template <typename E>
inline constexpr auto iota =
    make_index_sequence<simd_abi_traits<abi_tag, E>::size>{};
} // namespace datapar::xmm

__DPL_DEFAULT_NAMESPACE_END
