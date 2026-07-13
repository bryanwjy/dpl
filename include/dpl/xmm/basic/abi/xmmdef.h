// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

// IWYU pragma: private, include "dpl/xmm/basic/abi.h"

#if !DPL_MODULES
#  include "dpl/core/basic/basic_mask.h"
#  include "dpl/core/basic/basic_vector.h"
#  include "dpl/core/numbers/ext.h"
#  include "dpl/core/type_traits/simd_abi_traits.h"
#  include "dpl/core/type_traits/simd_element_representation.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

// NOLINTBEGIN(misc-unused-alias_decl)
namespace datapar::xmm {
namespace dx = __DPL datapar;
struct abi_tag;
template <typename E>
using vector = dx::basic_vector<E, abi_tag>;
template <typename E>
using mask = dx::basic_mask<E, abi_tag>;
} // namespace datapar::xmm
namespace xmm = datapar::xmm;
// NOLINTEND(misc-unused-alias_decl)

__DPL_DEFAULT_NAMESPACE_END
