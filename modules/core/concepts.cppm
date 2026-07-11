// Copyright 2025 Bryan Wong
module;

#define DPL_MODULES 1
#include "dpl/config.h"

export module dpl:core.concepts;
import :core.details.concepts;

// IWYU pragma: begin_exports
__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {
// NOLINTBEGIN(misc-unused-using-decls)
using __DPL datapar::broadcastable_to;
using __DPL datapar::canonical_mask;
using __DPL datapar::canonical_simd_type;
using __DPL datapar::canonical_vector;
using __DPL datapar::common_abi_with;
using __DPL datapar::common_simd_type_with;
using __DPL datapar::common_size_with;
using __DPL datapar::extended_mask;
using __DPL datapar::extended_simd_type;
using __DPL datapar::extended_vector;
using __DPL datapar::fixed_width_abi;
using __DPL datapar::scalable_abi;
using __DPL datapar::simd_abi;
using __DPL datapar::simd_element_for;
using __DPL datapar::simd_expression;
using __DPL datapar::simd_mask;
using __DPL datapar::simd_type;
using __DPL datapar::simd_vector;

// NOLINTEND(misc-unused-using-decls)
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END

// IWYU pragma: end_exports
