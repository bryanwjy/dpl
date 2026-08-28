// Copyright 2025 Bryan Wong
module;

#define DPL_MODULES 1
#include "dpl/config.h"

export module dpl:core.type_traits;
import :core.details.type_traits;

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {
// NOLINTBEGIN(misc-unused-using-decls)
using __DPL datapar::canonical_type;
using __DPL datapar::canonical_type_t;
using __DPL datapar::common_abi;
using __DPL datapar::common_abi_t;
using __DPL datapar::common_size_type;
using __DPL datapar::common_size_type_t;
using __DPL datapar::make_canonical_mask;
using __DPL datapar::make_canonical_mask_t;
using __DPL datapar::make_canonical_vector;
using __DPL datapar::make_canonical_vector_t;
using __DPL datapar::rebind_simd;
using __DPL datapar::rebind_simd_t;
using __DPL datapar::signed_representation;
using __DPL datapar::signed_representation_t;
using __DPL datapar::simd_abi_base;
using __DPL datapar::simd_abi_traits;
using __DPL datapar::simd_abi_type;
using __DPL datapar::simd_abi_type_t;
using __DPL datapar::simd_element_representation;
using __DPL datapar::simd_element_representation_t;
using __DPL datapar::simd_element_type;
using __DPL datapar::simd_element_type_t;
using __DPL datapar::simd_expression_result;
using __DPL datapar::simd_expression_result_t;
using __DPL datapar::simd_mask_base;
using __DPL datapar::simd_mask_type;
using __DPL datapar::simd_mask_type_t;
using __DPL datapar::simd_native_tuple;
using __DPL datapar::simd_native_tuple_t;
using __DPL datapar::simd_native_type;
using __DPL datapar::simd_native_type_t;
using __DPL datapar::simd_value_type;
using __DPL datapar::simd_value_type_t;
using __DPL datapar::simd_vector_base;
using __DPL datapar::simd_vector_type;
using __DPL datapar::simd_vector_type_t;
using __DPL datapar::unsigned_representation;
using __DPL datapar::unsigned_representation_t;
// NOLINTEND(misc-unused-using-decls)
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
