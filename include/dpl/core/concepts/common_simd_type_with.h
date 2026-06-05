// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/concepts/common_abi_with.h"
#include "dpl/core/concepts/simd_mask.h"
#include "dpl/core/concepts/simd_vector.h"

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

namespace atom {
template <typename A, typename B>
concept common_simd_type_with =
    (datapar::simd_vector<A> && datapar::simd_vector<B>) ||
    (datapar::simd_mask<A> && datapar::simd_mask<B>);
}

DPL_EXPORT template <typename A, typename B>
concept common_simd_type_with = atom::common_simd_type_with<A, B> &&
    common_abi_with<simd_abi_type_t<A>, simd_abi_type_t<B>>;

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
