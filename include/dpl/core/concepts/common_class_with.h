// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/concepts/common_abi_with.h"
#include "dpl/core/concepts/simd_class.h"

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

namespace atom {
template <typename A, typename B>
concept common_class_with = (atom::vector_type<A> && atom::vector_type<B>) ||
    (atom::simd_mask<A> && atom::simd_mask<B>);
}

DPL_EXPORT template <typename A, typename B>
concept common_class_with = atom::simd_basics<A> && atom::simd_basics<B> &&
    atom::common_class_with<A, B> &&
    atom::common_abi_with<typename A::abi_type, typename B::abi_type>;

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
