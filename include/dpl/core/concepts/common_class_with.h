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
concept common_class_with =
    (simd_vector<A> && simd_vector<B>) || (simd_mask<A> && simd_mask<B>);
}

DPL_EXPORT template <typename A, typename B>
concept common_class_with = atom::common_class_with<A, B> &&
    atom::common_abi_with<typename remove_cv_t<A>::abi_type,
        typename remove_cv_t<B>::abi_type>;

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
