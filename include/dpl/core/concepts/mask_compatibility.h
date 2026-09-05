// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/concepts/cpo_invocable.h"
#include "dpl/core/concepts/equivalence.h"
#include "dpl/core/concepts/simd_mask.h"
#include "dpl/core/concepts/simd_type.h"
#if !DPL_MODULES
#  include "dpl/core/type_traits/details/cpo_result.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::internal {

template <typename M, typename T>
concept compatible_mask_with =
    simd_mask<M> && simd_type<T> && common_mask_with<M, simd_mask_type_t<T>>;

template <typename M, typename T>
concept exact_mask_for =
    compatible_mask_with<M, T> && equivalent_mask_with<M, simd_mask_type_t<T>>;

template <typename M, typename D, typename... Ts>
concept result_mask_for =
    cpo_invocable<D, Ts...> && exact_mask_for<M, cpo_result_t<D, Ts...>>;

} // namespace datapar::internal

__DPL_DEFAULT_NAMESPACE_END
