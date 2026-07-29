// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/concepts/common_abi_with.h"
#include "dpl/core/concepts/common_size_with.h"
#include "dpl/core/concepts/cpo_invocable.h"
#include "dpl/core/concepts/simd_mask.h"
#if !DPL_MODULES
#  include "dpl/core/type_traits/details/cpo_result.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::internal {
/**
 * Scalable ABI workaround
 *
 * Since scalable backend types cannot be encapsulated,
 * the mask types may not be differentiated on certain backends,
 * e.g. ARM SVE
 */
template <typename A>
concept typed_mask = simd_abi<A> &&
    different_from<make_canonical_mask_t<int8, A>,
        make_canonical_mask_t<int32, A>>;

template <typename M, typename T>
concept compatible_mask_with = simd_mask<M> && simd_type<T> &&
    common_abi_with<simd_abi_type_t<M>, simd_abi_type_t<T>> &&
    (!typed_mask<simd_abi_type_t<T>> ||
        common_size_with<simd_element_type_t<M>, simd_element_type_t<T>>);

template <typename M, typename T>
concept exact_mask_for = compatible_mask_with<M, T> &&
    same_abi_as<simd_abi_type_t<M>, simd_abi_type_t<T>>;

template <typename M, typename D, typename... Ts>
concept result_mask_for =
    cpo_invocable<D, Ts...> && exact_mask_for<M, cpo_result_t<D, Ts...>>;

} // namespace datapar::internal

__DPL_DEFAULT_NAMESPACE_END
