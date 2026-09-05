// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/concepts/canonical.h"
#include "dpl/core/concepts/simd_mask.h"
#include "dpl/core/concepts/simd_type.h"
#include "dpl/core/concepts/simd_vector.h"

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar {

template <typename T>
concept extended_simd_type =
    simd_type<remove_cvref_t<T>> && !canonical_simd_type<remove_cvref_t<T>>;

template <typename T>
concept extended_vector =
    simd_vector<remove_cvref_t<T>> && extended_simd_type<T>;

template <typename T>
concept extended_mask = simd_mask<remove_cvref_t<T>> && extended_simd_type<T>;

namespace internal {
template <typename T>
concept unextended_type = !extended_simd_type<T>;

template <typename T, typename Op, typename... Ts>
concept unextended_terminal_of = unextended_type<T> &&
    (... && unextended_type<Ts>) && cpo_invocable<Op, Ts..., T>;
} // namespace internal

} // namespace datapar

__DPL_DEFAULT_NAMESPACE_END
