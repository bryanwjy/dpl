// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/dispatch/maskable/base.h"
#include "dpl/core/dispatch/operation/algorithm.h"
#include "dpl/core/dispatch/operation/base.h"
#include "dpl/core/dispatch/operation/math.h"
#include "dpl/core/dispatch/operation/primitive.h"
#include "dpl/core/dispatch/private/concepts.h"

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

DPL_EXPORT template <auto D>
concept simd_operation =
    internal::inherits_from<decltype(D), internal::operation_base<decltype(D)>>;

DPL_EXPORT template <auto D>
concept simd_basic_operation = simd_operation<D> &&
    internal::inherits_from<decltype(D),
        internal::basic_operation_base<decltype(D)>>;

DPL_EXPORT template <auto D>
concept simd_primitive_operation = simd_operation<D> &&
    internal::inherits_from<decltype(D),
        internal::primitive_operation_base<decltype(D)>>;

DPL_EXPORT template <auto D>
concept simd_algorithm_operation = simd_operation<D> &&
    internal::inherits_from<decltype(D), internal::algorithm_base<decltype(D)>>;

DPL_EXPORT template <auto D>
concept simd_math_operation = simd_operation<D> &&
    internal::inherits_from<decltype(D),
        internal::math_operation_base<decltype(D)>>;

DPL_EXPORT template <auto D>
concept maskable_simd_operation = simd_operation<D> &&
    internal::inherits_from<decltype(D),
        internal::maskable_operation_base<decltype(D)>>;

DPL_EXPORT template <auto L, auto R>
concept same_operation_as =
    simd_operation<L> && simd_operation<R> && same_as<decltype(L), decltype(R)>;

DPL_EXPORT template <auto D, typename... Ts>
concept simd_invocable =
    simd_operation<D> && internal::cpo_invocable<decltype(D), Ts...>;

DPL_EXPORT template <auto D, typename... Ts>
concept simd_extension_invocable =
    (extended_simd_type<Ts> || ...) && simd_invocable<D, Ts...> &&
    internal::extended_cpo_invocable<decltype(D), Ts...>;

DPL_EXPORT template <auto D, typename... Ts>
concept simd_canonical_invocable =
    ((!extended_simd_type<Ts>) && ...) && simd_invocable<D, Ts...> &&
    internal::cpo_invocable<internal::canonical_impl<decltype(D)>, Ts...>;

} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
