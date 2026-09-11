// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/dispatch/maskable/accumulation.h"
#include "dpl/core/dispatch/maskable/base.h"
#include "dpl/core/dispatch/maskable/predicate.h"
#include "dpl/core/dispatch/operation/algorithm.h"
#include "dpl/core/dispatch/operation/base.h"
#include "dpl/core/dispatch/operation/math.h"
#include "dpl/core/dispatch/operation/primitive.h"
#include "dpl/core/dispatch/private/concepts.h"

#if !DPL_MODULES
#  include "dpl/std/type_traits/remove_cvref.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

template <typename D>
concept simd_operation = internal::inherits_from<remove_cv_t<D>,
    internal::operation_base<remove_cv_t<D>>>;

template <typename D>
concept simd_basic_operation = simd_operation<D> &&
    internal::inherits_from<remove_cv_t<D>,
        internal::basic_operation_base<remove_cv_t<D>>>;

template <typename D>
concept simd_primitive_operation = simd_operation<D> &&
    internal::inherits_from<remove_cv_t<D>,
        internal::primitive_operation_base<remove_cv_t<D>>>;

template <typename D>
concept simd_algorithm_operation = simd_operation<D> &&
    internal::inherits_from<remove_cv_t<D>,
        internal::algorithm_base<remove_cv_t<D>>>;

template <typename D>
concept simd_math_operation = simd_operation<D> &&
    internal::inherits_from<remove_cv_t<D>,
        internal::math_operation_base<remove_cv_t<D>>>;

template <typename D>
concept maskable_simd_operation = simd_operation<D> &&
    internal::inherits_from<remove_cv_t<D>,
        internal::maskable_operation_base<remove_cv_t<D>>>;

template <typename D>
concept maskable_simd_accumulation = simd_operation<D> &&
    internal::inherits_from<remove_cv_t<D>,
        internal::maskable_accumulation_base<remove_cv_t<D>>>;

template <typename D>
concept maskable_simd_predicate = simd_operation<D> &&
    internal::inherits_from<remove_cv_t<D>,
        internal::maskable_predicate_base<remove_cv_t<D>>>;

template <typename L, auto R>
concept same_operation_as =
    simd_operation<L> && simd_operation<remove_cvref_t<decltype(R)>> &&
    same_as<remove_cv_t<L>, remove_cvref_t<decltype(R)>>;

template <typename D, typename... Ts>
concept simd_invocable =
    simd_operation<D> && internal::cpo_invocable<remove_cv_t<D>, Ts...>;

template <typename D, typename... Ts>
concept simd_extension_invocable =
    (extended_simd_type<Ts> || ...) && simd_invocable<D, Ts...> &&
    internal::extended_cpo_invocable<remove_cv_t<D>, Ts...>;

template <typename D, typename... Ts>
concept simd_canonical_invocable =
    ((!extended_simd_type<Ts>) && ...) && simd_invocable<D, Ts...> &&
    internal::cpo_invocable<internal::canonical_impl<remove_cv_t<D>>, Ts...>;

[[nodiscard]] consteval bool is_simd_operation(auto cpo) noexcept {
    return simd_operation<decltype(cpo)>;
}
[[nodiscard]] consteval bool is_simd_basic_operation(auto cpo) noexcept {
    return simd_basic_operation<decltype(cpo)>;
}
[[nodiscard]] consteval bool is_simd_primitive_operation(auto cpo) noexcept {
    return simd_primitive_operation<decltype(cpo)>;
}
[[nodiscard]] consteval bool is_simd_algorithm_operation(auto cpo) noexcept {
    return simd_algorithm_operation<decltype(cpo)>;
}
[[nodiscard]] consteval bool is_simd_math_operation(auto cpo) noexcept {
    return simd_math_operation<decltype(cpo)>;
}
[[nodiscard]] consteval bool is_maskable_simd_operation(auto cpo) noexcept {
    return maskable_simd_operation<decltype(cpo)>;
}
[[nodiscard]] consteval bool is_maskable_simd_accumulation(auto cpo) noexcept {
    return maskable_simd_accumulation<decltype(cpo)>;
}
[[nodiscard]] consteval bool is_maskable_simd_predicate(auto cpo) noexcept {
    return maskable_simd_predicate<decltype(cpo)>;
}

// Use == for same_operation_as

template <typename... Args>
[[nodiscard]] consteval bool is_simd_invocable(auto cpo) noexcept {
    return simd_invocable<decltype(cpo), Args...>;
}
template <typename... Args>
[[nodiscard]] consteval bool is_simd_extension_invocable(auto cpo) noexcept {
    return simd_extension_invocable<decltype(cpo), Args...>;
}
template <typename... Args>
[[nodiscard]] consteval bool is_simd_canonical_invocable(auto cpo) noexcept {
    return simd_canonical_invocable<decltype(cpo), Args...>;
}

// TODO how to integrate with reflection API?

} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
