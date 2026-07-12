// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/dispatch/interface.h"
#include "dpl/core/dispatch/private/type_traits.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/cpo_invocable.h"
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/type_traits/details/cpo_result.h"
#  include "dpl/std/type_traits/is_base_of.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::internal {

template <typename>
class basic_operation_base;

template <typename Derived, typename Base>
concept inherits_from = is_base_of_v<Base, Derived>;

template <typename D, typename... Ts>
concept extended_cpo_invocable = !inherits_from<D, basic_operation_base<D>> &&
    cpo_invocable<extended_impl<D>, Ts...>;

template <typename D, typename... Ts>
concept canonical_cpo_invocable = cpo_invocable<canonical_impl<D>, Ts...>;

template <typename D, typename... Ts>
concept fallback_cpo_invocable = cpo_invocable<fallback_impl<D>, Ts...>;

template <typename D, typename R, typename... Ts>
concept extended_cpo_invocable_r = extended_cpo_invocable<D, Ts...> &&
    equivalent_simd_type_with<R, cpo_result_t<extended_impl<D>, Ts...>>;

template <typename D, typename R, typename... Ts>
concept canonical_cpo_invocable_r = canonical_cpo_invocable<D, Ts...> &&
    canonical_simd_type<cpo_result_t<canonical_impl<D>, Ts...>> &&
    canonical_simd_type<R> &&
    same_as<R, cpo_result_t<canonical_impl<D>, Ts...>>;

template <typename D, typename R, typename... Ts>
concept fallback_cpo_invocable_r = fallback_cpo_invocable<D, Ts...> &&
    simd_type<cpo_result_t<fallback_impl<D>, Ts...>> && simd_type<R> &&
    same_as<R, cpo_result_t<fallback_impl<D>, Ts...>>;

template <auto>
void extended(...) noexcept = delete;

template <typename L, typename R>
concept common_abi_pair = !(simd_type<L> && simd_type<R>) ||
    common_abi_with<simd_abi_type_t<L>, simd_abi_type_t<R>>;

template <typename... Ts>
struct arguments_abi {};
template <typename... Ts>
using arguments_abi_t DPL_NODEBUG = typename arguments_abi<Ts...>::type;

template <simd_abi T>
struct arguments_abi<T> : simd_abi_type<T> {};

template <typename T, typename... Ts>
struct arguments_abi<T, Ts...> : arguments_abi<Ts...> {};
template <simd_type T, typename... Ts>
struct arguments_abi<T, Ts...> : arguments_abi<simd_abi_type_t<T>, Ts...> {};
template <simd_abi A, typename T, typename... Ts>
struct arguments_abi<A, T, Ts...> : arguments_abi<A, Ts...> {};
template <simd_abi A, simd_type T, typename... Ts>
requires common_abi_with<A, simd_abi_type_t<T>>
struct arguments_abi<A, T, Ts...> :
    arguments_abi<common_abi_t<A, simd_abi_type_t<T>>, Ts...> {};
template <simd_abi A, simd_type T, typename... Ts>
struct arguments_abi<A, T, Ts...> {};

template <typename T, typename... Ts>
concept common_abi_result = requires { typename arguments_abi_t<Ts...>; } &&
    common_abi_with<simd_abi_type_t<T>, common_abi_t<Ts...>>;

template <typename D, typename... Ts>
concept extended_nttp_invocable =
    !inherits_from<D, basic_operation_base<D>> && requires {
        typename integral_constant<D, D{}>;
        {
            extended<D{}>(internal::declarg<Ts>()...)
        } -> common_abi_result<Ts...>;
    };

template <typename D, typename R, typename... Ts>
concept extended_nttp_invocable_r =
    extended_nttp_invocable<D, Ts...> && requires {
        {
            extended<D{}>(internal::declarg<Ts>()...)
        } -> equivalent_simd_type_with<R>;
    };

template <typename D, typename R, typename... Ts>
concept extendable_operation_r = extended_cpo_invocable_r<D, R, Ts...> ||
    extended_nttp_invocable_r<D, R, Ts...>;

template <typename D, typename... Ts>
concept extendable_operation =
    extended_cpo_invocable<D, Ts...> || extended_nttp_invocable<D, Ts...>;

template <typename D>
struct operation_signature;

template <typename D, typename... Ts>
concept signature_compatible = requires(
    operation_signature<D> check) { check(internal::declarg<Ts>()...); };

template <typename T>
concept canonical_ornot_simd = canonical_simd_type<T> || !simd_type<T>;

template <typename D, typename... Ts>
concept cpo_eval_invocable = (... || simd_expression<Ts>) &&
    cpo_invocable<D, result_or_identity_t<Ts>...>;

template <typename... Ts>
concept extended_arguments = (... || extended_simd_type<Ts>);

template <typename... Ts>
inline constexpr bool all_same_abi_v = false;

template <simd_abi A>
inline constexpr bool all_same_abi_v<A> = true;

template <typename T, typename... Ts>
inline constexpr bool all_same_abi_v<T, Ts...> = all_same_abi_v<Ts...>;

template <simd_type T, typename... Ts>
inline constexpr bool all_same_abi_v<T, Ts...> =
    all_same_abi_v<simd_abi_type_t<T>, Ts...>;

template <simd_abi A, typename T, typename... Ts>
inline constexpr bool all_same_abi_v<A, T, Ts...> = all_same_abi_v<A, Ts...>;

template <simd_abi A, simd_type T, typename... Ts>
requires same_as<A, simd_abi_type_t<T>>
inline constexpr bool all_same_abi_v<A, T, Ts...> = all_same_abi_v<A, Ts...>;

template <simd_abi A, simd_type T, typename... Ts>
inline constexpr bool all_same_abi_v<A, T, Ts...> = false;

template <typename... Ts>
concept all_same_abi = all_same_abi_v<Ts...>;

template <typename... Ts>
concept has_simd_vector = (... || simd_vector<Ts>);

} // namespace datapar::internal

__DPL_DEFAULT_NAMESPACE_END
