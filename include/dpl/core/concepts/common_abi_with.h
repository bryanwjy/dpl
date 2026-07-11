// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/concepts/simd_abi.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/core/type_traits/common_abi.h"
#  include "dpl/std/concepts/same_as.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

namespace atom {
template <typename A, typename B>
concept monotonic_common_abi =
    (datapar::scalable_abi<common_abi_t<A, B>> || datapar::scalable_abi<A> ||
        datapar::scalable_abi<B>) ||
    (common_abi_t<A, B>::size >= A::size &&
        common_abi_t<A, B>::size >= B::size);

template <typename A, typename B>
concept common_abi_with =
    requires {
        typename common_abi_t<A, B>;
        typename common_abi_t<B, A>;
    } && same_as<common_abi_t<A, B>, common_abi_t<B, A>> &&
    datapar::simd_abi<common_abi_t<A, B>> &&
    datapar::simd_abi<common_abi_t<B, A>> && monotonic_common_abi<A, B> &&
    monotonic_common_abi<B, A>;
} // namespace atom

template <typename A, typename B>
concept common_abi_with = simd_abi<A> && simd_abi<B> &&
    atom::common_abi_with<remove_cvref_t<A>, remove_cvref_t<B>>;

template <typename A, typename B>
concept same_abi_as =
    common_abi_with<A, B> && same_as<remove_cvref_t<A>, remove_cvref_t<B>>;

namespace internal {

template <typename... Ts>
inline constexpr bool is_common_abi_with = false;

template <typename A>
inline constexpr bool is_common_abi_with<A> = true;

template <typename A, typename B>
inline constexpr bool is_common_abi_with<A, B> = common_abi_with<A, B>;

template <typename A, typename B, typename... Cs>
requires common_abi_with<A, B>
inline constexpr bool is_common_abi_with<A, B, Cs...> =
    is_common_abi_with<common_abi_t<A, B>, Cs...>;

} // namespace internal

template <typename... Ts>
concept all_common_abi = internal::is_common_abi_with<Ts...>;

} // namespace datapar

__DPL_DEFAULT_NAMESPACE_END
