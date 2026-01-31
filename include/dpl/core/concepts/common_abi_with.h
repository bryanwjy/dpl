// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/concepts/common_class_with.h"
#include "dpl/core/concepts/simd_abi.h"
#include "dpl/core/concepts/simd_class.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/std/concepts/same_as.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

/**
 * Pairwise trait of common_abi where both template arguments satisfies
 * simd_abi, may be partially specialized to define a common operating
 * ABI. The common abi functionalities are invoked whenever there are
 * differing ABIs between simd classes for non-basic binary or ternary
 * operations.
 */
DPL_EXPORT template <typename... T>
struct common_abi {};

namespace internal {
template <typename T>
concept simd_class_or_abi = simd_class<T> || simd_abi<T>;
}

DPL_EXPORT template <internal::simd_class_or_abi... Ts>
using common_abi_t = typename common_abi<Ts...>::type;

DPL_EXPORT template <simd_abi A>
struct common_abi<A> {
    using type DPL_NODEBUG = A;
};

DPL_EXPORT template <simd_class T>
struct common_abi<T> {
    using type DPL_NODEBUG = typename T::abi_type;
};

DPL_EXPORT template <simd_abi A>
struct common_abi<A, A> {
    using type DPL_NODEBUG = A;
};

DPL_EXPORT template <simd_class A, simd_abi B>
struct common_abi<A, B> : common_abi<typename A::abi_type, B> {};
DPL_EXPORT template <simd_abi A, simd_class B>
struct common_abi<A, B> : common_abi<A, typename B::abi_type> {};
DPL_EXPORT template <simd_class A, simd_class B>
struct common_abi<A, B> :
    common_abi<typename A::abi_type, typename B::abi_type> {};

DPL_EXPORT template <typename T, typename U, typename... Ts>
requires requires { typename common_abi_t<T, U>; }
struct common_abi<T, U, Ts...> : common_abi<common_abi_t<T, U>, Ts...> {};

DPL_EXPORT template <typename A, typename B>
concept common_abi_with =
    simd_abi<A> && simd_abi<B> && (same_as<A, B> || requires {
        typename common_abi_t<A, B>;
        typename common_abi_t<B, A>;
        requires simd_abi<common_abi_t<A, B>> && simd_abi<common_abi_t<B, A>>;
        requires same_as<common_abi_t<A, B>, common_abi_t<B, A>>;
    });

DPL_EXPORT template <typename A, typename B>
concept same_abi_as = common_abi_with<A, B> && same_as<A, B>;

DPL_EXPORT template <typename A, typename B>
concept common_abi_simd_with =
    simd_class<A> && simd_class<B> && common_class_with<A, B> &&
    common_abi_with<typename A::abi_type, typename B::abi_type>;

DPL_EXPORT template <typename A, typename B>
concept same_abi_simd_as = common_abi_simd_with<A, B> &&
    same_abi_as<typename A::abi_type, typename B::abi_type>;
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
