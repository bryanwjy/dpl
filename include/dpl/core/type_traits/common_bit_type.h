// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/type_traits/bit_type.h"
#include "dpl/core/type_traits/simd_element_type.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/common_bits_with.h"
#  include "dpl/core/concepts/simd_class.h"
#  include "dpl/core/concepts/simd_element.h"
#  include "dpl/std/concepts/common_with.h"
#  include "dpl/std/type_traits/common_type.h"
#  include "dpl/std/type_traits/make_signed.h"
#  include "dpl/std/type_traits/make_unsigned.h"
#  include "dpl/std/type_traits/type_identity.h"
#  include "dpl/std/type_traits/underlying_type.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

DPL_EXPORT template <typename... T>
struct common_bits_type {};

DPL_EXPORT template <simd_element T>
struct common_bits_type<T> : simd_element_type<T> {};
DPL_EXPORT template <simd_class T>
struct common_bits_type<T> : simd_element_type<T> {};
DPL_EXPORT template <simd_class T>
struct common_bits_type<T, T> : simd_element_type<T> {};
DPL_EXPORT template <simd_element T>
struct common_bits_type<T, T> : simd_element_type<T> {};

DPL_EXPORT template <simd_class A, simd_element B>
struct common_bits_type<A, B> : common_bits_type<simd_element_type_t<A>, B> {};
DPL_EXPORT template <simd_element A, simd_class B>
struct common_bits_type<A, B> : common_bits_type<A, simd_element_type_t<B>> {};
DPL_EXPORT template <simd_class A, simd_class B>
struct common_bits_type<A, B> :
    common_bits_type<simd_element_type_t<A>, simd_element_type_t<B>> {};

DPL_EXPORT template <typename... Ts>
using common_bits_type_t = typename common_bits_type<Ts...>::type;

DPL_EXPORT template <typename T, typename U, typename... Ts>
requires requires { typename common_bits_type_t<T, U>; }
struct common_bits_type<T, U, Ts...> :
    common_bits_type<common_bits_type_t<T, U>, Ts...> {};

DPL_EXPORT template <simd_element A, common_bits_with<A> B>
requires (enumeration<A> || enumeration<B>)
struct common_bits_type<A, B> {
private:
    static consteval auto choose_type() noexcept {
        if constexpr (same_as<A, B> && flag_enumeration<A> &&
            flag_enumeration<B>) {
            return type_identity<A>{};
        } else if constexpr (same_as<A, B> || enumeration<A>) {
            return underlying_type<A>{};
        } else {
            return underlying_type<B>{};
        }
    }

public:
    using type DPL_NODEBUG = typename decltype(choose_type())::type;
};

DPL_EXPORT template <simd_element A, common_bits_with<A> B>
struct common_bits_type<A, B> {
private:
    static consteval auto choose_type() noexcept {
        static_assert(!enumeration<A> && !enumeration<B>);
        if constexpr (same_as<A, B>) {
            return type_identity<A>{};
        } else if constexpr (common_with<A, B> &&
            common_bits_with<A, common_type_t<A, B>> &&
            common_bits_with<B, common_type_t<A, B>>) {
            return common_type<A, B>{};
        } else if constexpr (signed_integral<A> && signed_integral<B>) {
            return make_signed<internal::bit_type_for_t<A>>{};
        } else if constexpr (unsigned_integral<A> && unsigned_integral<B>) {
            return make_unsigned<internal::bit_type_for_t<A>>{};
        } else {
            return internal::bit_type_for<A>{};
        }
    }

public:
    using type DPL_NODEBUG = typename decltype(choose_type())::type;
};

DPL_EXPORT template <typename...>
struct common_bits_simd {};
DPL_EXPORT template <simd_class... Ts>
using common_bits_simd_t = typename common_bits_simd<Ts...>::type;

DPL_EXPORT template <simd_class T>
struct common_bits_simd<T> {
    using type DPL_NODEBUG = T;
};

DPL_EXPORT template <simd_class A, simd_class B, simd_class... Ts>
requires requires { typename common_bits_simd_t<A, B>; }
struct common_bits_simd<A, B, Ts...> :
    common_bits_simd<common_bits_simd_t<A, B>, Ts...> {};

DPL_EXPORT template <simd_element E1, simd_abi A1, simd_element E2, simd_abi A2>
requires requires { typename common_bits_type_t<E1, E2>; }
struct common_bits_simd<basic_simd<E1, A1>, basic_simd<E2, A2>> {
    using type DPL_NODEBUG =
        basic_simd<common_bits_type_t<E1, E2>, common_abi_t<A1, A2>>;
};

DPL_EXPORT template <simd_element E1, simd_abi A1, simd_element E2, simd_abi A2>
requires requires { typename common_bits_type_t<E1, E2>; }
struct common_bits_simd<simd_mask<E1, A1>, simd_mask<E2, A2>> {
    using type DPL_NODEBUG =
        simd_mask<common_bits_type_t<E1, E2>, common_abi_t<A1, A2>>;
};

DPL_EXPORT template <simd_type T1, common_class_with<T1> T2>
requires requires { typename common_bits_type_t<T1, T2>; }
struct common_bits_simd<T1, T2> {
    using type DPL_NODEBUG =
        basic_simd<common_bits_type_t<T1, T2>, common_abi_t<T1, T2>>;
};

DPL_EXPORT template <simd_mask_type T1, common_class_with<T1> T2>
requires requires { typename common_bits_type_t<T1, T2>; }
struct common_bits_simd<T1, T2> {
    using type DPL_NODEBUG =
        simd_mask<common_bits_type_t<T1, T2>, common_abi_t<T1, T2>>;
};

} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
