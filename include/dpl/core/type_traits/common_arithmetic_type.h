// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/type_traits/common_float_type.h"
#include "dpl/core/type_traits/representation.h"
#include "dpl/core/type_traits/simd_element_type.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/common_arithmetic_with.h"
#  include "dpl/core/concepts/common_bits_with.h"
#  include "dpl/core/concepts/common_float_with.h"
#  include "dpl/core/concepts/simd_class.h"
#  include "dpl/core/concepts/simd_element.h"
#  include "dpl/std/concepts/common_with.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

DPL_EXPORT template <typename... T>
struct common_arithmetic_type {};
DPL_EXPORT template <simd_element T>
struct common_arithmetic_type<T> : simd_element_type<T> {};
DPL_EXPORT template <simd_class T>
struct common_arithmetic_type<T> : simd_element_type<T> {};
DPL_EXPORT template <simd_class T>
struct common_arithmetic_type<T, T> : simd_element_type<T> {};
DPL_EXPORT template <simd_element T>
struct common_arithmetic_type<T, T> : simd_element_type<T> {};
DPL_EXPORT template <simd_class A, simd_element B>
struct common_arithmetic_type<A, B> :
    common_arithmetic_type<simd_element_type_t<A>, B> {};
DPL_EXPORT template <simd_element A, simd_class B>
struct common_arithmetic_type<A, B> :
    common_arithmetic_type<A, simd_element_type_t<B>> {};
DPL_EXPORT template <simd_class A, simd_class B>
struct common_arithmetic_type<A, B> :
    common_arithmetic_type<simd_element_type_t<A>, simd_element_type_t<B>> {};

DPL_EXPORT template <typename... Ts>
using common_arithmetic_type_t = typename common_arithmetic_type<Ts...>::type;

DPL_EXPORT template <typename T, typename U, typename... Ts>
requires requires { typename common_arithmetic_type_t<T, U>; }
struct common_arithmetic_type<T, U, Ts...> :
    common_arithmetic_type<common_arithmetic_type_t<T, U>, Ts...> {};

DPL_EXPORT template <simd_element A, common_arithmetic_with<A> B>
requires common_float_with<A, B>
struct common_arithmetic_type<A, B> : common_float_type<A, B> {};

DPL_EXPORT template <simd_element A, common_bits_with<A> B>
requires (!same_as<A, B> && !enumeration<B>) && enumeration<A>
struct common_arithmetic_type<A, B> : underlying_type<A> {};

DPL_EXPORT template <simd_element A, common_bits_with<A> B>
requires (!same_as<A, B> && !enumeration<A>) && enumeration<B>
struct common_arithmetic_type<A, B> : underlying_type<B> {};

DPL_EXPORT template <simd_element A, common_bits_with<A> B>
requires (enumeration<A> || enumeration<B>)
struct common_arithmetic_type<A, B> {
private:
    static consteval auto choose_type() noexcept {
        if constexpr (same_as<A, B> || enumeration<A>) {
            return underlying_type<A>{};
        } else {
            return underlying_type<B>{};
        }
    }

public:
    using type DPL_NODEBUG = typename decltype(choose_type())::type;
};

template <typename T, typename U>
concept common_arithmetic_with_common_type = common_with<T, U> && requires {
    requires common_arithmetic_with<T, common_type_t<T, U>> &&
        common_arithmetic_with<U, common_type_t<T, U>>;
};

DPL_EXPORT template <simd_element A, common_arithmetic_with<A> B>
struct common_arithmetic_type<A, B> {
private:
    static consteval auto choose_type() noexcept {
        static_assert(!enumeration<A> && !enumeration<B>);
        if constexpr (same_as<A, B>) {
            return type_identity<A>{};
        } else if constexpr (common_arithmetic_with_common_type<A, B>) {
            return common_type<A, B>{};
        } else if constexpr (signed_integral<A> && signed_integral<B>) {
            return signed_representation<A>{};
        } else if constexpr (unsigned_integral<A> && unsigned_integral<B>) {
            return unsigned_representation<A>{};
        } else {
            static_assert(same_as<basic_element_t<A>, basic_element_t<B>>);
            return basic_element<A>{};
        }
    }

public:
    using type DPL_NODEBUG = typename decltype(choose_type())::type;
};

DPL_EXPORT template <typename...>
struct common_arithmetic_simd {};
DPL_EXPORT template <simd_class... Ts>
using common_arithmetic_simd_t = typename common_arithmetic_simd<Ts...>::type;

DPL_EXPORT template <simd_class T>
struct common_arithmetic_simd<T> {
    using type DPL_NODEBUG = T;
};

DPL_EXPORT template <simd_class A, simd_class B, simd_class... Ts>
requires requires { typename common_arithmetic_simd_t<A, B>; }
struct common_arithmetic_simd<A, B, Ts...> :
    common_arithmetic_simd<common_arithmetic_simd_t<A, B>, Ts...> {};

DPL_EXPORT template <simd_element E1, simd_abi A1, simd_element E2, simd_abi A2>
requires requires { typename common_arithmetic_type_t<E1, E2>; }
struct common_arithmetic_simd<basic_simd<E1, A1>, basic_simd<E2, A2>> {
    using type DPL_NODEBUG =
        basic_simd<common_arithmetic_type_t<E1, E2>, common_abi_t<A1, A2>>;
};

DPL_EXPORT template <simd_element E1, simd_abi A1, simd_element E2, simd_abi A2>
requires requires { typename common_arithmetic_type_t<E1, E2>; }
struct common_arithmetic_simd<basic_simd_mask<E1, A1>,
    basic_simd_mask<E2, A2>> {
    using type DPL_NODEBUG =
        basic_simd_mask<common_arithmetic_type_t<E1, E2>, common_abi_t<A1, A2>>;
};

/**
 * These pairwise traits may be partially specialized to return
 * non-basic simd_types or simd_mask_types. However, if the partial
 * specialization returns a type that is does not satisfy equivalent_class_as
 * with the default definition, behaviour is undefined.
 */
DPL_EXPORT template <simd_type T1, common_class_with<T1> T2>
requires requires { typename common_arithmetic_type_t<T1, T2>; }
struct common_arithmetic_simd<T1, T2> {
    using type DPL_NODEBUG =
        basic_simd<common_arithmetic_type_t<T1, T2>, common_abi_t<T1, T2>>;
};

DPL_EXPORT template <simd_mask_type T1, common_class_with<T1> T2>
requires requires { typename common_arithmetic_type_t<T1, T2>; }
struct common_arithmetic_simd<T1, T2> {
    using type DPL_NODEBUG =
        basic_simd_mask<common_arithmetic_type_t<T1, T2>, common_abi_t<T1, T2>>;
};

} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
