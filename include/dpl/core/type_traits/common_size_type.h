// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/type_traits/representation.h"
#include "dpl/core/type_traits/simd_element_type.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/common_size_with.h"
#  include "dpl/core/concepts/simd_class.h"
#  include "dpl/core/concepts/simd_element.h"
#  include "dpl/std/bit/bit_type.h"
#  include "dpl/std/concepts/common_with.h"
#  include "dpl/std/type_traits/common_type.h"
#  include "dpl/std/type_traits/type_identity.h"
#  include "dpl/std/type_traits/underlying_type.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

DPL_EXPORT template <typename... T>
struct common_size_type {};

DPL_EXPORT template <simd_element T>
struct common_size_type<T> : simd_element_type<T> {};
DPL_EXPORT template <simd_class T>
struct common_size_type<T> : simd_element_type<T> {};
DPL_EXPORT template <simd_class T>
struct common_size_type<T, T> : simd_element_type<T> {};
DPL_EXPORT template <simd_element T>
struct common_size_type<T, T> : simd_element_type<T> {};

DPL_EXPORT template <simd_class A, simd_element B>
struct common_size_type<A, B> : common_size_type<simd_element_type_t<A>, B> {};
DPL_EXPORT template <simd_element A, simd_class B>
struct common_size_type<A, B> : common_size_type<A, simd_element_type_t<B>> {};
DPL_EXPORT template <simd_class A, simd_class B>
struct common_size_type<A, B> :
    common_size_type<simd_element_type_t<A>, simd_element_type_t<B>> {};

DPL_EXPORT template <typename... Ts>
using common_size_type_t = typename common_size_type<Ts...>::type;

DPL_EXPORT template <typename T, typename U, typename... Ts>
requires requires { typename common_size_type_t<T, U>; }
struct common_size_type<T, U, Ts...> :
    common_size_type<common_size_type_t<T, U>, Ts...> {};

template <typename T, typename U>
concept common_size_with_common_type = common_with<T, U> && requires {
    requires common_size_with<T, common_type_t<T, U>> &&
        common_size_with<U, common_type_t<T, U>>;
};

DPL_EXPORT template <simd_element A, common_size_with<A> B>
struct common_size_type<A, B> {
private:
    static consteval auto choose_type() noexcept {
        if constexpr (same_as<A, B>) {
            return type_identity<A>{};
        } else if constexpr (common_size_with_common_type<A, B>) {
            return common_type<A, B>{};
        } else if constexpr (signed_integral<A> && signed_integral<B>) {
            return signed_representation<A>{};
        } else if constexpr (unsigned_integral<A> && unsigned_integral<B>) {
            return unsigned_representation<A>{};
        } else if constexpr (enumeration<A> && enumeration<B>) {
            return common_size_type<underlying_type_t<A>,
                underlying_type_t<B>>{};
        } else if constexpr (enumeration<A>) {
            return common_size_type<underlying_type_t<A>, B>{};
        } else if constexpr (enumeration<B>) {
            return common_size_type<A, underlying_type_t<B>>{};
        } else {
            return bit_type<sizeof(A) * char_bit_v>{};
        }
    }

public:
    using type DPL_NODEBUG = typename decltype(choose_type())::type;
};

DPL_EXPORT template <typename...>
struct common_size_simd {};
DPL_EXPORT template <simd_class... Ts>
using common_size_simd_t = typename common_size_simd<Ts...>::type;

DPL_EXPORT template <simd_class T>
struct common_size_simd<T> {
    using type DPL_NODEBUG = T;
};

DPL_EXPORT template <simd_class A, simd_class B, simd_class... Ts>
requires requires { typename common_size_simd_t<A, B>; }
struct common_size_simd<A, B, Ts...> :
    common_size_simd<common_size_simd_t<A, B>, Ts...> {};

DPL_EXPORT template <simd_element E1, simd_abi A1, simd_element E2, simd_abi A2>
requires requires { typename common_size_type_t<E1, E2>; }
struct common_size_simd<basic_simd<E1, A1>, basic_simd<E2, A2>> {
    using type DPL_NODEBUG =
        basic_simd<common_size_type_t<E1, E2>, common_abi_t<A1, A2>>;
};

DPL_EXPORT template <simd_element E1, simd_abi A1, simd_element E2, simd_abi A2>
requires requires { typename common_size_type_t<E1, E2>; }
struct common_size_simd<basic_simd_mask<E1, A1>, basic_simd_mask<E2, A2>> {
    using type DPL_NODEBUG =
        basic_simd_mask<common_size_type_t<E1, E2>, common_abi_t<A1, A2>>;
};

DPL_EXPORT template <simd_type T1, common_class_with<T1> T2>
requires requires { typename common_size_type_t<T1, T2>; }
struct common_size_simd<T1, T2> {
    using type DPL_NODEBUG =
        basic_simd<common_size_type_t<T1, T2>, common_abi_t<T1, T2>>;
};

DPL_EXPORT template <simd_mask_type T1, common_class_with<T1> T2>
requires requires { typename common_size_type_t<T1, T2>; }
struct common_size_simd<T1, T2> {
    using type DPL_NODEBUG =
        basic_simd_mask<common_size_type_t<T1, T2>, common_abi_t<T1, T2>>;
};

} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
