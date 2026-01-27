// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/type_traits/bit_type.h"
#include "dpl/core/type_traits/common_float_type.h"
#include "dpl/core/type_traits/simd_element_type.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/common_arithmetic_with.h"
#  include "dpl/core/concepts/common_bits_with.h"
#  include "dpl/core/concepts/common_float_with.h"
#  include "dpl/core/concepts/simd_class.h"
#  include "dpl/core/concepts/simd_element.h"
#  include "dpl/std/concepts/common_with.h"
#  include "dpl/std/type_traits/make_signed.h"
#  include "dpl/std/type_traits/make_unsigned.h"
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
    using type DPL_NODEBUG = decltype(choose_type())::type;
};

DPL_EXPORT template <simd_element A, common_arithmetic_with<A> B>
struct common_arithmetic_type<A, B> {
private:
    static consteval auto choose_type() noexcept {
        static_assert(!enumeration<A> && !enumeration<B>);
        if constexpr (same_as<A, B>) {
            return type_identity<A>{};
        } else if constexpr (common_with<A, B> &&
            common_arithmetic_with<A, common_type_t<A, B>> &&
            common_arithmetic_with<B, common_type_t<A, B>>) {
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
    using type DPL_NODEBUG = decltype(choose_type());
};

} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
