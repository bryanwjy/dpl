// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/type_traits/bit_type.h"
#include "dpl/core/type_traits/common_float_type.h"
#include "dpl/core/type_traits/simd_element_type.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/common_float_with.h"
#  include "dpl/core/concepts/common_integral_with.h"
#  include "dpl/core/concepts/common_order_with.h"
#  include "dpl/core/concepts/simd_class.h"
#  include "dpl/core/concepts/simd_element.h"
#  include "dpl/std/type_traits/make_signed.h"
#  include "dpl/std/type_traits/make_unsigned.h"
#  include "dpl/std/type_traits/underlying_type.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

DPL_EXPORT template <typename...>
struct common_order_type {};
DPL_EXPORT template <simd_element T>
struct common_order_type<T> : simd_element_type<T> {};
DPL_EXPORT template <simd_class T>
struct common_order_type<T> : simd_element_type<T> {};
DPL_EXPORT template <simd_class T>
struct common_order_type<T, T> : simd_element_type<T> {};
DPL_EXPORT template <simd_element T>
struct common_order_type<T, T> : simd_element_type<T> {};

DPL_EXPORT template <simd_class A, simd_element B>
struct common_order_type<A, B> :
    common_order_type<simd_element_type_t<A>, B> {};
DPL_EXPORT template <simd_element A, simd_class B>
struct common_order_type<A, B> :
    common_order_type<A, simd_element_type_t<B>> {};
DPL_EXPORT template <simd_class A, simd_class B>
struct common_order_type<A, B> :
    common_order_type<simd_element_type_t<A>, simd_element_type_t<B>> {};

DPL_EXPORT template <typename... Ts>
using common_order_type_t = typename common_order_type<Ts...>::type;

DPL_EXPORT template <simd_element A, common_order_with<A> B>
requires common_float_with<A, B>
struct common_order_type<A, B> : common_float_type<A, B> {};

DPL_EXPORT template <simd_element A, common_order_with<A> B>
requires common_integral_with<A, B>
struct common_order_type<A, B> {
private:
    static consteval auto choose_type() noexcept {
        static_assert(!enumeration<A> && !enumeration<B>);
        if constexpr (same_as<A, B>) {
            return type_identity<A>{};
        } else if constexpr (signed_integral<A> && signed_integral<B>) {
            return make_signed<internal::bit_type_for_t<A>>{};
        } else {
            static_assert(unsigned_integral<A> && unsigned_integral<B>);
            return make_unsigned<internal::bit_type_for_t<A>>{};
        }
    }

public:
    using type DPL_NODEBUG = decltype(choose_type())::type;
};

DPL_EXPORT template <simd_element A, common_order_with<A> B>
requires (enumeration<A> || enumeration<B>)
struct common_order_type<A, B> {
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

} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
