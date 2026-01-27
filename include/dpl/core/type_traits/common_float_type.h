// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/type_traits/simd_element_type.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/common_float_with.h"
#  include "dpl/core/concepts/simd_class.h"
#  include "dpl/core/concepts/simd_element.h"
#  include "dpl/std/type_traits/type_identity.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

DPL_EXPORT template <typename... T>
struct common_float_type {};
DPL_EXPORT template <simd_element T>
struct common_float_type<T> : simd_element_type<T> {};
DPL_EXPORT template <simd_class T>
struct common_float_type<T> : simd_element_type<T> {};
DPL_EXPORT template <simd_class T>
struct common_float_type<T, T> : simd_element_type<T> {};
DPL_EXPORT template <simd_element T>
struct common_float_type<T, T> : simd_element_type<T> {};

DPL_EXPORT template <simd_class A, simd_element B>
struct common_float_type<A, B> :
    common_float_type<simd_element_type_t<A>, B> {};
DPL_EXPORT template <simd_element A, simd_class B>
struct common_float_type<A, B> :
    common_float_type<A, simd_element_type_t<B>> {};
DPL_EXPORT template <simd_class A, simd_class B>
struct common_float_type<A, B> :
    common_float_type<simd_element_type_t<A>, simd_element_type_t<B>> {};

DPL_EXPORT template <typename... Ts>
using common_float_type_t = typename common_float_type<Ts...>::type;

DPL_EXPORT template <typename T, typename U, typename... Ts>
requires requires { typename common_float_type_t<T, U>; }
struct common_float_type<T, U, Ts...> :
    common_float_type<common_float_type_t<T, U>, Ts...> {};

DPL_EXPORT template <simd_element A, common_float_with<A> B>
struct common_float_type<A, B> {
private:
    static consteval auto choose_type() noexcept {
        if constexpr (same_as<A, B>) {
            return type_identity<A>{};
        } else if constexpr (common_float_with<A, float> &&
            common_float_with<B, float>) {
            return type_identity<float>{};
        } else {
            static_assert(
                common_float_with<A, double> && common_float_with<B, double>);
            return type_identity<double>{};
        }
    }

public:
    using type DPL_NODEBUG = decltype(choose_type())::type;
};
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
