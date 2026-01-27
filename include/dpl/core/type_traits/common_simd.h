// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/type_traits/common_abi.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/simd_class.h"
#  include "dpl/core/concepts/simd_mask_type.h"
#  include "dpl/core/concepts/simd_type.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

DPL_EXPORT template <template <typename...> class, typename...>
struct common_simd;

DPL_EXPORT template <template <typename...> class CommonTrait, simd_class... Ts>
using common_simd_t = typename common_simd<CommonTrait, Ts...>::type;

DPL_EXPORT template <template <typename...> class CommonTrait, simd_class T>
struct common_simd<CommonTrait, T> : CommonTrait<T> {};

DPL_EXPORT template <template <typename...> class CommonTrait, simd_class A,
    simd_class B, simd_class... Ts>
requires requires { typename common_simd_t<CommonTrait, A, B>; }
struct common_simd<CommonTrait, A, B, Ts...> :
    common_simd<CommonTrait, common_simd_t<CommonTrait, A, B>, Ts...> {};

DPL_EXPORT template <template <typename...> class CommonTrait, simd_element AE,
    simd_abi AA, simd_element BE, simd_abi BA>
requires requires { typename CommonTrait<AE, AA>::type; }
struct common_simd<CommonTrait, basic_simd<AE, AA>, basic_simd<BE, BA>> {
    using type DPL_NODEBUG =
        basic_simd<typename CommonTrait<AE, BE>::type, common_abi_t<AA, BA>>;
};

DPL_EXPORT template <template <typename...> class CommonTrait, simd_element AE,
    simd_abi AA, simd_element BE, simd_abi BA>
requires requires { typename CommonTrait<AE, AA>::type; }
struct common_simd<CommonTrait, simd_mask<AE, AA>, simd_mask<BE, BA>> {
    using type DPL_NODEBUG =
        simd_mask<typename CommonTrait<AE, BE>::type, common_abi_t<AA, BA>>;
};

/**
 * Pairwise traits of common_simd may be partially specialized to return
 * non-basic simd_types or simd_mask_types. However, if the partial
 * specialization returns a type that is does not satisfy equivalent_class_as
 * with the default definition, behaviour is undefined.
 */
DPL_EXPORT template <template <typename...> class CommonTrait, simd_type A,
    common_class_with<A> B>
requires requires { typename CommonTrait<A, B>::type; }
struct common_simd<CommonTrait, A, B> {
    using type DPL_NODEBUG =
        basic_simd<typename CommonTrait<A, B>::type, common_abi_t<A, B>>;
};

DPL_EXPORT template <template <typename...> class CommonTrait, simd_mask_type A,
    common_class_with<A> B>
requires requires { typename CommonTrait<A, B>::type; }
struct common_simd<CommonTrait, A, B> {
    using type DPL_NODEBUG =
        simd_mask<typename CommonTrait<A, B>::type, common_abi_t<A, B>>;
};

} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
