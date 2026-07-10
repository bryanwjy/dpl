// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

#if !DPL_ARCH_x86_64 || !DPL_SIMD_X86_SSE4_2
#  error "Unsupported platform"
#endif

#if !DPL_MODULES
#  include "dpl/core/fwd.h"
// IWYU pragma: begin_exports
#  include "dpl/core/basic/basic_mask.h"
#  include "dpl/core/basic/basic_vector.h"
#  include "dpl/core/numbers/ext.h"
// IWYU pragma: end_exports
#  include "dpl/core/concepts/simd_element.h"
#  include "dpl/core/type_traits/enable_simd_abi.h"
#  include "dpl/core/type_traits/simd_element_representation.h"
#  include "dpl/std/concepts/same_as.h"
#  include "dpl/std/type_traits/sequence.h"

#  include <immintrin.h>
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {
namespace dx = __DPL datapar;       // NOLINT
}

DPL_EXPORT namespace xmm = datapar::xmm; // NOLINT
DPL_DEFAULT_NAMESPACE_END

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {
DPL_EXPORT struct abi_tag;

template <typename T>
using representation_t DPL_NODEBUG =
    dx::simd_element_representation_t<abi_tag, T>;

template <typename T>
struct native_vector {};
template <typename T>
using native_vector_t = typename native_vector<representation_t<T>>::type;
template <>
struct native_vector<float> {
    using type = __m128;
};
template <>
struct native_vector<double> {
    using type = __m128d;
};
template <>
struct native_vector<ext::bfloat16> {
    using type = __m128bh;
};
template <>
struct native_vector<ext::float16> {
    using type = __m128h;
};

template <integral T>
requires different_from<remove_cv_t<T>, bool>
struct native_vector<T> {
    using type = __m128i;
};

template <typename T>
concept native_vector_type = same_as<T, __m128> //
    || same_as<T, __m128d>                      //
    || same_as<T, __m128i>                      //
    || same_as<T, __m128bh>                     //
    || same_as<T, __m128h>;

template <typename T>
concept vectorizable = requires { typename native_vector_t<T>; };

DPL_EXPORT struct abi_tag : simd_abi_base<abi_tag> {

    static constexpr size_t size = 16;
    static constexpr size_t alignment = 16;

    template <typename E>
    requires requires {
        typename native_vector_t<E>;
        requires native_vector_type<typename xmm::native_vector_t<E>>;
    }
    using native_vector = typename xmm::native_vector_t<E>;
    template <typename E>
    requires requires { typename native_vector<E>; }
    using native_mask = native_vector<E>;
};

DPL_EXPORT inline constexpr abi_tag abi{};

DPL_EXPORT template <typename E>
using vector DPL_NODEBUG = dx::basic_vector<E, abi_tag>;
DPL_EXPORT template <typename E>
using mask DPL_NODEBUG = dx::basic_mask<E, abi_tag>;

template <typename E>
concept simd_element = simd_element_for<E, abi_tag>;

template <typename E, size_t N>
concept sized_element = sizeof(E) == N && simd_element<E>;

template <typename E>
inline constexpr auto iota = make_index_sequence<abi_tag::size / sizeof(E)>{};

template <simd_element E>
using imask_t DPL_NODEBUG = bit_type_t<simd_abi_traits<abi_tag, E>::size>;

template <simd_element E, imask_t<E> V>
using cmask_t DPL_NODEBUG =
    dx::const_mask<simd_abi_traits<abi_tag, E>::size, V>;

} // namespace datapar::xmm

DPL_DEFAULT_NAMESPACE_END
