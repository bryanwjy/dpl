// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

#if !DPL_ARCH_x86_64 || !DPL_SIMD_X86_SSE4_2
#  error "Unsupported platform"
#endif

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_element_for.h"
#  include "dpl/std/concepts/enumeration.h"
#  include "dpl/std/concepts/same_as.h"
#  include "dpl/std/type_traits/conditional.h"

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
using lane_representation_t DPL_NODEBUG =
    dx::simd_element_representation_t<abi_tag, T>;

template <typename T>
struct native_vector {};
template <typename T>
using native_vector_t = typename native_vector<lane_representation_t<T>>::type;
template <>
struct native_vector<float> {
    using type = __m128;
};
template <>
struct native_vector<double> {
    using type = __m128d;
};
template <brain_float T>
struct native_vector<T> {
    using type = __m128bh;
};
template <floating_point T>
requires (sizeof(T) == 2) && (!brain_float<T>)
struct native_vector<T> {
    using type = __m128h;
};

template <integral T>
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

DPL_EXPORT struct abi_tag {

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
using simd DPL_NODEBUG = dx::basic_vector<E, abi_tag>;
DPL_EXPORT template <typename E>
using mask DPL_NODEBUG = dx::basic_mask<E, abi_tag>;

template <typename T>
concept float16_like =
    floating_point<T> && !brain_float<T> && sizeof(T) == sizeof(int16);

template <typename T>
concept bfloat16_like = floating_point<T> && brain_float<T>;

namespace details {
// Used to defer template instantiation
template <typename T0, typename... Ts>
using front_t DPL_NODEBUG =
    dpl::conditional_t<(... && !is_same_v<Ts, T0>), T0, T0>;
} // namespace details

template <typename E, size_t N>
concept sized_element = sizeof(E) == N && simd_element_for<E, abi_tag>;
} // namespace datapar::xmm

namespace datapar {
template <>
inline constexpr bool enable_simd_abi<xmm::abi_tag> = true;
}

DPL_DEFAULT_NAMESPACE_END
