// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

#if !DPL_ARCH_x86_64 || !DPL_SIMD_X86_SSE4_2
#  error "Unsupported platform"
#endif

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/core/concepts/simd_element.h"
#  include "dpl/std/concepts/enumeration.h"
#  include "dpl/std/concepts/same_as.h"

#  include <immintrin.h>
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {
DPL_EXPORT namespace xmm {}
} // namespace datapar

namespace datapar::xmm {
namespace dx = __DPL datapar; // NOLINT
}

DPL_EXPORT namespace xmm = datapar::xmm; // NOLINT
DPL_DEFAULT_NAMESPACE_END

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {
template <typename T>
struct native_vector {};
template <typename T>
using native_vector_t = typename native_vector<T>::type;
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
template <enumeration T>
struct native_vector<T> {
    using type = __m128i;
};

template <typename T>
concept native_vector_type = same_as<T, decltype(_mm_setzero_ps())> //
    || same_as<T, decltype(_mm_setzero_pd())>                       //
    || same_as<T, decltype(_mm_setzero_si128())>
#if DPL_SIMD_X86_AVX2
    || same_as<T, decltype(_mm_cvtneps_avx_pbh(_mm_setzero_ps()))>
#endif
#if DPL_SIMD_X86_AVX512F16
    || same_as<T, decltype(_mm_setzero_ph())>
#endif
    ;

inline constexpr struct template_barrier_t {
    __DPL_HIDE_FROM_ABI explicit constexpr template_barrier_t() noexcept =
        default;
} barrier{};

DPL_EXPORT
struct abi_tag {
    static constexpr size_t size = 16;
    static constexpr size_t alignment = 16;

    template <simd_element E>
    requires requires {
        typename native_vector_t<E>;
        requires native_vector_type<typename xmm::native_vector_t<E>>;
    }
    using native_type = typename xmm::native_vector_t<E>;
    template <simd_element E>
    requires requires { typename native_type<E>; }
    using native_mask = native_type<E>;
};

DPL_EXPORT inline constexpr abi_tag abi{};

DPL_EXPORT template <simd_element E>
using simd DPL_NODEBUG = dx::basic_simd<E, abi_tag>;
DPL_EXPORT template <simd_element E>
using mask DPL_NODEBUG = dx::basic_simd_mask<E, abi_tag>;
} // namespace datapar::xmm

DPL_DEFAULT_NAMESPACE_END
