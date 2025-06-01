// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h" // IWYU pragma: keep

#include "dpl/blob.h"
#include "dpl/xmm/concepts.h"

#if !DPL_MODULES
#  include <immintrin.h>

#  include <concepts>
#endif

DPL_NAMESPACE_BEGIN

DPL_SIMD_ABI_NAMESPACE_BEGIN(xmm)

DPL_EXPORT_BEGIN

template <>
struct __DPL_PUBLIC_TEMPLATE native_type<std::byte> :
    native_type<unsigned char> {};

template <>
struct __DPL_PUBLIC_TEMPLATE native_type<b128> {
    using type = __m128i;
};

template <sized_floating_point<4> T>
struct __DPL_PUBLIC_TEMPLATE native_type<T> {
    using type = __m128;
};

template <sized_floating_point<8> T>
struct __DPL_PUBLIC_TEMPLATE native_type<T> {
    using type = __m128d;
};

template <std::integral I>
struct __DPL_PUBLIC_TEMPLATE native_type<I> {
    using type = __m128i;
};

#if DPL_SIMD_X86_AVX512FP16 & DPL_SUPPORTS_FLOAT16
template <>
struct __DPL_PUBLIC_TEMPLATE native_type<float16> {
    using type = __m128h;
};
#endif

#if DPL_SIMD_X86_AVX512BFP16 & DPL_SUPPORTS_BFLOAT16
template <>
struct __DPL_PUBLIC_TEMPLATE native_type<bfloat16> {
    using type = __m128bh;
};
#endif

template <typename To, typename From>
__DPL_HIDE_FROM_ABI void native_cast(...) noexcept = delete;

template <xmm_element To, xmm_element From>
requires std::same_as<native_type_t<To>, native_type_t<From>>
DPL_ATTRIBUTES(
    __HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) constexpr native_type_t<To>
native_cast(native_type_t<From> src) noexcept {
    return src;
}

template <std::same_as<b128> To, xmm_element From>
DPL_ATTRIBUTES(__HIDE_FROM_ABI, FLATTEN, NODISCARD) constexpr native_type_t<To> native_cast(
    native_type_t<From> src) noexcept {
    return native_cast<std::byte, From>(src);
}

template <sized_floating_point<4> To, std::integral From>
DPL_ATTRIBUTES(__HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) inline native_type_t<To>
native_cast(native_type_t<From> src) noexcept {
    return _mm_castsi128_ps(src);
}

template <sized_floating_point<4> To, sized_floating_point<8> From>
DPL_ATTRIBUTES(__HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) inline native_type_t<To>
native_cast(native_type_t<From> src) noexcept {
    return _mm_castpd_ps(src);
}

template <sized_floating_point<8> To, std::integral From>
DPL_ATTRIBUTES(__HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) inline native_type_t<To>
native_cast(native_type_t<From> src) noexcept {
    return _mm_castsi128_pd(src);
}

template <sized_floating_point<8> To, sized_floating_point<4> From>
DPL_ATTRIBUTES(__HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) inline native_type_t<To>
native_cast(native_type_t<From> src) noexcept {
    return _mm_castps_pd(src);
}

template <std::integral To, sized_floating_point<8> From>
DPL_ATTRIBUTES(__HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) inline native_type_t<To>
native_cast(native_type_t<From> src) noexcept {
    return _mm_castpd_si128(src);
}

template <std::integral To, sized_floating_point<4> From>
DPL_ATTRIBUTES(__HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) inline native_type_t<To>
native_cast(native_type_t<From> src) noexcept {
    return _mm_castps_si128(src);
}

template <typename From, typename To>
concept reinterpretable_as = xmm_element<From> && xmm_element<To> && requires {
    typename native_type_t<To>;
    typename native_type_t<From>;
} && requires(native_type_t<From> src) {
    { native_cast<To, From>(src) } -> std::same_as<native_type_t<To>>;
};

template <typename To, typename From>
concept reinterpretable_from = reinterpretable_as<From, To>;

#if DPL_SIMD_X86_AVX512FP16 & DPL_SUPPORTS_FLOAT16

template <std::integral To>
DPL_ATTRIBUTES(__HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) inline native_type_t<To>
native_cast(native_type_t<float16> src) noexcept {
    return _mm_castph_si128(src);
}

template <sized_floating_point<4> To>
DPL_ATTRIBUTES(__HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) inline native_type_t<To>
native_cast(native_type_t<float16> src) noexcept {
    return _mm_castph_ps(src);
}

template <sized_floating_point<8> To>
DPL_ATTRIBUTES(__HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) inline native_type_t<To>
native_cast(native_type_t<float16> src) noexcept {
    return _mm_castph_pd(src);
}

template <std::same_as<float16> To, std::integral From>
DPL_ATTRIBUTES(__HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) inline native_type_t<To>
native_cast(native_type_t<From> src) noexcept {
    return _mm_castsi128_ph(src);
}

template <std::same_as<float16> To, sized_floating_point<4> From>
DPL_ATTRIBUTES(__HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) inline native_type_t<To>
native_cast(native_type_t<From> src) noexcept {
    return _mm_castps_ph(src);
}

template <std::same_as<float16> To, sized_floating_point<8> From>
DPL_ATTRIBUTES(__HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) inline native_type_t<To>
native_cast(native_type_t<From> src) noexcept {
    return _mm_castpd_ph(src);
}

#endif

DPL_EXPORT_END

DPL_SIMD_ABI_NAMESPACE_END(xmm)

DPL_NAMESPACE_END
