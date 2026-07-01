// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

#if DPL_HAS_CXX26_EXTENSIONS
DPL_DISABLE_WARNING_PUSH()
DPL_DISABLE_WARNING("-Wc++26-extensions")
#endif

#if !DPL_ARCH_x86_64 || !DPL_SIMD_X86_SSE4_2
#  error "Unsupported platform"
#endif

#include "dpl/xmm/basic/abi.h"

#if !DPL_MODULES
#  include "dpl/core/type_traits/simd_abi_traits.h"
#  include "dpl/std/bit/bit_cast.h"
#  include "dpl/std/concepts/convertible_to.h"
#  include "dpl/std/utility/forward.h"

#  include <immintrin.h>
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::xmm {

DPL_EXPORT template <simd_element E, convertible_to<E>... Args>
requires (... && !same_as<Args, bool>) &&
    (sizeof...(Args) == simd_abi_traits<E, abi_tag>::size)
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr simd<E> initialize(abi_tag tag, Args&&... args) noexcept {
    if consteval {
#if !DPL_COMPILER_MSVC
        using array = E[abi_tag::size / sizeof(E)];
        alignas(abi_tag::alignment)
            array buffer{static_cast<E>(__DPL forward<Args>(args))...};
        return __DPL bit_cast<native_vector_t<E>>(buffer);
#else
        if constexpr (same_as<float, representation_t<E>>) {
            return native_vector_t<E>{
                .m128_f32 = {__DPL bit_cast<float>(
                    static_cast<E>( __DPL forward<Args>(args)))...},
            };
        } else if constexpr (same_as<double, representation_t<E>>) {
            return native_vector_t<E>{
                .m128_f64 = {__DPL bit_cast<double>(
                    static_cast<E>( __DPL forward<Args>(args)))...},
            };
        } else if constexpr (bfloat16_like<representation_t<E>> ||
            float16_like<representation_t<E>>) {
            return native_vector_t<E>{
                .m128_i16 = {__DPL bit_cast<int16>(
                    static_cast<E>( __DPL forward<Args>(args)))...},
            };
        } else if constexpr (same_as<representation_t<E>, int32>) {
            return native_vector_t<E>{
                .m128i_i32 = {__DPL bit_cast<int32>(static_cast<E>(args))...},
            };
        } else if constexpr (same_as<representation_t<E>, int16>) {
            return native_vector_t<E>{
                .m128i_i16 = {__DPL bit_cast<int16>(static_cast<E>(args))...},
            };
        } else if constexpr (same_as<representation_t<E>, int8>) {
            return native_vector_t<E>{
                .m128i_i8 = {__DPL bit_cast<int8>(static_cast<E>(args))...},
            };
        } else if constexpr (same_as<representation_t<E>, int64>) {
            return native_vector_t<E>{
                .m128i_i64 = {__DPL bit_cast<int64>(static_cast<E>(args))...},
            };
        } else if constexpr (same_as<representation_t<E>, uint32>) {
            return native_vector_t<E>{
                .m128i_u32 = {__DPL bit_cast<uint32>(
                    static_cast<E>(args))...},
            };
        } else if constexpr (same_as<representation_t<E>, uint16>) {
            return native_vector_t<E>{
                .m128i_u16 = {__DPL bit_cast<uint16>(
                    static_cast<E>(args))...},
            };
        } else if constexpr (same_as<representation_t<E>, uint8>) {
            return native_vector_t<E>{
                .m128i_u8 = {__DPL bit_cast<uint8>(static_cast<E>(args))...},
            };
        } else {
            return native_vector_t<E>{
                .m128i_u64 = {__DPL bit_cast<uint64>(
                    static_cast<E>(args))...},
            };
        }
#endif
    } else {

        if constexpr (same_as<float, representation_t<E>>) {
            return _mm_setr_ps( __DPL bit_cast<float>(
                static_cast<E>(__DPL forward<Args>(args)))...);
        } else if constexpr (same_as<double, representation_t<E>>) {
            return _mm_setr_pd( __DPL bit_cast<double>(
                static_cast<E>(__DPL forward<Args>(args)))...);
        } else if constexpr (sizeof(E) == sizeof(int64)) {
#if __cpp_pack_indexing >= 202311L
            static_assert(sizeof...(Args) == 2);
            return _mm_set_epi64x(
                __DPL bit_cast<int64>(static_cast<E>(args...[1])),
                __DPL bit_cast<int64>(static_cast<E>(args...[0])));
#else
            using array = E[abi_tag::size / sizeof(E)];
            alignas(abi_tag::alignment)
                array buffer{static_cast<E>(__DPL forward<Args>(args))...};
            return __DPL bit_cast<native_vector_t<E>>(buffer);
#endif
        } else if constexpr (sizeof(E) == sizeof(int32)) {
            return _mm_setr_epi32( __DPL bit_cast<int32>(
                static_cast<E>(__DPL forward<Args>(args)))...);
        } else if constexpr (sizeof(E) == sizeof(int16)) {
            return _mm_setr_epi16( __DPL bit_cast<int16>(
                static_cast<E>(__DPL forward<Args>(args)))...);
        } else {
            static_assert(sizeof(E) == sizeof(int8));
            return _mm_setr_epi8( __DPL bit_cast<int8>(
                static_cast<E>(__DPL forward<Args>(args)))...);
        }
    }
}

DPL_EXPORT template <simd_element E, different_from<abi_tag> Arg,
    core_convertible_to<E>... Args>
requires core_convertible_to<Arg, E> &&
    (different_from<Arg, bool> && ... && different_from<Args, bool>) &&
    (sizeof...(Args) + 1 == simd_abi_traits<E, abi_tag>::size)
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr simd<E> initialize(Arg&& arg, Args&&... args) noexcept {
    return xmm::initialize<E>(
        xmm::abi, __DPL forward<Arg>(arg), __DPL forward<Args>(args)...);
}

} // namespace datapar::xmm

DPL_DEFAULT_NAMESPACE_END

#if DPL_HAS_CXX26_EXTENSIONS
DPL_DISABLE_WARNING_POP()
#endif
