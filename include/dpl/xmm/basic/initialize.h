// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

#if !DPL_ARCH_x86_64 || !DPL_SIMD_X86_SSE4_2
#  error "Unsupported platform"
#endif

#include "dpl/xmm/basic/abi.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/common_float_with.h"
#  include "dpl/core/concepts/common_order_with.h" // IWYU pragma: keep
#  include "dpl/core/concepts/common_size_with.h"
#  include "dpl/core/concepts/simd_element.h"
#  include "dpl/core/constants/all_bits.h"
#  include "dpl/core/constants/zero.h"
#  include "dpl/std/bit/bit_cast.h"
#  include "dpl/std/type_traits/is_const.h"
#  include "dpl/std/type_traits/is_volatile.h"
#  include "dpl/std/type_traits/type_identity.h" // IWYU pragma: keep
#  include "dpl/std/utility/forward.h"           // IWYU pragma: keep
#  include "dpl/std/utility/to_underlying.h"     // IWYU pragma: keep

#  include <immintrin.h>
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {
DPL_EXPORT template <basic_simd_element E, core_convertible_to<E>... Args>
requires (... && !same_as<Args, bool>)
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr simd<E> initialize(abi_tag tag, Args&&... args) noexcept {
    static_assert(!is_const_v<E> && !is_volatile_v<E>);
    if consteval {
#if !DPL_COMPILER_MSVC
        using array = E[abi_tag::size / sizeof(E)];
        alignas(abi_tag::alignment)
            array buffer{static_cast<E>(__DPL forward<Args>(args))...};
        return __DPL bit_cast<native_vector_t<E>>(buffer);
#else
        if constexpr (common_float_with<float, E>) {
            return native_vector_t<E>{
                .m128_f32 = {__DPL bit_cast<float>(
                    static_cast<E>( __DPL forward<Args>(args)))...},
            };
        } else if constexpr (common_float_with<double, E>) {
            return native_vector_t<E>{
                .m128_f64 = {__DPL bit_cast<double>(
                    static_cast<E>( __DPL forward<Args>(args)))...},
            };
        } else if constexpr (floating_point<E> && sizeof(E) == 2) {
            return native_vector_t<E>{
                .m128_i16 = {__DPL bit_cast<int16>(
                    static_cast<E>( __DPL forward<Args>(args)))...},
            };
        } else if constexpr (enumeration<E>) {
            return dx::xmm::initialize(
                tag, __DPL to_underlying(static_cast<E>(args))...);
        } else if constexpr (common_order_with<E, int32>) {
            return native_vector_t<E>{
                .m128i_i32 = {__DPL bit_cast<int32>(static_cast<E>(args))...},
            };
        } else if constexpr (common_order_with<E, int16>) {
            return native_vector_t<E>{
                .m128i_i16 = {__DPL bit_cast<int16>(static_cast<E>(args))...},
            };
        } else if constexpr (common_order_with<E, int8>) {
            return native_vector_t<E>{
                .m128i_i8 = {__DPL bit_cast<int8>(static_cast<E>(args))...},
            };
        } else if constexpr (common_order_with<E, int64>) {
            return native_vector_t<E>{
                .m128i_i64 = {__DPL bit_cast<int64>(static_cast<E>(args))...},
            };
        } else if constexpr (common_order_with<E, uint32>) {
            return native_vector_t<E>{
                .m128i_u32 = {__DPL bit_cast<uint32>(
                    static_cast<E>(args))...},
            };
        } else if constexpr (common_order_with<E, uint16>) {
            return native_vector_t<E>{
                .m128i_u16 = {__DPL bit_cast<uint16>(
                    static_cast<E>(args))...},
            };
        } else if constexpr (common_order_with<E, uint8>) {
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

        if constexpr (common_float_with<E, float>) {
            return _mm_setr_ps( __DPL bit_cast<float>(static_cast<E>(args))...);
        } else if constexpr (common_float_with<E, double>) {
            return _mm_setr_pd(
                __DPL bit_cast<double>(static_cast<E>(args))...);
        } else if constexpr (common_size_with<int64, E>) {
#if __cpp_pack_indexing >= 202311L
#  if (DPL_COMPILER_CLANG | DPL_COMPILER_GCC) & !DPL_CXX26
            DPL_DISABLE_WARNING_PUSH()
            DPL_DISABLE_WARNING("-Wc++26-extensions")
#  endif
            static_assert(sizeof...(Args) == 2);
            return _mm_set_epi64x(
                __DPL bit_cast<int64>(static_cast<E>(args...[1])),
                __DPL bit_cast<int64>(static_cast<E>(args...[0])));
#  if DPL_COMPILER_CLANG & !DPL_CXX26
            DPL_DISABLE_WARNING_POP()
#  endif
#else
            using array = E[abi_tag::size / sizeof(E)];
            alignas(abi_tag::alignment)
                array buffer{static_cast<E>(__DPL forward<Args>(args))...};
            return __DPL bit_cast<native_vector_t<E>>(buffer);
#endif
        } else if constexpr (common_size_with<int32, E>) {
            return _mm_setr_epi32(
                __DPL bit_cast<int32>(static_cast<E>(args))...);
        } else if constexpr (common_size_with<int16, E>) {
            return _mm_setr_epi16(
                __DPL bit_cast<int16>(static_cast<E>(args))...);
        } else {
            static_assert(common_size_with<int8, E>);
            return _mm_setr_epi8(
                __DPL bit_cast<int8>(static_cast<E>(args))...);
        }
    }
}

DPL_EXPORT template <simd_element E, same_as<bool>... Args>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr mask<E> initialize(abi_tag tag, Args... scalars) noexcept {
    static_assert(!is_const_v<E> && !is_volatile_v<E>);
    return +dx::xmm::initialize<E>(
        tag, (scalars ? dx::all_bits_v<E> : dx::zero_v<E>)...);
}

DPL_EXPORT template <basic_simd_element E, core_convertible_to<E>... Args>
requires (... && !same_as<Args, bool>)
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr simd<E> initialize(Args&&... args) noexcept {
    return xmm::initialize(xmm::abi, __DPL forward<Args>(args)...);
}

DPL_EXPORT template <simd_element E, same_as<bool>... Args>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr mask<E> initialize(Args&&... args) noexcept {
    return +dx::xmm::initialize<E>(
        xmm::abi, (args ? dx::all_bits_v<E> : dx::zero_v<E>)...);
}

} // namespace datapar::xmm

DPL_DEFAULT_NAMESPACE_END
