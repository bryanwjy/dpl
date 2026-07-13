// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

#if DPL_HAS_CXX26_EXTENSIONS
DPL_DISABLE_WARNING_PUSH()
DPL_DISABLE_WARNING("-Wc++26-extensions")
#endif

#if DPL_SIMD_X86_SSE4_2

#  include "dpl/xmm/basic/abi.h"

#  if !DPL_MODULES
#    include "dpl/core/type_traits/simd_abi_traits.h"
#    include "dpl/std/bit/bit_cast.h"
#    include "dpl/std/concepts/convertible_to.h"
#    include "dpl/std/type_traits/decay.h"
#    include "dpl/std/utility/forward.h"

#    include <immintrin.h>
#  endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::xmm {

namespace internal {
template <simd_element E, typename... Args>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr vector<E> initialize(Args... args) noexcept {
    if consteval {
#  if !DPL_COMPILER_MSVC
        using array = E[abi_tag::size / sizeof(E)];
        alignas(abi_tag::alignment) array buffer{args...};
        return __DPL bit_cast<native_vector_t<E>>(buffer);
#  else
        if constexpr (is_same_v<float, E>) {
            return native_vector_t<E>{
                .m128_f32 = {__DPL bit_cast<float>(args)...},
            };
        } else if constexpr (is_same_v<double, E>) {
            return native_vector_t<E>{
                .m128_f64 = {__DPL bit_cast<double>(args)...},
            };
        } else if constexpr (is_same_v<ext::bfloat16, E> ||
            is_same_v<ext::float16, E>) {
            return native_vector_t<E>{
                .m128_i16 = {__DPL bit_cast<int16>(args)...},
            };
        } else if constexpr (is_same_v<E, int32>) {
            return native_vector_t<E>{
                .m128i_i32 = {__DPL bit_cast<int32>(args)...},
            };
        } else if constexpr (is_same_v<E, int16>) {
            return native_vector_t<E>{
                .m128i_i16 = {__DPL bit_cast<int16>(args)...},
            };
        } else if constexpr (is_same_v<E, int8>) {
            return native_vector_t<E>{
                .m128i_i8 = {__DPL bit_cast<int8>(args)...},
            };
        } else if constexpr (is_same_v<E, int64>) {
            return native_vector_t<E>{
                .m128i_i64 = {__DPL bit_cast<int64>(args)...},
            };
        } else if constexpr (is_same_v<E, uint32>) {
            return native_vector_t<E>{
                .m128i_u32 = {__DPL bit_cast<uint32>(args)...},
            };
        } else if constexpr (is_same_v<E, uint16>) {
            return native_vector_t<E>{
                .m128i_u16 = {__DPL bit_cast<uint16>(args)...},
            };
        } else if constexpr (is_same_v<E, uint8>) {
            return native_vector_t<E>{
                .m128i_u8 = {__DPL bit_cast<uint8>(args)...},
            };
        } else {
            return native_vector_t<E>{
                .m128i_u64 = {__DPL bit_cast<uint64>(args)...},
            };
        }
#  endif
    } else {

        if constexpr (same_as<float, E>) {
            return _mm_setr_ps( __DPL bit_cast<float>(args)...);
        } else if constexpr (same_as<double, E>) {
            return _mm_setr_pd( __DPL bit_cast<double>(args)...);
        } else if constexpr (sizeof(E) == sizeof(int64)) {
#  if __cpp_pack_indexing >= 202311L
            static_assert(sizeof...(Args) == 2);
            return _mm_set_epi64x(
                __DPL bit_cast<int64>(static_cast<E>(args...[1])),
                __DPL bit_cast<int64>(static_cast<E>(args...[0])));
#  else
            using array = E[abi_tag::size / sizeof(E)];
            alignas(abi_tag::alignment) array buffer{args...};
            return __DPL bit_cast<native_vector_t<E>>(buffer);
#  endif
        } else if constexpr (sizeof(E) == sizeof(int32)) {
            return _mm_setr_epi32( __DPL bit_cast<int32>(args)...);
        } else if constexpr (sizeof(E) == sizeof(int16)) {
            return _mm_setr_epi16( __DPL bit_cast<int16>(args)...);
        } else {
            static_assert(sizeof(E) == sizeof(int8));
            return _mm_setr_epi8( __DPL bit_cast<int8>(args)...);
        }
    }
}
} // namespace internal

template <simd_element E, core_convertible_to<E> Arg,
    core_convertible_to<E>... Args>
requires different_from<decay_t<Arg>, abi_tag> &&
    (different_from<decay_t<Arg>, bool> && ... &&
        different_from<decay_t<Args>, bool>) &&
    (sizeof...(Args) + 1 == simd_abi_traits<E, abi_tag>::size)
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr vector<E> initialize(Arg&& arg, Args&&... args) noexcept {
    return internal::initialize<E>(static_cast<E>(__DPL forward<Arg>(arg)),
        static_cast<E>(__DPL forward<Args>(args))...);
}

template <simd_element E, core_convertible_to<E>... Args>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr vector<E> initialize(abi_tag, Args&&... args) noexcept
requires requires { xmm::initialize<E>(__DPL forward<Args>(args)...); }
{
    return xmm::initialize<E>(__DPL forward<Args>(args)...);
}

} // namespace datapar::xmm

__DPL_DEFAULT_NAMESPACE_END

#  if DPL_HAS_CXX26_EXTENSIONS
DPL_DISABLE_WARNING_POP()
#  endif

#endif
