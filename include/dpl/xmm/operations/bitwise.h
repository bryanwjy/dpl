// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

#if !DPL_ARCH_x86_64 || !DPL_SIMD_X86_SSE4_2
#  error "Unsupported platform"
#endif

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/core/type_traits/common_bits_type.h"
#  include "dpl/core/type_traits/representation.h"
#  include "dpl/xmm/basic/abi.h"
#  include "dpl/xmm/basic/reinterpret.h"

#  include <immintrin.h>
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {

template <simd_element L, simd_element R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<common_bits_type_t<L, R>> DPL_VECTORCALL
    bwor(abi_tag, simd<L> lhs, simd<R> rhs) noexcept {
    using bits = common_bits_type_t<L, R>;
    if constexpr (same_as<__m128i, native_vector_t<bits>>) {
        return _mm_or_si128(
            +xmm::reinterpret<bits>(lhs), +xmm::reinterpret<bits>(rhs));
    } else if constexpr (same_as<__m128, native_vector_t<bits>>) {
        return _mm_or_ps(
            +xmm::reinterpret<bits>(lhs), +xmm::reinterpret<bits>(rhs));
    } else if constexpr (same_as<__m128d, native_vector_t<bits>>) {
        return _mm_or_pd(
            +xmm::reinterpret<bits>(lhs), +xmm::reinterpret<bits>(rhs));
    } else {
        using rep = signed_representation_t<L>;
        auto const result = _mm_or_si128(
            +xmm::reinterpret<rep>(lhs), +xmm::reinterpret<rep>(rhs));
        return xmm::reinterpret<bits>(simd<rep>(result));
    }
}

template <simd_element L, simd_element R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<common_bits_type_t<L, R>> DPL_VECTORCALL
    bwand(abi_tag, simd<L> lhs, simd<R> rhs) noexcept {
    using bits = common_bits_type_t<L, R>;
    if constexpr (same_as<__m128i, native_vector_t<bits>>) {
        return _mm_and_si128(
            +xmm::reinterpret<bits>(lhs), +xmm::reinterpret<bits>(rhs));
    } else if constexpr (same_as<__m128, native_vector_t<bits>>) {
        return _mm_and_ps(
            +xmm::reinterpret<bits>(lhs), +xmm::reinterpret<bits>(rhs));
    } else if constexpr (same_as<__m128d, native_vector_t<bits>>) {
        return _mm_and_pd(
            +xmm::reinterpret<bits>(lhs), +xmm::reinterpret<bits>(rhs));
    } else {
        using rep = signed_representation_t<L>;
        auto const result = _mm_and_si128(
            +xmm::reinterpret<rep>(lhs), +xmm::reinterpret<rep>(rhs));
        return xmm::reinterpret<bits>(simd<rep>(result));
    }
}

template <simd_element L, simd_element R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<common_bits_type_t<L, R>> DPL_VECTORCALL
    bwxor(abi_tag, simd<L> lhs, simd<R> rhs) noexcept {
    using bits = common_bits_type_t<L, R>;
    if constexpr (same_as<__m128i, native_vector_t<bits>>) {
        return _mm_xor_si128(
            +xmm::reinterpret<bits>(lhs), +xmm::reinterpret<bits>(rhs));
    } else if constexpr (same_as<__m128, native_vector_t<bits>>) {
        return _mm_xor_ps(
            +xmm::reinterpret<bits>(lhs), +xmm::reinterpret<bits>(rhs));
    } else if constexpr (same_as<__m128d, native_vector_t<bits>>) {
        return _mm_xor_pd(
            +xmm::reinterpret<bits>(lhs), +xmm::reinterpret<bits>(rhs));
    } else {
        using rep = signed_representation_t<L>;
        auto const result = _mm_xor_si128(
            +xmm::reinterpret<rep>(lhs), +xmm::reinterpret<rep>(rhs));
        return xmm::reinterpret<bits>(simd<rep>(result));
    }
}

template <simd_element L, simd_element R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<common_bits_type_t<L, R>> DPL_VECTORCALL
    bwandnot(abi_tag, simd<L> lhs, simd<R> rhs) noexcept {
    using bits = common_bits_type_t<L, R>;
    if constexpr (same_as<__m128i, native_vector_t<bits>>) {
        return _mm_andnot_si128(
            +xmm::reinterpret<bits>(rhs), +xmm::reinterpret<bits>(lhs));
    } else if constexpr (same_as<__m128, native_vector_t<bits>>) {
        return _mm_andnot_ps(
            +xmm::reinterpret<bits>(rhs), +xmm::reinterpret<bits>(lhs));
    } else if constexpr (same_as<__m128d, native_vector_t<bits>>) {
        return _mm_andnot_pd(
            +xmm::reinterpret<bits>(rhs), +xmm::reinterpret<bits>(lhs));
    } else {
        using rep = signed_representation_t<L>;
        auto const result = _mm_andnot_si128(
            +xmm::reinterpret<rep>(rhs), +xmm::reinterpret<rep>(lhs));
        return xmm::reinterpret<bits>(simd<rep>(result));
    }
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<E> DPL_VECTORCALL bwnot(abi_tag tag, simd<E> val) noexcept {
    auto const all = _mm_set1_epi32(-1);
    return bwxor(
        tag, val, xmm::reinterpret<E>(simd<signed_representation_t<E>>(all)));
}

template <simd_element L, simd_element R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<common_bits_type_t<L, R>> DPL_VECTORCALL
    bwornot(abi_tag tag, simd<L> lhs, simd<R> rhs) noexcept {
    return xmm::bwnot(tag, xmm::bwandnot(tag, rhs, lhs));
}

template <simd_element L, simd_element R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<common_bits_type_t<L, R>> DPL_VECTORCALL
    bwor(abi_tag tag, mask<L> lhs, mask<R> rhs) noexcept {
    return +xmm::bwor(tag, simd<L>(+lhs), simd<R>(+rhs));
}

template <simd_element L, simd_element R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<common_bits_type_t<L, R>> DPL_VECTORCALL
    bwand(abi_tag tag, mask<L> lhs, mask<R> rhs) noexcept {
    return +xmm::bwand(tag, simd<L>(+lhs), simd<R>(+rhs));
}

template <simd_element L, simd_element R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<common_bits_type_t<L, R>> DPL_VECTORCALL
    bwxor(abi_tag tag, mask<L> lhs, mask<R> rhs) noexcept {
    return +xmm::bwxor(tag, simd<L>(+lhs), simd<R>(+rhs));
}

template <simd_element L, simd_element R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<common_bits_type_t<L, R>> DPL_VECTORCALL
    bwandnot(abi_tag tag, mask<L> lhs, mask<R> rhs) noexcept {
    return +xmm::bwandnot(tag, simd<L>(+lhs), simd<R>(+rhs));
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<E> DPL_VECTORCALL bwnot(abi_tag tag, mask<E> val) noexcept {
    return +xmm::bwnot(tag, simd<E>(+val));
}

template <simd_element L, simd_element R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<common_bits_type_t<L, R>> DPL_VECTORCALL
    bwornot(abi_tag tag, mask<L> lhs, mask<R> rhs) noexcept {
    return +xmm::bwornot(tag, simd<L>(+lhs), simd<L>(+rhs));
}

namespace details {

alignas(16) inline constexpr char iota_epi8[]{
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

alignas(16) inline constexpr char niota_epi8[]{112, 113, 114, 115, 116, 117,
    118, 119, 120, 121, 122, 123, 124, 125, 126, 127};

} // namespace details

template <template_barrier_t = barrier, simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<E> DPL_VECTORCALL
    bwshift_left(abi_tag tag, mask<E> val, int shift) noexcept {
    if constexpr (floating_point<E>) {
        using bit = unsigned_representation_t<E>;
        return xmm::reinterpret<E>(tag,
            xmm::bwshift_left(tag, xmm::reinterpret<bit>(tag, val), shift));
    } else if constexpr (enumeration<E>) {
        using bit = underlying_type_t<E>;
        return xmm::reinterpret<E>(tag,
            xmm::bwshift_left(tag, xmm::reinterpret<bit>(tag, val), shift));
    } else {
        auto const idx = _mm_load_si128(
            reinterpret_cast<__m128i const*>(details::iota_epi8));
        auto const vshift = _mm_set1_epi8(static_cast<char>(shift));
        return _mm_shuffle_epi8(+val, _mm_sub_epi8(idx, vshift));
    }
}

template <template_barrier_t = barrier, simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<E> DPL_VECTORCALL
    bwshift_right(abi_tag tag, mask<E> val, int shift) noexcept {
    if constexpr (floating_point<E>) {
        using bit = unsigned_representation_t<E>;
        return xmm::reinterpret<E>(tag,
            xmm::bwshift_right(tag, xmm::reinterpret<bit>(tag, val), shift));
    } else if constexpr (enumeration<E>) {
        using bit = underlying_type_t<E>;
        return xmm::reinterpret<E>(tag,
            xmm::bwshift_right(tag, xmm::reinterpret<bit>(tag, val), shift));
    } else {
        auto const idx = _mm_load_si128(
            reinterpret_cast<__m128i const*>(details::niota_epi8));
        auto const vshift = _mm_set1_epi8(static_cast<char>(shift));

        // if msb is set: dst = 0; else: dst = src[idx % 16]
        return _mm_shuffle_epi8(+val, _mm_add_epi8(idx, vshift));
    }
}

template <integral auto V, simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<E> DPL_VECTORCALL bwshift_left(
    abi_tag tag, mask<E> val, immediate<V> imm = dx::imm<V>) noexcept {
    if constexpr (floating_point<E>) {
        using bit = unsigned_representation_t<E>;
        return xmm::reinterpret<E>(
            tag, xmm::bwshift_left(tag, xmm::reinterpret<bit>(tag, val), imm));
    } else if constexpr (enumeration<E>) {
        using bit = underlying_type_t<E>;
        return xmm::reinterpret<E>(
            tag, xmm::bwshift_left(tag, xmm::reinterpret<bit>(tag, val), imm));
    } else {
        return _mm_bslli_si128(+val, static_cast<int>(V * sizeof(E)));
    }
}

template <integral auto V, simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<E> DPL_VECTORCALL bwshift_right(
    abi_tag tag, mask<E> val, immediate<V> imm = dx::imm<V>) noexcept {
    if constexpr (floating_point<E>) {
        using bit = unsigned_representation_t<E>;
        return xmm::reinterpret<E>(
            tag, xmm::bwshift_right(tag, xmm::reinterpret<bit>(tag, val), imm));
    } else if constexpr (enumeration<E>) {
        using bit = underlying_type_t<E>;
        return xmm::reinterpret<E>(
            tag, xmm::bwshift_right(tag, xmm::reinterpret<bit>(tag, val), imm));
    } else {
        return _mm_bsrli_si128(+val, static_cast<int>(V * sizeof(E)));
    }
}

template <template_barrier_t = barrier, simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<E> DPL_VECTORCALL
    bwshift_left(abi_tag tag, simd<E> val, int shift) noexcept {
    if constexpr (floating_point<E>) {
        using bit = unsigned_representation_t<E>;
        return xmm::reinterpret<E>(tag,
            xmm::bwshift_left(tag, xmm::reinterpret<bit>(tag, val), shift));
    } else if constexpr (enumeration<E>) {
        using bit = underlying_type_t<E>;
        return xmm::reinterpret<E>(tag,
            xmm::bwshift_left(tag, xmm::reinterpret<bit>(tag, val), shift));
    } else if constexpr (sizeof(E) == sizeof(int64)) {
        return _mm_sll_epi64(+val, _mm_cvtsi32_si128(shift));
    } else if constexpr (sizeof(E) == sizeof(int32)) {
        return _mm_sll_epi32(+val, _mm_cvtsi32_si128(shift));
    } else if constexpr (sizeof(E) == sizeof(int16)) {
        return _mm_sll_epi16(+val, _mm_cvtsi32_si128(shift));
    } else {
        static_assert(sizeof(E) == sizeof(int8));
        auto const vval = +val;
        auto const zero = _mm_setzero_si128();
        auto lower = _mm_unpacklo_epi8(vval, zero);
        auto upper = _mm_unpackhi_epi8(vval, zero);

        auto const vshift = _mm_cvtsi32_si128(shift);
        lower = _mm_sll_epi16(lower, vshift);
        upper = _mm_sll_epi16(upper, vshift);

        auto const mask = _mm_set1_epi16(0x00FF);
        lower = _mm_and_si128(lower, mask);
        upper = _mm_and_si128(upper, mask);
        return _mm_packus_epi16(lower, upper);
    }
}

template <integral auto V, simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<E> DPL_VECTORCALL bwshift_left(
    abi_tag tag, simd<E> val, immediate<V> imm = dx::imm<V>) noexcept {
    static_assert(V >= 0 && V < sizeof(E) * char_bit_v);
    if constexpr (floating_point<E>) {
        using bit = unsigned_representation_t<E>;
        return xmm::reinterpret<E>(
            tag, xmm::bwshift_left(tag, xmm::reinterpret<bit>(tag, val), imm));
    } else if constexpr (enumeration<E>) {
        using bit = underlying_type_t<E>;
        return xmm::reinterpret<E>(
            tag, xmm::bwshift_left(tag, xmm::reinterpret<bit>(tag, val), imm));
    } else if constexpr (sizeof(E) == sizeof(int64)) {
        return _mm_slli_epi64(+val, static_cast<int>(V));
    } else if constexpr (sizeof(E) == sizeof(int32)) {
        return _mm_slli_epi32(+val, static_cast<int>(V));
    } else if constexpr (sizeof(E) == sizeof(int16)) {
        return _mm_slli_epi16(+val, static_cast<int>(V));
    } else {
        static_assert(sizeof(E) == sizeof(int8));
        auto const vval = +val;
        auto const zero = _mm_setzero_si128();
        auto lower = _mm_unpacklo_epi8(vval, zero);
        auto upper = _mm_unpackhi_epi8(vval, zero);

        constexpr int shift = static_cast<int>(V);
        lower = _mm_slli_epi16(lower, shift);
        upper = _mm_slli_epi16(upper, shift);

        auto const mask = _mm_set1_epi16(0x00FF);
        lower = _mm_and_si128(lower, mask);
        upper = _mm_and_si128(upper, mask);
        return _mm_packus_epi16(lower, upper);
    }
}

template <template_barrier_t = barrier, simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<E> DPL_VECTORCALL
    bwshift_right(abi_tag tag, simd<E> val, int shift) noexcept {
    if constexpr (floating_point<E>) {
        using bit = unsigned_representation_t<E>;
        return xmm::reinterpret<E>(tag,
            xmm::bwshift_right(tag, xmm::reinterpret<bit>(tag, val), shift));
    } else if constexpr (enumeration<E>) {
        using bit = underlying_type_t<E>;
        return xmm::reinterpret<E>(tag,
            xmm::bwshift_right(tag, xmm::reinterpret<bit>(tag, val), shift));
    } else if constexpr (sizeof(E) == sizeof(int64)) {
        if constexpr (unsigned_integral<E>) {
            return _mm_srl_epi64(+val, _mm_cvtsi32_si128(shift));
        } else {
#if DPL_SIMD_X86_AVX512F & DPL_SIMD_X86_AVX512VL
            return _mm_srav_epi64(
                +val, _mm_set1_epi64x(static_cast<int64>(shift)));
#else
            auto const vval = +val;
            auto const shifted = _mm_srl_epi64(vval, _mm_cvtsi32_si128(shift));
            auto const sign = _mm_cmpgt_epi64(_mm_setzero_si128(), vval);
            auto const fill =
                _mm_sll_epi64(sign, _mm_cvtsi32_si128(64 - shift));
            return _mm_or_si128(shifted, fill);
#endif
        }
    } else if constexpr (sizeof(E) == sizeof(int32)) {
        if constexpr (unsigned_integral<E>) {
            return _mm_srl_epi32(+val, _mm_cvtsi32_si128(shift));
        } else {
            return _mm_sra_epi32(+val, _mm_cvtsi32_si128(shift));
        }
    } else if constexpr (sizeof(E) == sizeof(int16)) {
        if constexpr (unsigned_integral<E>) {
            return _mm_srl_epi16(+val, _mm_cvtsi32_si128(shift));
        } else {
            return _mm_sra_epi16(+val, _mm_cvtsi32_si128(shift));
        }
    } else {
        static_assert(sizeof(E) == sizeof(int8));
        auto const vval = +val;
        auto const vshift = _mm_cvtsi32_si128(shift);
        if constexpr (unsigned_integral<E>) {
            auto const zero = _mm_setzero_si128();
            auto lower = _mm_unpacklo_epi8(vval, zero);
            auto upper = _mm_unpackhi_epi8(vval, zero);
            lower = _mm_srl_epi16(lower, vshift);
            upper = _mm_srl_epi16(upper, vshift);

            return _mm_packus_epi16(lower, upper);
        } else {
            auto const sign = _mm_cmplt_epi8(vval, _mm_setzero_si128());
            auto lower = _mm_unpacklo_epi8(vval, sign);
            auto upper = _mm_unpackhi_epi8(vval, sign);

            lower = _mm_sra_epi16(lower, vshift);
            upper = _mm_sra_epi16(upper, vshift);
            return _mm_packs_epi16(lower, upper);
        }
    }
}

template <integral auto V, simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<E> DPL_VECTORCALL bwshift_right(
    abi_tag tag, simd<E> val, immediate<V> imm = dx::imm<V>) noexcept {
    static_assert(V >= 0 && V < sizeof(E) * char_bit_v);
    if constexpr (floating_point<E>) {
        using bit = unsigned_representation_t<E>;
        return xmm::reinterpret<E>(
            tag, xmm::bwshift_left(tag, xmm::reinterpret<bit>(tag, val), imm));
    } else if constexpr (enumeration<E>) {
        using bit = underlying_type_t<E>;
        return xmm::reinterpret<E>(
            tag, xmm::bwshift_left(tag, xmm::reinterpret<bit>(tag, val), imm));
    } else if constexpr (sizeof(E) == sizeof(int64)) {
        if constexpr (unsigned_integral<E>) {
            return _mm_srli_epi64(+val, static_cast<int>(V));
        } else {
#if DPL_SIMD_X86_AVX512F & DPL_SIMD_X86_AVX512VL
            return _mm_srai_epi64(+val, static_cast<int>(V));
#else
            auto const vval = +val;
            auto const shifted = _mm_srli_epi64(vval, static_cast<int>(V));
            auto const sign = _mm_cmpgt_epi64(_mm_setzero_si128(), vval);
            auto const fill = _mm_slli_epi64(sign, static_cast<int>(64 - V));
            return _mm_or_si128(shifted, fill);
#endif
        }
    } else if constexpr (sizeof(E) == sizeof(int32)) {
        if constexpr (unsigned_integral<E>) {
            return _mm_srli_epi32(+val, static_cast<int>(V));
        } else {
            return _mm_srai_epi32(+val, static_cast<int>(V));
        }
    } else if constexpr (sizeof(E) == sizeof(int16)) {
        if constexpr (unsigned_integral<E>) {
            return _mm_srli_epi16(+val, static_cast<int>(V));
        } else {
            return _mm_srai_epi16(+val, static_cast<int>(V));
        }
    } else {
        static_assert(sizeof(E) == sizeof(int8));
        auto const vval = +val;
        if constexpr (unsigned_integral<E>) {
            auto const zero = _mm_setzero_si128();
            auto lower = _mm_unpacklo_epi8(vval, zero);
            auto upper = _mm_unpackhi_epi8(vval, zero);
            lower = _mm_srli_epi16(lower, static_cast<int>(V));
            upper = _mm_srli_epi16(upper, static_cast<int>(V));

            return _mm_packus_epi16(lower, upper);
        } else {
            auto const sign = _mm_cmplt_epi8(vval, _mm_setzero_si128());
            auto lower = _mm_unpacklo_epi8(vval, sign);
            auto upper = _mm_unpackhi_epi8(vval, sign);

            lower = _mm_srai_epi16(lower, static_cast<int>(V));
            upper = _mm_srai_epi16(upper, static_cast<int>(V));
            return _mm_packs_epi16(lower, upper);
        }
    }
}

#if DPL_SIMD_X86_AVX2

template <template_barrier_t = barrier, simd_element L, simd_element R>
requires common_size_with<L, R> && integral<R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<L> DPL_VECTORCALL
    bwshift_left(abi_tag tag, simd<L> lhs, simd<R> rhs) noexcept {
    if constexpr (floating_point<L>) {
        using bit = unsigned_representation_t<L>;
        return xmm::reinterpret<L>(
            tag, xmm::bwshift_left(tag, xmm::reinterpret<bit>(tag, lhs), rhs));
    } else if constexpr (enumeration<L>) {
        using bit = underlying_type_t<L>;
        return xmm::reinterpret<L>(
            tag, xmm::bwshift_left(tag, xmm::reinterpret<bit>(tag, lhs), rhs));
    } else if constexpr (sizeof(L) == sizeof(int64)) {
        return _mm_sllv_epi64(+lhs, +rhs);
    } else if constexpr (sizeof(L) == sizeof(int32)) {
        return _mm_sllv_epi32(+lhs, +rhs);
    } else if constexpr (sizeof(L) == sizeof(int16)) {
#  if DPL_SIMD_X86_AVX512BW & DPL_SIMD_X86_AVX512VL
        return _mm_sllv_epi16(+lhs, +rhs);
#  else
        // Maybe just use the ymm registers?
        auto xmm0 = +lhs;
        auto xmm1 = +rhs;

        auto xmm3 = _mm_cvtepi16_epi32(xmm1);
        auto xmm2 = _mm_cvtepi16_epi32(xmm0);
        xmm2 = _mm_sllv_epi32(xmm2, xmm3);

        xmm1 = _mm_cvtepi16_epi32(_mm_srli_si128(xmm1, 8));
        xmm0 = _mm_cvtepi16_epi32(_mm_srli_si128(xmm0, 8));
        xmm0 = _mm_sllv_epi32(xmm0, xmm1);
        xmm1 = _mm_xor_si128(xmm1, xmm1); // zero
        constexpr auto mask = 0b01010101;
        xmm2 = _mm_blend_epi16(xmm1, xmm2, mask);
        xmm1 = _mm_blend_epi16(xmm1, xmm0, mask);
        return _mm_packus_epi32(xmm2, xmm1);
#  endif
    } else {
        static_assert(sizeof(L) == sizeof(int8));
        auto xmm0 = +lhs;
        auto xmm1 = +rhs;
        using i16 = make_signed_t<bit_type_t<sizeof(L) * 2 * char_bit_v>>;
        auto xmm3 = _mm_cvtepi8_epi16(xmm1);
        auto xmm2 = _mm_cvtepi8_epi16(xmm0);
        xmm2 = +xmm::bwshift_left(tag, simd<i16>(xmm2), simd<i16>(xmm3));

        xmm1 = _mm_cvtepi8_epi16(_mm_srli_si128(xmm1, 8));
        xmm0 = _mm_cvtepi8_epi16(_mm_srli_si128(xmm0, 8));
        xmm0 = +xmm::bwshift_left(tag, simd<i16>(xmm0), simd<i16>(xmm1));
        // broadcast 0x00ff00ff
        xmm1 = _mm_srli_epi16(_mm_cmpeq_epi32(xmm1, xmm1), 8);

        xmm2 = _mm_and_si128(xmm2, xmm1);
        xmm1 = _mm_and_si128(xmm0, xmm1);
        return _mm_packus_epi16(xmm2, xmm1);
    }
}

template <template_barrier_t = barrier, simd_element L, simd_element R>
requires common_size_with<L, R> && integral<R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<L> DPL_VECTORCALL
    bwshift_right(abi_tag tag, simd<L> lhs, simd<R> rhs) noexcept {
    if constexpr (floating_point<L>) {
        using bit = unsigned_representation_t<L>;
        return xmm::reinterpret<L>(
            tag, xmm::bwshift_left(tag, xmm::reinterpret<bit>(tag, lhs), rhs));
    } else if constexpr (enumeration<L>) {
        using bit = underlying_type_t<L>;
        return xmm::reinterpret<L>(
            tag, xmm::bwshift_left(tag, xmm::reinterpret<bit>(tag, lhs), rhs));
    } else if constexpr (sizeof(L) == sizeof(int64)) {
        if constexpr (unsigned_integral<L>) {
            return _mm_srlv_epi64(+lhs, +rhs);
        } else {
#  if DPL_SIMD_X86_AVX512F & DPL_SIMD_X86_AVX512VL
            return _mm_srav_epi64(+lhs, +rhs);
#  else
            auto const shifted = _mm_srlv_epi64(+lhs, +rhs);
            auto const sign = _mm_cmpgt_epi64(_mm_setzero_si128(), +lhs);
            auto const fill =
                _mm_slli_epi64(sign, _mm_sub_epi64(_mm_set1_epi64x(64), +rhs));
            return _mm_or_si128(shifted, fill);
#  endif
        }
    } else if constexpr (sizeof(L) == sizeof(int32)) {
        if constexpr (unsigned_integral<L>) {
            return _mm_srlv_epi32(+lhs, +rhs);
        } else {
            return _mm_srav_epi32(+lhs, +rhs);
        }
    } else if constexpr (sizeof(L) == sizeof(int16)) {
#  if DPL_SIMD_X86_AVX512BW & DPL_SIMD_X86_AVX512VL
        if constexpr (unsigned_integral<L>) {
            return _mm_srlv_epi16(+lhs, +rhs);
        } else {
            return _mm_srav_epi16(+lhs, +rhs);
        }
#  else
        if constexpr (unsigned_integral<L>) {
            auto xmm0 = +lhs;
            auto xmm1 = +rhs;
            auto xmm3 = _mm_cvtepu16_epi32(xmm1);
            auto xmm2 = _mm_cvtepu16_epi32(xmm0);
            xmm2 = _mm_srlv_epi32(xmm2, xmm3);
            xmm1 = _mm_srli_si128(xmm1, 8);
            xmm0 = _mm_srli_si128(xmm0, 8);
            xmm1 = _mm_cvtepu16_epi32(xmm1);
            xmm0 = _mm_cvtepu16_epi32(xmm0);
            xmm0 = _mm_srlv_epi32(xmm0, xmm1);
            xmm1 = _mm_xor_si128(xmm1, xmm1);
            xmm2 = _mm_blend_epi16(xmm1, xmm2, 0x55);
            xmm1 = _mm_blend_epi16(xmm1, xmm0, 0x55);
            return _mm_packus_epi32(xmm2, xmm1);
        } else {
            auto xmm0 = +lhs;
            auto xmm1 = +rhs;
            auto xmm3 = _mm_cvtepi16_epi32(xmm1);
            auto xmm2 = _mm_cvtepi16_epi32(xmm0);
            xmm2 = _mm_srav_epi32(xmm2, xmm3);
            xmm1 = _mm_srli_si128(xmm1, 8);
            xmm0 = _mm_srli_si128(xmm0, 8);
            xmm1 = _mm_cvtepi16_epi32(xmm1);
            xmm0 = _mm_cvtepi16_epi32(xmm0);
            xmm0 = _mm_srav_epi32(xmm0, xmm1);
            xmm1 = _mm_xor_si128(xmm1, xmm1);
            xmm2 = _mm_blend_epi16(xmm1, xmm2, 0x55);
            xmm1 = _mm_blend_epi16(xmm1, xmm0, 0x55);
            return _mm_packus_epi32(xmm2, xmm1);
        }
#  endif
    } else {
        static_assert(common_bits_with<L, int8>);
        if constexpr (unsigned_integral<L>) {
            using i16 = make_unsigned_t<bit_type_t<sizeof(L) * 2 * char_bit_v>>;
            auto xmm0 = +lhs;
            auto xmm1 = +rhs;
            auto xmm3 = _mm_cvtepu8_epi16(xmm1);
            auto xmm2 = _mm_cvtepu8_epi16(xmm0);
            xmm2 = +xmm::bwshift_right(tag, simd<i16>(xmm2), simd<i16>(xmm3));
            xmm1 = _mm_srli_si128(xmm1, 8);
            xmm0 = _mm_srli_si128(xmm0, 8);
            xmm1 = _mm_cvtepu8_epi16(xmm1);
            xmm0 = _mm_cvtepu8_epi16(xmm0);
            xmm0 = +xmm::bwshift_right(tag, simd<i16>(xmm0), simd<i16>(xmm1));
            xmm1 = _mm_srli_epi16(_mm_cmpeq_epi32(xmm1, xmm1), 8);
            xmm2 = _mm_and_si128(xmm2, xmm1);
            xmm1 = _mm_and_si128(xmm0, xmm1);
            return _mm_packus_epi16(xmm2, xmm1);
        } else {
            using i16 = make_signed_t<bit_type_t<sizeof(L) * 2 * char_bit_v>>;
            auto xmm0 = +lhs;
            auto xmm1 = +rhs;
            auto xmm3 = _mm_cvtepi8_epi16(xmm1);
            auto xmm2 = _mm_cvtepi8_epi16(xmm0);
            xmm2 = +xmm::bwshift_right(tag, simd16(xmm2), simd16(xmm3));
            xmm1 = _mm_srli_si128(xmm1, 8);
            xmm0 = _mm_srli_si128(xmm0, 8);
            xmm1 = _mm_cvtepi8_epi16(xmm1);
            xmm0 = _mm_cvtepi8_epi16(xmm0);
            xmm0 = +xmm::bwshift_right(tag, simd16(xmm0), simd16(xmm1));
            xmm1 = _mm_srli_epi16(_mm_cmpeq_epi32(xmm1, xmm1), 8);
            xmm2 = _mm_and_si128(xmm2, xmm1);
            xmm1 = _mm_and_si128(xmm0, xmm1);
            return _mm_packus_epi16(xmm2, xmm1);
        }
    }
}

#else
// SSE 4.2
template <template_barrier_t = barrier, simd_element L, simd_element R>
requires common_size_with<L, R> && integral<R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<L> DPL_VECTORCALL
    bwshift_left(abi_tag tag, simd<L> lhs, simd<R> rhs) noexcept {
    if constexpr (floating_point<L>) {
        using bit = unsigned_representation_t<L>;
        return xmm::reinterpret<L>(
            tag, xmm::bwshift_left(tag, xmm::reinterpret<bit>(tag, lhs), rhs));
    } else if constexpr (enumeration<L>) {
        using bit = underlying_type_t<L>;
        return xmm::reinterpret<L>(
            tag, xmm::bwshift_left(tag, xmm::reinterpret<bit>(tag, lhs), rhs));
    } else if constexpr (sizeof(L) == sizeof(int64)) {
        auto xmm0 = +lhs;
        auto xmm1 = +rhs;
        auto xmm2 = _mm_sll_epi64(xmm0, xmm1);
        xmm1 = _mm_shuffle_epi32(xmm1, _MM_SHUFFLE(3, 2, 3, 2));
        xmm0 = _mm_sll_epi64(xmm0, xmm1);
        return _mm_castpd_si128(
            _mm_move_sd(_mm_castsi128_pd(xmm0), _mm_castsi128_pd(xmm2)));
    } else if constexpr (sizeof(L) == sizeof(int32)) {
        constexpr auto magic = 1065353216;
        auto xmm0 = +lhs;
        auto xmm1 = _mm_slli_epi32(+rhs, 23);
        xmm1 = _mm_add_epi32(xmm1, _mm_set1_epi32(magic));
        xmm1 = _mm_cvttps_epi32(_mm_castsi128_ps(xmm1));
        xmm1 = _mm_and_si128(_mm_cmplt_epi32(rhs, _mm_set1_epi32(32)), xmm1);
        return _mm_mullo_epi32(xmm0, xmm1);
    } else if constexpr (sizeof(L) == sizeof(int16)) {
        auto xmm0 = +lhs;
        auto xmm1 = +rhs;

        auto xmm3 = _mm_cvtepi16_epi32(xmm1);
        auto xmm2 = _mm_cvtepi16_epi32(xmm0);
        using i32 = make_signed_t<bit_type_t<sizeof(L) * 2 * char_bit_v>>;
        xmm2 = +xmm::bwshift_left(tag, simd<i32>(xmm2), simd<i32>(xmm3));

        xmm1 = _mm_cvtepi16_epi32(_mm_srli_si128(xmm1, 8));
        xmm0 = _mm_cvtepi16_epi32(_mm_srli_si128(xmm0, 8));
        xmm0 = +xmm::bwshift_left(tag, simd<i32>(xmm0), simd<i32>(xmm1));

        xmm1 = _mm_xor_si128(xmm1, xmm1); // zero
        constexpr auto mask = 0b01010101;
        xmm2 = _mm_blend_epi16(xmm1, xmm2, mask);
        xmm1 = _mm_blend_epi16(xmm1, xmm0, mask);
        return _mm_packus_epi32(xmm2, xmm1);
    } else {
        static_assert(sizeof(L) == sizeof(int8));
        auto xmm0 = +lhs;
        auto xmm1 = +rhs;

        using i16 = make_signed_t<bit_type_t<sizeof(L) * 2 * char_bit_v>>;

        auto xmm3 = _mm_cvtepi8_epi16(xmm1);
        auto xmm2 = _mm_cvtepi8_epi16(xmm0);
        xmm2 = +bwshift_left(tag, simd<i16>(xmm2), simd<i16>(xmm3));

        xmm1 = _mm_cvtepi8_epi16(_mm_srli_si128(xmm1, 8));
        xmm0 = _mm_cvtepi8_epi16(_mm_srli_si128(xmm0, 8));
        xmm0 = +bwshift_left(tag, simd<i16>(xmm0), simd<i16>(xmm1));

        // broadcast 0x00ff00ff
        xmm1 = _mm_srli_epi16(_mm_cmpeq_epi32(xmm1, xmm1), 8);

        xmm2 = _mm_and_si128(xmm2, xmm1);
        xmm1 = _mm_and_si128(xmm0, xmm1);
        return _mm_packus_epi16(xmm2, xmm1);
    }
}

// No efficient way to right shift without avx2
#endif

} // namespace datapar::xmm

DPL_DEFAULT_NAMESPACE_END
