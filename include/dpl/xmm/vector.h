// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h" // IWYU pragma: keep

#include "dpl/xmm/concepts.h"

// IWYU pragma: begin_exports
#include "dpl/xmm/fwd.h"

#include "dpl/blob.h"
#include "dpl/immediate.h"
#include "dpl/utility.h"
#include "dpl/xmm/constants.h"
#include "dpl/xmm/native.h"
#include "dpl/xmm/vector_mask.h"
// IWYU pragma: end_exports

#if !DPL_MODULES
#  include <immintrin.h>

#  include <concepts>
#endif

DPL_NAMESPACE_BEGIN

DPL_SIMD_ABI_NAMESPACE_BEGIN(xmm)

DPL_ATTRIBUTES(
    _HIDE_FROM_ABI, NODISCARD, PURE) native_type_t<blob<xmm_abi::size>>
operator+(blob<xmm_abi::size> const& val) noexcept {
    return _mm_load_si128(reinterpret_cast<__m128i const*>(&val));
}

DPL_EXPORT_BEGIN

template <xmm_element T>
class __DPL_PUBLIC_TEMPLATE vector<T> {
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, PURE, NODISCARD) static inline native_type_t<T>
    broadcast(T value) noexcept {
        if constexpr (sized_floating_point<T, 8>) {
            return _mm_set1_pd(value);
        } else if constexpr (sized_floating_point<T, 4>) {
            return _mm_set1_ps(value);
        } else if constexpr (sized_integral<T, 8>) {
            return _mm_set1_epi64x(value);
        } else if constexpr (sized_integral<T, 4>) {
            return _mm_set1_epi32(value);
        } else if constexpr (sized_integral<T, 2>) {
            return _mm_set1_epi16(value);
        } else {
            static_assert(sized_integral<T, 1>, "unsupported operation");
            return _mm_set1_epi8(value);
        }
    }

public:
    using element_type = T;
    using native_type = native_type_t<T>;
    using mask_type = vector_mask<T>;
    using scalar_type = scalar<T>;

    __DPL_HIDE_FROM_ABI inline constexpr vector() noexcept = default;
    __DPL_HIDE_FROM_ABI inline constexpr vector(
        vector const&) noexcept = default;
    __DPL_HIDE_FROM_ABI inline constexpr vector& operator=(
        vector const&) noexcept = default;
    __DPL_HIDE_FROM_ABI inline constexpr ~vector() noexcept = default;
    __DPL_HIDE_FROM_ABI inline vector(zero_t<T> val) noexcept : val{+val} {}
    __DPL_HIDE_FROM_ABI inline vector(all_t<T> val) noexcept : val{+val} {}
    __DPL_HIDE_FROM_ABI inline vector(msb_t<T> val) noexcept : val{+val} {}
    __DPL_HIDE_FROM_ABI inline vector(lsb_t<T> val) noexcept : val{+val} {}
    __DPL_HIDE_FROM_ABI inline vector(iota_t<T> val) noexcept : val{+val} {}

    __DPL_HIDE_FROM_ABI explicit inline vector(
        aligned_t, T const* data) noexcept
        : val{_mm_load_ps(data)} {}
    __DPL_HIDE_FROM_ABI explicit inline vector(T const* data) noexcept
        : val{_mm_loadu_ps(data)} {}
    __DPL_HIDE_FROM_ABI explicit inline vector(T val) noexcept
        : vector{broadcast(val)} {}
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL) explicit inline vector(
        blob<xmm_abi::size> const& val) noexcept
    requires std::same_as<T, blob<xmm_abi::size>>
        : val{+val} {}
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL) inline constexpr vector(
        native_type val) noexcept
        : val{val} {}

    template <std::unsigned_integral U>
    requires xmm_element<U> && std::same_as<std::make_signed_t<U>, T>
    DPL_ATTRIBUTES(
        _HIDE_FROM_ABI, VECTORCALL) explicit inline vector(vector<U> val)
        : val{+val} {}
    template <std::signed_integral U>
    requires xmm_element<U> && std::same_as<std::make_unsigned_t<U>, T>
    DPL_ATTRIBUTES(
        _HIDE_FROM_ABI, VECTORCALL) explicit inline vector(vector<U> val)
        : val{+val} {}

    template <std::unsigned_integral U>
    requires xmm_element<U> && std::same_as<std::make_signed_t<U>, T>
    DPL_ATTRIBUTES(
        _HIDE_FROM_ABI, VECTORCALL) explicit inline vector(scalar<U> val)
        : val{+val} {}
    template <std::signed_integral U>
    requires xmm_element<U> && std::same_as<std::make_unsigned_t<U>, T>
    DPL_ATTRIBUTES(
        _HIDE_FROM_ABI, VECTORCALL) explicit inline vector(scalar<U> val)
        : val{+val} {}

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL) explicit inline vector(
        scalar<T> val) noexcept
        : val{+val} {}
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL) explicit inline vector(
        vector_mask<T> val) noexcept
        : val{+val} {}
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL) explicit inline vector(
        scalar_mask<T> val) noexcept
        : val{+val} {}

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) inline native_type operator+(
        this vector vec) noexcept {
        return vec.val;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) explicit inline
    operator native_type(this vector vec) noexcept {
        return vec.val;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline vector operator+(
        this vector left, vector right) noexcept {
        if constexpr (sized_floating_point<T, 4>) {
            return _mm_add_ps(left.val, right.val);
        } else if constexpr (sized_floating_point<T, 8>) {
            return _mm_add_pd(left.val, right.val);
        } else if constexpr (sized_integral<T, 1>) {
            return _mm_add_epi8(left.val, right.val);
        } else if constexpr (sized_integral<T, 2>) {
            return _mm_add_epi16(left.val, right.val);
        } else if constexpr (sized_integral<T, 4>) {
            return _mm_add_epi32(left.val, right.val);
        } else {
            static_assert(sized_integral<T, 8>);
            return _mm_add_epi64(left.val, right.val);
        }
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL) inline vector& operator+=(
        vector right) noexcept {
        return *this = *this + right;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline vector operator-(
        this vector left, vector right) noexcept {
        if constexpr (sized_floating_point<T, 4>) {
            return _mm_sub_ps(left.val, right.val);
        } else if constexpr (sized_floating_point<T, 8>) {
            return _mm_sub_pd(left.val, right.val);
        } else if constexpr (sized_integral<T, 1>) {
            return _mm_sub_epi8(left.val, right.val);
        } else if constexpr (sized_integral<T, 2>) {
            return _mm_sub_epi16(left.val, right.val);
        } else if constexpr (sized_integral<T, 4>) {
            return _mm_sub_epi32(left.val, right.val);
        } else {
            static_assert(sized_integral<T, 8>);
            return _mm_sub_epi64(left.val, right.val);
        }
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL) inline vector& operator-=(
        vector right) noexcept {
        return *this = *this - right;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline vector operator[](
        this vector self, vector_mask<T> where) noexcept;

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline vector operator*(
        this vector self, vector right) noexcept {
        static_assert(std::integral<T>, "unsupported operation");
        if constexpr (sized_signed_integral<T, 2>) {
            return _mm_mullo_epi16(self.val, right.val);
        } else if constexpr (sized_signed_integral<T, 4>) {
            return _mm_mullo_epi32(self.val, right.val);
        } else {
            static_assert(sizeof(T) == 8, "unsupported operation");
            return _mm_mullo_epi64(self.val, right.val);
        }
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) inline multiplication<vector>
    operator*(this vector left, vector right) noexcept
    requires std::floating_point<T>
    {
#if DPL_SUPPORTS_BFLOAT16
        static_assert(
            !std::same_as<bfloat16, element_type>, "unsupported operation");
#endif
        return multiplication<vector>{left.val, right.val};
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL) inline vector& operator*=(
        vector right) noexcept {
        return *this = *this * right;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline vector operator/(
        this vector self, vector right) noexcept
    requires std::floating_point<T>
    {
        if constexpr (sized_floating_point<T, 4>) {
            return _mm_div_ps(self.val, right.val);
        } else {
            static_assert(sized_floating_point<T, 8>, "unsupported operation");
            return _mm_div_pd(self.val, right.val);
        }
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL) inline vector& operator/=(
        vector right) noexcept {
        return *this = *this / right;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline vector operator&(
        this vector self, vector right) noexcept {
        if constexpr (sized_floating_point<T, 4>) {
            return _mm_and_ps(self.val, right.val);
        } else if constexpr (sized_floating_point<T, 8>) {
            return _mm_and_pd(self.val, right.val);
        } else {
            return _mm_and_si128(native_cast<b128, element_type>(self.val),
                native_cast<b128, element_type>(right.val));
        }
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline vector operator&(
        this vector self, vector_mask<T> right) noexcept {
        if constexpr (sized_floating_point<T, 4>) {
            return _mm_and_ps(self.val, +right);
        } else if constexpr (sized_floating_point<T, 8>) {
            return _mm_and_pd(self.val, +right);
        } else {
            return _mm_and_si128(
                native_cast<b128, element_type>(self.val), +right);
        }
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) friend inline vector
    operator&(vector_mask<T> left, vector right) noexcept {
        if constexpr (sized_floating_point<T, 4>) {
            return _mm_and_ps(+left, right.val);
        } else if constexpr (sized_floating_point<T, 8>) {
            return _mm_and_pd(+left, right.val);
        } else {
            return _mm_and_si128(
                +left, native_cast<b128, element_type>(right.val));
        }
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL) inline vector& operator&=(
        vector right) noexcept {
        return *this = *this & right;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL) inline vector& operator&=(
        vector_mask<T> right) noexcept {
        return *this = *this & right;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline vector operator^(
        this vector self, vector right) noexcept {
        if constexpr (sized_floating_point<T, 4>) {
            return _mm_xor_ps(self.val, right.val);
        } else if constexpr (sized_floating_point<T, 8>) {
            return _mm_xor_pd(self.val, right.val);
        } else {
            return _mm_xor_si128(native_cast<b128, element_type>(self.val),
                native_cast<b128, element_type>(right.val));
        }
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline vector operator^(
        this vector self, vector_mask<T> right) noexcept {
        if constexpr (sized_floating_point<T, 4>) {
            return _mm_xor_ps(self.val, +right);
        } else if constexpr (sized_floating_point<T, 8>) {
            return _mm_xor_pd(self.val, +right);
        } else {
            return _mm_xor_si128(native_cast<b128, element_type>(self.val),
                native_cast<b128, element_type>(+right));
        }
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) friend inline vector
    operator^(vector_mask<T> left, vector self) noexcept {
        if constexpr (sized_floating_point<T, 4>) {
            return _mm_xor_ps(+left, self.val);
        } else if constexpr (sized_floating_point<T, 8>) {
            return _mm_xor_pd(+left, self.val);
        } else {
            return _mm_xor_si128(native_cast<b128, element_type>(+left),
                native_cast<b128, element_type>(self.val));
        }
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL) inline vector& operator^=(
        vector right) noexcept {
        return *this = *this ^ right;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL) inline vector& operator^=(
        vector_mask<T> right) noexcept {
        return *this = *this ^ right;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline vector operator|(
        this vector self, vector right) noexcept {
        if constexpr (sized_floating_point<T, 4>) {
            return _mm_or_ps(self.val, right.val);
        } else if constexpr (sized_floating_point<T, 8>) {
            return _mm_or_pd(self.val, right.val);
        } else {
            return _mm_xor_si128(native_cast<b128, element_type>(self.val),
                native_cast<b128, element_type>(right.val));
        }
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline vector operator|(
        this vector self, vector_mask<T> right) noexcept {
        if constexpr (sized_floating_point<T, 4>) {
            return _mm_or_ps(self.val, +right);
        } else if constexpr (sized_floating_point<T, 8>) {
            return _mm_or_pd(self.val, +right);
        } else {
            return _mm_xor_si128(native_cast<b128, element_type>(self.val),
                native_cast<b128, element_type>(+right));
        }
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) friend inline vector
    operator|(vector_mask<T> left, vector self) noexcept {
        if constexpr (sized_floating_point<T, 4>) {
            return _mm_or_ps(+left, self.val);
        } else if constexpr (sized_floating_point<T, 8>) {
            return _mm_or_pd(+left, self.val);
        } else {
            return _mm_xor_si128(native_cast<b128, element_type>(+left),
                native_cast<b128, element_type>(self.val));
        }
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL) inline vector& operator|=(
        vector right) noexcept {
        return *this = *this | right;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL) inline vector& operator|=(
        vector_mask<T> right) noexcept {
        return *this = *this | right;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline bw_negation<vector>
    operator~(this vector self) noexcept {
        return bw_negation<vector>{self.val};
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) inline vector operator-(
        this vector self) noexcept;

    template <int V>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline vector operator>>(
        this vector self, immediate<V>)
    requires std::integral<T>
    {
        static_assert(V > 0, "Invalid input");
        if constexpr (sized_signed_integral<T, 2>) {
            return _mm_srai_epi16(self.val, V);
        } else if constexpr (sized_signed_integral<T, 4>) {
            return _mm_srai_epi32(self.val, V);
        } else if constexpr (sized_unsigned_integral<T, 16>) {
            return _mm_srli_si128(self.val, V);
        } else if constexpr (sized_unsigned_integral<T, 8>) {
            return _mm_srli_epi64(self.val, V);
        } else if constexpr (sized_unsigned_integral<T, 4>) {
            return _mm_srli_epi32(self.val, V);
        } else {
            static_assert(
                sized_unsigned_integral<T, 2>, "unsupported operation");
            return _mm_srli_epi16(self.val, V);
        }
    }

    template <int V>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL) inline vector& operator>>=(immediate<V> imm)
    requires std::integral<T>
    {
        return *this = *this >> imm;
    }

    template <int V>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline vector operator<<(
        this vector self, immediate<V>)
    requires std::integral<T>
    {
        static_assert(V > 0, "Invalid input");
        if constexpr (sizeof(T) == 16) {
            return _mm_slli_si128(self.val, V);
        } else if constexpr (sizeof(T) == 8) {
            return _mm_slli_epi64(self.val, V);
        } else if constexpr (sizeof(T) == 4) {
            return _mm_slli_epi32(self.val, V);
        } else {
            static_assert(sizeof(T) == 2, "unsupported operation");
            return _mm_slli_epi16(self.val, V);
        }
    }

    template <int V>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL) inline vector& operator<<=(immediate<V> imm)
    requires std::integral<T>
    {
        return *this = *this << imm;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline mask_type
    operator==(this vector self, vector right) noexcept {
        if constexpr (sized_integral<T, 1>) {
            return mask_type{_mm_cmpeq_epi8(self.val, right.val)};
        } else if constexpr (sized_integral<T, 2>) {
            return mask_type{_mm_cmpeq_epi16(self.val, right.val)};
        } else if constexpr (sized_integral<T, 4>) {
            return mask_type{_mm_cmpeq_epi32(self.val, right.val)};
        } else if constexpr (sized_integral<T, 8>) {
            static_assert(sized_integral<T, 8>);
            return mask_type{_mm_cmpeq_epi64(self.val, right.val)};
        } else if constexpr (sized_floating_point<T, 4>) {
            return mask_type{_mm_cmpeq_ps(self.val, right.val)};
        } else {
            static_assert(sized_floating_point<T, 8>, "unsupported operation");
            return mask_type{_mm_cmpeq_pd(self.val, right.val)};
        }
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline mask_type
    operator!=(this vector self, vector right) noexcept {
        if constexpr (sized_integral<T, 1>) {
            return mask_type{_mm_cmpneq_epi8(self.val, right.val)};
        } else if constexpr (sized_integral<T, 2>) {
            return mask_type{_mm_cmpneq_epi16(self.val, right.val)};
        } else if constexpr (sized_integral<T, 4>) {
            return mask_type{_mm_cmpneq_epi32(self.val, right.val)};
        } else if constexpr (sized_integral<T, 8>) {
            static_assert(sized_integral<T, 8>);
            return mask_type{_mm_cmpneq_epi64(self.val, right.val)};
        } else if constexpr (sized_floating_point<T, 4>) {
            return mask_type{_mm_cmpneq_ps(self.val, right.val)};
        } else {
            static_assert(sized_floating_point<T, 8>, "unsupported operation");
            return mask_type{_mm_cmpneq_pd(self.val, right.val)};
        }
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline mask_type operator<(
        this vector self, vector right) noexcept
    requires (!std::unsigned_integral<T>)
    {
        if constexpr (sized_integral<T, 1>) {
            return mask_type{_mm_cmplt_epi8(self.val, right.val)};
        } else if constexpr (sized_integral<T, 2>) {
            return mask_type{_mm_cmplt_epi16(self.val, right.val)};
        } else if constexpr (sized_integral<T, 4>) {
            return mask_type{_mm_cmplt_epi32(self.val, right.val)};
        } else if constexpr (sized_integral<T, 8>) {
            static_assert(sized_integral<T, 4>, "unsupported operation");
            DPL_BUILTIN_unreachable();
        } else if constexpr (sized_floating_point<T, 4>) {
            return mask_type{_mm_cmplt_ps(self.val, right.val)};
        } else {
            static_assert(sized_floating_point<T, 8>, "unsupported operation");
            return mask_type{_mm_cmplt_pd(self.val, right.val)};
        }
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline mask_type
    operator<=(this vector self, vector right) noexcept
    requires (!std::unsigned_integral<T>)
    {
        if constexpr (std::integral<T>) {
            return mask_type{~(right < self)};
        } else if constexpr (sized_floating_point<T, 4>) {
            return mask_type{_mm_cmple_ps(self.val, right.val)};
        } else {
            static_assert(sized_floating_point<T, 8>, "unsupported operation");
            return mask_type{_mm_cmple_pd(self.val, right.val)};
        }
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline mask_type operator>(
        this vector self, vector right) noexcept
    requires (!std::unsigned_integral<T>)
    {
        if constexpr (std::integral<T>) {
            return right < self;
        } else if constexpr (sized_floating_point<T, 4>) {
            return mask_type{_mm_cmpgt_ps(self.val, right.val)};
        } else {
            static_assert(sized_floating_point<T, 8>, "unsupported operation");
            return mask_type{_mm_cmpgt_pd(self.val, right.val)};
        }
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline mask_type
    operator>=(this vector self, vector right) noexcept
    requires (!std::unsigned_integral<T>)
    {
        if constexpr (std::integral<T>) {
            return mask_type{_mm_xor_si128(+(self < right), +all<T>)};
        } else if constexpr (sized_floating_point<T, 4>) {
            return mask_type{_mm_cmpge_ps(self.val, right.val)};
        } else {
            static_assert(sized_floating_point<T, 8>, "unsupported operation");
            return mask_type{_mm_cmpge_pd(self.val, right.val)};
        }
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline mask_type operator<(
        this vector self, vector right) noexcept {
        return ~(right <= self);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline mask_type
    operator<=(this vector self, vector right) noexcept {
        if constexpr (sizeof(T) == 1) {
            return mask_type{
                _mm_cmpeq_epi8(_mm_max_epu8(+self, +right), +right)};
        } else if constexpr (sizeof(T) == 2) {
            return mask_type{
                _mm_cmpeq_epi16(_mm_max_epu16(+self, +right), +right)};
        } else if constexpr (sizeof(T) == 4) {
            return mask_type{
                _mm_cmpeq_epi32(_mm_max_epu32(+self, +right), +right)};
        } else {
#if DPL_SIMD_X86_AVX512 & DPL_SIMD_X86_AVX512DQ
            static_assert(sizeof(T) == 8, "unsupported operation");
            return mask_type{
                _mm_cmpeq_epi64(_mm_max_epu64(+self, +right), +right)};
#else
            static_assert(sizeof(T) == 4, "unsupported operation");
            DPL_BUILTIN_unreachable();
#endif
        }
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline mask_type operator>(
        this vector self, vector right) noexcept {
        return ~(self <= right);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline mask_type
    operator>=(this vector self, vector right) noexcept {
        if constexpr (sizeof(T) == 1) {
            return mask_type{
                _mm_cmpeq_epi8(_mm_min_epu8(+self, +right), +right)};
        } else if constexpr (sizeof(T) == 2) {
            return mask_type{
                _mm_cmpeq_epi16(_mm_min_epu16(+self, +right), +right)};
        } else if constexpr (sizeof(T) == 4) {
            return mask_type{
                _mm_cmpeq_epi32(_mm_min_epu32(+self, +right), +right)};
        } else {
#if DPL_SIMD_X86_AVX512 & DPL_SIMD_X86_AVX512DQ
            static_assert(sizeof(T) == 8, "unsupported operation");
            return mask_type{
                _mm_cmpeq_epi64(_mm_min_epu64(+self, +right), +right)};
#else
            static_assert(sizeof(T) == 4, "unsupported operation");
            DPL_BUILTIN_unreachable();
#endif
        }
    }

#if DPL_SIMD_X86_AVX512FP16 & DPL_SUPPORTS_FLOAT16
    __DPL_HIDE_FROM_ABI explicit inline vector(vector_mask<T> val) noexcept
    requires std::same_as<float16, T>
    = delete; // TODO
    __DPL_HIDE_FROM_ABI explicit inline vector(scalar_mask<T> val) noexcept
    requires std::same_as<float16, T>
    = delete; // TODO

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline vector operator+(
        this vector left, vector right) noexcept
    requires std::floating_point<T> && std::same_as<float16, T>
    {
        return _mm_add_ph(left.val, right.val);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline vector operator-(
        this vector left, vector right) noexcept
    requires std::floating_point<T> && std::same_as<float16, T>
    {
        return _mm_sub_ph(left.val, right.val);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline vector operator/(
        this vector self, vector right) noexcept
    requires std::floating_point<T> && std::same_as<float16, T>
    {
        return _mm_div_ph(self.val, right.val);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline mask_type
    operator==(this vector self, vector right) noexcept
    requires std::same_as<float16, T>
    {
        return mask_type{_mm_cmp_ph_mask(+left, self.val, 0)};
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline mask_type
    operator!=(this vector self, vector right) noexcept
    requires std::same_as<float16, T>
    {
        return mask_type{_mm_cmp_ph_mask(+left, self.val, 12)};
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) friend inline mask_type
    operator<(this vector self, vector right) noexcept
    requires std::same_as<float16, T>
    {
        return mask_type{_mm_cmp_ph_mask(+left, self.val, 17)};
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) friend inline mask_type
    operator<=(this vector self, vector right) noexcept
    requires std::same_as<float16, T>
    {
        return mask_type{_mm_cmp_ph_mask(+left, self.val, 18)};
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) friend inline mask_type
    operator>(this vector self, vector right) noexcept
    requires std::same_as<float16, T>
    {
        return mask_type{_mm_cmp_ph_mask(+left, self.val, 30)};
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) friend inline mask_type
    operator>=(this vector self, vector right) noexcept
    requires std::same_as<float16, T>
    {
        return mask_type{_mm_cmp_ph_mask(+left, self.val, 29)};
    }
#endif

private:
    native_type val;
};

template <size_t N>
requires (N == xmm_abi::size)
explicit vector(blob<xmm_abi::size> const&) -> vector<blob<xmm_abi::size>>;
template <xmm_element T>
explicit vector(zero_t<T>) -> vector<T>;
template <xmm_element T>
explicit vector(all_t<T>) -> vector<T>;
template <xmm_element T>
explicit vector(msb_t<T>) -> vector<T>;
template <xmm_element T>
explicit vector(lsb_t<T>) -> vector<T>;
template <xmm_element T>
explicit vector(iota_t<T>) -> vector<T>;

template <xmm_element T>
explicit vector(scalar<T>) -> vector<T>;
template <xmm_element T>
explicit vector(vector_mask<T>) -> vector<T>;
template <xmm_element T>
explicit vector(scalar_mask<T>) -> vector<T>;

template <xmm_element To, reinterpretable_as<To> From>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline vector<To>
reinterpret_as(vector<From> vec) noexcept {
    return vector<To>{native_cast<To, From>(+vec)};
}

template <simd_element T>
class multiplication<vector<T>> : public binary_vector_expression {
    static_assert(std::floating_point<T>);

public:
    using element_type = T;
    using native_type = native_type_t<T>;

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL) explicit inline multiplication(
        native_type left, native_type right) noexcept
        : left{left}
        , right{right} {}

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) explicit inline
    operator native_type(this multiplication self) noexcept {
        if constexpr (std::same_as<T, float>) {
            return _mm_mul_ps(self.left, self.right);
        } else {
            static_assert(std::same_as<T, double>, "unsupported operation");
            return _mm_mul_pd(self.left, self.right);
        }
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline vector<T> operator+(
        this multiplication self, vector<T> right) noexcept {
        if constexpr (std::same_as<T, float>) {
            return _mm_fmadd_ps(self.left, self.right, +right);
        } else {
            static_assert(std::same_as<T, double>, "unsupported operation");
            return _mm_fmadd_pd(self.left, self.right, +right);
        }
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline vector<T> operator-(
        this multiplication self, vector<T> right) noexcept {
        if constexpr (sized_floating_point<T, 4>) {
            return _mm_fmsub_ps(self.left, self.right, +right);
        } else {
            static_assert(sized_floating_point<T, 8>, "unsupported operation");
            return _mm_fmsub_pd(self.left, self.right, +right);
        }
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) friend inline vector<T>
    operator+(vector<T> left, multiplication mul) noexcept {
        return mul + left;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) friend inline vector<T>
    operator-(vector<T> left, multiplication mul) noexcept {
        if constexpr (sized_floating_point<T, 4>) {
            return _mm_fnmadd_ps(mul.left, mul.right, +left);
        } else {
            static_assert(sized_floating_point<T, 8>, "unsupported operation");
            return _mm_fnmadd_pd(mul.left, mul.right, +left);
        }
    }

#if DPL_SIMD_X86_AVX512FP16 & DPL_SUPPORTS_FLOAT16
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) explicit inline
    operator native_type(this multiplication self) noexcept
    requires std::same_as<float16, element_type>
    {
        return _mm_mul_ph(self.left, self.right);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline vector<T> operator+(
        this multiplication self, vector<T> right) noexcept {
        return _mm_fmadd_ph(self.left, self.right, +right);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline vector<T> operator-(
        this multiplication self, vector<T> right) noexcept {
        return _mm_fmsub_ph(self.left, self.right, +right);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) friend inline vector<T>
    operator+(vector<T> left, multiplication mul) noexcept {
        return mul + left;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) friend inline vector<T>
    operator-(vector<T> left, multiplication mul) noexcept {
        return _mm_fnmadd_ph(mul.left, mul.right, +left);
    }
#endif

private:
    native_type left, right;
};

template <xmm_element T>
class bw_negation<vector<T>> : public unary_vector_expression {
public:
    using element_type = T;
    using native_type = native_type_t<T>;

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL) explicit inline bw_negation(
        vector<T> value) noexcept
        : val{value} {}

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) friend inline vector<T>
    evaluate(bw_negation self) noexcept {
        if constexpr (sized_floating_point<T, 4>) {
            return _mm_xor_ps(self.val, +all<T>);
        } else if constexpr (sized_floating_point<T, 8>) {
            return _mm_xor_pd(self.val, +all<T>);
        } else {
            return native_cast<T, int>(_mm_xor_si128(self.val, +all<T>));
        }
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline vector<T> operator&(
        this bw_negation self, vector<T> right) noexcept {
        if constexpr (sized_floating_point<T, 4>) {
            return _mm_andnot_ps(self.val, +right);
        } else if constexpr (sized_floating_point<T, 8>) {
            return _mm_andnot_pd(self.val, +right);
        } else {
            return native_cast<T, int>(_mm_andnot_si128(self.val, +right));
        }
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline bw_negation
    operator|(this bw_negation self, vector<T> right) noexcept {
        if constexpr (sized_floating_point<T, 4>) {
            return bw_negation{_mm_andnot_ps(self.val, +right)};
        } else if constexpr (sized_floating_point<T, 8>) {
            return bw_negation{_mm_andnot_pd(self.val, +right)};
        } else {
            return bw_negation{
                native_cast<T, int>(_mm_andnot_si128(self.val, +right))};
        }
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline bw_negation
    operator&(this bw_negation self, bw_negation right) noexcept {
        if constexpr (sized_floating_point<T, 4>) {
            return bw_negation{_mm_or_ps(self.val, right.val)};
        } else if constexpr (sized_floating_point<T, 8>) {
            return bw_negation{_mm_or_pd(self.val, right.val)};
        } else {
            return bw_negation{
                native_cast<T, int>(_mm_or_si128(self.val, right.val))};
        }
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline bw_negation
    operator|(this bw_negation self, bw_negation right) noexcept {
        if constexpr (sized_floating_point<T, 4>) {
            return bw_negation{_mm_and_ps(self.val, right.val)};
        } else if constexpr (sized_floating_point<T, 8>) {
            return bw_negation{_mm_and_pd(self.val, right.val)};
        } else {
            return bw_negation{
                native_cast<T, int>(_mm_and_si128(self.val, right.val))};
        }
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) friend inline vector<T>
    operator&(vector<T> left, bw_negation self) noexcept {
        if constexpr (sized_floating_point<T, 4>) {
            return _mm_andnot_ps(self.val, left);
        } else if constexpr (sized_floating_point<T, 8>) {
            return _mm_andnot_pd(self.val, left);
        } else {
            return native_cast<T, int>(_mm_andnot_si128(self.val, left));
        }
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) friend inline bw_negation
    operator|(vector<T> left, bw_negation self) noexcept {
        if constexpr (sized_floating_point<T, 4>) {
            return _mm_andnot_ps(self.val, left);
        } else if constexpr (sized_floating_point<T, 8>) {
            return _mm_andnot_pd(self.val, left);
        } else {
            return native_cast<T, int>(_mm_andnot_si128(self.val, left));
        }
    }

private:
    native_type val;
};

DPL_EXPORT_END

DPL_SIMD_ABI_NAMESPACE_END(xmm)

DPL_NAMESPACE_END