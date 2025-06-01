// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h" // IWYU pragma: keep

#include "dpl/xmm/concepts.h"

// IWYU pragma: begin_exports
#include "dpl/xmm/fwd.h"

#include "dpl/xmm/bitset.h"
#include "dpl/xmm/constants.h"
#include "dpl/xmm/immediate_mask.h"
#include "dpl/xmm/native.h"
#include "dpl/xmm/scalar_mask.h"
// IWYU pragma: end_exports

#if !DPL_MODULES
#  include <immintrin.h>

#  include <bit>
#endif

DPL_NAMESPACE_BEGIN

DPL_SIMD_ABI_NAMESPACE_BEGIN(xmm)

DPL_EXPORT_BEGIN

template <xmm_element T>
class __DPL_PUBLIC_TEMPLATE vector_mask<T> {

    template <bit_type_for<T> V>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, PURE, NODISCARD) static inline native_type_t<T> load(
        mask_for<T, V>) noexcept {
        static constexpr auto set = +(bitset_for<T>{V} * all<T>);
        return native_cast<T, blob<xmm_abi::size>>(
            _mm_load_si128(reinterpret_cast<__m128i const*>(&set)));
    }

public:
    using element_type = T;
    using native_type = native_type_t<T>;

    __DPL_HIDE_FROM_ABI inline constexpr vector_mask() noexcept = default;
    __DPL_HIDE_FROM_ABI inline constexpr vector_mask(
        vector_mask const&) noexcept = default;
    __DPL_HIDE_FROM_ABI inline constexpr vector_mask& operator=(
        vector_mask const&) noexcept = default;
    __DPL_HIDE_FROM_ABI inline constexpr ~vector_mask() noexcept = default;
    __DPL_HIDE_FROM_ABI explicit inline constexpr vector_mask(
        native_type val) noexcept
        : val{val} {}
    __DPL_HIDE_FROM_ABI inline vector_mask(all_t<T> all) noexcept : val{+all} {}
    __DPL_HIDE_FROM_ABI inline vector_mask(zero_t<T> zero) noexcept
        : val{+zero} {}

    template <bit_type_for<T> V>
    __DPL_HIDE_FROM_ABI inline vector_mask(mask_for<T, V> mask) noexcept
        : val{load(mask)} {}

    template <simd_element U>
    requires (sizeof(T) == sizeof(U))
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL) inline vector_mask(
        vector_mask<U> vec) noexcept
        : val{native_cast<T, U>(+vec)} {}

    template <xmm_scalar_element U>
    requires (sizeof(T) == sizeof(U))
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL) inline vector_mask(
        scalar_mask<U> vec) noexcept
        : val{native_cast<T, U>(+vec)} {}

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) explicit inline
    operator native_type(this vector_mask self) noexcept {
        return self.val;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) inline native_type operator+(
        this vector_mask self) noexcept {
        return self.val;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, VECTORCALL, NODISCARD) explicit inline
    operator int(this vector_mask self) noexcept {
        // WARNING 16bit elements do not have appropriate masks
        if constexpr (sizeof(T) < 4) {
            return _mm_movemask_epi8(self.val);
        } else if constexpr (sized_integral<T, 4>) {
            return _mm_movemask_ps(native_cast<float, T>(self.val));
        } else if constexpr (sized_integral<T, 8>) {
            return _mm_movemask_pd(native_cast<double, T>(self.val));
        } else if constexpr (sized_floating_point<T, 2>) {
            return _mm_movemask_epi8(native_cast<short, T>(self.val));
        } else if constexpr (sized_floating_point<T, 4>) {
            return _mm_movemask_ps(self.val);
        } else {
            static_assert(sized_floating_point<T, 8>, "unsupported operation");
            return _mm_movemask_pd(self.val);
        }
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, VECTORCALL, NODISCARD) inline vector_mask
    operator&(this vector_mask self, vector_mask other) noexcept {
        return vector_mask{native_cast<T, int>(_mm_and_si128(
            native_cast<int, T>(self.val), native_cast<int, T>(+other)))};
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, VECTORCALL, NODISCARD) inline vector_mask
    operator|(this vector_mask self, vector_mask other) noexcept {
        return vector_mask{native_cast<T, int>(_mm_or_si128(
            native_cast<int, T>(self.val), native_cast<int, T>(+other)))};
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, VECTORCALL, NODISCARD) inline vector_mask
    operator^(this vector_mask self, vector_mask other) noexcept {
        return vector_mask{native_cast<T, int>(_mm_xor_si128(
            native_cast<int, T>(self.val), native_cast<int, T>(+other)))};
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, VECTORCALL, NODISCARD) inline bw_negation<vector_mask>
    operator~(this vector_mask self) noexcept {
        return bw_negation<vector_mask>{+self};
    }

    template <xmm_element U>
    requires (sizeof(U) == sizeof(T))
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, VECTORCALL, NODISCARD) inline bool match_all(
        this vector_mask self, vector_mask<U> other) noexcept {
        vector_mask const result{native_cast<T, int>(_mm_cmpeq_si128(
            native_cast<int, T>(self.val), native_cast<int, T>(+other)))};
        return all_of(result);
    }

    template <xmm_element U>
    requires (sizeof(U) == sizeof(T))
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, VECTORCALL, NODISCARD) inline bool match_any(
        this vector_mask self, vector_mask<U> other) noexcept {
        vector_mask const result{native_cast<T, int>(_mm_cmpeq_si128(
            native_cast<int, T>(self.val), native_cast<int, T>(+other)))};
        return any_of(result);
    }

    template <xmm_element U>
    requires (sizeof(U) == sizeof(T))
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, VECTORCALL, NODISCARD) inline bool match_none(
        this vector_mask self, vector_mask<U> other) noexcept {
        vector_mask const result{native_cast<T, int>(_mm_cmpeq_si128(
            native_cast<int, T>(self.val), native_cast<int, T>(+other)))};
        return none_of(result);
    }

    template <xmm_element U>
    requires (sizeof(U) == sizeof(T))
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, VECTORCALL, NODISCARD) inline bool match_some(
        this vector_mask self, vector_mask<U> other) noexcept {
        vector_mask const result{native_cast<T, int>(_mm_cmpeq_si128(
            native_cast<int, T>(self.val), native_cast<int, T>(+other)))};
        return some_of(result);
    }

    template <size_t W, bit_type_t<W> V>
    requires (W == element_count<T>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, VECTORCALL, NODISCARD) inline bool match_all(
        this vector_mask self, basic_mask<W, V>) noexcept {
        if constexpr (mask_width == register_size && mask == all<T>) {
            return _mm_test_all_ones(self.val);
        } else if constexpr (sizeof(T) == 2) {
            // only check upper bit
            static constexpr auto tiled =
                +(bitset<8>{V}.template zero_extend<2>() << 1);
            return (static_cast<int>(self) & tiled) == tiled;
        } else {
            return (static_cast<int>(self) & V) == V;
        }
    }

    template <size_t W, bit_type_t<W> V>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, VECTORCALL, NODISCARD) inline bool match_any(
        this vector_mask self, basic_mask<W, V>) noexcept {
        if constexpr (sizeof(T) == 2) {
            static constexpr auto tiled =
                +(bitset<8>{V}.template zero_extend<2>() << 1);
            return (static_cast<int>(self) & tiled) != 0;
        } else {
            return (static_cast<int>(self) & V) != 0;
        }
    }

    template <size_t W, bit_type_t<W> V>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, VECTORCALL, NODISCARD) inline bool match_none(
        this vector_mask self, basic_mask<W, V>) noexcept {
        if constexpr (sizeof(T) == 2) {
            static constexpr auto tiled =
                +(bitset<8>{V}.template zero_extend<2>() << 1);
            return (static_cast<int>(self) & tiled) == 0;
        } else {
            return (static_cast<int>(self) & V) == 0;
        }
    }

    template <size_t W, bit_type_t<W> V>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, VECTORCALL, NODISCARD) inline bool match_some(
        this vector_mask self, basic_mask<W, V>) noexcept {

        if constexpr (sizeof(T) == 2) {
            static constexpr auto tiled =
                +(bitset<8>{V}.template zero_extend<2>() << 1);
            auto const result = static_cast<int>(self) & tiled;
            return result && std::popcount(result) < std::popcount(V);
        } else {
            auto const result = static_cast<int>(self) & V;
            return result && std::popcount(result) < std::popcount(V);
        }
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, VECTORCALL, NODISCARD) inline bool match_all(
        this vector_mask self, all_t<T>) noexcept {
        return self.match_all(mask(all<T>));
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, VECTORCALL, NODISCARD) inline bool match_any(
        this vector_mask self, all_t<T> g) noexcept {
        return self.match_any(mask(all<T>));
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, VECTORCALL, NODISCARD) inline bool match_none(
        this vector_mask self, all_t<T>) noexcept {
        return self.match_none(mask(all<T>));
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, VECTORCALL, NODISCARD) inline bool match_some(
        this vector_mask self, all_t<T>) noexcept {
        return self.match_some(mask(all<T>));
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, VECTORCALL, NODISCARD) inline bool match_all(
        this vector_mask self, zero_t<T>) noexcept {
        return self.match_all(mask(zero<T>));
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, VECTORCALL, NODISCARD) inline bool match_any(
        this vector_mask self, zero_t<T>) noexcept {
        return self.match_any(mask(zero<T>));
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, VECTORCALL, NODISCARD) inline bool match_none(
        this vector_mask self, zero_t<T>) noexcept {
        return self.match_none(mask(zero<T>));
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, VECTORCALL, NODISCARD) inline bool match_some(
        this vector_mask self, zero_t<T>) noexcept {
        return self.match_some(mask(zero<T>));
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, VECTORCALL, NODISCARD) friend inline bool any_of(
        vector_mask self) noexcept {
        return static_cast<int>(self);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, VECTORCALL, NODISCARD) friend inline bool none_of(
        vector_mask self) noexcept {
        return static_cast<int>(self) == 0;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, VECTORCALL, NODISCARD) friend inline bool all_of(
        vector_mask self) noexcept {
        return match_all(all<T>);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, VECTORCALL, NODISCARD) friend inline bool some_of(
        vector_mask self) noexcept {
        return match_some(all<T>);
    }

    template <typename U>
    requires (vector_mask self, U val)
    {
        self.match_all(val);
    }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, VECTORCALL, NODISCARD) inline bool operator==(
        this vector_mask self, U val) noexcept {
        return self.match_all(val);
    }

    template <typename U>
    requires (vector_mask self, U val)
    {
        self.match_all(val);
    }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, VECTORCALL, NODISCARD) inline bool operator!=(
        this vector_mask self, U val) noexcept {
        return !(self == val);
    }

    template <typename U>
    requires (vector_mask self, U val)
    {
        self.match_all(val);
    }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, VECTORCALL, NODISCARD) friend inline bool
    operator==(U val, vector_mask self) noexcept {
        return self == val;
    }

    template <typename U>
    requires (vector_mask self, U val)
    {
        self.match_all(val);
    }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, VECTORCALL, NODISCARD) friend inline bool
    operator!=(U val, vector_mask self) noexcept {
        return self != val;
    }

private:
    native_type val;
};

template <xmm_element To, reinterpretable_as<To> From>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline vector_mask<To>
reinterpret_as(vector_mask<From> vec) noexcept {
    return vector_mask<To>{native_cast<To, From>(+vec)};
}

template <xmm_element T>
class bw_negation<vector_mask<T>> : public unary_vector_mask_expression {
public:
    using element_type = T;
    using native_type = native_type_t<T>;

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL) inline bw_negation(
        native_type_t<V> value) noexcept
        : val{value} {}

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, PURE, VECTORCALL, NODISCARD) explicit inline
    operator native_type(this bw_negation self) noexcept {
        if constexpr (sized_floating_point<T, 4>) {
            return _mm_xor_ps(self.val, all<T>);
        } else if constexpr (sized_floating_point<T, 8>) {
            return _mm_xor_pd(self.val, all<T>);
        } else {
            return _mm_xor_si128(self.val, all<T>);
        }
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, VECTORCALL, NODISCARD) inline vector_mask<T>
    operator&(this bw_negation self, vector_mask<T> right) noexcept {
        if constexpr (sized_floating_point<T, 4>) {
            return _mm_andnot_ps(self.val, +right);
        } else if constexpr (sized_floating_point<T, 8>) {
            return _mm_andnot_pd(self.val, +right);
        } else {
            return _mm_andnot_si128(self.val, +right);
        }
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, VECTORCALL, NODISCARD) inline bw_negation
    operator|(this bw_negation self, vector_mask<T> right) noexcept {
        if constexpr (sized_floating_point<T, 4>) {
            return _mm_andnot_ps(self.val, +right);
        } else if constexpr (sized_floating_point<T, 8>) {
            return _mm_andnot_pd(self.val, +right);
        } else {
            return _mm_andnot_si128(self.val, +right);
        }
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, VECTORCALL, NODISCARD) inline bw_negation
    operator&(this bw_negation self, bw_negation right) noexcept {
        if constexpr (sized_floating_point<T, 4>) {
            return _mm_or_ps(self.val, right.val);
        } else if constexpr (sized_floating_point<T, 8>) {
            return _mm_or_pd(self.val, right.val);
        } else {
            return _mm_or_si128(self.val, right.val);
        }
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, VECTORCALL, NODISCARD) inline bw_negation
    operator|(this bw_negation self, bw_negation right) noexcept {
        if constexpr (sized_floating_point<T, 4>) {
            return _mm_and_ps(self.val, right.val);
        } else if constexpr (sized_floating_point<T, 8>) {
            return _mm_and_pd(self.val, right.val);
        } else {
            return _mm_and_si128(self.val, right.val);
        }
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, VECTORCALL, NODISCARD) friend inline vector_mask<T>
    operator&(vector_mask<T> left, bw_negation self) noexcept {
        if constexpr (sized_floating_point<T, 4>) {
            return _mm_andnot_ps(self.val, left);
        } else if constexpr (sized_floating_point<T, 8>) {
            return _mm_andnot_pd(self.val, left);
        } else {
            return _mm_andnot_si128(self.val, left);
        }
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, VECTORCALL, NODISCARD) friend inline bw_negation
    operator|(vector_mask<T> left, bw_negation self) noexcept {
        if constexpr (sized_floating_point<T, 4>) {
            return _mm_andnot_ps(self.val, left);
        } else if constexpr (sized_floating_point<T, 8>) {
            return _mm_andnot_pd(self.val, left);
        } else {
            return _mm_andnot_si128(self.val, left);
        }
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, VECTORCALL, NODISCARD) inline vector_mask<T>
    operator&(this bw_negation self, vector_mask<T> right) noexcept {
        return _mm_andnot_si128(self.val, +right);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, VECTORCALL, NODISCARD) friend inline auto
    operator&(vector_mask<T> left, bw_negation self) noexcept {
        return _mm_andnot_si128(self.val, +left);
    }

private:
    native_type val;
};

#if DPL_SIMD_X86_AVX512FP16 & DPL_SUPPORTS_FLOAT16

template <>
class __DPL_PUBLIC_TEMPLATE vector_mask<float16> {
public:
    using element_type = float16;
    using native_type = __mmask8;

    __DPL_HIDE_FROM_ABI inline constexpr vector_mask() noexcept = default;
    __DPL_HIDE_FROM_ABI inline constexpr vector_mask(
        vector_mask const&) noexcept = default;
    __DPL_HIDE_FROM_ABI inline constexpr vector_mask& operator=(
        vector_mask const&) noexcept = default;
    __DPL_HIDE_FROM_ABI inline constexpr ~vector_mask() noexcept = default;
    __DPL_HIDE_FROM_ABI explicit inline constexpr vector_mask(
        native_type val) noexcept
        : val{val} {}
    __DPL_HIDE_FROM_ABI inline vector_mask(all_t<element_type> all) noexcept
        : val{0xFF} {}
    __DPL_HIDE_FROM_ABI inline vector_mask(zero_t<element_type> zero) noexcept
        : val{} {}

    template <size_t W, bit_type_t<W> V>
    requires (W == element_count<element_type>)
    __DPL_HIDE_FROM_ABI inline vector_mask(basic_mask<W, V> mask) noexcept
        : val{static_cast<native_type>(V)} {}

    template <simd_element U>
    requires (sizeof(element_type) == sizeof(U))
    __DPL_HIDE_FROM_ABI inline vector_mask(vector_mask<U> vec) noexcept
        : val{static_cast<native_type>(static_cast<int>(vec))} {}

    template <xmm_scalar_element U>
    requires (sizeof(element_type) == sizeof(U))
    __DPL_HIDE_FROM_ABI inline vector_mask(scalar_mask<U> vec) noexcept
        : val{static_cast<native_type>(static_cast<int>(vec))} {}

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) explicit inline
    operator native_type(this vector_mask self) noexcept {
        return self.val;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) inline native_type operator+(
        this vector_mask self) noexcept {
        return self.val;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD) explicit inline operator int(
        this vector_mask self) noexcept {
        return static_cast<int>(self.val);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD) inline vector_mask operator&(
        this vector_mask self, vector_mask other) noexcept {
        return self.val & other.val;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD) inline vector_mask operator|(
        this vector_mask self, vector_mask other) noexcept {
        return self.val | other.val;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD) inline vector_mask operator^(
        this vector_mask self, vector_mask other) noexcept {
        return self.val ^ other.val;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD) inline vector_mask operator~(
        this vector_mask self) noexcept {
        return vector_mask{~self.val};
    }

    template <xmm_element U>
    requires (sizeof(U) == sizeof(element_type))
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD) inline bool match_all(
        this vector_mask self, vector_mask<U> other) noexcept {
        return self.val == static_cast<native_type>(static_cast<int>(other));
    }

    template <xmm_element U>
    requires (sizeof(U) == sizeof(element_type))
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD) inline bool match_any(
        this vector_mask self, vector_mask<U> other) noexcept {
        auto const result =
            self.val ^ static_cast<native_type>(static_cast<int>(other));
        return result != 0xFF;
    }

    template <xmm_element U>
    requires (sizeof(U) == sizeof(element_type))
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD) inline bool match_none(
        this vector_mask self, vector_mask<U> other) noexcept {
        auto const result =
            self.val ^ static_cast<native_type>(static_cast<int>(other));
        return result == 0xFF;
    }

    template <xmm_element U>
    requires (sizeof(U) == sizeof(element_type))
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD) inline bool match_some(
        this vector_mask self, vector_mask<U> other) noexcept {
        auto const result =
            self.val ^ static_cast<native_type>(static_cast<int>(other));
        return result != 0xFF && result != 0;
    }

    template <size_t W, bit_type_t<W> V>
    requires (W == element_count<element_type>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD) inline bool match_all(
        this vector_mask self, basic_mask<W, V>) noexcept {
        return self.match_all(vector_mask{mask});
    }

    template <size_t W, bit_type_t<W> V>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD) inline bool match_any(
        this vector_mask self, basic_mask<W, V> mask) noexcept {
        return self.match_any(vector_mask{mask});
    }

    template <size_t W, bit_type_t<W> V>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD) inline bool match_none(
        this vector_mask self, basic_mask<W, V>) noexcept {
        return self.match_none(vector_mask{mask});
    }

    template <size_t W, bit_type_t<W> V>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD) inline bool match_some(
        this vector_mask self, basic_mask<W, V>) noexcept {
        return self.match_some(vector_mask{mask});
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD) inline bool match_all(
        this vector_mask self, all_t<T>) noexcept {
        return _mm_test_all_ones(reinterpret_as<int>(+self));
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD) inline bool match_any(
        this vector_mask self, all_t<T> g) noexcept {
        return self.match_any(vector_mask{all<T>});
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD) inline bool match_none(
        this vector_mask self, all_t<T>) noexcept {
        return self.match_none(vector_mask{all<T>});
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD) inline bool match_some(
        this vector_mask self, all_t<T>) noexcept {
        return self.match_some(vector_mask{all<T>});
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD) inline bool match_all(
        this vector_mask self, zero_t<T>) noexcept {
        return self.match_all(vector_mask{zero<T>});
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD) inline bool match_any(
        this vector_mask self, zero_t<T>) noexcept {
        return self.match_any(vector_mask{zero<T>});
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD) inline bool match_none(
        this vector_mask self, zero_t<T>) noexcept {
        return self.match_none(vector_mask{zero<T>});
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD) inline bool match_some(
        this vector_mask self, zero_t<T>) noexcept {
        return self.match_some(vector_mask{zero<T>});
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD) friend inline bool any_of(
        vector_mask self) noexcept {
        return static_cast<int>(self) != 0;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD) friend inline bool none_of(
        vector_mask self) noexcept {
        return static_cast<int>(self) == 0;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD) friend inline bool all_of(
        vector_mask self) noexcept {
        return self.match_all(all<T>);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD) friend inline bool some_of(
        vector_mask self) noexcept {
        return self.match_some(all<T>);
    }

    template <typename U>
    requires (vector_mask self, U val)
    {
        self.match_all(val);
    }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD) inline bool operator==(
        this vector_mask self, U val) noexcept {
        return self.match_all(val);
    }

    template <typename U>
    requires (vector_mask self, U val)
    {
        self.match_all(val);
    }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD) inline bool operator!=(
        this vector_mask self, U val) noexcept {
        return !(self == val);
    }

    template <typename U>
    requires (vector_mask self, U val)
    {
        self.match_all(val);
    }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD) friend inline bool operator==(
        U val, vector_mask self) noexcept {
        return self == val;
    }

    template <typename U>
    requires (vector_mask self, U val)
    {
        self.match_all(val);
    }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD) friend inline bool operator!=(
        U val, vector_mask self) noexcept {
        return self != val;
    }

private:
    native_type val;
};

#endif

template <xmm_element T>
explicit vector_mask(all_t<T>) -> vector_mask<T>;
template <xmm_element T>
explicit vector_mask(zero_t<T>) -> vector_mask<T>;
template <xmm_scalar_element T>
explicit vector_mask(scalar_mask<T>) -> vector_mask<T>;

DPL_EXPORT_END

DPL_SIMD_ABI_NAMESPACE_END(xmm)

DPL_NAMESPACE_END
