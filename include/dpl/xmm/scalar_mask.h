// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h" // IWYU pragma: keep

// IWYU pragma: begin_exports
#include "dpl/xmm/fwd.h"

#include "dpl/xmm/constants.h"
#include "dpl/xmm/native.h"
// IWYU pragma: end_exports

#include "dpl/xmm/concepts.h"

#if !DPL_MODULES
#  include <immintrin.h>
#endif

DPL_NAMESPACE_BEGIN

DPL_SIMD_ABI_NAMESPACE_BEGIN(xmm)

DPL_EXPORT_BEGIN

template <xmm_element T>
class __DPL_PUBLIC_TEMPLATE scalar_mask<T> {
public:
    using element_type = T;
    using native_type = native_type_t<T>;

    __DPL_HIDE_FROM_ABI inline constexpr scalar_mask() noexcept = default;
    __DPL_HIDE_FROM_ABI inline constexpr scalar_mask(
        scalar_mask const&) noexcept = default;
    __DPL_HIDE_FROM_ABI inline constexpr scalar_mask& operator=(
        scalar_mask const&) noexcept = default;
    __DPL_HIDE_FROM_ABI inline constexpr ~scalar_mask() noexcept = default;
    __DPL_HIDE_FROM_ABI explicit inline constexpr scalar_mask(
        native_type val) noexcept
        : val{val} {}

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) explicit inline
    operator native_type(this scalar_mask self) noexcept {
        return self.val;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) inline native_type operator+(
        this scalar_mask self) noexcept {
        return self.val;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, VECTORCALL, NODISCARD) explicit inline
    operator bool(this scalar_mask self) noexcept {
        if constexpr (sizeof(T) == 1) {
            return _mm_movemask_epi8(val) & 0b1;
        } else if constexpr (sizeof(T) == 2) {
            // WARNING 16bit elements do not have appropriate masks
            return (_mm_movemask_epi8(val) & 0b11) == 0b11;
        } else if constexpr (sized_integral<T, 4>) {
            return _mm_movemask_ps(native_cast<float, T>(self.val)) & 0x1;
        } else if constexpr (sized_integral<T, 8>) {
            return _mm_movemask_pd(native_cast<double, T>(self.val)) & 0x1;
        } else if constexpr (sized_floating_point<T, 4>) {
            return _mm_movemask_ps(self.val) & 0x1;
        } else {
            static_assert(sized_floating_point<T, 8>, "unsupported operation");
            return _mm_movemask_pd(self.val) & 0x1;
        }
    }

private:
    native_type val;
};

#if DPL_SIMD_X86_AVX512FP16 & DPL_SUPPORTS_FLOAT16
template <>
class __DPL_PUBLIC_TEMPLATE scalar_mask<float16> {
public:
    using element_type = T;
    using native_type = __mmask8;

    __DPL_HIDE_FROM_ABI inline constexpr scalar_mask() noexcept = default;
    __DPL_HIDE_FROM_ABI inline constexpr scalar_mask(
        scalar_mask const&) noexcept = default;
    __DPL_HIDE_FROM_ABI inline constexpr scalar_mask& operator=(
        scalar_mask const&) noexcept = default;
    __DPL_HIDE_FROM_ABI inline constexpr ~scalar_mask() noexcept = default;
    __DPL_HIDE_FROM_ABI explicit inline constexpr scalar_mask(
        native_type val) noexcept
        : val{val} {}

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) explicit inline
    operator native_type(this scalar_mask self) noexcept {
        return self.val;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) inline native_type operator+(
        this scalar_mask self) noexcept {
        return self.val;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD) explicit inline operator bool(
        this scalar_mask self) noexcept {
        return val & 0b1;
    }

private:
    native_type val;
};
#endif

template <xmm_scalar_element To, reinterpretable_as<To> From>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline scalar_mask<To>
reinterpret_as(scalar_mask<From> vec) noexcept {
    return scalar_mask<To>{native_cast<To, From>(+vec)};
}

DPL_EXPORT_END

DPL_SIMD_ABI_NAMESPACE_END(xmm)

DPL_NAMESPACE_END
