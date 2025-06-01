// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h" // IWYU pragma: keep

// IWYU pragma: begin_exports
#include "dpl/xmm/fwd.h"

#include "dpl/xmm/constants.h"
#include "dpl/xmm/native.h"
#include "dpl/xmm/scalar_mask.h"
// IWYU pragma: end_exports

#include "dpl/xmm/concepts.h"

#if !DPL_MODULES
#  include <immintrin.h>

#  include <concepts>
#endif

DPL_NAMESPACE_BEGIN

DPL_SIMD_ABI_NAMESPACE_BEGIN(xmm)

DPL_EXPORT_BEGIN

template <xmm_scalar_element T>
class scalar<T> {
public:
    using element_type = T;
    using native_type = native_type_t<T>;
    using mask_type = scalar_mask<T>;
    using vector_type = vector<T>;

    inline constexpr scalar() noexcept = default;
    inline constexpr scalar(scalar const&) noexcept = default;
    inline constexpr scalar& operator=(scalar const&) noexcept = default;
    inline constexpr ~scalar() noexcept = default;
    inline scalar(native_type val) noexcept : val{val} {}
    inline scalar(T val) noexcept
    requires sized_floating_point<T, 4>
        : scalar{_mm_set_ss(val)} {}
    inline scalar(T val) noexcept
    requires sized_floating_point<T, 8>
        : scalar{_mm_set_sd(val)} {}
    explicit inline scalar(vector<T> val) noexcept : val{+val} {}
    explicit inline scalar(scalar_mask<T> val) noexcept : val{+val} {}
    explicit inline scalar(vector_mask<T> val) noexcept : val{+val} {}

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, NODISCARD) explicit inline operator T(
        this scalar self) noexcept {
        T result;
        // cast to deal with float32 and float64 which are distinct from float
        // and double
        if constexpr (sized_floating_point<T, 4>) {
            _mm_store_ss((float*)&result, self.val);
        } else {
            static_assert(sized_floating_point<T, 8>, "unsupported operation");
            _mm_store_sd((double*)&result, self.val);
        }

        return result;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) T operator*(
        this scalar self) noexcept {
        return static_cast<T>(self);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) native_type operator+(
        this scalar self) noexcept {
        return self.val;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) explicit operator native_type(
        this scalar self) noexcept {
        return self.val;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) inline scalar operator-(
        this scalar self) noexcept;

    template <xmm_type_for<T> R = scalar>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline scalar operator+(
        this scalar self, R right) noexcept {
        if constexpr (sized_floating_point<T, 4>) {
            return _mm_add_ss(self.val, +right);
        } else {
            static_assert(sized_floating_point<T, 8>, "unsupported operation");
            return _mm_add_sd(self.val, +right);
        }
    }

    template <xmm_type_for<T> L = scalar>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) friend inline scalar
    operator+(L left, scalar self) noexcept {
        if constexpr (sized_floating_point<T, 4>) {
            return _mm_add_ss(+left, self.val);
        } else {
            static_assert(sized_floating_point<T, 8>, "unsupported operation");
            return _mm_add_sd(+left, self.val);
        }
    }

    template <xmm_type_for<T> R = scalar>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL) inline scalar& operator+=(R right) noexcept {
        return *this = *this + right;
    }

    template <xmm_type_for<T> R = scalar>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline scalar operator-(
        this scalar self, R right) noexcept {
        if constexpr (sized_floating_point<T, 4>) {
            return _mm_sub_ss(self.val, +right);
        } else {
            static_assert(sized_floating_point<T, 8>, "unsupported operation");
            return _mm_sub_sd(self.val, +right);
        }
    }

    template <xmm_type_for<T> L = scalar>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) friend inline scalar
    operator-(L left, scalar self) noexcept {
        if constexpr (sized_floating_point<T, 4>) {
            return _mm_sub_ss(+left, self.val);
        } else {
            static_assert(sized_floating_point<T, 8>, "unsupported operation");
            return _mm_sub_sd(+left, self.val);
        }
    }

    template <xmm_type_for<T> R = scalar>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL) inline scalar& operator-=(R right) noexcept {
        return *this = *this - right;
    }

    template <xmm_type_for<T> R = scalar>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline scalar operator/(
        this scalar self, R right) noexcept {
        if constexpr (sized_floating_point<T, 4>) {
            return _mm_div_ss(self.val, +right);
        } else {
            static_assert(sized_floating_point<T, 8>, "unsupported operation");
            return _mm_div_sd(self.val, +right);
        }
    }

    template <xmm_type_for<T> R = scalar>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL) inline scalar& operator/=(R right) noexcept {
        return *this = *this / right;
    }

    template <xmm_type_for<T> L = scalar>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) friend inline scalar
    operator/(L left, scalar self) noexcept {
        if constexpr (sized_floating_point<T, 4>) {
            return _mm_div_ss(+left, self.val);
        } else {
            static_assert(sized_floating_point<T, 8>, "unsupported operation");
            return _mm_div_sd(+left, self.val);
        }
    }

    template <xmm_type_for<T> R = scalar>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) inline multiplication<scalar>
    operator*(this scalar self, R right) noexcept {
        static_assert(std::floating_point<T>, "unsupported operation");
        return multiplication<scalar>{self.val, +right};
    }

    template <xmm_type_for<T> R = scalar>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL) inline scalar& operator*=(R right) noexcept {
        return *this = *this * right;
    }

    template <xmm_type_for<T> L = scalar>
    [[gnu::always_inline, nodiscard]] friend inline multiplication<scalar>
    operator*(L left, scalar self) noexcept {
        static_assert(std::floating_point<T>, "unsupported operation");
        return multiplication<scalar>{+left, self.val};
    }

    template <xmm_type_for<T> R = scalar>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline mask_type
    operator==(this scalar self, R right) noexcept {
        if constexpr (sized_floating_point<T, 4>) {
            return mask_type{_mm_cmpeq_ss(self.val, +right)};
        } else {
            static_assert(sized_floating_point<T, 8>, "unsupported operation");
            return mask_type{_mm_cmpeq_sd(self.val, +right)};
        }
    }

    template <xmm_type_for<T> R = scalar>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline mask_type operator<(
        this scalar self, R right) noexcept {
        if constexpr (sized_floating_point<T, 4>) {
            return mask_type{_mm_cmplt_ss(self.val, +right)};
        } else {
            static_assert(sized_floating_point<T, 8>, "unsupported operation");
            return mask_type{_mm_cmplt_sd(self.val, +right)};
        }
    }

    template <xmm_type_for<T> R = scalar>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline mask_type
    operator<=(this scalar self, R right) noexcept {
        if constexpr (sized_floating_point<T, 4>) {
            return mask_type{_mm_cmple_ss(self.val, +right)};
        } else {
            static_assert(sized_floating_point<T, 8>, "unsupported operation");
            return mask_type{_mm_cmple_sd(self.val, +right)};
        }
    }

    template <xmm_type_for<T> R = scalar>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline mask_type operator>(
        this scalar self, R right) noexcept {
        if constexpr (sized_floating_point<T, 4>) {
            return mask_type{_mm_cmpgt_ss(self.val, +right)};
        } else {
            static_assert(sized_floating_point<T, 8>, "unsupported operation");
            return mask_type{_mm_cmpgt_sd(self.val, +right)};
        }
    }

    template <xmm_type_for<T> R = scalar>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline mask_type
    operator>=(this scalar self, R right) noexcept {
        if constexpr (sized_floating_point<T, 4>) {
            return mask_type{_mm_cmpge_ss(self.val, +right)};
        } else {
            static_assert(sized_floating_point<T, 8>, "unsupported operation");
            return mask_type{_mm_cmpge_sd(self.val, +right)};
        }
    }

    template <xmm_type_for<T> L>
    requires (!std::same_as<L, scalar>)
    DPL_ATTRIBUTES(
        _HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) friend inline mask_type
    operator==(L left, scalar self) noexcept {
        if constexpr (sized_floating_point<T, 4>) {
            return mask_type{_mm_cmpeq_ss(+left, self.val)};
        } else {
            static_assert(sized_floating_point<T, 8>, "unsupported operation");
            return mask_type{_mm_cmpeq_sd(+left, self.val)};
        }
    }

    template <xmm_type_for<T> L>
    requires (!std::same_as<L, scalar>)
    DPL_ATTRIBUTES(
        _HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) friend inline mask_type
    operator<(L left, scalar self) noexcept {
        if constexpr (sized_floating_point<T, 4>) {
            return mask_type{_mm_cmplt_ss(+left, self.val)};
        } else {
            static_assert(sized_floating_point<T, 8>, "unsupported operation");
            return mask_type{_mm_cmplt_sd(+left, self.val)};
        }
    }

    template <xmm_type_for<T> L>
    requires (!std::same_as<L, scalar>)
    DPL_ATTRIBUTES(
        _HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) friend inline mask_type
    operator<=(L left, scalar self) noexcept {
        if constexpr (sized_floating_point<T, 4>) {
            return mask_type{_mm_cmple_ss(+left, self.val)};
        } else {
            static_assert(sized_floating_point<T, 8>, "unsupported operation");
            return mask_type{_mm_cmple_sd(+left, self.val)};
        }
    }

    template <xmm_type_for<T> L>
    requires (!std::same_as<L, scalar>)
    DPL_ATTRIBUTES(
        _HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) friend inline mask_type
    operator>(L left, scalar self) noexcept {
        if constexpr (sized_floating_point<T, 4>) {
            return mask_type{_mm_cmpgt_ss(+left, self.val)};
        } else {
            static_assert(sized_floating_point<T, 8>, "unsupported operation");
            return mask_type{_mm_cmpgt_sd(+left, self.val)};
        }
    }

    template <xmm_type_for<T> L>
    requires (!std::same_as<L, scalar>)
    DPL_ATTRIBUTES(
        _HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) friend inline mask_type
    operator>=(L left, scalar self) noexcept {
        if constexpr (sized_floating_point<T, 4>) {
            return mask_type{_mm_cmpge_ss(+left, self.val)};
        } else {
            static_assert(sized_floating_point<T, 8>, "unsupported operation");
            return mask_type{_mm_cmpge_sd(+left, self.val)};
        }
    }

#if DPL_SIMD_X86_AVX512FP16 & DPL_SUPPORTS_FLOAT16
    template <xmm_type_for<T> R = scalar>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline scalar operator+(
        this scalar self, R right) noexcept
    requires std::same_as<float16, T>
    {
        return _mm_add_sh(self.val, +right);
    }

    template <xmm_type_for<T> L = scalar>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) friend inline scalar
    operator+(L left, scalar self) noexcept
    requires std::same_as<float16, T>
    {
        return _mm_add_sh(+left, self.val);
    }

    template <xmm_type_for<T> R = scalar>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline scalar operator-(
        this scalar self, R right) noexcept
    requires std::same_as<float16, T>
    {
        return _mm_sub_sh(self.val, +right);
    }

    template <xmm_type_for<T> L = scalar>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) friend inline scalar
    operator-(L left, scalar self) noexcept
    requires std::same_as<float16, T>
    {
        return _mm_sub_sh(+left, self.val);
    }

    template <xmm_type_for<T> R = scalar>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline scalar operator/(
        this scalar self, R right) noexcept
    requires std::same_as<float16, T>
    {
        return _mm_div_sh(self.val, +right);
    }

    template <xmm_type_for<T> L = scalar>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) friend inline scalar
    operator/(L left, scalar self) noexcept
    requires std::same_as<float16, T>
    {
        return _mm_div_sh(+left, self.val);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD) explicit inline operator T(
        this scalar self) noexcept
    requires std::same_as<float16, T>
    {
        T result;
        // cast to deal with float16 and bfloat16 which are distinct from float
        // and double
        _mm_store_sh(&result, self.val);
        return result;
    }

    template <xmm_type_for<T> R = scalar>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline mask_type
    operator==(this scalar self, R right) noexcept
    requires std::same_as<float16, T>
    {
        // TODO make cmp imm enum
        // TODO allow configurable quiet/signaling and ordered/unordered
        return mask_type{_mm_cmp_sh_mask(self.val, +right, 0)};
    }

    template <xmm_type_for<T> R = scalar>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline mask_type
    operator!=(this scalar self, R right) noexcept
    requires std::same_as<float16, T>
    {
        // TODO make cmp imm enum
        // TODO allow configurable quiet/signaling and ordered/unordered
        return mask_type{_mm_cmp_sh_mask(self.val, +right, 12)};
    }

    template <xmm_type_for<T> R = scalar>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline mask_type operator<(
        this scalar self, R right) noexcept
    requires std::same_as<float16, T>
    {
        // TODO allow configurable quiet/signaling and ordered/unordered
        return mask_type{_mm_cmp_sh_mask(self.val, +right, 17)};
    }

    template <xmm_type_for<T> R = scalar>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline mask_type
    operator<=(this scalar self, R right) noexcept
    requires std::same_as<float16, T>
    {
        return mask_type{_mm_cmp_sh_mask(self.val, +right, 18)};
    }

    template <xmm_type_for<T> R = scalar>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline mask_type operator>(
        this scalar self, R right) noexcept
    requires std::same_as<float16, T>
    {
        return mask_type{_mm_cmp_sh_mask(self.val, +right, 30)};
    }

    template <xmm_type_for<T> R = scalar>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline mask_type
    operator>=(this scalar self, R right) noexcept
    requires std::same_as<float16, T>
    {
        return mask_type{_mm_cmp_sh_mask(self.val, +right, 29)};
    }

    template <xmm_type_for<T> L>
    requires (!std::same_as<L, scalar>)
    DPL_ATTRIBUTES(
        _HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) friend inline mask_type
    operator==(L left, scalar self) noexcept
    requires std::same_as<float16, T>
    {
        return mask_type{_mm_cmp_sh_mask(+left, self.val, 0)};
    }

    template <xmm_type_for<T> L>
    requires (!std::same_as<L, scalar>)
    DPL_ATTRIBUTES(
        _HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) friend inline mask_type
    operator!=(L left, scalar self) noexcept
    requires std::same_as<float16, T>
    {
        return mask_type{_mm_cmp_sh_mask(+left, self.val, 12)};
    }

    template <xmm_type_for<T> L>
    requires (!std::same_as<L, scalar>)
    DPL_ATTRIBUTES(
        _HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) friend inline mask_type
    operator<(L left, scalar self) noexcept
    requires std::same_as<float16, T>
    {
        return mask_type{_mm_cmp_sh_mask(+left, self.val, 17)};
    }

    template <xmm_type_for<T> L>
    requires (!std::same_as<L, scalar>)
    DPL_ATTRIBUTES(
        _HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) friend inline mask_type
    operator<=(L left, scalar self) noexcept
    requires std::same_as<float16, T>
    {
        return mask_type{_mm_cmp_sh_mask(+left, self.val, 18)};
    }

    template <xmm_type_for<T> L>
    requires (!std::same_as<L, scalar>)
    DPL_ATTRIBUTES(
        _HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) friend inline mask_type
    operator>(L left, scalar self) noexcept
    requires std::same_as<float16, T>
    {
        return mask_type{_mm_cmp_sh_mask(+left, self.val, 30)};
    }

    template <xmm_type_for<T> L>
    requires (!std::same_as<L, scalar>)
    DPL_ATTRIBUTES(
        _HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) friend inline mask_type
    operator>=(L left, scalar self) noexcept
    requires std::same_as<float16, T>
    {
        return mask_type{_mm_cmp_sh_mask(+left, self.val, 29)};
    }
#endif

#if DPL_SIMD_X86_AVX512BFP16 & DPL_SUPPORTS_BFLOAT16
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD) explicit inline operator T(
        this scalar self) noexcept
    requires std::same_as<bfloat16, T>
    {
        T result;
        // cast to deal with float16 and bfloat16 which are distinct from float
        // and double
        _mm_store_sh((float16*)&result, self.val);
        return result;
    }
#endif

private:
    native_type val;
};

template <xmm_scalar_element T>
explicit scalar(T) -> scalar<T>;
template <xmm_scalar_element T>
explicit scalar(vector<T>) -> scalar<T>;
template <xmm_scalar_element T>
explicit scalar(vector_mask<T>) -> scalar<T>;
template <xmm_scalar_element T>
explicit scalar(scalar_mask<T>) -> scalar<T>;

template <xmm_scalar_element To, reinterpretable_as<To> From>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline scalar<To>
reinterpret_as(scalar<From> vec) noexcept {
    return scalar<To>{native_cast<To, From>(+vec)};
}

template <xmm_scalar_element T>
class multiplication<scalar<T>> : public binary_scalar_expression {
public:
    static_assert(std::floating_point<T>);
    using element_type = T;
    using native_type = native_type_t<T>;

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL) inline multiplication(
        native_type left, native_type right) noexcept
        : left{left}
        , right{right} {}

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) explicit inline
    operator native_type(this multiplication self) noexcept {
        if constexpr (sized_floating_point<T, 4>) {
            return _mm_mul_ss(self.left, self.right);
        } else {
            static_assert(sized_floating_point<T, 8>, "unsupported operation");
            return _mm_mul_sd(self.left, self.right);
        }
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline scalar<T> operator+(
        this multiplication self, scalar<T> right) noexcept {
        if constexpr (sized_floating_point<T, 4>) {
            return _mm_fmadd_ss(self.left, self.right, right);
        } else {
            static_assert(sized_floating_point<T, 8>, "unsupported operation");
            return _mm_fmadd_sd(self.left, self.right, right);
        }
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline scalar<T> operator-(
        this multiplication self, scalar<T> right) noexcept {
        if constexpr (sized_floating_point<T, 4>) {
            return _mm_fmsub_ss(self.left, self.right, right);
        } else {
            static_assert(sized_floating_point<T, 8>, "unsupported operation");
            return _mm_fmsub_sd(self.left, self.right, right);
        }
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) friend inline scalar<T>
    operator+(scalar<T> left, multiplication mul) noexcept {
        return mul + left;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) friend inline scalar<T>
    operator-(scalar<T> left, multiplication mul) noexcept {
        if constexpr (sized_floating_point<T, 4>) {
            return _mm_fnmadd_ss(mul.left, mul.right, left);
        } else {
            static_assert(sized_floating_point<T, 8>, "unsupported operation");
            return _mm_fnmadd_sd(mul.left, mul.right, left);
        }
    }

#if DPL_SIMD_X86_AVX512FP16 & DPL_SUPPORTS_FLOAT16
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline scalar<T> operator+(
        this multiplication self, scalar<T> right) noexcept
    requires std::same_as<float16, T>
    {
        return _mm_fmadd_sh(self.left, self.right, right);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline scalar<T> operator-(
        this multiplication self, scalar<T> right) noexcept
    requires std::same_as<float16, T>
    {
        return _mm_fmsub_sh(self.left, self.right, right);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) friend inline scalar<T>
    operator-(scalar<T> left, multiplication mul) noexcept
    requires std::same_as<float16, T>
    {
        return _mm_fnmadd_sh(mul.left, mul.right, left);
    }
#endif

private:
    native_type left, right;
};

DPL_EXPORT_END

DPL_SIMD_ABI_NAMESPACE_END(xmm)

DPL_NAMESPACE_END