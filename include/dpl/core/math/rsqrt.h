// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/details/accuracy.h"
#include "dpl/core/math/details/ldexp.h"
#include "dpl/core/math/details/rsqrt2.h"
#include "dpl/core/math/fixup.h"
#include "dpl/core/math/frexp.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/extended.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_vector.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/maskable/transform.h"
#  include "dpl/core/dispatch/operation/math.h"
#  include "dpl/core/operations/arithmetic.h" // IWYU pragma: keep
#  include "dpl/core/operations/bitwise.h"    // IWYU pragma: keep
#  include "dpl/core/operations/compare.h"    // IWYU pragma: keep
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void rsqrt(...) noexcept = delete;

struct DPL_EMPTY_BASES rsqrt_t :
    public math_operation_base<rsqrt_t>,
    public maskable_transform_base<rsqrt_t> {
    using math_operation_base<rsqrt_t>::operator();
    using maskable_transform_base<rsqrt_t>::operator();
};

template <>
struct operation_signature<rsqrt_t> {
    static consteval void operator()(simd_vector auto&&) noexcept {}
};

template <typename S, typename M, typename T>
concept unqualified_canonical_mrsqrt = cpo_invocable<rsqrt_t, T> &&
    (!simd_type<S> || same_as<S, cpo_result_t<rsqrt_t, T>>) &&
    requires(S src, M mask, T val) {
        {
            rsqrt(internal::abi<T>, src, mask, val)
        } -> same_as<cpo_result_t<rsqrt_t, T>>;
    };

template <>
struct canonical_impl<rsqrt_t> {
    template <canonical_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val) noexcept
    requires requires { rsqrt(internal::abi<T>, val); }
    {
        return rsqrt(internal::abi<T>, val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mrsqrt<T, simd_mask_type_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, simd_mask_type_t<T> mask, T val) noexcept {
        return rsqrt(internal::abi<T>, src, mask, val);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mrsqrt<T, launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, M cmask, T val) noexcept {
        return rsqrt(internal::abi<T>, src, dx::to_const_mask<T>(cmask), val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mrsqrt<dx::zero_t, simd_mask_type_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, simd_mask_type_t<T> mask, T val) noexcept {
        return rsqrt(internal::abi<T>, zero, mask, val);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mrsqrt<dx::zero_t, launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(dx::zero_t zero, M cmask, T val) noexcept {
        return rsqrt(internal::abi<T>, zero, dx::to_const_mask<T>(cmask), val);
    }
};

template <typename T, typename A = simd_abi_type_t<T>>
concept unqualified_extended_rsqrt = requires {
    { rsqrt(internal::declarg<T>()) } -> vector_with_common_abi<A>;
};

template <typename S, typename M, typename T>
concept unqualified_extended_mrsqrt = cpo_invocable<rsqrt_t, T> &&
    (!simd_type<S> || equivalent_vector_with<S, cpo_result_t<rsqrt_t, T>>) &&
    requires {
        {
            rsqrt(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>())
        } -> equivalent_vector_with<cpo_result_t<rsqrt_t, T>>;
    };

template <>
struct extended_impl<rsqrt_t> {
public:
    template <extended_vector T>
    requires unqualified_extended_rsqrt<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val) {
        return rsqrt(__DPL forward<T>(val));
    }

    template <simd_vector S, exact_mask_for<S> M, equivalent_vector_with<S> T>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mrsqrt<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, T&& val) {
        return rsqrt( __DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val));
    }

    template <simd_vector S, const_mask_for<S> M, equivalent_vector_with<S> T>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_mrsqrt<S, launder_cmask_t<S, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M cmask, T&& val) {
        return rsqrt( __DPL forward<S>(src), dx::to_const_mask<S>(cmask),
            __DPL forward<T>(val));
    }

    template <simd_vector T, result_mask_for<rsqrt_t, T> M>
    requires (extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mrsqrt<dx::zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M&& mask, T&& val) {
        return rsqrt(zero, __DPL forward<M>(mask), __DPL forward<T>(val));
    }

    template <simd_vector T, result_cmask_for<rsqrt_t, T> M>
    requires extended_vector<T> &&
        unqualified_extended_mrsqrt<dx::zero_t,
            launder_cmask_t<cpo_result_t<rsqrt_t, T>, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M cmask, T&& val) {
        return rsqrt(zero, dx::to_const_mask<cpo_result_t<rsqrt_t, T>>(cmask),
            __DPL forward<T>(val));
    }
};

template <>
struct fallback_impl<rsqrt_t> {
public:
    // TODO: float16/bfloat16
    template <canonical_vector T>
    requires same_as<float, simd_element_type_t<T>> ||
        same_as<double, simd_element_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(T val) noexcept {
        constexpr auto rsqrt_v = mx::rsqrt2(dx::one);
        auto const inv_sqrt2 = dx::broadcast<T>(rsqrt_v);

        using vexp_t = mx::exponent_vector_t<T>;
        vexp_t exp;
        auto const fr = dx::frexp(val, exp);
        auto const remtwo = exp & dx::one;
        auto const reduced = mx::rsqrt2(mx::accuracy::speed, fr);
        auto const rhs =
            dx::negate(dx::bwshift_right(dx::subtract(exp, dx::one), imm<1>));
        auto result = mx::ldexp(mx::compliance::unsafe, reduced, rhs);
        result = dx::multiply(
            result, dx::cmpeq(remtwo, dx::zero), result, inv_sqrt2);
        constexpr auto fixflags = fpfix::condition<fpfix::nan, fpfix::copy> //
            | fpfix::condition<fpfix::infinite, dx::zero>                   //
            | fpfix::condition<fpfix::negative, dx::nan>;
        if constexpr (dx::is_simd_canonical_invocable<T, T, decltype(fixflags)>(
                          dx::fixup)) {
            return dx::fixup(result, val, fixflags);
        } else {
            result = dx::select(
                dx::logical_or(dx::isnan(val), dx::cmplt(val, dx::zero)),
                dx::all_bits, result);
            result = dx::select(dx::isinf(val), dx::zero, result);
            return result;
        }
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::rsqrt_t rsqrt{};
}
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
