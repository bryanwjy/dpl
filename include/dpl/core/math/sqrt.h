// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/details/accuracy.h"
#include "dpl/core/math/details/apply_widened.h"
#include "dpl/core/math/details/ldexp.h"
#include "dpl/core/math/details/rsqrt2.h"
#include "dpl/core/math/frexp.h"
#include "dpl/core/math/isfinite.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/concepts/extended.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_vector.h"
#  include "dpl/core/operations/arithmetic.h"
#  include "dpl/core/operations/bitwise.h" // IWYU pragma: keep
#  include "dpl/core/operations/compare.h"
#  include "dpl/core/operations/select.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void sqrt(...) noexcept = delete;

struct DPL_EMPTY_BASES sqrt_t :
    public math_operation_base<sqrt_t>,
    public maskable_transform_base<sqrt_t> {
    using math_operation_base<sqrt_t>::operator();
    using maskable_transform_base<sqrt_t>::operator();
};

template <>
struct operation_signature<sqrt_t> {
    static consteval void operator()(simd_vector auto&&) noexcept {}
};

template <typename S, typename M, typename T>
concept unqualified_canonical_msqrt = cpo_invocable<sqrt_t, T> &&
    (!simd_type<S> || same_as<S, cpo_result_t<sqrt_t, T>>) &&
    requires(S src, M mask, T val) {
        {
            sqrt(internal::abi<T>, src, mask, val)
        } -> same_as<cpo_result_t<sqrt_t, T>>;
    };

template <>
struct canonical_impl<sqrt_t> {
    template <canonical_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val) noexcept
    requires requires { sqrt(internal::abi<T>, val); }
    {
        return sqrt(internal::abi<T>, val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_msqrt<T, simd_mask_type_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, simd_mask_type_t<T> mask, T val) noexcept {
        return sqrt(internal::abi<T>, src, mask, val);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_msqrt<T, launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, M cmask, T val) noexcept {
        return sqrt(internal::abi<T>, src, dx::to_const_mask<T>(cmask), val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_msqrt<dx::zero_t, simd_mask_type_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, simd_mask_type_t<T> mask, T val) noexcept {
        return sqrt(internal::abi<T>, zero, mask, val);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_msqrt<dx::zero_t, launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(dx::zero_t zero, M cmask, T val) noexcept {
        return sqrt(internal::abi<T>, zero, dx::to_const_mask<T>(cmask), val);
    }
};

template <typename T, typename A = simd_abi_type_t<T>>
concept unqualified_extended_sqrt = requires {
    { sqrt(internal::declarg<T>()) } -> vector_with_common_abi<A>;
};

template <typename S, typename M, typename T>
concept unqualified_extended_msqrt = cpo_invocable<sqrt_t, T> &&
    (!simd_type<S> || equivalent_vector_with<S, cpo_result_t<sqrt_t, T>>) &&
    requires {
        {
            sqrt(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>())
        } -> equivalent_vector_with<cpo_result_t<sqrt_t, T>>;
    };

template <>
struct extended_impl<sqrt_t> {
public:
    template <extended_vector T>
    requires unqualified_extended_sqrt<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val) {
        return sqrt(__DPL forward<T>(val));
    }

    template <simd_vector S, exact_mask_for<S> M, equivalent_vector_with<S> T>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_msqrt<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, T&& val) {
        return sqrt( __DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val));
    }

    template <simd_vector S, const_mask_for<S> M, equivalent_vector_with<S> T>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_msqrt<S, launder_cmask_t<S, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M cmask, T&& val) {
        return sqrt( __DPL forward<S>(src), dx::to_const_mask<S>(cmask),
            __DPL forward<T>(val));
    }

    template <simd_vector T, result_mask_for<sqrt_t, T> M>
    requires (extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_msqrt<dx::zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M&& mask, T&& val) {
        return sqrt(zero, __DPL forward<M>(mask), __DPL forward<T>(val));
    }

    template <simd_vector T, result_cmask_for<sqrt_t, T> M>
    requires extended_vector<T> &&
        unqualified_extended_msqrt<dx::zero_t,
            launder_cmask_t<cpo_result_t<sqrt_t, T>, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M cmask, T&& val) {
        return sqrt(zero, dx::to_const_mask<cpo_result_t<sqrt_t, T>>(cmask),
            __DPL forward<T>(val));
    }
};

template <>
struct fallback_impl<sqrt_t> {
    template <canonical_vector T>
    requires floating_point_like<simd_element_type_t<T>> &&
        same_as<simd_element_type_t<T>, ext::bfloat16>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL sqrt(T val) noexcept {
        // should we do this properly?
        return mx::apply_widened<float>(fallback_impl{}, val);
    }

    template <canonical_vector T>
    requires floating_point_like<simd_element_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL sqrt(T val) noexcept {
        using vexp_t = mx::exponent_vector_t<T>;
        using E = simd_element_type_t<T>;
        auto const vsqrt8 = dx::broadcast<T>(2.8284271247461900976033774484);
        auto const vtwo = dx::broadcast<T>(2.0);
        auto const vone = dx::broadcast<vexp_t>(dx::one);

        auto exp = dx::undefined<vexp_t>();
        auto const fr = dx::frexp(val, exp);
        auto const isodd = dx::cmpeq(dx::bwand(exp, vone), dx::one);

        auto const k =
            dx::bwshift_right(dx::subtract(exp, isodd, exp, vone), imm<1>);

        auto const f = dx::add(fr, isodd, fr, fr);
        auto const vrsqrt2 = dx::broadcast<T>(mx::rsqrt2(dx::one));
        auto const vhalf = dx::broadcast<T>(0.5);
        auto const scale = dx::select(isodd, vhalf, vrsqrt2);
        auto const rsqrt2fr = mx::rsqrt2(mx::accuracy::maximum, fr);

        auto result = [&] {
            // TODO: implement constexpr fmadd. Once we add support for 'real'
            // fmadd at compile-time if not consteval won't be needed
            if not consteval {
                if constexpr (dx::is_simd_canonical_invocable<T, T, T>(
                                  dx::fmadd)) {
                    auto const half_rsqrtf = dx::multiply(rsqrt2fr, scale);
                    auto const sqrtf = dx::multiply(half_rsqrtf, dx::add(f, f));
                    auto result = dx::nmulsub(sqrtf, sqrtf, f);
                    return dx::muladd(result, half_rsqrtf, sqrtf);
                }
            }

            auto const half_rsqrtf = mx::single{rsqrt2fr} * scale;
            auto const sqrtf = mx::pair_ref{half_rsqrtf} * dx::add(f, f);

            auto result = mx::pair_ref{sqrtf} * sqrtf;
            result = f - mx::pair_ref{result};
            result = mx::pair_ref{result} * half_rsqrtf;
            result = mx::pair_ref{result} + sqrtf;
            return mx::recombine(result);
        }();

        return mx::ldexp(mx::compliance::unsafe, //
            val, dx::isfinite(val), result, k);
    }

public:
    template <canonical_vector T>
    requires floating_point_like<simd_element_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(T val) noexcept {
        auto const vzero = dx::broadcast<T>(dx::zero);
        auto result = [&] {
            if consteval {
                return fallback_impl::sqrt(dx::select(
                    dx::logical_and(dx::isfinite(val), dx::cmpge(val, vzero)),
                    val, dx::zero));
            } else {
                return fallback_impl::sqrt(val);
            }
        }();

        constexpr auto fixflags = fpfix::condition<fpfix::negative, dx::nan> |
            fpfix::condition<fpfix::zero, fpfix::copy> |
            fpfix::condition<fpfix::infinite, fpfix::copy> |
            fpfix::condition<fpfix::nan, fpfix::copy>;
        if constexpr (dx::is_simd_canonical_invocable<T, T, decltype(fixflags)>(
                          dx::fixup)) {
            return dx::fixup(result, val, fixflags);
        } else {
            result = dx::select(dx::isinf(val), dx::infinity, result);
            result = dx::select(dx::cmpgt(val, vzero), result, dx::all_bits);
            result = dx::select(dx::cmpeq(val, vzero), val, result);
            return result;
        }
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::sqrt_t sqrt{};
}
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
