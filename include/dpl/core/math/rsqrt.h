// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/accuracy.h"
#include "dpl/core/math/details/apply_widened.h"
#include "dpl/core/math/fixup.h"

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

/**
 * @brief Computes the reciprocal square root, 1/√x, of each element.
 *
 * Two accuracy levels are available, selected with a trailing tag:
 *
 * @code
 * dx::rsqrt(x);                      // approximate (default)
 * dx::rsqrt(x, accuracy::approximate);
 * dx::rsqrt(x, accuracy::precise);
 * @endcode
 *
 * <b>Approximate</b> (default, @c accuracy::approximate)
 * - Accurate to at least half the precision of the element type
 *   (for example, at least 12 bits for @c float).
 * - Subnormal arguments are handled.
 * - Results that would be subnormal may be flushed to zero, depending on the
 *   target instruction set.
 * - The exact result may differ between platforms within the stated accuracy.
 * - Intended for uses where speed matters more than the last bits of accuracy.
 *
 * <b>Precise</b> (@c accuracy::precise)
 * - More accurate (0.75 ULP) than computing @c 1 / sqrt(x).
 * - May have higher latency than @c 1 /sqrt(x).
 * - Throughput depends on the platform. On targets with a fast hardware
 *   reciprocal square root estimate (accurate to at least half the precision
 *   of the element type) and native fused multiply-add, such as AVX-512,
 *   throughput is typically equal to or better than @c 1 / sqrt(x), making it
 *   well suited to processing large amounts of independent data. In
 *   latency-critical dependency chains, prefer the approximate variant.
 * - On other targets, a generic fallback implementation is used. It favours
 *   correctness and constant evaluation over speed: it is not optimized and
 *   is not intended for performance-sensitive runtime use. If accuracy beyond
 *   @c 1 / sqrt(x) is not required there, use @c 1 / sqrt(x) or the
 *   approximate variant instead with appropriate number of newton steps.
 *
 * @tparam T  A floating-point SIMD vector type.
 *
 * @param x   The input values.
 * @param tag Optional. @c accuracy::approximate (default) or
 *            @c accuracy::precise.
 *
 * @return A vector of the same type as @p x containing 1/√x for each element.
 */
struct DPL_EMPTY_BASES rsqrt_t :
    public math_operation_base<rsqrt_t>,
    public maskable_transform_base<rsqrt_t> {
    using math_operation_base<rsqrt_t>::operator();
    using maskable_transform_base<rsqrt_t>::operator();
};

template <>
struct operation_signature<rsqrt_t> {
    static consteval void operator()(simd_vector auto&&) noexcept {}
    static consteval void operator()(
        simd_vector auto&&, accuracy_tag auto) noexcept {}
};

template <typename T, typename O>
concept unqualified_canonical_rsqrt = accuracy_tag<O> && requires {
    {
        rsqrt(internal::abi<T>, internal::declarg<T>(), internal::declarg<O>())
    } -> same_as<T>;
};

template <typename S, typename M, typename T, typename O>
concept unqualified_canonical_mrsqrt_base =
    cpo_invocable<rsqrt_t, T, O> && requires {
        {
            rsqrt(internal::abi<T>, internal::declarg<S>(),
                internal::declarg<M>(), internal::declarg<T>(),
                internal::declarg<O>())
        } -> same_as<T>;
    };

template <typename M, typename T, typename O>
concept unqualified_canonical_mrsqrt =
    unqualified_canonical_mrsqrt_base<T, M, T, O>;

template <typename M, typename T, typename O>
concept unqualified_canonical_zmrsqrt =
    unqualified_canonical_mrsqrt_base<dx::zero_t, M, T, O>;

template <>
struct canonical_impl<rsqrt_t> {
    template <canonical_vector T, accuracy_tag O = accuracy::approximate_t>
    requires unqualified_canonical_rsqrt<T, O>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        T val, O opt = accuracy::approximate) noexcept {
        return rsqrt(internal::abi<T>, val, opt);
    }

    template <canonical_vector T, accuracy_tag O = accuracy::approximate_t>
    requires unqualified_canonical_mrsqrt<simd_mask_type_t<T>, T, O>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(type_identity_t<T> src,
        simd_mask_type_t<T> mask, T val,
        O opt = accuracy::approximate) noexcept {
        return rsqrt(internal::abi<T>, src, mask, val, opt);
    }

    template <canonical_vector T, const_mask_for<T> M,
        accuracy_tag O = accuracy::approximate_t>
    requires unqualified_canonical_mrsqrt<launder_cmask_t<T, M>, T, O>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(type_identity_t<T> src, M mask, T val,
        O opt = accuracy::approximate) noexcept {
        return rsqrt(
            internal::abi<T>, src, dx::to_const_mask<T>(mask), val, opt);
    }

    template <canonical_vector T, accuracy_tag O = accuracy::approximate_t>
    requires unqualified_canonical_zmrsqrt<simd_mask_type_t<T>, T, O>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(dx::zero_t zero, simd_mask_type_t<T> mask,
        T val, O opt = accuracy::approximate) noexcept {
        return rsqrt(internal::abi<T>, zero, mask, val, opt);
    }

    template <canonical_vector T, const_mask_for<T> M,
        accuracy_tag O = accuracy::approximate_t>
    requires unqualified_canonical_zmrsqrt<launder_cmask_t<T, M>, T, O>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(dx::zero_t zero, M mask, T val,
        O opt = accuracy::approximate) noexcept {
        return rsqrt(
            internal::abi<T>, zero, dx::to_const_mask<T>(mask), val, opt);
    }
};

template <typename T, typename O>
concept unqualified_extended_rsqrt = accuracy_tag<O> && requires {
    {
        rsqrt(internal::declarg<T>(), internal::declarg<O>())
    } -> equivalent_vector_with<T>;
};

template <typename S, typename M, typename T, typename O>
concept unqualified_extended_mrsqrt_base =
    cpo_invocable<rsqrt_t, T, O> && requires {
        {
            rsqrt(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>(), internal::declarg<O>())
        } -> equivalent_vector_with<T>;
    };

template <typename S, typename M, typename T, typename O>
concept unqualified_extended_mrsqrt = equivalent_vector_with<T, S> &&
    unqualified_extended_mrsqrt_base<S, M, T, O>;

template <typename M, typename T, typename O>
concept unqualified_extended_zmrsqrt =
    unqualified_extended_mrsqrt_base<dx::zero_t, M, T, O>;

template <>
struct extended_impl<rsqrt_t> {
    template <extended_vector T, accuracy_tag O = accuracy::approximate_t>
    requires unqualified_extended_rsqrt<T, O>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val, O opt = accuracy::approximate) {
        return rsqrt(__DPL forward<T>(val), opt);
    }

    template <simd_vector S, exact_mask_for<S> M, equivalent_vector_with<S> T,
        accuracy_tag O = accuracy::approximate_t>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mrsqrt<S, M, T, O>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        S&& src, M&& mask, T&& val, O opt = accuracy::approximate) {
        return rsqrt( __DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val), opt);
    }

    template <simd_vector S, const_mask_for<S> M, equivalent_vector_with<S> T,
        accuracy_tag O = accuracy::approximate_t>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_mrsqrt<S, launder_cmask_t<S, M>, T, O>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        S&& src, M mask, T&& val, O opt = accuracy::approximate) {
        return rsqrt( __DPL forward<S>(src), dx::to_const_mask<S>(mask),
            __DPL forward<T>(val), opt);
    }

    template <simd_vector T, exact_mask_for<T> M,
        accuracy_tag O = accuracy::approximate_t>
    requires (extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_zmrsqrt<M, T, O>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, T&& val, O opt = accuracy::approximate) {
        return rsqrt(
            zero, __DPL forward<M>(mask), __DPL forward<T>(val), opt);
    }

    template <extended_vector T, const_mask_for<T> M,
        accuracy_tag O = accuracy::approximate_t>
    requires unqualified_extended_zmrsqrt<launder_cmask_t<T, M>, T, O>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M mask, T&& val, O opt = accuracy::approximate) {
        return rsqrt(
            zero, dx::to_const_mask<T>(mask), __DPL forward<T>(val), opt);
    }
};

struct sqrt_t;

template <>
struct fallback_impl<rsqrt_t> {
private:
    template <canonical_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T fix(T result, T val) noexcept {
        constexpr auto fixflags = fpfix::condition<fpfix::nan, fpfix::copy> //
            | fpfix::condition<fpfix::pos_inf, dx::zero>                    //
            | fpfix::condition<fpfix::neg_inf, dx::nan>                     //
            | fpfix::condition<fpfix::negative, dx::nan>                    //
            | fpfix::condition<fpfix::one, fpfix::copy>;
        if constexpr (dx::is_simd_canonical_invocable<T, T, decltype(fixflags)>(
                          dx::fixup)) {
            return dx::fixup(result, val, fixflags);
        } else {
            result = dx::select(dx::cmpgt(val, dx::zero), result, dx::all_bits);
            result = dx::select(dx::cmpeq(val, dx::zero), dx::infinity, result);
            result = dx::select(dx::cmpeq(val, dx::one), val, result);
            result = dx::select(dx::cmpeq(val, dx::infinity), dx::zero, result);
            return result;
        }
    }

    template <floating_point_like E>
    static constexpr unsigned_representation_t<E> quake_lut = [] {
        if constexpr (is_same_v<E, double>) {
            return 0x5fe6eb50c7b537a9ull;
        } else {
            static_assert(is_same_v<E, float>);
            return 0x5f3759dfu;
        }
    }();

    template <floating_point_like E>
    static constexpr auto prenormalizer = []() {
        if constexpr (floating_point_traits<E>::digits >=
            floating_point_traits<float>::digits) {
            return static_cast<E>(0x1.p64);
        } else {
            return static_cast<E>(0x1.p12);
        }
    }();

    template <floating_point_like E>
    static constexpr auto normalizer = []() {
        if constexpr (floating_point_traits<E>::digits >=
            floating_point_traits<float>::digits) {
            return static_cast<E>(0x1.p32);
        } else {
            return static_cast<E>(0x1.p6);
        }
    }();

    template <canonical_vector T, int N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL refine_precise(
        T y, T x, immediate<N>) noexcept {
        static_assert(N >= 1);
        using E = simd_element_type_t<T>;
        constexpr E half = 0.5;
        if constexpr (N > 1) { // cheap steps
            auto r = dx::nmulsub(dx::multiply(x, y), y, dx::one);
            return refine_precise(
                dx::muladd(dx::multiply(y, half), r, y), x, imm<N - 1>);
        } else { // last step: exact residual
            auto const h = mx::single(x) * y;
            // TODO: implement constexpr fmadd
            if not consteval {
                if constexpr (dx::is_simd_canonical_invocable<T, T, T>(
                                  dx::fmadd)) {
                    auto const [upper, lower] = dx::to_tuple_like(h);
                    auto r = dx::nmulsub(upper, y, dx::one);
                    r = dx::nmulsub(lower, y, r);
                    return dx::muladd(dx::multiply(y, half), r, y);
                }
            }

            auto const p = mx::single(dx::get_element<0>(h)) * y;
            auto r = dx::subtract(dx::subtract(dx::one, dx::get_element<0>(p)),
                dx::get_element<1>(p));
            r = dx::subtract(r, dx::multiply(dx::get_element<1>(h), y));
            return dx::muladd(dx::multiply(y, half), r, y);
        }
    }

    template <canonical_vector T, int N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL refine_approx(
        T y, T x, immediate<N>) noexcept {
        if constexpr (N == 0) {
            return y;
        } else {
            using E = simd_element_type_t<T>;
            constexpr E half = 0.5;
            auto result = dx::multiply(x, y);
            result = dx::nmulsub(result, y, dx::one);
            result = dx::muladd(dx::multiply(y, half), result, y);
            return refine_approx(result, x, imm<N - 1>);
        }
    }

    friend fallback_impl<sqrt_t>;

    static consteval int refine_steps(int target) noexcept {
        auto const lut_accuracy = 4.8f;
        auto const log2threehalves = 0.585f;
        int n = 0;
        for (auto b = lut_accuracy; b < target; b = 2 * b - log2threehalves) {
            ++n;
        }
        return n;
    }

    template <canonical_vector T, accuracy_tag O>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL normal_rsqrt(T val, O opt) noexcept {
        // normal: i.e. does not deal with subnormals
        using E = simd_element_type_t<T>;
        using A = simd_abi_type_t<T>;
        using uint_t = unsigned_representation_t<E>;
        auto const vlut = dx::broadcast<uint_t, A>(quake_lut<E>);
        auto y = dx::reinterpret<E>(dx::subtract(
            vlut, dx::bwshift_right(dx::reinterpret<uint_t>(val), imm<1>)));
        if constexpr (opt == accuracy::approximate) {
            constexpr int steps =
                refine_steps(floating_point_traits<E>::digits / 2);
            return refine_approx(y, val, imm<steps>);
        } else {
            constexpr int steps =
                refine_steps(floating_point_traits<E>::digits);
            return refine_precise(y, val, imm<steps>);
        }
    }

    template <canonical_vector T, accuracy_tag O>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL rsqrt_impl(T val, O opt) noexcept {
        using E = simd_element_type_t<T>;
        auto const issubnormal = dx::cmplt(val, dx::min_value);
        val = dx::multiply(
            val, issubnormal, val, fallback_impl<rsqrt_t>::prenormalizer<E>);
        auto const result = fallback_impl::normal_rsqrt(val, opt);
        return dx::multiply(
            result, issubnormal, result, fallback_impl<rsqrt_t>::normalizer<E>);
    }

    template <canonical_vector T, accuracy_tag O>
    requires same_as<simd_element_type_t<T>, float> ||
        same_as<simd_element_type_t<T>, double>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL rsqrt(T val, O opt) noexcept {
        using E = simd_element_type_t<T>;
        // TODO: implement constexpr fmadd
        if not consteval {
            if constexpr (dx::is_simd_canonical_invocable<T, T, T>(dx::fmadd)) {
                return rsqrt_impl(val, opt);
            }
        }

        constexpr auto upper_bound = [] {
            auto const bias = floating_point_traits<E>::exponent_bias;
            auto const digits = floating_point_traits<E>::digits;
            auto const shift =
                __DPL countr_zero(floating_point_traits<E>::exponent_mask);
            return __DPL bit_cast<E>(
                bit_representation_t<E>(bias * 2 - digits) << shift);
        }();
        constexpr auto shrink =
            static_cast<E>(1.0 / fallback_impl<rsqrt_t>::prenormalizer<E>);
        constexpr auto grow =
            static_cast<E>(1.0 / fallback_impl<rsqrt_t>::normalizer<E>);
        auto const islarge = dx::cmpge(val, upper_bound);
        val = dx::multiply(val, islarge, val, shrink);
        auto result = rsqrt_impl(val, opt);
        result = dx::multiply(result, islarge, result, grow);
        return result;
    }

    template <canonical_vector T, accuracy_tag O>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL rsqrt(T val, O opt) noexcept {
        using E = simd_element_type_t<T>;
        using vf32_t = rebind_simd_t<T, float>;
        return mx::apply_widened<float>(
            [opt](vf32_t val) {
                auto const result = [&] {
                    if not consteval {
                        if constexpr (dx::simd_canonical_invocable<rsqrt_t,
                                          vf32_t, accuracy::approximate_t>) {
                            constexpr rsqrt_t impl;
                            return impl(val, accuracy::approximate);
                        }
                    }

                    if constexpr (is_same_v<E, ext::float16>) {
                        // No need to deal with large/small values
                        return normal_rsqrt(val, accuracy::approximate);
                    } else {
                        return fallback_impl::rsqrt(val, accuracy::approximate);
                    }
                }();

                if constexpr (is_same_v<E, ext::float16> &&
                    opt == accuracy::precise) {
                    // DPL guarantees rsqrt is accurate to 12 bits
                    // but 12 bit accuracy is insufficient for float16
                    // at least 13 bits needed
                    return refine_approx(result, val, imm<1>);
                } else {
                    return result;
                }
            },
            val);
    }

public:
    template <canonical_vector T>
    requires same_as<simd_element_type_t<T>, double> ||
        same_as<simd_element_type_t<T>, float> ||
        same_as<simd_element_type_t<T>, ext::float16> ||
        same_as<simd_element_type_t<T>, ext::bfloat16>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val,
        accuracy::approximate_t approximate = accuracy::approximate) noexcept {
        auto const result = [&] {
            if consteval {
                return fallback_impl::rsqrt(
                    dx::select(dx::logical_and(
                                   dx::isfinite(val), dx::cmpgt(val, dx::zero)),
                        val, dx::one),
                    approximate);
            } else {
                return fallback_impl::rsqrt(val, approximate);
            }
        }();
        return fallback_impl::fix(result, val);
    }

    template <canonical_vector T>
    requires same_as<simd_element_type_t<T>, double> ||
        same_as<simd_element_type_t<T>, float> ||
        same_as<simd_element_type_t<T>, ext::float16> ||
        same_as<simd_element_type_t<T>, ext::bfloat16>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val, accuracy::precise_t precise) noexcept {
        auto const result = [&] {
            if consteval {
                return fallback_impl::rsqrt(
                    dx::select(dx::logical_and(
                                   dx::isfinite(val), dx::cmpgt(val, dx::zero)),
                        val, dx::one),
                    precise);
            } else {
                return fallback_impl::rsqrt(val, precise);
            }
        }();
        return fallback_impl::fix(result, val);
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::rsqrt_t rsqrt{};
}
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
