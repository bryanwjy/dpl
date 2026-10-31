// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/details/apply_widened.h"
#include "dpl/core/math/frexp.h"
#include "dpl/core/math/isfinite.h"
#include "dpl/core/math/rsqrt.h"

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

template <typename T>
concept unqualified_canonical_sqrt = requires {
    { sqrt(internal::abi<T>, internal::declarg<T>()) } -> same_as<T>;
};

template <typename S, typename M, typename T>
concept unqualified_canonical_msqrt_base =
    cpo_invocable<sqrt_t, T> && requires {
        {
            sqrt(internal::abi<T>, internal::declarg<S>(),
                internal::declarg<M>(), internal::declarg<T>())
        } -> same_as<T>;
    };

template <typename M, typename T>
concept unqualified_canonical_msqrt = unqualified_canonical_msqrt_base<T, M, T>;

template <typename M, typename T>
concept unqualified_canonical_zmsqrt =
    unqualified_canonical_msqrt_base<dx::zero_t, M, T>;

template <>
struct canonical_impl<sqrt_t> {
    template <canonical_vector T>
    requires unqualified_canonical_sqrt<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val) noexcept {
        return sqrt(internal::abi<T>, val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_msqrt<simd_mask_type_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, simd_mask_type_t<T> mask, T val) noexcept {
        return sqrt(internal::abi<T>, src, mask, val);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_msqrt<launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, M mask, T val) noexcept {
        return sqrt(internal::abi<T>, src, dx::to_const_mask<T>(mask), val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_zmsqrt<simd_mask_type_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, simd_mask_type_t<T> mask, T val) noexcept {
        return sqrt(internal::abi<T>, zero, mask, val);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_zmsqrt<launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(dx::zero_t zero, M mask, T val) noexcept {
        return sqrt(internal::abi<T>, zero, dx::to_const_mask<T>(mask), val);
    }
};

template <typename T>
concept unqualified_extended_sqrt = requires {
    { sqrt(internal::declarg<T>()) } -> equivalent_vector_with<T>;
};

template <typename S, typename M, typename T>
concept unqualified_extended_msqrt_base = cpo_invocable<sqrt_t, T> && requires {
    {
        sqrt(internal::declarg<S>(), internal::declarg<M>(),
            internal::declarg<T>())
    } -> equivalent_vector_with<T>;
};

template <typename S, typename M, typename T>
concept unqualified_extended_msqrt =
    equivalent_vector_with<T, S> && unqualified_extended_msqrt_base<S, M, T>;

template <typename M, typename T>
concept unqualified_extended_zmsqrt =
    unqualified_extended_msqrt_base<dx::zero_t, M, T>;

template <>
struct extended_impl<sqrt_t> {
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
    static constexpr auto operator()(S&& src, M mask, T&& val) {
        return sqrt( __DPL forward<S>(src), dx::to_const_mask<S>(mask),
            __DPL forward<T>(val));
    }

    template <simd_vector T, exact_mask_for<T> M>
    requires (extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_zmsqrt<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M&& mask, T&& val) {
        return sqrt(zero, __DPL forward<M>(mask), __DPL forward<T>(val));
    }

    template <extended_vector T, const_mask_for<T> M>
    requires unqualified_extended_zmsqrt<launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M mask, T&& val) {
        return sqrt(zero, dx::to_const_mask<T>(mask), __DPL forward<T>(val));
    }
};

template <>
struct fallback_impl<sqrt_t> {
    template <canonical_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL rsqrt(T val) noexcept {
        using E = simd_element_type_t<T>;
        static_assert(!is_same_v<E, ext::bfloat16>);

        if not consteval {
            using O = accuracy::approximate_t;
            if constexpr (dx::is_simd_canonical_invocable<T, O>(dx::rsqrt)) {
                return dx::rsqrt(val, accuracy::approximate);
            }
        }

        if constexpr (is_same_v<E, float> || is_same_v<E, double>) {
            // Don't need to deal with large/small/special args
            return fallback_impl<rsqrt_t>::normal_rsqrt(
                val, accuracy::approximate);
        } else {
            using vf32_t = rebind_simd_t<T, float>;
            return mx::apply_widened<float>(
                [](vf32_t val) {
                    return fallback_impl<rsqrt_t>::normal_rsqrt(
                        val, accuracy::approximate);
                },
                val);
        }
    }

    template <canonical_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL sqrt_impl(T val) noexcept {
        using E = simd_element_type_t<T>;
        using rsqrt_impl = fallback_impl<rsqrt_t>;
        constexpr auto lower_bound = [] {
            auto const digits = floating_point_traits<E>::digits;
            auto const shift =
                __DPL countr_zero(floating_point_traits<E>::exponent_mask);
            return __DPL bit_cast<E>(bit_representation_t<E>(digits) << shift);
        }();

        auto const issmall = dx::cmplt(val, lower_bound);
        val = dx::multiply(val, issmall, val, rsqrt_impl::prenormalizer<E>);
        constexpr E half = 0.5;
        auto g = fallback_impl::rsqrt(val);
        auto y = dx::multiply(val, g);
        auto result = [&] {
            // TODO: implement constexpr fmadd
            if not consteval {
                if constexpr (dx::is_simd_canonical_invocable<T, T, T>(
                                  dx::fmadd)) {
                    return dx::nmulsub(y, y, val);
                }
            }

            auto result = mx::single(y) * y;
            result = val - mx::pair_ref{result};
            return mx::recombine(result);
        }();
        result = dx::muladd(result, dx::multiply(half, g), y);
        constexpr auto normalizer =
            static_cast<E>(1.0 / rsqrt_impl::normalizer<E>);
        return dx::multiply(result, issmall, result, normalizer);
    }

    template <canonical_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL sqrt(T val) noexcept {
        using E = simd_element_type_t<T>;
        using rsqrt_impl = fallback_impl<rsqrt_t>;
        // TODO: implement constexpr fmadd
        if not consteval {
            if constexpr (dx::is_simd_canonical_invocable<T, T, T>(dx::fmadd)) {
                return sqrt_impl(val);
            }
        }

        constexpr auto preshrink =
            static_cast<E>(1.0 / rsqrt_impl::prenormalizer<E>);
        constexpr auto upper_bound = [] {
            auto const bias = floating_point_traits<E>::exponent_bias;
            auto const shift =
                __DPL countr_zero(floating_point_traits<E>::exponent_mask);
            return __DPL bit_cast<E>(
                bit_representation_t<E>(bias * 2 - 1) << shift);
        }();

        auto const islarge = dx::cmpge(val, upper_bound);
        val = dx::multiply(val, islarge, val, preshrink);
        auto result = sqrt_impl(val);
        result =
            dx::multiply(result, islarge, result, rsqrt_impl::normalizer<E>);
        return result;
    }

    template <canonical_vector T>
    requires same_as<simd_element_type_t<T>, ext::bfloat16>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL sqrt(T val) noexcept {
        using vf32_t = rebind_simd_t<T, float>;
        return mx::apply_widened<float>(
            [](vf32_t val) { return fallback_impl::sqrt(val); }, val);
    }

public:
    template <canonical_vector T>
    requires same_as<simd_element_type_t<T>, double> ||
        same_as<simd_element_type_t<T>, float> ||
        same_as<simd_element_type_t<T>, ext::float16> ||
        same_as<simd_element_type_t<T>, ext::bfloat16>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(T val) noexcept {
#if DPL_HAS_CONSTEXPR_BUILTIN(__builtin_sqrt)
        if consteval {
            if constexpr (fixed_width_abi<simd_abi_type_t<T>>) {
                return internal::transform<T>(
                    [](E val) {
                        if constexpr (is_same_v<E, double>) {
                            return __builtin_sqrt(val);
                        } else {
                            return __builtin_sqrtf(static_cast<float>(val));
                        }
                    },
                    val);
            }
        }
#endif

        auto const vzero = dx::broadcast<T>(dx::zero);
        auto result = [&] {
            if consteval {
                return fallback_impl::sqrt(dx::select(
                    dx::logical_and(dx::isfinite(val), dx::cmpgt(val, vzero)),
                    val, dx::one));
            } else {
                return fallback_impl::sqrt(val);
            }
        }();

        constexpr auto fixflags = fpfix::condition<fpfix::negative, dx::nan> |
            fpfix::condition<fpfix::neg_inf, dx::nan> |
            fpfix::condition<fpfix::zero, fpfix::copy> |
            fpfix::condition<fpfix::pos_inf, fpfix::copy> |
            fpfix::condition<fpfix::nan, fpfix::copy> |
            fpfix::condition<fpfix::one, fpfix::copy>;
        if constexpr (dx::is_simd_canonical_invocable<T, T, decltype(fixflags)>(
                          dx::fixup)) {
            return dx::fixup(result, val, fixflags);
        } else {
            result =
                dx::select(dx::cmpeq(val, dx::infinity), dx::infinity, result);
            result = dx::select(dx::cmpgt(val, vzero), result, dx::all_bits);
            result = dx::select(
                dx::logical_or(dx::cmpeq(val, vzero), dx::cmpeq(val, dx::one)),
                val, result);
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
