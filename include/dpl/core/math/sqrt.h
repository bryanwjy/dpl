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
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL rsqrt(T val) noexcept {
        if consteval {
            return fallback_impl<rsqrt_t>::normal_rsqrt(val);
        } else {
            if constexpr (dx::is_simd_canonical_invocable<T>(dx::rsqrt)) {
                // approximate
                return dx::rsqrt(val);
            } else {
                // skips the fixup instructions
                return fallback_impl<rsqrt_t>::normal_rsqrt(val);
            }
        }
    }

    template <canonical_vector T>
    requires same_as<simd_element_type_t<T>, ext::bfloat16>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL sqrt(T val) noexcept {
        using vf32_t = rebind_simd_t<T, float>;
        return mx::apply_widened<float>(
            [](vf32_t val) {
                constexpr auto half = 0.5f;
                auto g = fallback_impl::rsqrt(val);
                auto y = dx::multiply(val, g);
                auto result = dx::nmulsub(y, y, val);
                return dx::muladd(result, dx::multiply(half, g), y);
            },
            val);
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
        auto g = rsqrt(val);
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

        auto const islarge = dx::cmpgt(val, upper_bound);
        val = dx::multiply(val, islarge, val, preshrink);
        auto result = sqrt_impl(val);
        result =
            dx::multiply(result, islarge, result, rsqrt_impl::normalizer<E>);
        return result;
    }

public:
    template <canonical_vector T>
    requires binary_layout_floating_point<simd_element_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(T val) noexcept {
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
