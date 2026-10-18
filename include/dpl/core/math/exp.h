// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/details/constants.h"
#include "dpl/core/math/details/ldexp.h"
#include "dpl/core/math/details/pair.h"
#include "dpl/core/math/details/polynomial.h"
#include "dpl/core/math/isnan.h"
#include "dpl/core/math/mulx.h"
#include "dpl/core/math/roundeven.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/extended.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_vector.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/maskable/transform.h"
#  include "dpl/core/dispatch/operation/math.h"
#  include "dpl/core/immediate/constants/ln2.h"
#  include "dpl/core/numbers/ext.h"
#  include "dpl/core/operations/arithmetic.h"
#  include "dpl/core/operations/bitwise.h"
#  include "dpl/core/operations/cast.h"
#  include "dpl/core/operations/compare.h"
#  include "dpl/core/operations/logical.h"
#  include "dpl/core/operations/select.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void exp(...) noexcept = delete;

struct DPL_EMPTY_BASES exp_t :
    public math_operation_base<exp_t>,
    public maskable_transform_base<exp_t> {
    using math_operation_base<exp_t>::operator();
    using maskable_transform_base<exp_t>::operator();
};

template <>
struct operation_signature<exp_t> {
    static consteval void operator()(simd_vector auto&&) noexcept {}
};

template <typename S, typename M, typename T>
concept unqualified_canonical_mexp = cpo_invocable<exp_t, T> &&
    (!simd_type<S> || same_as<S, cpo_result_t<exp_t, T>>) &&
    requires(S src, M mask, T val) {
        {
            exp(internal::abi<T>, src, mask, val)
        } -> same_as<cpo_result_t<exp_t, T>>;
    };

template <>
struct canonical_impl<exp_t> {
public:
    template <canonical_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val) noexcept
    requires requires { exp(internal::abi<T>, val); }
    {
        return exp(internal::abi<T>, val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mexp<T, simd_mask_type_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, simd_mask_type_t<T> mask, T val) noexcept {
        return exp(internal::abi<T>, src, mask, val);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mexp<T, launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, M cmask, T val) noexcept {
        return exp(internal::abi<T>, src, dx::to_const_mask<T>(cmask), val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mexp<dx::zero_t, simd_mask_type_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, simd_mask_type_t<T> mask, T val) noexcept {
        return exp(internal::abi<T>, zero, mask, val);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mexp<dx::zero_t, launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(dx::zero_t zero, M cmask, T val) noexcept {
        return exp(internal::abi<T>, zero, dx::to_const_mask<T>(cmask), val);
    }
};

template <typename T, typename A = simd_abi_type_t<T>>
concept unqualified_extended_exp = requires {
    { exp(internal::declarg<T>()) } -> vector_with_common_abi<A>;
};

template <typename S, typename M, typename T>
concept unqualified_extended_mexp = cpo_invocable<exp_t, T> &&
    (!simd_type<S> || equivalent_vector_with<S, cpo_result_t<exp_t, T>>) &&
    requires {
        {
            exp(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>())
        } -> equivalent_vector_with<cpo_result_t<exp_t, T>>;
    };

template <>
struct extended_impl<exp_t> {
public:
    template <extended_vector T>
    requires unqualified_extended_exp<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val) {
        return exp(__DPL forward<T>(val));
    }

    template <simd_vector S, exact_mask_for<S> M, equivalent_vector_with<S> T>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mexp<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, T&& val) {
        return exp( __DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val));
    }

    template <simd_vector S, const_mask_for<S> M, equivalent_vector_with<S> T>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_mexp<S, launder_cmask_t<S, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M cmask, T&& val) {
        return exp( __DPL forward<S>(src), dx::to_const_mask<S>(cmask),
            __DPL forward<T>(val));
    }

    template <simd_vector T, result_mask_for<exp_t, T> M>
    requires (extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mexp<dx::zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M&& mask, T&& val) {
        return exp(zero, __DPL forward<M>(mask), __DPL forward<T>(val));
    }

    template <simd_vector T, result_cmask_for<exp_t, T> M>
    requires extended_vector<T> &&
        unqualified_extended_mexp<dx::zero_t,
            launder_cmask_t<cpo_result_t<exp_t, T>, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M cmask, T&& val) {
        return exp(zero, dx::to_const_mask<cpo_result_t<exp_t, T>>(cmask),
            __DPL forward<T>(val));
    }
};

template <>
struct fallback_impl<exp_t> {
private:
    template <canonical_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL solve_poly(T val) noexcept {
        using E = simd_element_type_t<T>;
        if constexpr (is_same_v<E, double>) {
            constexpr fmath::polynomial<
                0x1.0000000000001p-1,  // 0.5000000000000001
                0x1.5555555555558p-3,  // 0.16666666666666674
                0x1.5555555553d67p-5,  // 0.04166666666662416
                0x1.111111110f809p-7,  // 0.008333333333322217
                0x1.6c16c1788a1f1p-10, // 0.0013888888917198384
                0x1.a01a01b0094c6p-13, // 0.0001984126988655908
                0x1.a019b913fb328p-16, // 2.4801521320161297e-05
                0x1.71ddf6bc1f18bp-19, // 2.7557242373006254e-06
                0x1.28917d2ea7404p-22, // 2.7620076795982907e-07
                0x1.af631b5be3131p-26  // 2.511003614718777e-08
                >
                poly;
            return poly(val);
        } else if constexpr (is_same_v<E, float>) {
            constexpr fmath::polynomial<0x1p-1f, // 0.5
                0x1.5554dcp-3f,                  // 0.16666576
                0x1.5554b8p-5f,                  // 0.041666374
                0x1.120b66p-7f,                  // 0.008363175
                0x1.6d753ap-10f                  // 0.0013941113
                >
                poly;
            return poly(val);
        } else {
            static_assert(is_same_v<E, ext::float16>);
            using ext::literals::operator""_f16;
            constexpr fmath::polynomial<0x1p-1_f16, // 0.5
                0x1.56cp-3_f16,                     // 0.1674
                0x1.57p-5_f16                       // 0.04187
                >
                poly;
            return poly(val);
        }
    }

    template <floating_point_like E>
    static constexpr E ln2upper = [] {
        if constexpr (is_same_v<E, double>) {
            return 0x1.62e42fefa38p-1;
        } else if constexpr (is_same_v<E, float>) {
            return 0x1.62e4p-1f;
        } else {
            static_assert(is_same_v<E, ext::float16>);
            using ext::literals::operator""_f16;
            return 0x1.6p-1_f16;
        }
    }();

    template <floating_point_like E>
    static constexpr E ln2lower = [] {
        if constexpr (is_same_v<E, double>) {
            return 0x1.ef35793c7673p-45;
        } else if constexpr (is_same_v<E, float>) {
            return 0x1.7f7d1cp-20f;
        } else {
            static_assert(is_same_v<E, ext::float16>);
            using ext::literals::operator""_f16;
            return 0x1.72p-8_f16;
        }
    }();

    template <floating_point_like E>
    static constexpr auto upper_bound = [] {
        if constexpr (is_same_v<E, double>) {
            return 0x1.62e42fefa39efp+9;
        } else if constexpr (is_same_v<E, float>) {
            return 0x1.62e42ep+6f;
        } else if constexpr (is_same_v<E, ext::float16>) {
            using ext::literals::operator""_f16;
            return 0x1.62cp+3_f16;
        } else {
            static_assert(is_same_v<E, ext::bfloat16>);
            return 0x1.64p+6f;
        }
    }();
    template <floating_point_like E>
    static constexpr auto lower_bound = [] {
        if constexpr (is_same_v<E, double>) {
            return -0x1.74910d52d3051p+9;
        } else if constexpr (is_same_v<E, float>) {
            return -0x1.9fe368p+6f;
        } else if constexpr (is_same_v<E, ext::float16>) {
            using ext::literals::operator""_f16;
            return -0x1.154p+4_f16;
        } else {
            static_assert(is_same_v<E, ext::bfloat16>);
            return -0x1.74p+6f;
        }
    }();

    template <canonical_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL exp(T val) noexcept {
        using E = simd_element_type_t<T>;
        using sint_t = signed_representation_t<E>;
        auto const qf =
            dx::roundeven(dx::multiply(val, mx::inv_ln2), rounding::no_exc);
        auto const q = dx::element_cast<sint_t>(qf);
        auto const s =
            dx::nmulsub(qf, ln2lower<E>, dx::nmulsub(qf, ln2upper<E>, val));
        // x2 * f + x + 1
        auto u = dx::muladd(dx::multiply(s, s), solve_poly(s), s);
        u = dx::add(u, dx::one);
        u = fmath::ldexp(fmath::compliance::speed, u, q);
        u = dx::select(dx::cmpge(val, upper_bound<E>), dx::infinity, u);
        return dx::select(dx::cmple(val, lower_bound<E>), dx::zero, u);
    }

    template <canonical_vector T>
    requires same_as<simd_element_type_t<T>, ext::bfloat16>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL exp(T val) noexcept {
        auto const shift = [] {
            if constexpr (fixed_width_abi<simd_abi_type_t<T>>) {
                return dx::imm<dx::simd_abi_traits<T>::size() / 2>;
            } else {
                return dx::simd_abi_traits<T>::size() / 2;
            }
        }();

        using E = simd_element_type_t<T>;
        auto const lo = dx::element_cast<float>(val);
        auto const hi = dx::element_cast<float>(dx::shift_left(val, shift));
        constexpr auto exp = [](dx::rebind_simd_t<T, float> fp32) {
            using vf32_t = dx::rebind_simd_t<T, float>;
            using sint_t = signed_representation_t<float>;

            // 0.50390625, 0.16796875
            constexpr fmath::polynomial<0x1.02p-1f, 0x1.58p-3f> poly;
            auto const qf = dx::roundeven(
                dx::multiply(fp32, mx::inv_ln2), rounding::no_exc);
            auto const q = dx::element_cast<sint_t>(qf);
            auto const fr = dx::nmulsub(qf, dx::ln2, fp32);
            // x2 * f + x + 1
            auto u = dx::muladd(dx::multiply(fr, fr), solve_poly(fr), fr);
            u = dx::add(u, dx::one);
            u = fmath::ldexp(fmath::compliance::speed, u, q);
            u = dx::select(dx::cmpge(fp32, upper_bound<E>), dx::infinity, u);
            return dx::select(dx::cmple(fp32, lower_bound<E>), dx::zero, u);
        };

        auto const r_lo = dx::element_cast<E>(exp(lo));
        auto const r_hi = dx::element_cast<E>(exp(hi));
        return dx::slide_left(dx::rotate_right(r_lo, shift), r_hi, shift);
    }

public:
    template <canonical_vector T>
    requires same_as<simd_element_type_t<T>, double> ||
        same_as<simd_element_type_t<T>, float> ||
        same_as<simd_element_type_t<T>, ext::float16> ||
        same_as<simd_element_type_t<T>, ext::bfloat16>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val) noexcept {
        if consteval {
            using E = simd_element_type_t<T>;
            auto const isnan = dx::isnan(val);
            // prevent NAN arithmetics
            auto safe = dx::select(isnan, dx::zero, val);
            // prevent OOB cast
            safe = dx::min(safe, upper_bound<E>);
            safe = dx::max(safe, lower_bound<E>);
            return dx::select(isnan, val, fallback_impl::exp(safe));
        } else {
            return fallback_impl::exp(val);
        }
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::exp_t exp{};
}
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
