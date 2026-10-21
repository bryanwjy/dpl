// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/details/apply_widened.h"
#include "dpl/core/math/details/constants.h"
#include "dpl/core/math/details/ldexp.h"
#include "dpl/core/math/details/pair.h"
#include "dpl/core/math/details/polynomial.h"
#include "dpl/core/math/isnan.h"
#include "dpl/core/math/mulx.h"
#include "dpl/core/math/roundeven.h"

#if !DPL_MODULES
#  include "dpl/core/algorithm/rotate_.h"
#  include "dpl/core/algorithm/slide.h"
#  include "dpl/core/concepts/canonical.h"
#  include "dpl/core/concepts/extended.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_vector.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/maskable/transform.h"
#  include "dpl/core/dispatch/operation/math.h"
#  include "dpl/core/numbers/ext.h"
#  include "dpl/core/operations/arithmetic.h"
#  include "dpl/core/operations/cast.h"
#  include "dpl/core/operations/compare.h"
#  include "dpl/core/operations/logical.h"
#  include "dpl/core/operations/select.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void exp2(...) noexcept = delete;

struct DPL_EMPTY_BASES exp2_t :
    public math_operation_base<exp2_t>,
    public maskable_transform_base<exp2_t> {
    using math_operation_base<exp2_t>::operator();
    using maskable_transform_base<exp2_t>::operator();
};

template <>
struct operation_signature<exp2_t> {
    static consteval void operator()(simd_vector auto&&) noexcept {}
};

template <typename S, typename M, typename T>
concept unqualified_canonical_mexp2 = cpo_invocable<exp2_t, T> &&
    (!simd_type<S> || same_as<S, cpo_result_t<exp2_t, T>>) &&
    requires(S src, M mask, T val) {
        {
            exp2(internal::abi<T>, src, mask, val)
        } -> same_as<cpo_result_t<exp2_t, T>>;
    };

template <>
struct canonical_impl<exp2_t> {
    template <canonical_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val) noexcept
    requires requires { exp2(internal::abi<T>, val); }
    {
        return exp2(internal::abi<T>, val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mexp2<T, simd_mask_type_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, simd_mask_type_t<T> mask, T val) noexcept {
        return exp2(internal::abi<T>, src, mask, val);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mexp2<T, launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, M cmask, T val) noexcept {
        return exp2(internal::abi<T>, src, dx::to_const_mask<T>(cmask), val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mexp2<dx::zero_t, simd_mask_type_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, simd_mask_type_t<T> mask, T val) noexcept {
        return exp2(internal::abi<T>, zero, mask, val);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mexp2<dx::zero_t, launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(dx::zero_t zero, M cmask, T val) noexcept {
        return exp2(internal::abi<T>, zero, dx::to_const_mask<T>(cmask), val);
    }
};

template <typename T, typename A = simd_abi_type_t<T>>
concept unqualified_extended_exp2 = requires {
    { exp2(internal::declarg<T>()) } -> vector_with_common_abi<A>;
};

template <typename S, typename M, typename T>
concept unqualified_extended_mexp2 = cpo_invocable<exp2_t, T> &&
    (!simd_type<S> || equivalent_vector_with<S, cpo_result_t<exp2_t, T>>) &&
    requires {
        {
            exp2(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>())
        } -> equivalent_vector_with<cpo_result_t<exp2_t, T>>;
    };

template <>
struct extended_impl<exp2_t> {
public:
    template <extended_vector T>
    requires unqualified_extended_exp2<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val) {
        return exp2(__DPL forward<T>(val));
    }

    template <simd_vector S, exact_mask_for<S> M, equivalent_vector_with<S> T>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mexp2<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, T&& val) {
        return exp2( __DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val));
    }

    template <simd_vector S, const_mask_for<S> M, equivalent_vector_with<S> T>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_mexp2<S, launder_cmask_t<S, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M cmask, T&& val) {
        return exp2( __DPL forward<S>(src), dx::to_const_mask<S>(cmask),
            __DPL forward<T>(val));
    }

    template <simd_vector T, result_mask_for<exp2_t, T> M>
    requires (extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mexp2<dx::zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M&& mask, T&& val) {
        return exp2(zero, __DPL forward<M>(mask), __DPL forward<T>(val));
    }

    template <extended_vector T, result_cmask_for<exp2_t, T> M>
    requires unqualified_extended_mexp2<dx::zero_t,
        launder_cmask_t<cpo_result_t<exp2_t, T>, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M cmask, T&& val) {
        return exp2(zero, dx::to_const_mask<cpo_result_t<exp2_t, T>>(cmask),
            __DPL forward<T>(val));
    }
};

struct pow_t;

template <>
struct fallback_impl<exp2_t> {
private:
    friend fallback_impl<pow_t>;

    template <canonical_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL solve_poly(T val) noexcept {
        using E = simd_element_type_t<T>;
        if constexpr (is_same_v<E, double>) {
            constexpr fmath::polynomial<
                0x1.62e42fefa39efp-1,  // 0.6931471805599453
                0x1.ebfbdff82c598p-3,  // 0.24022650695910097
                0x1.c6b08d704a0cbp-5,  // 0.05550410866482166
                0x1.3b2ab6fba1dd9p-7,  // 0.009618129107606887
                0x1.5d87fe78a2cbdp-10, // 0.0013333558146395995
                0x1.43091309700dap-13, // 0.00015403530441738291
                0x1.ffcbfc7441c97p-17, // 1.5252733853289477e-05
                0x1.62bfd476e9a95p-20, // 1.3215442586728665e-06
                0x1.b524dcb878ef4p-24, // 1.0178051724308302e-07
                0x1.e606406677be9p-28, // 7.072586161230777e-09
                0x1.ea0447380274bp-32  // 4.456676030826705e-10
                >
                poly;
            return poly(val);
        } else if constexpr (is_same_v<E, float>) {
            constexpr fmath::polynomial<0x1.62e43p-1f, // 0.6931472
                0x1.ebfbep-3f,                         // 0.2402265
                0x1.c6af6cp-5f,                        // 0.05550357
                0x1.3b29e4p-7f,                        // 0.009618031
                0x1.5f0896p-10f,                       // 0.0013390867
                0x1.446c82p-13f                        // 0.00015469735
                >
                poly;
            return poly(val);
        } else {
            static_assert(is_same_v<E, ext::float16>);
            using ext::literals::operator""_f16;
            constexpr fmath::polynomial<0x1.63p-1_f16, // 0.6934
                0x1.ecp-3_f16,                         // 0.2402
                0x1.c98p-5_f16,                        // 0.05585
                0x1.3dp-7_f16                          // 0.009674
                >
                poly;
            return poly(val);
        }
    }

    template <floating_point_like E>
    static constexpr E upper_bound =
        floating_point_traits<E>::exponent_bias + 1;

    template <floating_point_like E>
    static constexpr E lower_bound =
        1 - floating_point_traits<E>::exponent_bias -
        static_cast<int>(floating_point_traits<E>::digits);

    template <canonical_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL exp2(T val) noexcept {
        using E = simd_element_type_t<T>;

        auto const qf = dx::roundeven(val, rounding::no_exc);
        auto const fr = dx::subtract(val, qf);
        auto u = dx::muladd(solve_poly(fr), fr, dx::one);
        u = fmath::ldexp(fmath::compliance::speed, u, qf);

        // Use ge to ensure NANs propogates
        u = dx::select(dx::cmpge(val, upper_bound<E>), dx::infinity, u);
        return dx::select(dx::cmple(val, lower_bound<E>), dx::zero, u);
    }

    template <canonical_vector T>
    requires same_as<simd_element_type_t<T>, ext::bfloat16>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL exp2(T val) noexcept {
        constexpr auto exp2f = [](dx::rebind_simd_t<T, float> fp32) {
            using vf32_t = dx::rebind_simd_t<T, float>;

            using ext::literals::operator""_bf16;
            constexpr fmath::polynomial<0x1.62p-1f, // 0.69140625
                0x1.fp-3f,                          // 0.2421875
                0x1.cap-5f                          // 0.055908203125
                >
                poly;

            auto const qf = dx::roundeven(fp32, rounding::no_exc);
            auto const fr = dx::subtract(fp32, qf);
            auto u = dx::muladd(poly(fr), fr, dx::one);
            u = fmath::ldexp(fmath::compliance::speed, u, qf);

            // Use ge to ensure NANs propogates
            u = dx::select(
                dx::cmpge(fp32, upper_bound<ext::bfloat16>), dx::infinity, u);
            return dx::select(
                dx::cmpge(fp32, lower_bound<ext::bfloat16>), u, dx::zero);
        };

        return fmath::apply_widened<float>(exp2f, val);
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
            return dx::select(isnan, val, fallback_impl::exp2(safe));
        } else {
            return fallback_impl::exp2(val);
        }
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::exp2_t exp2{};
}
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
