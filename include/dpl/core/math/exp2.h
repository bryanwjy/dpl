// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/details/constants.h"
#include "dpl/core/math/details/ldexp.h"
#include "dpl/core/math/details/pair.h"
#include "dpl/core/math/details/polynomial.h"
#include "dpl/core/math/mulx.h"
#include "dpl/core/math/round.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/extended.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_vector.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/maskable/transform.h"
#  include "dpl/core/dispatch/operation/math.h"
#  include "dpl/core/numbers/ext.h" // IWYU pragma: keep
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
    private math_operation_base<exp2_t>,
    private maskable_transform_base<exp2_t> {
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
private:
    template <typename T>
    using mask_t DPL_NODEBUG =
        basic_mask<simd_element_type_t<T>, simd_abi_type_t<T>>;

public:
    template <simd_abi A, simd_element_for<A> E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val) noexcept
    requires requires { exp2(internal::abi<A>, val); }
    {
        return exp2(internal::abi<A>, val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mexp2<type_identity_t<T>, mask_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, mask_t<T> mask, T val) noexcept {
        return exp2(internal::abi<T>, src, mask, val);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mexp2<type_identity_t<T>,
        launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, M cmask, T val) noexcept {
        return exp2(internal::abi<T>, src, dx::to_const_mask<T>(cmask), val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mexp2<dx::zero_t, mask_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, mask_t<T> mask, T val) noexcept {
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

    template <simd_vector S, exact_mask_for<S> M, common_vector_with<S> T>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mexp2<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, T&& val) {
        return exp2( __DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val));
    }

    template <simd_vector S, const_mask_for<S> M, common_vector_with<S> T>
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

    template <simd_vector T, result_cmask_for<exp2_t, T> M>
    requires extended_vector<T> &&
        unqualified_extended_mexp2<dx::zero_t,
            launder_cmask_t<cpo_result_t<exp2_t, T>, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M cmask, T&& val) {
        return exp2(zero, dx::to_const_mask<cpo_result_t<exp2_t, T>>(cmask),
            __DPL forward<T>(val));
    }
};

template <>
struct fallback_impl<exp2_t> {
private:
    static constexpr auto rounding_opt =
        rounding::no_exc | rounding::to_nearest_int;

public:
    template <simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        basic_vector<float, A> val) noexcept {
        using sint = signed_representation_t<float>;
        auto const qf = dx::round(
            val * fmath::inv_ln2, rounding::no_exc | rounding::to_nearest_int);
        auto const q = dx::element_cast<sint>(qf);
        using fpair = fmath::pair<float, A>;
        constexpr auto ln2 = fmath::ln2_v<fpair>;
        auto const s =
            dx::nmuladd(qf, ln2.lower, dx::nmuladd(qf, ln2.upper, val));
        static constexpr fmath::polynomial<0.6931471825f, //
            0.2402264476f,                                //
            0.5550347269e-1f,                             //
            0.9618384764e-2f,                             //
            0.1339262701e-2f,                             //
            0.1535920892e-3f>
            polynomial;
        auto u = dx::muladd(polynomial(s), s, dx::one);
        u = fmath::ldexp(fmath::compliance::speed, u, q);
        u = dx::select(val >= 128.0f, dx::infinity, u);
        // underflow
        return dx::select(val >= -149.0f, u, dx::zero);
    }

    template <simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        basic_vector<double, A> val) noexcept {
        using sint = signed_representation_t<double>;
        auto const q = dx::element_cast<sint>(val * fmath::inv_ln2);
        auto const qf = dx::element_cast<double>(q);
        using fpair = fmath::pair<double, A>;
        constexpr auto ln2 = fmath::ln2_v<fpair>;
        auto const s =
            dx::nmuladd(qf, ln2.lower, dx::nmuladd(qf, ln2.upper, val));
        static constexpr fmath::polynomial<0.6931471805599452862,
            0.2402265069591012214e+0, 0.5550410866482046596e-1,
            0.9618129107597600536e-2, 0.1333355814670499073e-2,
            0.1540353045101147808e-3, 0.1525273353517584730e-4,
            0.1321543872511327615e-5, 0.1017819260921760451e-6,
            0.7073164598085707425e-8, 0.4434359082926529454e-9>
            polynomial;
        auto u = dx::muladd(polynomial(s), s, dx::one);
        u = fmath::ldexp(fmath::compliance::speed, u, q);
        u = dx::select(val >= 1024.0, dx::infinity, u);
        // underflow
        return dx::select(val >= -1074.0, u, dx::zero);
    }

    /*
    template <canonical_vector T>
    requires (same_as<ext::float16, simd_element_type_t<T>> ||
    same_as<ext::bfloat16, simd_element_type_t<T>>) && cpo_invocable<round_t, T,
    decltype(rounding_opt)> DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(T val) noexcept {
        using sint = signed_representation_t<E>;
        auto const qf = dx::round(
            val * fmath::inv_ln2, rounding::no_exc | rounding::to_nearest_int);
        auto const q = dx::element_cast<sint>(qf);
        using fpair = fmath::pair<E, A>;
        constexpr auto ln2 = fmath::ln2_v<fpair>;
        auto const s =
            dx::nmuladd(qf, ln2.lower, dx::nmuladd(qf, ln2.upper, val));
        static constexpr fmath::polynomial<0.6931471825f, //
            0.2402264476f,                                //
            0.5550347269e-1f,                             //
            0.9618384764e-2f>
            polynomial;
        static constexpr E max_log = 11;
        auto u = dx::muladd(polynomial(s), s, dx::one);
        u = fmath::ldexp(fmath::compliance::speed, u, q);
        if constexpr (same_as<ext::bfloat16>) {
            u = dx::select(val >= 128.0f, dx::infinity, u);
            // underflow
            return dx::select(val >= -133.0, u, dx::zero);
        } else {
            u = dx::select(val >= 16.0, dx::infinity, u);
            // underflow
            return dx::select(val >= -24.0, u, dx::zero);
        }
    }
    */
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::exp2_t exp2{};
}
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
