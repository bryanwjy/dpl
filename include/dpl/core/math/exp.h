// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/fma.h"
#include "dpl/core/math/internal/constants.h"
#include "dpl/core/math/internal/ldexp.h"
#include "dpl/core/math/internal/pair.h"
#include "dpl/core/math/internal/polynomial.h"
#include "dpl/core/math/round.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/extended.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_vector.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/maskable/transform.h"
#  include "dpl/core/dispatch/operation/math.h"
#  include "dpl/core/operations/arithmetic.h" // IWYU pragma: keep
#  include "dpl/core/operations/bitwise.h"    // IWYU pragma: keep
#  include "dpl/core/operations/cast.h"
#  include "dpl/core/operations/compare.h" // IWYU pragma: keep
#  include "dpl/core/operations/logical.h"
#  include "dpl/core/operations/select.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void exp(...) noexcept = delete;

struct DPL_EMPTY_BASES exp_t :
    private math_operation_base<exp_t>,
    private maskable_transform_base<exp_t> {
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
private:
    template <typename T>
    using mask_t DPL_NODEBUG =
        basic_mask<simd_element_type_t<T>, simd_abi_type_t<T>>;

public:
    template <simd_abi A, simd_element_for<A> E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val) noexcept
    requires requires { exp(internal::abi<A>, val); }
    {
        return exp(internal::abi<A>, val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mexp<type_identity_t<T>, mask_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, mask_t<T> mask, T val) noexcept {
        return exp(internal::abi<T>, src, mask, val);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mexp<type_identity_t<T>,
        launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, M cmask, T val) noexcept {
        return exp(internal::abi<T>, src, dx::to_const_mask<T>(cmask), val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mexp<dx::zero_t, mask_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, mask_t<T> mask, T val) noexcept {
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

    template <simd_vector S, exact_mask_for<S> M, common_vector_with<S> T>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mexp<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, T&& val) {
        return exp( __DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val));
    }

    template <simd_vector S, const_mask_for<S> M, common_vector_with<S> T>
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
    static constexpr auto rounding_opt =
        rounding::no_exc | rounding::to_nearest_int;
    template <typename E>
    static constexpr auto float16_like =
        brain_float<E> || (digits_v<E> == 12 && sizeof(E) == 2);

public:
    template <simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        basic_vector<float, A> val) noexcept {
        using sint = signed_representation_t<float>;
        auto const qf = dx::round(val * fmath::inv_ln2, rounding_opt);
        auto const q = dx::element_cast<sint>(qf);
        using fpair = fmath::pair<float, A>;
        constexpr auto ln2 = fmath::ln2_v<fpair>;
        auto const s =
            dx::fnmadd(qf, ln2.lower, dx::fnmadd(qf, ln2.upper, val));
        constexpr fmath::polynomial<0.5f,
            0.166666671633720397949219f,   //
            0.0416664853692054748535156f,  //
            0.00833336077630519866943359f, //
            0.00139304355252534151077271f, //
            0.000198527617612853646278381f>
            polynomial;
        // x2 * f + x + 1
        auto u = dx::fmadd(dx::multiply(s, s), polynomial(s), s) + dx::one;
        u = fmath::ldexp(fmath::compliance::speed, u, q);
        u = dx::select(val > 100.0f, dx::infinity, u);
        // underflow
        return dx::select(val < -103.97208f, dx::zero, u);
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
            dx::fnmadd(qf, ln2.lower, dx::fnmadd(qf, ln2.upper, val));
        constexpr fmath::polynomial<0.5, 0.1666666666666669072e+0,
            0.4166666666666602598e-1, 0.8333333333314938210e-2,
            0.1388888888914497797e-2, 0.1984126989855865850e-3,
            0.2480158687479686264e-4, 0.2755723402025388239e-5,
            0.2755762628169491192e-6, 0.2511210703042288022e-7,
            0.2081276378237164457e-8>
            polynomial;
        auto u = dx::fmadd(dx::multiply(s, s), polynomial(s), s) + dx::one;
        u = fmath::ldexp(fmath::compliance::speed, u, q);
        constexpr auto max_log = 0x1.62e42fefa39efp+9;
        u = dx::select(val > max_log, dx::infinity, u);
        // underflow
        return dx::select(val < -745.133, dx::zero, u);
    }

    /*
    template <canonical_vector T>
    requires float16_like<simd_element_type_t<T>> &&
        convertible_to<float, simd_element_type_t<T>> &&
        cpo_invocable<round_t, T, decltype(rounding_opt)>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(T val) noexcept {
        using E = simd_element_type_t<T>;
        using sint = signed_representation_t<E>;
        auto const qf = dx::round(val * fmath::inv_ln2, rounding_opt);
        auto const q = dx::element_cast<sint>(qf);
        using fpair = fmath::pair<E, A>;
        constexpr auto ln2 = fmath::ln2_v<fpair>;
        auto const s =
            dx::fnmadd(qf, ln2.lower, dx::fnmadd(qf, ln2.upper, val));
        constexpr fmath::polynomial<0.5f,
            0.166666671633720397949219f,   //
            0.0416664853692054748535156f,  //
            0.00833336077630519866943359f, //
            0.00139304355252534151077271f>
            polynomial;
        auto u = dx::fmadd(dx::multiply(s, s), polynomial(s), s) + dx::one;
        u = fmath::ldexp(fmath::compliance::speed, u, q);
        if constexpr (brain_float<E>) {
            u = dx::select(val > 100.0f, dx::infinity, u);
            // underflow
            return dx::select(val < -92.186785f, dx::zero, u);
        } else {
            static_assert(digits_v<E> == 12 && sizeof(E) == 2);
            constexpr E max_ln = 11.089866f;
            constexpr E min_ln = -16.63553f;
            u = dx::select(val > max_ln, dx::infinity, u);
            // underflow
            return dx::select(val < min_ln, dx::zero, u);
        }
    }
    */
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::exp_t exp{};
}
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
