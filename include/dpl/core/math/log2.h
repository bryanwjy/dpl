// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/fixup.h"
#include "dpl/core/math/frexp.h"
#include "dpl/core/math/internal/pair.h"
#include "dpl/core/math/internal/polynomial.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/extended.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_vector.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/maskable/transform.h"
#  include "dpl/core/dispatch/operation/math.h"
#  include "dpl/core/immediate/constants/infinity.h"
#  include "dpl/core/immediate/constants/nan.h"
#  include "dpl/core/operations/arithmetic.h" // IWYU pragma: keep
#  include "dpl/core/operations/bitwise.h"    // IWYU pragma: keep
#  include "dpl/core/operations/compare.h"    // IWYU pragma: keep
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void log2(...) noexcept = delete;

struct DPL_EMPTY_BASES log2_t :
    private math_operation_base<log2_t>,
    private maskable_transform_base<log2_t> {
    using math_operation_base<log2_t>::operator();
    using maskable_transform_base<log2_t>::operator();
};

template <>
struct operation_signature<log2_t> {
    static consteval void operator()(simd_vector auto&&) noexcept {}
};

template <typename S, typename M, typename T>
concept unqualified_canonical_mlog2 =
    (!simd_type<S> || same_as<S, cpo_result_t<log2_t, T>>) &&
    requires(S src, M mask, T val) {
        {
            log2(internal::abi<T>, src, mask, val)
        } -> same_as<cpo_result_t<log2_t, T>>;
    };

template <>
struct canonical_impl<log2_t> {
private:
    template <typename T>
    using mask_t DPL_NODEBUG =
        basic_mask<simd_element_type_t<T>, simd_abi_type_t<T>>;

    template <typename T>
    using imask_t DPL_NODEBUG = mask_value_t<simd_abi_traits<T>::size>;

    template <typename T, imask_t<T> M>
    using cmask_t DPL_NODEBUG = const_mask<simd_abi_traits<T>::size, M>;

public:
    template <simd_abi A, simd_element_for<A> E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val) noexcept
    requires requires { log2(internal::abi<A>, val); }
    {
        return log2(internal::abi<A>, val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mlog2<type_identity_t<T>, mask_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, mask_t<T> mask, T val) noexcept {
        return log2(internal::abi<T>, src, mask, val);
    }

    template <fixed_width_vector T, imask_t<T> M>
    requires canonical_vector<T> &&
        unqualified_canonical_mlog2<type_identity_t<T>, cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, cmask_t<T, M> cmask, T val) noexcept {
        return log2(internal::abi<T>, src, cmask, val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mlog2<dx::zero_t, mask_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, mask_t<T> mask, T val) noexcept {
        return log2(internal::abi<T>, zero, mask, val);
    }

    template <fixed_width_vector T, imask_t<T> M>
    requires canonical_vector<T> &&
        unqualified_canonical_mlog2<dx::zero_t, cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, cmask_t<T, M> cmask, T val) noexcept {
        return log2(internal::abi<T>, zero, cmask, val);
    }
};

template <typename T, typename A = simd_abi_type_t<T>>
concept unqualified_extended_log2 = requires {
    { log2(internal::declarg<T>()) } -> vector_with_common_abi<A>;
};

template <typename S, typename M, typename T>
concept unqualified_extended_mlog2 =
    (!simd_type<S> || equivalent_vector_with<S, cpo_result_t<log2_t, T>>) &&
    requires {
        {
            log2(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>())
        } -> equivalent_vector_with<cpo_result_t<log2_t, T>>;
    };

template <>
struct extended_impl<log2_t> {
private:
    template <typename T>
    using imask_t DPL_NODEBUG = mask_value_t<simd_abi_traits<T>::size>;

    template <typename T, imask_t<T> M>
    using cmask_t DPL_NODEBUG = const_mask<simd_abi_traits<T>::size, M>;

    template <typename T>
    using mask_t DPL_NODEBUG =
        basic_mask<simd_element_type_t<T>, simd_abi_type_t<T>>;

public:
    template <extended_vector T>
    requires unqualified_extended_log2<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val) {
        return log2(__DPL forward<T>(val));
    }

    template <simd_vector S, common_vector_with<S> T,
        equivalent_mask_with<mask_t<S>> M>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mlog2<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, T&& val) {
        return log2( __DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val));
    }

    template <fixed_width_vector S, imask_t<S> M, common_vector_with<S> T>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_mlog2<S, cmask_t<S, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, cmask_t<S, M> cmask, T&& val) {
        return log2( __DPL forward<S>(src), cmask, __DPL forward<T>(val));
    }

    template <simd_vector T, common_mask_with<mask_t<T>> M>
    requires (extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mlog2<dx::zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M&& mask, T&& val) {
        return log2(zero, __DPL forward<M>(mask), __DPL forward<T>(val));
    }

    template <fixed_width_vector T, imask_t<T> M>
    requires extended_vector<T> &&
        unqualified_extended_mlog2<dx::zero_t, cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, cmask_t<T, M> cmask, T&& val) {
        return log2(zero, cmask, __DPL forward<T>(val));
    }
};

template <>
struct fallback_impl<log2_t> {
private:
    template <typename E>
    static constexpr auto float16_like =
        brain_float<E> || (digits_v<E> == 12 && sizeof(E) == 2);

public:
    template <simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        basic_vector<float, A> val) noexcept {
        using simdf = basic_vector<float, A>;
        using pairf = fmath::pair<float, A>;
        auto const decomp =
            dx::frexp(val, frexp_reduced | frexp_floating_point);
        auto const one = fmath::single(dx::broadcast<A, float>(1.0f));
        pairf const x = (decomp.fr - one) / (one + decomp.fr);
        auto const x2 = x.upper * x.upper;
        constexpr fmath::polynomial<0.9618012905120f, //
            0.5764790177e+0f,                         //
            0.4374550283e+0f>
            polynomial;
        auto const t = polynomial(x2);
        auto const inv_halfln2 = fmath::make_pair<float, A>(
            2.8853900432586669922f, 3.2734474483568488616e-08f);
        auto s = decomp.exp + x * inv_halfln2;
        s = s + x2 * x * t;

        return dx::fixup(val, s.upper + s.lower,
            fpfix::condition<fpfix::negative, dx::nan> |
                fpfix::condition<fpfix::zero, -dx::infinity> |
                fpfix::condition<fpfix::infinity, fpfix::revert> |
                fpfix::condition<fpfix::nan, fpfix::revert>);
    }

    template <simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        basic_vector<double, A> val) noexcept {
        auto const decomp =
            dx::frexp(val, frexp_reduced | frexp_floating_point);
        auto const one = fmath::single(dx::broadcast<A, double>(1));
        auto const x = (decomp.fr - one) / (one + decomp.fr);
        auto const x2 = x.upper * x.upper;
        constexpr fmath::polynomial<0.96179669392608091449,
            0.5770780162997058982, 0.4121985945485324709, 0.3205977477944495502,
            0.2623708057488514656, 0.2200768693152277689, 0.2211941750456081490>
            polynomial;
        auto const t = polynomial(x2);
        auto const inv_halfln2 = fmath::make_pair<double, A>(
            2.885390081777926774, 6.0561604995516736434e-18);

        auto s = decomp.exp + x * inv_halfln2;
        s = s + x2 * x * t;

        return dx::fixup(val, s.upper + s.lower,
            fpfix::condition<fpfix::negative, dx::nan> |
                fpfix::condition<fpfix::zero, -dx::infinity> |
                fpfix::condition<fpfix::infinity, fpfix::revert> |
                fpfix::condition<fpfix::nan, fpfix::revert>);
    }

    /*
    template <canonical_vector T>
    requires float16_like<simd_element_type_t<T>> &&
        convertible_to<float, simd_element_type_t<T>> &&
        cpo_invocable<round_t, T, decltype(rounding_opt)>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(T val) noexcept {
        using simdf = basic_vector<E, A>;
        auto const decomp =
            dx::frexp(val, frexp_reduced | frexp_floating_point);
        auto const one = fmath::single(dx::broadcast<A, E>(1));
        auto const x = (decomp.fr - one) / (one + decomp.fr);
        auto const x2 = x.upper * x.upper;
        constexpr fmath::polynomial<0.9618012905120f, //
            0.5764790177e+0f,                         //
            0.4374550283e+0f>
            polynomial;
        auto const t = polynomial(x2);
        auto const inv_halfln2 = fmath::make_pair<E, A>(
            2.8853900432586669922f, 3.2734474483568488616e-08f);
        auto s = decomp.exp + x * inv_halfln2;
        s = s + x2 * x * t;
        auto result = s.upper + s.lower;

        return dx::fixup(val, result,
            fpfix::condition<fpfix::negative, dx::nan> |
                fpfix::condition<fpfix::zero, -dx::infinity> |
                fpfix::condition<fpfix::infinity, fpfix::revert> |
                fpfix::condition<fpfix::nan, fpfix::revert>);
    }
    */
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::log2_t log2{};
}
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
