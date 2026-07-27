// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/details/compliance.h"
#include "dpl/core/math/details/ilogb.h"
#include "dpl/core/math/rounding.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/extended.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_vector.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/maskable/transform.h"
#  include "dpl/core/dispatch/operation/math.h"
#  include "dpl/core/immediate/constants/one.h"
#  include "dpl/core/numbers/binary_layout_floating_point.h"
#  include "dpl/core/operations/arithmetic.h"
#  include "dpl/core/operations/bitwise/bwandnot.h"
#  include "dpl/core/operations/compare.h"
#  include "dpl/core/operations/select.h"
#  include "dpl/core/type_traits/representation.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

struct trunc_t :
    private math_operation_base<trunc_t>,
    private maskable_transform_base<trunc_t> {
    using math_operation_base<trunc_t>::operator();
    using maskable_transform_base<trunc_t>::operator();
};

inline constexpr auto trunc_noexc_v = rounding::to_zero | rounding::no_exc;
inline constexpr auto trunc_exc_v = rounding::to_zero;

template <>
struct operation_signature<trunc_t> {
    static consteval void operator()(simd_vector auto&&) noexcept {}

    static consteval void operator()(
        simd_vector auto&&, rounding::no_exc_t) noexcept {}
};

template <typename S, typename M, typename T>
concept unqualified_canonical_mtrunc = cpo_invocable<trunc_t, T> &&
    (!simd_type<S> || same_as<S, cpo_result_t<trunc_t, T>>) && requires {
        {
            round(internal::abi<cpo_result_t<trunc_t, T>>,
                internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>(), trunc_exc_v)
        } -> same_as<cpo_result_t<trunc_t, T>>;
    };

template <typename S, typename M, typename T>
concept unqualified_canonical_mtruncne =
    cpo_invocable<trunc_t, T, rounding::no_exc_t> &&
    (!simd_type<S> ||
        same_as<S, cpo_result_t<trunc_t, T, rounding::no_exc_t>>) &&
    requires {
        {
            round(internal::abi<cpo_result_t<trunc_t, T, rounding::no_exc_t>>,
                internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>(), trunc_noexc_v)
        } -> same_as<cpo_result_t<trunc_t, T, rounding::no_exc_t>>;
    };

template <>
struct canonical_impl<trunc_t> {
private:
    template <typename T>
    using mask_t DPL_NODEBUG =
        basic_mask<simd_element_type_t<T>, simd_abi_type_t<T>>;

public:
    template <simd_abi A, simd_element_for<A> E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val) noexcept
    requires requires { round(internal::abi<A>, val, trunc_exc_v); }
    {
        return round(internal::abi<A>, val, trunc_exc_v);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mtrunc<type_identity_t<T>, mask_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, mask_t<T> mask, T val) noexcept {
        return round(internal::abi<T>, src, mask, val, trunc_exc_v);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mtrunc<type_identity_t<T>,
        launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, M cmask, T val) noexcept {
        return round(internal::abi<T>, src, dx::to_const_mask<T>(cmask), val,
            trunc_exc_v);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mtrunc<dx::zero_t, mask_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, mask_t<T> mask, T val) noexcept {
        return round(internal::abi<T>, zero, mask, val, trunc_exc_v);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mtrunc<dx::zero_t, launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(dx::zero_t zero, M cmask, T val) noexcept {
        return round(internal::abi<T>, zero, dx::to_const_mask<T>(cmask), val,
            trunc_exc_v);
    }
    ///

    template <simd_abi A, simd_element_for<A> E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val, rounding::no_exc_t) noexcept
    requires requires { round(internal::abi<A>, val, trunc_noexc_v); }
    {
        return round(internal::abi<A>, val, trunc_noexc_v);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mtruncne<type_identity_t<T>, mask_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(type_identity_t<T> src, mask_t<T> mask, T val,
        rounding::no_exc_t) noexcept {
        return round(internal::abi<T>, src, mask, val, trunc_noexc_v);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mtruncne<type_identity_t<T>,
        launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, M cmask, T val, rounding::no_exc_t) noexcept {
        return round(internal::abi<T>, src, dx::to_const_mask<T>(cmask), val,
            trunc_noexc_v);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mtruncne<dx::zero_t, mask_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, mask_t<T> mask, T val, rounding::no_exc_t) noexcept {
        return round(internal::abi<T>, zero, mask, val, trunc_noexc_v);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mtruncne<dx::zero_t, launder_cmask_t<T, M>,
        T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, M cmask, T val, rounding::no_exc_t) noexcept {
        return round(internal::abi<T>, zero, dx::to_const_mask<T>(cmask), val,
            trunc_noexc_v);
    }
};

template <typename T>
concept unqualified_extended_trunc = requires {
    {
        round(internal::declarg<T>(), trunc_exc_v)
    } -> vector_with_common_abi<simd_abi_type_t<T>>;
};

template <typename T>
concept unqualified_extended_truncne = requires {
    {
        round(internal::declarg<T>(), trunc_noexc_v)
    } -> vector_with_common_abi<simd_abi_type_t<T>>;
};

template <typename S, typename M, typename T>
concept unqualified_extended_mtrunc = cpo_invocable<trunc_t, T> &&
    (!simd_type<S> || equivalent_vector_with<S, cpo_result_t<trunc_t, T>>) &&
    requires {
        {
            round(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>(), trunc_exc_v)
        } -> equivalent_vector_with<cpo_result_t<trunc_t, T>>;
    };

template <typename S, typename M, typename T>
concept unqualified_extended_mtruncne =
    cpo_invocable<trunc_t, T, rounding::no_exc_t> &&
    (!simd_type<S> ||
        equivalent_vector_with<S,
            cpo_result_t<trunc_t, T, rounding::no_exc_t>>) &&
    requires {
        {
            round(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>(), trunc_noexc_v)
        }
        -> equivalent_vector_with<cpo_result_t<trunc_t, T, rounding::no_exc_t>>;
    };

template <>
struct extended_impl<trunc_t> {
public:
    template <extended_vector T>
    requires unqualified_extended_trunc<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val) {
        return round(__DPL forward<T>(val), trunc_exc_v);
    }

    template <simd_vector S, exact_mask_for<S> M, common_vector_with<S> T>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mtrunc<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, T&& val) {
        return round( __DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val), trunc_exc_v);
    }

    template <simd_vector S, const_mask_for<S> M, common_vector_with<S> T>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_mtrunc<S, launder_cmask_t<S, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M cmask, T&& val) {
        return round( __DPL forward<S>(src), dx::to_const_mask<S>(cmask),
            __DPL forward<T>(val), trunc_exc_v);
    }

    template <simd_vector T, result_mask_for<trunc_t, T> M>
    requires (extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mtrunc<dx::zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M&& mask, T&& val) {
        return round(
            zero, __DPL forward<M>(mask), __DPL forward<T>(val), trunc_exc_v);
    }

    template <simd_vector T, result_cmask_for<trunc_t, T> M>
    requires extended_vector<T> &&
        unqualified_extended_mtrunc<dx::zero_t,
            launder_cmask_t<cpo_result_t<trunc_t, T>, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M cmask, T&& val) {
        return round(zero, dx::to_const_mask<cpo_result_t<trunc_t, T>>(cmask),
            __DPL forward<T>(val), trunc_exc_v);
    }

    template <extended_vector T>
    requires unqualified_extended_truncne<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val, rounding::no_exc_t) {
        return round( __DPL forward<T>(val), trunc_noexc_v);
    }

    template <simd_vector S, exact_mask_for<S> M, common_vector_with<S> T>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mtruncne<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        S&& src, M&& mask, T&& val, rounding::no_exc_t) {
        return round( __DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val), trunc_noexc_v);
    }

    template <simd_vector S, const_mask_for<S> M, common_vector_with<S> T>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_mtruncne<S, launder_cmask_t<S, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        S&& src, M cmask, T&& val, rounding::no_exc_t) {
        return round( __DPL forward<S>(src), dx::to_const_mask<S>(cmask),
            __DPL forward<T>(val), trunc_noexc_v);
    }

    template <simd_vector T, result_mask_for<trunc_t, T, rounding::no_exc_t> M>
    requires (extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mtruncne<dx::zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, T&& val, rounding::no_exc_t) {
        return round(zero, __DPL forward<M>(mask), __DPL forward<T>(val),
            trunc_noexc_v);
    }

    template <extended_vector T,
        result_cmask_for<trunc_t, T, rounding::no_exc_t> M>
    requires extended_vector<T> &&
        unqualified_extended_mtruncne<dx::zero_t,
            launder_cmask_t<cpo_result_t<trunc_t, T, rounding::no_exc_t>, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M cmask, T&& val, rounding::no_exc_t) {
        return round(zero,
            dx::to_const_mask<cpo_result_t<trunc_t, T, rounding::no_exc_t>>(
                cmask),
            __DPL forward<T>(val), trunc_noexc_v);
    }
};

template <>
struct fallback_impl<trunc_t> {
private:
    template <typename E>
    static constexpr auto nonmantissa_v = __DPL type_bit_v<E> -
        __DPL countr_zero(floating_point_traits<E>::exponent_mask);

public:
    template <simd_abi A, simd_element_for<A> E>
    requires binary_layout_floating_point<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        basic_vector<E, A> val) noexcept {
        // Based on musl libm
        using sint = signed_representation_t<E>;
        auto const nonmantissa = dx::broadcast<sint, A>(nonmantissa_v<E>);
        auto const exp = [](auto exp, auto nonmantissa) {
            return dx::select(exp < nonmantissa, dx::one, exp);
        }(mx::ilogb(mx::compliance::unsafe, val) + nonmantissa, nonmantissa);

        // Operation might be slow for backends without vector right shift
        auto const m = dx::broadcast<E, A>(dx::all_bits) >> exp;
        return dx::bwandnot(val, m);
    }

    template <simd_abi A, simd_element_for<A> E>
    requires binary_layout_floating_point<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val, rounding::no_exc_t) noexcept {
        return operator()(val);
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::trunc_t trunc{};
}
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
