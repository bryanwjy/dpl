// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/maskable/transform.h"
#  include "dpl/core/dispatch/operation/math.h"
#  include "dpl/core/immediate/constants/exponent_bias.h"
#  include "dpl/core/immediate/constants/exponent_mask.h"
#  include "dpl/core/immediate/constants/mantissa_width.h"
#  include "dpl/core/operations/arithmetic.h"
#  include "dpl/core/operations/minmax.h"
#  include "dpl/std/bit/popcount.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

void ldexp(...) noexcept = delete;

struct DPL_EMPTY_BASES ldexp_t :
    public math_operation_base<ldexp_t>,
    public maskable_transform_base<ldexp_t> {
    using math_operation_base<ldexp_t>::operator();
    using maskable_transform_base<ldexp_t>::operator();
};

template <>
struct operation_signature<ldexp_t> {
    template <simd_vector L, typename R>
    static consteval void operator()(L&&, R&&) noexcept {}
};

template <typename L, typename R, typename T>
concept unqualified_canonical_ldexp = requires {
    {
        ldexp(internal::abi<T>, internal::declarg<L>(), internal::declarg<R>())
    } -> same_as<T>;
};

template <typename S, typename M, typename L, typename R, typename T>
concept unqualified_canonical_mldexp_base =
    cpo_invocable<ldexp_t, L, R> && requires {
        {
            ldexp(internal::abi<T>, internal::declarg<S>(),
                internal::declarg<M>(), internal::declarg<L>(),
                internal::declarg<R>())
        } -> same_as<T>;
    };

template <typename M, typename L, typename R,
    typename T = common_canonical_simd_t<L, R>>
concept unqualified_canonical_mldexp =
    unqualified_canonical_mldexp_base<T, M, L, R, T>;

template <typename M, typename L, typename R,
    typename T = common_canonical_simd_t<L, R>>
concept unqualified_canonical_zmldexp =
    unqualified_canonical_mldexp_base<dx::zero_t, M, L, R, T>;

template <typename R, typename L>
concept broadcastable_to_exp_of = broadcastable_to<R, mx::exponent_vector_t<L>>;

template <>
struct canonical_impl<ldexp_t> {
private:
    template <typename L, typename R>
    using result_t DPL_NODEBUG = cpo_result_t<ldexp_t, L, R>;
    template <typename L, typename R>
    using mask_t DPL_NODEBUG = simd_mask_type_t<result_t<L, R>>;

public:
    template <canonical_vector L, canonical_vector R>
    requires same_abi_as<simd_abi_type_t<L>, simd_abi_type_t<R>> &&
        unqualified_canonical_ldexp<L, R, L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr common_canonical_simd_t<L, R> operator()(
        L lhs, R rhs) noexcept {
        return ldexp(internal::abi<common_abi_t<L, R>>, lhs, rhs);
    }

    template <canonical_vector L, broadcastable_to_exp_of<L> R>
    requires unqualified_canonical_ldexp<L, R, L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr L operator()(L lhs, R&& rhs) noexcept {
        return ldexp(internal::abi<L>, lhs, __DPL forward<R>(rhs));
    }

    template <unextended_type L, unextended_terminal_of<ldexp_t, L> R>
    requires unqualified_canonical_mldexp<mask_t<L, R>, L, R, result_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<L, R> operator()(
        result_t<L, R> src, mask_t<L, R> mask, L&& lhs, R&& rhs) noexcept {
        return ldexp(internal::abi<result_t<L, R>>, src, mask,
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <unextended_type L, unextended_type R, result_cmask_for<L, R> M>
    requires unqualified_canonical_mldexp<launder_cmask_t<result_t<L, R>, M>, L,
        R, result_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<L, R> operator()(
        result_t<L, R> src, M mask, L&& lhs, R&& rhs) noexcept {
        using T = result_t<L, R>;
        return ldexp(internal::abi<T>, src, dx::to_const_mask<T>(mask),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <unextended_type L, unextended_terminal_of<ldexp_t, L> R>
    requires unqualified_canonical_zmldexp<mask_t<L, R>, L, R, result_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<L, R> operator()(
        dx::zero_t zero, mask_t<L, R> mask, L&& lhs, R&& rhs) noexcept {
        return ldexp(internal::abi<result_t<L, R>>, zero, mask,
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <unextended_type L, unextended_type R, result_cmask_for<L, R> M>
    requires unqualified_canonical_zmldexp<launder_cmask_t<result_t<L, R>, M>,
        L, R, result_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<L, R> operator()(
        dx::zero_t zero, M mask, L&& lhs, R&& rhs) noexcept {
        using T = result_t<L, R>;
        return ldexp(internal::abi<T>, zero, dx::to_const_mask<T>(mask),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }
};

template <typename L, typename R, typename T>
concept unqualified_extended_ldexp = requires {
    {
        ldexp(internal::declarg<L>(), internal::declarg<R>())
    } -> equivalent_vector_with<T>;
};

template <typename S, typename M, typename L, typename R>
concept unqualified_extended_mldexp_base =
    cpo_invocable<ldexp_t, L, R> && requires {
        {
            ldexp(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<L>(), internal::declarg<R>())
        } -> equivalent_vector_with<cpo_result_t<ldexp_t, L, R>>;
    };

template <typename S, typename M, typename L, typename R>
concept unqualified_extended_mldexp =
    equivalent_vector_with<S, cpo_result_t<ldexp_t, L, R>> &&
    unqualified_extended_mldexp_base<S, M, L, R>;

template <typename M, typename L, typename R>
concept unqualified_extended_zmldexp =
    unqualified_extended_mldexp_base<dx::zero_t, M, L, R>;

template <>
struct extended_impl<ldexp_t> {
public:
    template <simd_vector L, simd_vector R>
    requires (extended_vector<L> || extended_vector<R>) &&
        same_abi_as<simd_abi_type_t<L>, simd_abi_type_t<R>> &&
        unqualified_extended_ldexp<L, R, L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R&& rhs) {
        return ldexp(__DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <extended_vector L, broadcastable_to_exp_of<L> R>
    requires unqualified_extended_ldexp<L, R, L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R&& rhs) {
        return ldexp(__DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <simd_vector S, exact_mask_for<S> M, typename L, typename R>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<L> ||
                 extended_vector<R>) &&
        unqualified_extended_mldexp<S, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, L&& lhs, R&& rhs) {
        return ldexp(__DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <simd_vector S, const_mask_for<S> M, typename L, typename R>
    requires (extended_vector<S> || extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_mldexp<S, launder_cmask_t<S, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M mask, L&& lhs, R&& rhs) {
        return ldexp(src, dx::to_const_mask<S>(mask), __DPL forward<L>(lhs),
            __DPL forward<R>(rhs));
    }

    template <typename L, typename R, result_mask_for<ldexp_t, L, R> M>
    requires (extended_mask<M> || extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_zmldexp<M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, L&& lhs, R&& rhs) {
        return ldexp(zero, __DPL forward<M>(mask), __DPL forward<L>(lhs),
            __DPL forward<R>(rhs));
    }

    template <typename L, typename R, result_cmask_for<ldexp_t, L, R> M>
    requires (extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_zmldexp<
            launder_cmask_t<cpo_result_t<ldexp_t, L, R>, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M mask, L&& lhs, R&& rhs) {
        return ldexp(zero, dx::to_const_mask<cpo_result_t<ldexp_t, L, R>>(mask),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }
};

template <>
struct fallback_impl<ldexp_t> {
private:
    template <canonical_vector T>
    requires signed_integral<simd_element_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T signbit(T val) noexcept {
        using E = simd_element_type_t<T>;
        constexpr auto shift = __DPL type_bit_v<E> - 1;
        return dx::bwshift_right(val, imm<shift>);
    }

public:
    template <canonical_vector T>
    requires binary_layout_floating_point<simd_element_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(
        T num, mx::exponent_vector_t<T> exp) noexcept {
        using E = simd_element_type_t<T>;
        using traits = floating_point_traits<E>;
        constexpr auto shift = __DPL countr_zero(traits::exponent_mask);
        constexpr auto bexp_max = __DPL to_signed(
            __DPL to_underlying(traits::exponent_mask >> shift));
        constexpr auto chunk =
            __DPL popcount(__DPL to_unsigned(traits::exponent_bias)) - 1;
        constexpr auto lshift = imm<chunk>;
        constexpr auto rshift = imm<chunk - 2>;

        auto const sign = fallback_impl::signbit(exp);
        auto m = dx::add(sign, exp);
        m = dx::bwshift_right(m, lshift);
        m = dx::subtract(m, sign);
        m = dx::bwshift_left(m, rshift);

        exp = dx::subtract(exp, dx::bwshift_left(m, imm<2zu>));
        m = dx::add(m, traits::exponent_bias);
        m = dx::max(dx::zero, m);
        m = dx::min(bexp_max, m);
        m = dx::bwshift_left(m, imm<shift>);
        auto u = dx::reinterpret<E>(m);
        auto const u4 = [](T u2) { return dx::multiply(u2, u2); }(
                            dx::multiply(u, u));
        num = dx::multiply(num, u4);
        exp = dx::add(exp, traits::exponent_bias);
        exp = dx::bwshift_left(exp, imm<shift>);
        return dx::multiply(num, dx::reinterpret<E>(exp));
    }

    template <canonical_vector L, broadcastable_to_exp_of<L> R>
    requires cpo_invocable<ldexp_t, L, mx::exponent_vector_t<L>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr L operator()(L lhs, R&& rhs) noexcept {
        using I DPL_NODEBUG = mx::exponent_vector_t<L>;
        return ldexp_t::operator()(
            lhs, dx::broadcast<I>(__DPL forward<R>(rhs)));
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::ldexp_t ldexp{};
}
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
