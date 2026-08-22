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
    static consteval void operator()(
        simd_vector auto&&, simd_vector auto&&) noexcept {}
};

template <typename L, typename R>
concept unqualified_canonical_ldexp = requires {
    {
        ldexp(internal::abi<L>, internal::declarg<L>(), internal::declarg<R>())
    } -> same_as<L>;
};

template <typename S, typename M, typename L, typename R>
concept unqualified_canonical_mldexp = cpo_invocable<ldexp_t, L, R> &&
    (!simd_type<S> || same_as<S, cpo_result_t<ldexp_t, L, R>>) && requires {
        {
            ldexp(internal::abi<cpo_result_t<ldexp_t, L, R>>,
                internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<L>(), internal::declarg<R>())
        } -> same_as<cpo_result_t<ldexp_t, L, R>>;
    };

template <>
struct canonical_impl<ldexp_t> {
    template <canonical_vector L, canonical_vector R>
    requires same_abi_as<simd_abi_type_t<L>, simd_abi_type_t<R>> &&
        unqualified_canonical_ldexp<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr L operator()(L lhs, R rhs) noexcept {
        return ldexp(internal::abi<L>, lhs, rhs);
    }

    template <canonical_vector L, canonical_vector R>
    requires same_abi_as<simd_abi_type_t<L>, simd_abi_type_t<R>> &&
        unqualified_canonical_mldexp<L, simd_mask_type_t<L>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr L operator()(type_identity_t<L> src,
        simd_mask_type_t<L> mask, L lhs, R rhs) noexcept {
        return ldexp(internal::abi<L>, src, mask, lhs, rhs);
    }

    template <canonical_vector L, const_mask_for<L> M, canonical_vector R>
    requires same_abi_as<simd_abi_type_t<L>, simd_abi_type_t<R>> &&
        unqualified_canonical_mldexp<L, launder_cmask_t<L, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr L operator()(
        type_identity_t<L> src, M cmask, L lhs, R rhs) noexcept {
        return ldexp(
            internal::abi<L>, src, dx::to_const_mask<L>(cmask), lhs, rhs);
    }

    template <canonical_vector L, canonical_vector R>
    requires same_abi_as<simd_abi_type_t<L>, simd_abi_type_t<R>> &&
        unqualified_canonical_mldexp<dx::zero_t, simd_mask_type_t<L>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr L operator()(
        dx::zero_t zero, simd_mask_type_t<L> mask, L lhs, R rhs) noexcept {
        return ldexp(internal::abi<L>, zero, mask, lhs, rhs);
    }

    template <canonical_vector L, const_mask_for<L> M, canonical_vector R>
    requires same_abi_as<simd_abi_type_t<L>, simd_abi_type_t<R>> &&
        unqualified_canonical_mldexp<dx::zero_t, launder_cmask_t<L, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr L operator()(
        dx::zero_t zero, M cmask, L lhs, R rhs) noexcept {
        return ldexp(
            internal::abi<L>, zero, dx::to_const_mask<L>(cmask), lhs, rhs);
    }
};

template <typename L, typename R = L, typename A = common_abi_t<L, R>>
concept unqualified_extended_ldexp = requires {
    {
        ldexp(internal::declarg<L>(), internal::declarg<R>())
    } -> vector_with_common_abi<A>;
};

template <typename S, typename M, typename L, typename R>
concept unqualified_extended_mldexp = cpo_invocable<ldexp_t, L, R> &&
    (!simd_type<S> || equivalent_vector_with<S, cpo_result_t<ldexp_t, L, R>>) &&
    requires {
        {
            ldexp(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<L>(), internal::declarg<R>())
        } -> equivalent_vector_with<cpo_result_t<ldexp_t, L, R>>;
    };

template <>
struct extended_impl<ldexp_t> {
private:
    template <typename L, typename R, typename M>
    using vcmask_t DPL_NODEBUG =
        launder_cmask_t<cpo_result_t<ldexp_t, L, R>, M>;

public:
    template <simd_vector L, simd_vector R>
    requires (extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_ldexp<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R&& rhs) {
        return ldexp(__DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <simd_vector S, exact_mask_for<S> M, vector_subsumed_by<S> L,
        simd_vector R>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<L> ||
                 extended_vector<R>) &&
        unqualified_extended_mldexp<S, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, L&& lhs, R&& rhs) {
        return ldexp(__DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <simd_vector S, const_mask_for<S> M, vector_subsumed_by<S> L,
        simd_vector R>
    requires (extended_vector<S> || extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_mldexp<S, launder_cmask_t<S, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M cmask, L&& lhs, R&& rhs) {
        return ldexp(src, dx::to_const_mask<S>(cmask), __DPL forward<L>(lhs),
            __DPL forward<R>(rhs));
    }

    template <simd_vector L, simd_vector R, result_mask_for<ldexp_t, L, R> M>
    requires (extended_mask<M> || extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_mldexp<dx::zero_t, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, L&& lhs, R&& rhs) {
        return ldexp(zero, __DPL forward<M>(mask), __DPL forward<L>(lhs),
            __DPL forward<R>(rhs));
    }

    template <simd_vector L, simd_vector R, result_cmask_for<ldexp_t, L, R> M>
    requires (extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_mldexp<dx::zero_t, vcmask_t<L, R, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M cmask, L&& lhs, R&& rhs) {
        return ldexp(zero,
            dx::to_const_mask<cpo_result_t<ldexp_t, L, R>>(cmask),
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
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::ldexp_t ldexp{};
}
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
