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

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

void ldexp(...) noexcept = delete;

struct DPL_EMPTY_BASES ldexp_t :
    private math_operation_base<ldexp_t>,
    private maskable_transform_base<ldexp_t> {
    using math_operation_base<ldexp_t>::operator();
    using maskable_transform_base<ldexp_t>::operator();
};

template <>
struct operation_signature<ldexp_t> {
    static consteval void operator()(
        simd_vector auto&&, simd_vector auto&&) noexcept {}
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_canonical_ldexp = requires {
    {
        ldexp(internal::abi<A>, internal::declarg<L>(), internal::declarg<R>())
    } -> same_as<basic_vector<simd_element_type_t<L>, A>>;
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
private:
    template <typename L, typename R>
    using vresult_t DPL_NODEBUG =
        basic_vector<simd_element_type_t<L>, common_abi_t<L, R>>;

    template <typename L, typename R>
    using vmask_t DPL_NODEBUG =
        basic_mask<simd_element_type_t<L>, common_abi_t<L, R>>;

    template <typename L, typename R, typename M>
    using vcmask_t DPL_NODEBUG =
        launder_cmask_t<cpo_result_t<ldexp_t, L, R>, M>;

public:
    template <canonical_vector L, canonical_vector R>
    requires canonical_vector<R> && unqualified_canonical_ldexp<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr vresult_t<L, R> operator()(L lhs, R rhs) noexcept {
        return ldexp(internal::abi<common_abi_t<L, R>>, lhs, rhs);
    }

    template <canonical_vector L, canonical_vector R>
    requires canonical_vector<R> &&
        unqualified_canonical_mldexp<vresult_t<L, R>, vmask_t<L, R>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr vresult_t<L, R> operator()(
        vresult_t<L, R> src, vmask_t<L, R> mask, L lhs, R rhs) noexcept {
        return ldexp(internal::abi<common_abi_t<L, R>>, src, mask, lhs, rhs);
    }

    template <canonical_vector L, canonical_vector R,
        const_mask_for<vresult_t<L, R>> M>
    requires canonical_vector<R> &&
        unqualified_canonical_mldexp<vresult_t<L, R>, vcmask_t<L, R, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr vresult_t<L, R> operator()(
        vresult_t<L, R> src, M cmask, L lhs, R rhs) noexcept {
        return ldexp(internal::abi<common_abi_t<L, R>>, src,
            dx::to_const_mask<vresult_t<L, R>>(cmask), lhs, rhs);
    }

    template <canonical_vector L, canonical_vector R>
    requires canonical_vector<R> &&
        unqualified_canonical_mldexp<dx::zero_t, vmask_t<L, R>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr vresult_t<L, R> operator()(
        dx::zero_t zero, vmask_t<L, R> mask, L lhs, R rhs) noexcept {
        return ldexp(internal::abi<common_abi_t<L, R>>, zero, mask, lhs, rhs);
    }

    template <canonical_vector L, canonical_vector R,
        const_mask_for<vresult_t<L, R>> M>
    requires canonical_vector<R> &&
        unqualified_canonical_mldexp<dx::zero_t, vcmask_t<L, R, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr vresult_t<L, R> operator()(
        dx::zero_t zero, M cmask, L lhs, R rhs) noexcept {
        return ldexp(internal::abi<common_abi_t<L, R>>, zero,
            dx::to_const_mask<vresult_t<L, R>>(cmask), lhs, rhs);
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

    template <simd_vector S, exact_mask_for<S> M, simd_vector L, simd_vector R>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<L> ||
                 extended_vector<R>) &&
        unqualified_extended_mldexp<S, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, L&& lhs, R&& rhs) {
        return ldexp(__DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <simd_vector S, const_mask_for<S> M, simd_vector L, simd_vector R>
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
    template <signed_integral E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> signbit(
        basic_vector<E, A> val) noexcept {
        constexpr auto shift = sizeof(E) * char_bit_v - 1;
        return val >> imm<shift>;
    }

public:
    template <floating_point E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_vector<E, A>
        DPL_VECTORCALL operator()(basic_vector<E, A> num,
            basic_vector<signed_representation_t<E>, A> exp) noexcept {

        constexpr auto mantissa_shift = imm<dx::mantissa_width_v<E>>;
        using sint = signed_representation_t<E>;
        constexpr auto exp_mask =
            dx::exponent_mask_v<E, sint> >> mantissa_shift;
        constexpr auto exp_bias = dx::exponent_bias<E>;
        constexpr auto chunk =
            __DPL popcount(static_cast<unsigned>(exp_bias)) - 1;
        constexpr auto lshift = imm<chunk>;
        constexpr auto rshift = imm<chunk - 2>;
        auto const sign = fallback_impl::signbit(exp);
        auto m = (((sign + exp) >> lshift) - sign) << rshift;
        exp = exp - (m << imm<2>);

        m += exp_bias;
        m = dx::max(dx::zero, m);
        m = dx::min(exp_mask, m);

        using simdi = basic_vector<sint, A>;
        auto u = dx::reinterpret<E>(m << mantissa_shift);

        num *= [](auto u2) { return u2 * u2; }(u * u);

        return num * dx::reinterpret<E>((exp + exp_bias) << mantissa_shift);
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::ldexp_t ldexp{};
}
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
