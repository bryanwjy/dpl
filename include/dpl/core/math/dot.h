// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/basic/broadcast.h"
#  include "dpl/core/basic/lane_index.h"
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/maskable/accumulation.h"
#  include "dpl/core/dispatch/operation/math.h"
#  include "dpl/core/operations/arithmetic.h"
#  include "dpl/core/operations/bitwise.h"
#  include "dpl/core/operations/cast.h"
#  include "dpl/core/operations/permute.h"
#  include "dpl/core/type_traits/common_abi.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
/**
 * @brief Computes the bfloat16 dot product with float32 accumulation.
 *
 * Performs a paired bfloat16 dot product, internally promoting each bfloat16
 * operand to float32 before multiplication and accumulation. The result is a
 * vector of float32 values, where each float32 element accumulates the dot
 * product of two consecutive bfloat16 pairs from the input vectors.
 *
 * @par Mathematical Definition
 * For each float32 lane j in the destination register:
 * @code
 * FOR j := 0 to N
 *     dst.fp32[j] += make_fp32(a.bf16[2*j+1]) * make_fp32(b.bf16[2*j+1])
 *     dst.fp32[j] += make_fp32(a.bf16[2*j+0]) * make_fp32(b.bf16[2*j+0])
 * ENDFOR
 * @endcode
 * where N is the number of float32 lanes in the vector register.
 */
void dot_product(...) noexcept = delete;
struct dot_product_t :
    private math_operation_base<dot_product_t>,
    private maskable_accumulation_base<dot_product_t> {
    using math_operation_base<dot_product_t>::operator();
    using maskable_accumulation_base<dot_product_t>::operator();
};

template <>
struct operation_signature<dot_product_t> {
    template <simd_vector S, simd_vector L, simd_vector R>
    static consteval void operator()(S&&, L&&, R&&) noexcept {}
};

template <typename S, typename L, typename R>
concept unqualified_canonical_dot_product =
    same_as<simd_abi_type_t<S>, common_abi_t<L, R>> && requires {
        {
            dot_product(internal::abi<common_abi_t<L, R>>,
                internal::declarg<S>(), internal::declarg<L>(),
                internal::declarg<R>())
        } -> same_as<S>;
    };

template <typename S, typename M, typename L, typename R>
concept unqualified_canonical_mdot_product =
    same_as<S, cpo_result_t<dot_product_t, S, L, R>> && requires {
        {
            dot_product(internal::abi<cpo_result_t<dot_product_t, S, L, R>>,
                internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<L>(), internal::declarg<R>())
        } -> same_as<cpo_result_t<dot_product_t, S, L, R>>;
    };

template <typename M, typename S, typename L, typename R>
concept unqualified_canonical_mzdot_product =
    cpo_invocable<dot_product_t, S, L, R> && requires {
        {
            dot_product(internal::abi<cpo_result_t<dot_product_t, S, L, R>>,
                dx::zero, internal::declarg<M>(), internal::declarg<S>(),
                internal::declarg<L>(), internal::declarg<R>())
        } -> same_as<cpo_result_t<dot_product_t, S, L, R>>;
    };

template <>
struct canonical_impl<dot_product_t> {
public:
    template <typename E, typename L, typename R>
    using source_t DPL_NODEBUG = basic_vector<E, common_abi_t<L, R>>;

    template <typename E, typename L, typename R>
    using mask_t DPL_NODEBUG = basic_mask<E, common_abi_t<L, R>>;

    template <typename E, typename L, typename R>
    using imask_t DPL_NODEBUG =
        mask_value_t<simd_abi_traits<E, common_abi_t<L, R>>::size>;

    template <typename E, typename L, typename R, imask_t<E, L, R> V>
    using cmask_t DPL_NODEBUG =
        const_mask<simd_abi_traits<E, common_abi_t<L, R>>::size, V>;

private:
    template <canonical_vector L, common_vector_with<L> R,
        simd_element_for<common_abi_t<L, R>> SE>
    requires canonical_vector<R> &&
        unqualified_canonical_dot_product<source_t<SE, L, R>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr source_t<SE, L, R> operator()(
        source_t<SE, L, R> src, L lhs, R rhs) noexcept {
        return dot_product(internal::abi<common_abi_t<L, R>>, src, lhs, rhs);
    }

    template <canonical_vector L, common_vector_with<L> R,
        simd_element_for<common_abi_t<L, R>> SE>
    requires canonical_vector<R> &&
        unqualified_canonical_mdot_product<source_t<SE, L, R>, mask_t<SE, L, R>,
            L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr source_t<SE, L, R> operator()(
        source_t<SE, L, R> src, mask_t<SE, L, R> mask, L lhs, R rhs) noexcept {
        return dot_product(
            internal::abi<common_abi_t<L, R>>, src, mask, lhs, rhs);
    }

    template <canonical_vector L, common_vector_with<L> R,
        simd_element_for<common_abi_t<L, R>> SE, imask_t<SE, L, R> M>
    requires canonical_vector<R> &&
        unqualified_canonical_mdot_product<source_t<SE, L, R>,
            cmask_t<SE, L, R, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr source_t<SE, L, R> operator()(source_t<SE, L, R> src,
        cmask_t<SE, L, R, M> cmask, L lhs, R rhs) noexcept {
        return dot_product(
            internal::abi<common_abi_t<L, R>>, src, cmask, lhs, rhs);
    }

    template <canonical_vector L, common_vector_with<L> R,
        simd_element_for<common_abi_t<L, R>> SE>
    requires canonical_vector<R> &&
        unqualified_canonical_mzdot_product<mask_t<SE, L, R>,
            source_t<SE, L, R>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, mask_t<SE, L, R> mask,
        source_t<SE, L, R> src, L lhs, R rhs) noexcept {
        return dot_product(
            internal::abi<common_abi_t<L, R>>, zero, mask, src, lhs, rhs);
    }

    template <canonical_vector L, common_vector_with<L> R,
        simd_element_for<common_abi_t<L, R>> SE, imask_t<SE, L, R> M>
    requires canonical_vector<R> &&
        unqualified_canonical_mzdot_product<cmask_t<SE, L, R, M>,
            source_t<SE, L, R>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero,
        cmask_t<SE, L, R, M> cmask, source_t<SE, L, R> src, L lhs,
        R rhs) noexcept {
        return dot_product(
            internal::abi<common_abi_t<L, R>>, zero, cmask, src, lhs, rhs);
    }
};

template <typename S, typename L, typename R>
concept unqualified_extended_dot_product =
    same_as<simd_abi_type_t<S>, common_abi_t<L, R>> && requires {
        {
            dot_product(internal::declarg<S>(), internal::declarg<L>(),
                internal::declarg<R>())
        } -> equivalent_vector_with<S>;
    };

template <typename S, typename M, typename L, typename R>
concept unqualified_extended_mdot_product =
    equivalent_vector_with<S, cpo_result_t<dot_product_t, S, L, R>> &&
    requires {
        {
            dot_product(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<L>(), internal::declarg<R>())
        } -> equivalent_vector_with<cpo_result_t<dot_product_t, S, L, R>>;
    };

template <typename S, typename M, typename L, typename R>
concept unqualified_extended_mzdot_product =
    equivalent_vector_with<S, cpo_result_t<dot_product_t, S, L, R>> &&
    requires {
        {
            dot_product(dx::zero, internal::declarg<M>(),
                internal::declarg<S>(), internal::declarg<L>(),
                internal::declarg<R>())
        } -> equivalent_vector_with<cpo_result_t<dot_product_t, S, L, R>>;
    };

template <>
struct extended_impl<dot_product_t> {
private:
    template <typename S>
    using simask_t DPL_NODEBUG = mask_value_t<simd_abi_type_t<S>::size>;

    template <typename S, simask_t<S> V>
    using scmask_t DPL_NODEBUG = const_mask<simd_abi_type_t<S>::size, V>;

public:
    template <simd_vector S, simd_vector L, common_vector_with<L> R>
    requires (extended_vector<S> || extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_dot_product<S, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, L&& lhs, R&& rhs) {
        return dot_product(__DPL forward<S>(src), __DPL forward<L>(lhs),
            __DPL forward<R>(rhs));
    }

    template <simd_vector S, exact_mask_for<S> M, simd_vector L,
        common_vector_with<L> R>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<L> ||
                 extended_vector<R>) &&
        unqualified_extended_mdot_product<S, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, L&& lhs, R&& rhs) {
        return dot_product(__DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <simd_vector S, simask_t<S> M, simd_vector L,
        common_vector_with<L> R>
    requires (extended_vector<S> || extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_mdot_product<S, scmask_t<S, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        S&& src, scmask_t<S, M> cmask, L&& lhs, R&& rhs) {
        return dot_product( __DPL forward<S>(src), cmask, __DPL forward<L>(lhs),
            __DPL forward<R>(rhs));
    }

    template <simd_vector S, exact_mask_for<S> M, simd_vector L,
        common_vector_with<L> R>
    requires (extended_vector<S> || extended_vector<L> || extended_vector<R> ||
                 extended_mask<M>) &&
        unqualified_extended_mzdot_product<S, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, S&& src, L&& lhs, R&& rhs) {
        return dot_product(zero, __DPL forward<M>(mask), __DPL forward<S>(src),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <simd_vector S, simask_t<S> M, simd_vector L,
        common_vector_with<L> R>
    requires (extended_vector<S> || extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_mzdot_product<S, scmask_t<S, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, scmask_t<S, M> cmask, S&& src, L&& lhs, R&& rhs) {
        return dot_product(zero, cmask, __DPL forward<S>(src),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }
};

template <>
struct fallback_impl<dot_product_t> {
    template <brain_float E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_vector<float, A>
        DPL_VECTORCALL operator()(basic_vector<float, A> src,
            basic_vector<E, A> lhs, basic_vector<E, A> rhs) noexcept {
        auto const idx = dx::lane_index<E, A>();
        auto const lower_half = idx < basic_vector<float, A>::size();
        auto const even = dx::bwshift_left(lower_half, idx, imm<1zu>);
        auto const odd = dx::add(lower_half, even,
            dx::broadcast<decltype(dx::lane_index<E, A>())>(dx::one));
        auto const odd_vals =
            dx::multiply(dx::element_cast<float>(dx::permute(lhs, odd)),
                dx::element_cast<float>(dx::permute(rhs, odd)));
        auto const even_vals =
            dx::multiply(dx::element_cast<float>(dx::permute(lhs, even)),
                dx::element_cast<float>(dx::permute(rhs, even)));
        return dx::add(src, dx::add(odd_vals, even_vals));
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::dot_product_t dot_product{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
