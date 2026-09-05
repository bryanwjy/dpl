// Copyright 2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/dot_product.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/dispatch/maskable/accumulation.h"
#  include "dpl/core/dispatch/operation/math.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::internal {
void inner_product(...) noexcept = delete;

/**
 * @brief Accumulating dot product.
 *
 * Multiplies corresponding lanes of @p lhs and @p rhs and adds the products
 * into @p src. Equivalent to a dot product with a running accumulator, in the
 * sense of @c std::inner_product -- <tt>dot_product(lhs, rhs)</tt> is
 * <tt>inner_product(dx::zero, lhs, rhs)</tt>.
 *
 * @par Widening
 * @p src may have a wider element type than the operands. For integers,
 * adjacent operand lanes are multiplied and summed into one accumulator lane,
 * with the grouping given by the ratio of element widths -- four @c int8_t
 * lanes per @c int32_t accumulator lane, and so on. The same applies to
 * @c bfloat16 operands with a @c float accumulator, where products are
 * accumulated at single precision.
 *
 * Widening ratios must divide the operand lane count evenly. The accumulator
 * width follows from @p src rather than being deduced from the operands, since
 * more than one valid width generally exists.
 *
 * @param src Accumulator, and the merge target under masking. @c dx::zero
 *            starts a fresh accumulation and elides the initial add.
 * @param lhs First operand.
 * @param rhs Second operand.
 * @return The updated accumulator.
 *
 * @pre Operand lane count is a multiple of the widening ratio.
 *
 * @par Classification
 * Domain @c math_operation; maskability @c maskable_accumulation.
 *
 * The mask is accumulator-granular: active lanes receive the updated value,
 * inactive lanes are taken from @p src unchanged or zeroed.
 *
 * @sa dot_product Non-accumulating form.
 */
struct inner_product_t :
    public math_operation_base<inner_product_t>,
    public maskable_accumulation_base<inner_product_t> {
    using operation_base<inner_product_t>::operator();
    using maskable_accumulation_base<inner_product_t>::operator();

    template <typename L, typename R>
    requires cpo_invocable<dot_product_t, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr cpo_result_t<dot_product_t, L, R> operator()(
        dx::zero_t zero, L&& lhs,
        R&& rhs) noexcept(unextended_type<L> && unextended_type<R>) {
        return dx::dot_product( __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }
};

template <>
struct operation_signature<inner_product_t> {
    template <simd_vector S, typename L, typename R>
    requires simd_vector<L> || simd_vector<R>
    static consteval void operator()(S&&, L&&, R&&) noexcept {}
};

template <typename S, typename L, typename R>
concept inner_product_result = simd_vector<L> && simd_vector<R> &&
    simd_vector<S> && same_abi_as<simd_abi_type_t<S>, common_abi_t<L, R>>;

template <typename S, typename L, typename R>
concept canonical_inner_product_result = canonical_vector<L> &&
    canonical_vector<R> && canonical_vector<S> && inner_product_result<S, L, R>;

template <typename S, typename L, typename R>
concept unqualified_canonical_inner_product = requires {
    {
        inner_product(internal::abi<S>, internal::declarg<S>(),
            internal::declarg<L>(), internal::declarg<R>())
    } -> same_as<S>;
};

template <typename S, typename M, typename L, typename R>
concept unqualified_canonical_minner_product =
    cpo_invocable<inner_product_t, S, L, R> &&
    same_as<S, cpo_result_t<inner_product_t, S, L, R>> && requires {
        {
            inner_product(internal::abi<S>, internal::declarg<S>(),
                internal::declarg<M>(), internal::declarg<L>(),
                internal::declarg<R>())
        } -> same_as<S>;
    };

template <typename M, typename S, typename L, typename R>
concept unqualified_canonical_zminner_product =
    cpo_invocable<inner_product_t, S, L, R> &&
    same_as<S, cpo_result_t<inner_product_t, S, L, R>> && requires {
        {
            inner_product(internal::abi<S>, dx::zero, internal::declarg<M>(),
                internal::declarg<S>(), internal::declarg<L>(),
                internal::declarg<R>())
        } -> same_as<S>;
    };

template <>
struct canonical_impl<inner_product_t> {
    template <canonical_vector L, common_vector_with<L> R,
        canonical_inner_product_result<L, R> S>
    requires canonical_vector<R> && unqualified_canonical_inner_product<S, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr S operator()(S src, L lhs, R rhs) noexcept {
        return inner_product(internal::abi<S>, src, lhs, rhs);
    }

    template <canonical_vector L, broadcastable_to<L> R,
        canonical_inner_product_result<L, L> S>
    requires unqualified_canonical_inner_product<S, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr S operator()(S src, L lhs, R&& rhs) noexcept {
        return inner_product(
            internal::abi<S>, src, lhs, __DPL forward<R>(rhs));
    }

    template <canonical_vector R, broadcastable_to<R> L,
        canonical_inner_product_result<R, R> S>
    requires unqualified_canonical_inner_product<S, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr S operator()(S src, L&& lhs, R rhs) noexcept {
        return inner_product(
            internal::abi<S>, src, __DPL forward<L>(lhs), rhs);
    }

    template <canonical_vector S, broadcastable_to<S> L, broadcastable_to<S> R>
    requires unqualified_canonical_inner_product<S, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr S operator()(S src, L&& lhs, R&& rhs) noexcept {
        return inner_product(internal::abi<S>, src, __DPL forward<L>(lhs),
            __DPL forward<R>(rhs));
    }

    template <unextended_type L, unextended_type R, canonical_vector S>
    requires unqualified_canonical_minner_product<S, simd_mask_type_t<S>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr S operator()(
        S src, simd_mask_type_t<S> mask, L&& lhs, R&& rhs) noexcept {
        return inner_product(internal::abi<S>, src, mask, __DPL forward<L>(lhs),
            __DPL forward<R>(rhs));
    }

    template <unextended_type L, unextended_type R, canonical_vector S,
        const_mask_for<S> M>
    requires unqualified_canonical_minner_product<S, launder_cmask_t<S, M>, L,
        R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr S operator()(S src, M mask, L&& lhs, R&& rhs) noexcept {
        return inner_product(internal::abi<S>, src, dx::to_const_mask<S>(mask),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <unextended_type L, unextended_type R, canonical_vector S>
    requires unqualified_canonical_zminner_product<simd_mask_type_t<S>, S, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr S operator()(dx::zero_t zero, simd_mask_type_t<S> mask,
        S src, L&& lhs, R&& rhs) noexcept {
        return inner_product(internal::abi<S>, zero, mask, src,
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <unextended_type L, unextended_type R, canonical_vector S,
        const_mask_for<S> M>
    requires unqualified_canonical_zminner_product<launder_cmask_t<S, M>, S, L,
        R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M mask, S src, L&& lhs, R&& rhs) noexcept {
        return inner_product(internal::abi<S>, zero, dx::to_const_mask<S>(mask),
            src, __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }
};

template <typename S, typename L, typename R>
concept unqualified_extended_inner_product = requires {
    {
        inner_product(internal::declarg<S>(), internal::declarg<L>(),
            internal::declarg<R>())
    } -> equivalent_vector_with<S>;
};

template <typename S, typename M, typename L, typename R>
concept unqualified_extended_minner_product =
    cpo_invocable<inner_product_t, S, L, R> &&
    equivalent_vector_with<cpo_result_t<inner_product_t, S, L, R>, S> &&
    requires {
        {
            inner_product(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<L>(), internal::declarg<R>())
        } -> equivalent_vector_with<S>;
    };

template <typename M, typename S, typename L, typename R>
concept unqualified_extended_zminner_product =
    cpo_invocable<inner_product_t, S, L, R> &&
    equivalent_vector_with<cpo_result_t<inner_product_t, S, L, R>, S> &&
    requires {
        {
            inner_product(dx::zero, internal::declarg<M>(),
                internal::declarg<S>(), internal::declarg<L>(),
                internal::declarg<R>())
        } -> equivalent_vector_with<S>;
    };

template <>
struct extended_impl<inner_product_t> {
public:
    template <simd_vector L, common_vector_with<L> R,
        inner_product_result<L, R> S>
    requires (extended_vector<S> || extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_inner_product<S, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, L&& lhs, R&& rhs) {
        return inner_product(__DPL forward<S>(src), __DPL forward<L>(lhs),
            __DPL forward<R>(rhs));
    }

    template <simd_vector L, broadcastable_to<L> R,
        inner_product_result<L, L> S>
    requires (extended_vector<S> || extended_vector<L>) &&
        unqualified_extended_inner_product<S, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr S operator()(S src, L lhs, R&& rhs) noexcept {
        return inner_product(__DPL forward<S>(src), __DPL forward<L>(lhs),
            __DPL forward<R>(rhs));
    }

    template <simd_vector R, broadcastable_to<R> L,
        inner_product_result<R, R> S>
    requires (extended_vector<S> || extended_vector<R>) &&
        unqualified_extended_inner_product<S, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr S operator()(S src, L&& lhs, R rhs) noexcept {
        return inner_product(__DPL forward<S>(src), __DPL forward<L>(lhs),
            __DPL forward<R>(rhs));
    }

    template <extended_vector S, broadcastable_to<S> L, broadcastable_to<S> R>
    requires unqualified_extended_inner_product<S, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr S operator()(S src, L&& lhs, R&& rhs) noexcept {
        return inner_product(__DPL forward<S>(src), __DPL forward<L>(lhs),
            __DPL forward<R>(rhs));
    }

    template <typename L, typename R, simd_vector S, exact_mask_for<S> M>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<L> ||
                 extended_vector<R>) &&
        unqualified_extended_minner_product<S, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, L&& lhs, R&& rhs) {
        return inner_product(__DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <typename L, typename R, simd_vector S, const_mask_for<S> M>
    requires (extended_vector<S> || extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_minner_product<S, launder_cmask_t<S, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M mask, L&& lhs, R&& rhs) {
        return inner_product( __DPL forward<S>(src),
            dx::to_const_mask<S>(mask), __DPL forward<L>(lhs),
            __DPL forward<R>(rhs));
    }

    template <typename L, typename R, simd_vector S, exact_mask_for<S> M>
    requires (extended_vector<S> || extended_vector<L> || extended_vector<R> ||
                 extended_mask<M>) &&
        unqualified_extended_zminner_product<M, S, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, S&& src, L&& lhs, R&& rhs) {
        return inner_product(zero, __DPL forward<M>(mask),
            __DPL forward<S>(src), __DPL forward<L>(lhs),
            __DPL forward<R>(rhs));
    }

    template <typename L, typename R, simd_vector S, const_mask_for<S> M>
    requires (extended_vector<S> || extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_zminner_product<launder_cmask_t<S, M>, S, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero,
        launder_cmask_t<S, M> mask, S&& src, L&& lhs, R&& rhs) {
        return inner_product(zero, dx::to_const_mask<S>(mask),
            __DPL forward<S>(src), __DPL forward<L>(lhs),
            __DPL forward<R>(rhs));
    }
};

template <typename From, typename To>
concept integral_widen_to = ((signed_integral<From> && signed_integral<To>) ||
                                (unsigned_integral<From> &&
                                    unsigned_integral<To>)) &&
    sizeof(To) > sizeof(From);

template <>
struct fallback_impl<inner_product_t> {
    // Same element
    template <canonical_vector L, common_vector_with<L> R>
    requires cpo_invocable<hsum_t, common_canonical_simd_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr common_canonical_simd_t<L, R> operator()(
        common_canonical_simd_t<L, R> src, L lhs, R rhs) noexcept {
        return dx::add(src, dx::dot_product(lhs, rhs));
    }

    // Widening
    template <canonical_vector S, canonical_vector T>
    requires same_abi_as<simd_abi_type_t<S>, simd_abi_type_t<T>> &&
        same_as<float, simd_element_type_t<S>> &&
        same_as<ext::bfloat16, simd_element_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr S DPL_VECTORCALL operator()(S src, T lhs, T rhs) noexcept {
        using vidx_t = signed_canonical_vector_t<T>;
        auto const idx = dx::lane_index<vidx_t>();
        auto const lower_half = dx::cmplt(idx, simd_abi_traits<S>::size());
        auto const even = dx::bwshift_left(lower_half, idx, imm<1zu>);
        auto const odd =
            dx::add(lower_half, even, dx::broadcast<vidx_t>(dx::one));
        auto const odd_vals =
            dx::multiply(dx::element_cast<float>(dx::permute(lhs, odd)),
                dx::element_cast<float>(dx::permute(rhs, odd)));
        auto const even_vals =
            dx::multiply(dx::element_cast<float>(dx::permute(lhs, even)),
                dx::element_cast<float>(dx::permute(rhs, even)));
        return dx::add(src, dx::add(odd_vals, even_vals));
    }

    template <canonical_vector S, canonical_vector T>
    requires same_abi_as<simd_abi_type_t<S>, simd_abi_type_t<T>> &&
        integral_widen_to<simd_element_type_t<T>, simd_element_type_t<S>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr S DPL_VECTORCALL operator()(S src, T lhs, T rhs) noexcept {
        using vidx_t = signed_canonical_vector_t<T>;

        using To = simd_element_type_t<S>;
        using From = simd_element_type_t<T>;
        constexpr auto shift =
            __DPL bit_width(sizeof(To)) - __DPL bit_width(sizeof(From));
        constexpr auto ratio = sizeof(To) / sizeof(From);

        auto const idx = dx::lane_index<vidx_t>();
        auto const idx_mask = dx::cmplt(idx, simd_abi_traits<S>::size());
        auto const base = dx::bwshift_left(idx_mask, idx, imm<shift>);
        auto addend = dx::multiply( //
            dx::element_cast<To>(dx::permute(lhs, base)),
            dx::element_cast<To>(dx::permute(rhs, base)));

        return [&]<size_t I = 1>(this auto self, immediate<I> = imm<1zu>) {
            if constexpr (I == ratio) {
                return dx::add(src, addend);
            } else {
                auto const next_idx =
                    dx::add(idx_mask, base, dx::broadcast<vidx_t>(I));
                // TODO define wmultiply?
                auto const next = dx::multiply(
                    dx::element_cast<To>(dx::permute(lhs, next_idx)),
                    dx::element_cast<To>(dx::permute(rhs, next_idx)));
                addend = dx::add(next, addend);
                return self(imm<I + 1>);
            }
        }();
    }

    template <canonical_vector L, broadcastable_to<L> R,
        canonical_inner_product_result<L, L> S>
    requires cpo_invocable<inner_product_t, S, L, L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr S operator()(S src, L lhs, R&& rhs) noexcept {
        return inner_product_t::operator()(
            src, lhs, dx::broadcast<L>(__DPL forward<R>(rhs)));
    }

    template <canonical_vector R, broadcastable_to<R> L,
        canonical_inner_product_result<R, R> S>
    requires cpo_invocable<inner_product_t, S, R, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr S operator()(S src, L&& lhs, R rhs) noexcept {
        return inner_product_t::operator()(
            src, dx::broadcast<R>(__DPL forward<L>(lhs)), rhs);
    }

    template <canonical_vector S, broadcastable_to<S> L, broadcastable_to<S> R>
    requires cpo_invocable<inner_product_t, S, S, S>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr S operator()(S src, L&& lhs, R&& rhs) noexcept {
        return inner_product_t::operator()(src,
            dx::broadcast<S>(__DPL forward<L>(lhs)),
            dx::broadcast<S>(__DPL forward<R>(rhs)));
    }
};

} // namespace datapar::internal
namespace datapar {
inline namespace cpo {
inline constexpr internal::inner_product_t inner_product{};
} // namespace cpo
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
