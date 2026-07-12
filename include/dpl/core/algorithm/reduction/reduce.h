// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/algorithm/compress.h"
#include "dpl/core/algorithm/reduction/common.h"

#if !DPL_MODULES
#  include "dpl/core/basic/lane_index.h"
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/dispatch/maskable/transform.h"
#  include "dpl/core/dispatch/operation/algorithm.h"
#  include "dpl/core/immediate/const_mask.h"
#  include "dpl/core/immediate/immediate.h"
#  include "dpl/core/operations/bit.h"
#  include "dpl/core/operations/logical.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void reduce(...) noexcept = delete;
struct reduce_t :
    private reduction_base<reduce_t>,
    private maskable_transform_base<reduce_t> {
    using operation_base<reduce_t>::operator();
    using maskable_transform_base<reduce_t>::operator();
};

template <>
struct operation_signature<reduce_t> {
    template <simd_vector T, reduction_operator_for<T> Op>
    static consteval void operator()(T&&, Op&&) noexcept {}
};

template <typename T, typename Op>
concept unqualified_canonical_reduce = requires {
    {
        reduce(
            internal::abi<T>, internal::declarg<T>(), internal::declarg<Op>())
    } -> canonical_vector;
};

template <typename S, typename M, typename T, typename Op>
concept unqualified_canonical_mreduce = cpo_invocable<reduce_t, T, Op> &&
    (!simd_type<S> ||
        equivalent_vector_with<S, cpo_result_t<reduce_t, T, Op>>) &&
    requires {
        {
            reduce(internal::abi<conditional_t<simd_type<S>, S, T>>,
                internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>(), internal::declarg<Op>())
        } -> equivalent_vector_with<cpo_result_t<reduce_t, T, Op>>;
    };

template <>
struct canonical_impl<reduce_t> {
private:
    template <typename T, typename Op>
    using result_t DPL_NODEBUG =
        canonical_type_t<cpo_result_t<reduce_t, T, Op>>;

    template <typename T, typename Op>
    using mask_t DPL_NODEBUG = basic_mask<simd_element_type_t<result_t<T, Op>>,
        simd_abi_type_t<result_t<T, Op>>>;

public:
    template <canonical_vector T, reduction_operator_for<T> Op>
    requires unqualified_canonical_reduce<T, Op>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val, Op&& func) noexcept {
        return reduce(internal::abi<T>, val, __DPL forward<Op>(func));
    }

    template <canonical_vector T, reduction_operator_for<T> Op>
    requires unqualified_canonical_mreduce<result_t<T, Op>, mask_t<T, Op>, T,
        Op>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        result_t<T, Op> src, mask_t<T, Op> mask, T val, Op&& func) noexcept {
        return reduce(
            internal::abi<T>, src, mask, val, __DPL forward<Op>(func));
    }

    template <canonical_vector T, reduction_operator_for<T> Op,
        result_cmask_for<reduce_t, T, Op> M>
    requires unqualified_canonical_mreduce<result_t<T, Op>,
        launder_cmask_t<cpo_result<reduce_t, T, Op>, M>, T, Op>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        result_t<T, Op> src, M cmask, T val, Op&& func) noexcept {
        return reduce(internal::abi<T>, src,
            dx::to_const_mask<cpo_result<reduce_t, T, Op>>(cmask), val,
            __DPL forward<Op>(func));
    }

    template <canonical_vector T, reduction_operator_for<T> Op>
    requires unqualified_canonical_mreduce<dx::zero_t, mask_t<T, Op>, T, Op>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, mask_t<T, Op> mask, T val, Op&& func) noexcept {
        return reduce(
            internal::abi<T>, zero, mask, val, __DPL forward<Op>(func));
    }

    template <canonical_vector T, reduction_operator_for<T> Op,
        result_cmask_for<reduce_t, T, Op> M>
    requires unqualified_canonical_mreduce<dx::zero_t,
        launder_cmask_t<cpo_result<reduce_t, T, Op>, M>, T, Op>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, M cmask, T val, Op&& func) noexcept {
        return reduce(internal::abi<T>, zero,
            dx::to_const_mask<cpo_result<reduce_t, T, Op>>(cmask), val,
            __DPL forward<Op>(func));
    }
};

template <typename T, typename Op>
concept unqualified_extended_reduce = requires {
    {
        reduce(internal::declarg<T>(), internal::declarg<Op>())
    } -> equivalent_vector_with<T>;
};

template <typename S, typename M, typename T, typename Op>
concept unqualified_extended_mreduce = cpo_invocable<reduce_t, T, Op> &&
    (!simd_type<S> ||
        equivalent_vector_with<S, cpo_result_t<reduce_t, T, Op>>) &&
    requires {
        {
            reduce(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>(), internal::declarg<Op>())
        } -> equivalent_vector_with<cpo_result_t<reduce_t, T, Op>>;
    };

template <>
struct extended_impl<reduce_t> {
public:
    template <extended_vector T, reduction_operator_for<T> Op>
    requires unqualified_extended_reduce<T, Op>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val, Op&& func) {
        return reduce(__DPL forward<T>(val), __DPL forward<Op>(func));
    }

    template <simd_vector S, exact_mask_for<S> M, common_vector_with<S> T,
        broadcastable_to<T> V, reduction_operator_for<T> Op>
    requires (extended_vector<S> || extended_vector<T> || extended_mask<M>) &&
        unqualified_extended_mreduce<S, M, T, Op>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, T&& val, Op&& func) {
        return reduce(__DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val), __DPL forward<Op>(func));
    }

    template <simd_vector S, const_mask_for<S> M, common_vector_with<S> T,
        broadcastable_to<T> V, reduction_operator_for<T> Op>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_mreduce<S, launder_cmask_t<S, M>, T, Op>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M cmask, T&& val, Op&& func) {
        return reduce(__DPL forward<S>(src), dx::to_const_mask<S>(cmask),
            __DPL forward<T>(val), __DPL forward<Op>(func));
    }

    template <simd_vector T, reduction_operator_for<T> Op,
        result_mask_for<reduce_t, T, Op> M>
    requires (extended_vector<T> || extended_mask<M>) &&
        unqualified_extended_mreduce<dx::zero_t, M, T, Op>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, T&& val, Op&& func) {
        return reduce(zero, __DPL forward<M>(mask), __DPL forward<T>(val),
            __DPL forward<Op>(func));
    }

    template <extended_vector T, reduction_operator_for<T> Op,
        result_cmask_for<reduce_t, T, Op> M>
    requires unqualified_extended_mreduce<dx::zero_t,
        launder_cmask_t<cpo_result_t<reduce_t, T, Op>, M>, T, Op>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M cmask, T&& val, Op&& func) {
        return reduce(zero,
            dx::to_const_mask<cpo_result_t<reduce_t, T, Op>>(cmask),
            __DPL forward<T>(val), __DPL forward<Op>(func));
    }
};

template <>
struct fallback_impl<reduce_t> {
public:
    template <simd_vector T, reduction_operator_for<T> Op>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val, Op&& func) {
        return internal::reduction(
            __DPL forward<T>(val), __DPL forward<Op>(func));
    }

    template <simd_vector S, exact_mask_for<S> M, simd_vector T, typename Op>
    requires cpo_invocable<compress_t, M, T> && cpo_invocable<popcount_t, M> &&
        reduction_operator_for<Op, cpo_result_t<compress_t, M, T>> &&
        equivalent_vector_with<S, cpo_result_t<compress_t, M, T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        S&& src, M&& mask, T&& val, Op&& func) {
        auto const pop = dx::popcount(mask);
        auto compressed =
            dx::compress(__DPL forward<M>(mask), __DPL forward<T>(val));
        return dx::select(dx::lane_index<S>() < pop,
            internal::reduction(
                pop, __DPL move(compressed), __DPL forward<Op>(func)),
            __DPL forward<S>(src));
    }

    template <simd_vector S, const_mask_for<S> M, simd_vector T, typename Op>
    requires cpo_invocable<compress_t, M, T> &&
        reduction_operator_for<Op, cpo_result_t<compress_t, M, T>> &&
        equivalent_vector_with<S, cpo_result_t<compress_t, M, T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        S&& src, M cmask, T&& val, Op&& func) {
        constexpr auto pop = imm<dx::popcount(dx::to_const_mask<S>(cmask))>;
        constexpr auto V = (1zu << pop()) - 1;
        constexpr make_const_mask_t<T, V> mask{};
        return dx::select(mask,
            internal::reduction(pop,
                dx::compress(cmask, __DPL forward<T>(val)),
                __DPL forward<Op>(func)),
            __DPL forward<S>(src));
    }

    template <simd_vector T, exact_mask_for<T> M, typename Op>
    requires cpo_invocable<compress_t, M, T> && cpo_invocable<popcount_t, M> &&
        reduction_operator_for<Op, cpo_result_t<compress_t, M, T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        dx::zero_t zero, M&& mask, T&& val, Op&& func) {
        auto const pop = dx::popcount(mask);
        auto compressed =
            dx::compress(__DPL forward<M>(mask), __DPL forward<T>(val));
        return dx::select(dx::lane_index<T>() < pop,
            internal::reduction(
                pop, __DPL move(compressed), __DPL forward<Op>(func)),
            zero);
    }

    template <simd_vector T, typename Op, const_mask_for<T> M>
    requires cpo_invocable<compress_t, M, T> &&
        reduction_operator_for<Op, cpo_result_t<compress_t, M, T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        dx::zero_t zero, M cmask, T&& val, Op&& func) {
        constexpr auto pop = imm<dx::popcount(dx::to_const_mask<T>(cmask))>;
        constexpr auto V = (1zu << pop()) - 1;
        constexpr make_const_mask_t<T, V> mask{};
        return dx::select(mask,
            internal::reduction(pop,
                dx::compress(cmask, __DPL forward<T>(val)),
                __DPL forward<Op>(func)),
            zero);
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::reduce_t reduce{};
} // namespace cpo
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
