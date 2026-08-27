// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/algorithm/compress.h"

#if !DPL_MODULES
#  include "dpl/core/basic/extract.h"
#  include "dpl/core/basic/lane_index.h"
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/dispatch/maskable/base.h"
#  include "dpl/core/dispatch/operation/algorithm.h"
#  include "dpl/core/immediate/const_mask.h"
#  include "dpl/core/immediate/immediate.h"
#  include "dpl/core/operations/bit.h"
#  include "dpl/core/operations/internal/reduction.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void reduce(...) noexcept = delete;
struct reduce_t :
    public reduction_base<reduce_t>,
    public maskable_operation_base<reduce_t> {
    using operation_base<reduce_t>::operator();
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
    } -> same_as<simd_element_type_t<T>>;
};

template <typename M, typename T, typename Op>
concept unqualified_canonical_mreduce =
    cpo_invocable<reduce_t, T, Op> && requires {
        {
            reduce(internal::abi<T>, internal::declarg<M>(),
                internal::declarg<T>(), internal::declarg<Op>())
        } -> same_as<simd_element_type_t<T>>;
    };

template <>
struct canonical_impl<reduce_t> {
public:
    template <canonical_vector T, reduction_operator_for<T> Op>
    requires unqualified_canonical_reduce<T, Op>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr simd_element_type_t<T> operator()(
        T val, Op&& func) noexcept {
        return reduce(internal::abi<T>, val, __DPL forward<Op>(func));
    }

    template <canonical_vector T, reduction_operator_for<T> Op>
    requires unqualified_canonical_mreduce<simd_mask_type_t<T>, T, Op>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr simd_element_type_t<T> operator()(
        simd_mask_type_t<T> mask, T val, Op&& func) noexcept {
        return reduce(internal::abi<T>, mask, val, __DPL forward<Op>(func));
    }

    template <canonical_vector T, const_mask_for<T> M,
        reduction_operator_for<T> Op>
    requires unqualified_canonical_mreduce<launder_cmask_t<T, M>, T, Op>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr simd_element_type_t<T> operator()(
        M mask, T val, Op&& func) noexcept {
        return reduce(internal::abi<T>, dx::to_const_mask<T>(mask), val,
            __DPL forward<Op>(func));
    }
};

template <typename T, typename Op>
concept unqualified_extended_reduce = requires {
    {
        reduce(internal::declarg<T>(), internal::declarg<Op>())
    } -> core_convertible_to<simd_element_type_t<T>>;
};

template <typename M, typename T, typename Op>
concept unqualified_extended_mreduce =
    cpo_invocable<reduce_t, T, Op> && requires {
        {
            reduce(internal::declarg<M>(), internal::declarg<T>(),
                internal::declarg<Op>())
        } -> core_convertible_to<simd_element_type_t<T>>;
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

    template <simd_vector T, exact_mask_for<T> M, reduction_operator_for<T> Op>
    requires (extended_vector<T> || extended_mask<M>) &&
        unqualified_extended_mreduce<M, T, Op>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M&& mask, T&& val, Op&& func) {
        return reduce(__DPL forward<M>(mask), __DPL forward<T>(val),
            __DPL forward<Op>(func));
    }

    template <simd_vector T, const_mask_for<T> M, reduction_operator_for<T> Op>
    requires (extended_vector<T> || extended_mask<M>) &&
        unqualified_extended_mreduce<launder_cmask_t<T, M>, T, Op>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T&& val, Op&& func) {
        return reduce(dx::to_const_mask<T>(mask), __DPL forward<T>(val),
            __DPL forward<Op>(func));
    }
};

template <>
struct fallback_impl<reduce_t> {
public:
    template <simd_vector T, reduction_operator_for<T> Op>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr simd_element_type_t<T> operator()(T&& val, Op&& func) {
        auto const result = internal::reduction(
            __DPL forward<T>(val), __DPL forward<Op>(func));
        return dx::extract(result, imm<0>);
    }

    template <simd_vector T, exact_mask_for<T> M, typename Op>
    requires cpo_invocable<compress_t, M, T> && cpo_invocable<popcount_t, M> &&
        reduction_operator_for<Op, cpo_result_t<compress_t, M, T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr simd_element_type_t<T> operator()(
        M&& mask, T&& val, Op&& func) {
        auto compressed =
            dx::compress(__DPL forward<M>(mask), __DPL forward<T>(val));
        auto const result = internal::reduction(dx::popcount(mask),
            __DPL move(compressed), __DPL forward<Op>(func));
        return dx::extract(result, imm<0>);
    }

    template <simd_vector T, const_mask_for<T> M, typename Op>
    requires cpo_invocable<compress_t, M, T> &&
        reduction_operator_for<Op, cpo_result_t<compress_t, M, T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr simd_element_type_t<T> operator()(
        M mask, T&& val, Op&& func) {
        constexpr auto pop = imm<dx::popcount(dx::to_const_mask<T>(mask))>;
        auto const result =
            internal::reduction(pop, dx::compress(mask, __DPL forward<T>(val)),
                __DPL forward<Op>(func));
        return dx::extract(result, imm<0>);
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::reduce_t reduce{};
} // namespace cpo
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
