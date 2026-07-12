// Copyright 2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/algorithm/compress.h"
#include "dpl/core/algorithm/scan/common.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/dispatch/maskable/transform.h"
#  include "dpl/core/dispatch/operation/algorithm.h"
#  include "dpl/core/immediate/const_mask.h"
#  include "dpl/core/immediate/immediate.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::internal {
void scan(...) noexcept = delete;

struct scan_t :
    private inclusive_scan_base<scan_t>,
    private maskable_transform_base<scan_t> {
    using operation_base<scan_t>::operator();
    using maskable_transform_base<scan_t>::operator();
};

template <>
struct operation_signature<scan_t> {
    template <simd_vector T, scan_operator_for<T> Op>
    static consteval void operator()(T&&, Op&&) noexcept {}
};

template <typename T, typename Op>
concept unqualified_canonical_scan = requires {
    {
        scan(internal::abi<T>, internal::declarg<T>(), internal::declarg<Op>())
    } -> canonical_vector;
};

template <typename S, typename M, typename T, typename Op>
concept unqualified_canonical_mscan = cpo_invocable<scan_t, T, Op> &&
    (!simd_type<S> || equivalent_vector_with<S, cpo_result_t<scan_t, T, Op>>) &&
    requires {
        {
            scan(internal::abi<conditional_t<simd_type<S>, S, T>>,
                internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>(), internal::declarg<Op>())
        } -> equivalent_vector_with<cpo_result_t<scan_t, T, Op>>;
    };

template <>
struct canonical_impl<scan_t> {
private:
    template <typename T, typename Op>
    using result_t DPL_NODEBUG = canonical_type_t<cpo_result_t<scan_t, T, Op>>;

    template <typename T, typename Op>
    using mask_t DPL_NODEBUG = basic_mask<simd_element_type_t<result_t<T, Op>>,
        simd_abi_type_t<result_t<T, Op>>>;

public:
    template <canonical_vector T, scan_operator_for<T> Op>
    requires unqualified_canonical_scan<T, Op>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val, Op&& func) noexcept {
        return scan(internal::abi<T>, val, __DPL forward<Op>(func));
    }

    template <canonical_vector T, scan_operator_for<T> Op>
    requires unqualified_canonical_mscan<result_t<T, Op>, mask_t<T, Op>, T, Op>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        result_t<T, Op> src, mask_t<T, Op> mask, T val, Op&& func) noexcept {
        return scan(internal::abi<T>, src, mask, val, __DPL forward<Op>(func));
    }

    template <canonical_vector T, scan_operator_for<T> Op,
        result_cmask_for<scan_t, T, Op> M>
    requires unqualified_canonical_mscan<result_t<T, Op>,
        launder_cmask_t<cpo_result_t<scan_t, T, Op>, M>, T, Op>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        result_t<T, Op> src, M cmask, T val, Op&& func) noexcept {
        return scan(internal::abi<T>, src,
            dx::to_const_mask<cpo_result_t<scan_t, T, Op>>(cmask), val,
            __DPL forward<Op>(func));
    }

    template <canonical_vector T, scan_operator_for<T> Op>
    requires unqualified_canonical_mscan<dx::zero_t, mask_t<T, Op>, T, Op>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, mask_t<T, Op> mask, T val, Op&& func) noexcept {
        return scan(
            internal::abi<T>, zero, mask, val, __DPL forward<Op>(func));
    }

    template <canonical_vector T, scan_operator_for<T> Op,
        result_cmask_for<scan_t, T, Op> M>
    requires unqualified_canonical_mscan<dx::zero_t,
        launder_cmask_t<cpo_result_t<scan_t, T, Op>, M>, T, Op>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, M cmask, T val, Op&& func) noexcept {
        return scan(internal::abi<T>, zero,
            dx::to_const_mask<cpo_result_t<scan_t, T, Op>>(cmask), val,
            __DPL forward<Op>(func));
    }
};

template <typename T, typename Op>
concept unqualified_extended_scan = requires {
    {
        scan(internal::declarg<T>(), internal::declarg<Op>())
    } -> equivalent_vector_with<T>;
};

template <typename S, typename M, typename T, typename Op>
concept unqualified_extended_mscan = cpo_invocable<scan_t, T, Op> &&
    (!simd_type<S> || equivalent_vector_with<S, cpo_result_t<scan_t, T, Op>>) &&
    requires {
        {
            scan(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>(), internal::declarg<Op>())
        } -> equivalent_vector_with<cpo_result_t<scan_t, T, Op>>;
    };

template <>
struct extended_impl<scan_t> {
public:
    template <extended_vector T, scan_operator_for<T> Op>
    requires unqualified_extended_scan<T, Op>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val, Op&& func) {
        return scan(__DPL forward<T>(val), __DPL forward<Op>(func));
    }

    template <simd_vector S, exact_mask_for<S> M, common_vector_with<S> T,
        broadcastable_to<T> V, scan_operator_for<T> Op>
    requires (extended_vector<S> || extended_vector<T> || extended_mask<M>) &&
        unqualified_extended_mscan<S, M, T, Op>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, T&& val, Op&& func) {
        return scan(__DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val), __DPL forward<Op>(func));
    }

    template <simd_vector S, const_mask_for<S> M, common_vector_with<S> T,
        broadcastable_to<T> V, scan_operator_for<T> Op>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_mscan<S, launder_cmask_t<S, M>, T, Op>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M cmask, T&& val, Op&& func) {
        return scan(__DPL forward<S>(src), dx::to_const_mask<S>(cmask),
            __DPL forward<T>(val), __DPL forward<Op>(func));
    }

    template <simd_vector T, scan_operator_for<T> Op,
        result_mask_for<scan_t, T, Op> M>
    requires (extended_vector<T> || extended_mask<M>) &&
        unqualified_extended_mscan<dx::zero_t, M, T, Op>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, T&& val, Op&& func) {
        return scan(zero, __DPL forward<M>(mask), __DPL forward<T>(val),
            __DPL forward<Op>(func));
    }

    template <extended_vector T, scan_operator_for<T> Op,
        result_cmask_for<scan_t, T, Op> M>
    requires unqualified_extended_mscan<dx::zero_t,
        launder_cmask_t<cpo_result_t<scan_t, T, Op>, M>, T, Op>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M cmask, T&& val, Op&& func) {
        return scan(zero, dx::to_const_mask<cpo_result_t<scan_t, T, Op>>(cmask),
            __DPL forward<T>(val), __DPL forward<Op>(func));
    }
};

template <>
struct fallback_impl<scan_t> {
public:
    template <simd_vector T, scan_operator_for<T> Op>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val, Op&& func) {
        using type = remove_cvref_t<T>;
        return internal::inclusive_scan<type>(
            __DPL forward<T>(val), __DPL forward<Op>(func));
    }

    template <simd_vector S, exact_mask_for<S> M, simd_vector T, typename Op>
    requires cpo_invocable<compress_t, M, T> && cpo_invocable<popcount_t, M> &&
        scan_operator_for<Op, cpo_result_t<compress_t, M, T>> &&
        equivalent_vector_with<S, cpo_result_t<compress_t, M, T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        S&& src, M&& mask, T&& val, Op&& func) {
        auto const pop = dx::popcount(mask);
        auto compressed =
            dx::compress(__DPL forward<M>(mask), __DPL forward<T>(val));
        return dx::select(dx::lane_index<S>() < pop,
            internal::inclusive_scan(
                pop, __DPL move(compressed), __DPL forward<Op>(func)),
            __DPL forward<S>(src));
    }

    template <simd_vector S, const_mask_for<S> M, simd_vector T, typename Op>
    requires cpo_invocable<compress_t, M, T> &&
        scan_operator_for<Op, cpo_result_t<compress_t, M, T>> &&
        equivalent_vector_with<S, cpo_result_t<compress_t, M, T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        S&& src, M cmask, T&& val, Op&& func) {
        constexpr auto pop = imm<dx::popcount(dx::to_const_mask<S>(cmask))>;
        constexpr auto V = (1zu << pop()) - 1;
        constexpr make_const_mask_t<T, V> mask{};
        return dx::select(mask,
            internal::inclusive_scan(pop,
                dx::compress(cmask, __DPL forward<T>(val)),
                __DPL forward<Op>(func)),
            __DPL forward<S>(src));
    }

    template <simd_vector T, exact_mask_for<T> M, typename Op>
    requires cpo_invocable<compress_t, M, T> && cpo_invocable<popcount_t, M> &&
        scan_operator_for<Op, cpo_result_t<compress_t, M, T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        dx::zero_t zero, M&& mask, T&& val, Op&& func) {
        auto const pop = dx::popcount(mask);
        auto compressed =
            dx::compress(__DPL forward<M>(mask), __DPL forward<T>(val));
        return dx::select(dx::lane_index<T>() < pop,
            internal::inclusive_scan(
                pop, __DPL move(compressed), __DPL forward<Op>(func)),
            zero);
    }

    template <simd_vector T, const_mask_for<T> M, typename Op>
    requires cpo_invocable<compress_t, M, T> &&
        scan_operator_for<Op, cpo_result_t<compress_t, M, T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        dx::zero_t zero, M cmask, T&& val, Op&& func) {
        constexpr auto pop = imm<dx::popcount(dx::to_const_mask<T>(cmask))>;
        constexpr auto V = (1zu << pop()) - 1;
        constexpr make_const_mask_t<T, V> mask{};
        return dx::select(mask,
            internal::inclusive_scan(pop,
                dx::compress(cmask, __DPL forward<T>(val)),
                __DPL forward<Op>(func)),
            zero);
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::scan_t scan{};
}
} // namespace datapar

__DPL_DEFAULT_NAMESPACE_END
