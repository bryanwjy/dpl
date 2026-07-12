// Copyright 2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/algorithm/compress.h"
#include "dpl/core/algorithm/rotate.h"
#include "dpl/core/algorithm/scan/common.h"
#include "dpl/core/algorithm/slide.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/dispatch/maskable/transform.h"
#  include "dpl/core/dispatch/operation/algorithm.h"
#  include "dpl/core/immediate/immediate.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::internal {
void exscan(...) noexcept = delete;

struct exscan_t :
    private exclusive_scan_base<exscan_t>,
    private maskable_transform_base<exscan_t> {
    using operation_base<exscan_t>::operator();
    using maskable_transform_base<exscan_t>::operator();
};

template <>
struct operation_signature<exscan_t> {
    template <simd_vector T, broadcastable_to<T> V, scan_operator_for<T> Op>
    static consteval void operator()(T&&, V&&, Op&&) noexcept {}
};

template <typename T, typename V, typename Op>
concept unqualified_canonical_exscan = requires {
    {
        exscan(internal::abi<T>, internal::declarg<T>(), internal::declarg<V>(),
            internal::declarg<Op>())
    } -> canonical_vector;
};

template <typename S, typename M, typename T, typename V, typename Op>
concept unqualified_canonical_mexscan = cpo_invocable<exscan_t, T, V, Op> &&
    (!simd_type<S> ||
        equivalent_vector_with<S, cpo_result_t<exscan_t, T, V, Op>>) &&
    requires {
        {
            exscan(internal::abi<conditional_t<simd_type<S>, S, T>>,
                internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>(), internal::declarg<V>(),
                internal::declarg<Op>())
        } -> equivalent_vector_with<cpo_result_t<exscan_t, T, V, Op>>;
    };

template <>
struct canonical_impl<exscan_t> {
private:
    template <typename T, typename V, typename Op>
    using result_t DPL_NODEBUG =
        canonical_type_t<cpo_result_t<exscan_t, T, V, Op>>;

    template <typename T, typename V, typename Op>
    using mask_t DPL_NODEBUG =
        basic_mask<simd_element_type_t<result_t<T, V, Op>>,
            simd_abi_type_t<result_t<T, V, Op>>>;

public:
    template <canonical_vector T, broadcastable_to<T> V,
        scan_operator_for<T> Op>
    requires unqualified_canonical_exscan<T, V, Op>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val, V&& init, Op&& func) noexcept {
        return exscan(internal::abi<T>, val, __DPL forward<V>(init),
            __DPL forward<Op>(func));
    }

    template <canonical_vector T, broadcastable_to<T> V,
        scan_operator_for<T> Op>
    requires unqualified_canonical_mexscan<result_t<T, V, Op>, mask_t<T, V, Op>,
        T, V, Op>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(result_t<T, V, Op> src, mask_t<T, V, Op> mask,
        T val, V&& init, Op&& func) noexcept {
        return exscan(internal::abi<T>, src, mask, val, __DPL forward<V>(init),
            __DPL forward<Op>(func));
    }

    template <canonical_vector T, broadcastable_to<T> V,
        scan_operator_for<T> Op, result_cmask_for<exscan_t, T, V, Op> M>
    requires unqualified_canonical_mexscan<result_t<T, V, Op>,
        launder_cmask_t<cpo_result_t<exscan_t, T, V, Op>, M>, T, V, Op>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        result_t<T, V, Op> src, M cmask, T val, V&& init, Op&& func) noexcept {
        return exscan(internal::abi<T>, src,
            dx::to_const_mask<cpo_result_t<exscan_t, T, V, Op>>(cmask), val,
            __DPL forward<V>(init), __DPL forward<Op>(func));
    }

    template <canonical_vector T, broadcastable_to<T> V,
        scan_operator_for<T> Op>
    requires unqualified_canonical_mexscan<dx::zero_t, mask_t<T, V, Op>, T, V,
        Op>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(dx::zero_t zero, mask_t<T, V, Op> mask, T val,
        V&& init, Op&& func) noexcept {
        return exscan(internal::abi<T>, zero, mask, val, __DPL forward<V>(init),
            __DPL forward<Op>(func));
    }

    template <canonical_vector T, broadcastable_to<T> V,
        scan_operator_for<T> Op, result_cmask_for<exscan_t, T, V, Op> M>
    requires unqualified_canonical_mexscan<dx::zero_t,
        launder_cmask_t<cpo_result_t<exscan_t, T, V, Op>, M>, T, V, Op>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, M cmask, T val, V&& init, Op&& func) noexcept {
        return exscan(internal::abi<T>, zero,
            dx::to_const_mask<cpo_result_t<exscan_t, T, V, Op>>(cmask), val,
            __DPL forward<V>(init), __DPL forward<Op>(func));
    }
};

template <typename T, typename V, typename Op>
concept unqualified_extended_exscan = requires {
    {
        exscan(internal::declarg<T>(), internal::declarg<V>(),
            internal::declarg<Op>())
    } -> equivalent_vector_with<T>;
};

template <typename S, typename M, typename T, typename V, typename Op>
concept unqualified_extended_mexscan = cpo_invocable<exscan_t, T, V, Op> &&
    (!simd_type<S> ||
        equivalent_vector_with<S, cpo_result_t<exscan_t, T, V, Op>>) &&
    requires {
        {
            exscan(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>(), internal::declarg<V>(),
                internal::declarg<Op>())
        } -> equivalent_vector_with<cpo_result_t<exscan_t, T, V, Op>>;
    };

template <>
struct extended_impl<exscan_t> {
public:
    template <extended_vector T, broadcastable_to<T> V, scan_operator_for<T> Op>
    requires unqualified_extended_exscan<T, V, Op>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val, V&& init, Op&& func) {
        return exscan(__DPL forward<T>(val), __DPL forward<V>(init),
            __DPL forward<Op>(func));
    }

    template <simd_vector S, exact_mask_for<S> M, common_vector_with<S> T,
        broadcastable_to<T> V, scan_operator_for<T> Op>
    requires (extended_vector<S> || extended_vector<T> || extended_mask<M>) &&
        unqualified_extended_mexscan<S, M, T, V, Op>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        S&& src, M&& mask, T&& val, V&& init, Op&& func) {
        return exscan(__DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val), __DPL forward<V>(init),
            __DPL forward<Op>(func));
    }

    template <simd_vector S, const_mask_for<S> M, common_vector_with<S> T,
        broadcastable_to<T> V, scan_operator_for<T> Op>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_mexscan<S, launder_cmask_t<S, M>, T, V, Op>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        S&& src, M cmask, T&& val, V&& init, Op&& func) {
        return exscan(__DPL forward<S>(src), dx::to_const_mask<S>(cmask),
            __DPL forward<T>(val), __DPL forward<V>(init),
            __DPL forward<Op>(func));
    }

    template <simd_vector T, broadcastable_to<T> V, scan_operator_for<T> Op,
        result_mask_for<exscan_t, T, V, Op> M>
    requires (extended_vector<T> || extended_mask<M>) &&
        unqualified_extended_mexscan<dx::zero_t, M, T, V, Op>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, T&& val, V&& init, Op&& func) {
        return exscan(zero, __DPL forward<M>(mask), __DPL forward<T>(val),
            __DPL forward<V>(init), __DPL forward<Op>(func));
    }

    template <extended_vector T, broadcastable_to<T> V, scan_operator_for<T> Op,
        result_cmask_for<exscan_t, T, V, Op> M>
    requires unqualified_extended_mexscan<dx::zero_t,
        launder_cmask_t<cpo_result_t<exscan_t, T, V, Op>, M>, T, V, Op>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M cmask, T&& val, V&& init, Op&& func) {
        return exscan(zero,
            dx::to_const_mask<cpo_result_t<exscan_t, T, V, Op>>(cmask),
            __DPL forward<T>(val), __DPL forward<V>(init),
            __DPL forward<Op>(func));
    }
};

template <>
struct fallback_impl<exscan_t> {
private:
    template <typename S>
    using simask_t DPL_NODEBUG = mask_value_t<simd_abi_type_t<S>::size>;

    template <typename S, simask_t<S> M>
    using scmask_t DPL_NODEBUG = const_mask<simd_abi_type_t<S>::size, M>;

public:
    template <simd_vector T, broadcastable_to<T> V, scan_operator_for<T> Op>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val, V&& init, Op&& func) noexcept {
        auto slid = dx::slide_right(__DPL forward<T>(val),
            dx::broadcast<T>(__DPL forward<V>(init)), imm<1zu>);
        return internal::inclusive_scan(
            __DPL move(slid), __DPL forward<Op>(func));
    }

    template <simd_vector S, exact_mask_for<S> M, simd_vector T,
        broadcastable_to<T> V, typename Op>
    requires cpo_invocable<compress_t, canonical_type_t<T>, M, T> &&
        cpo_invocable<popcount_t, M> &&
        scan_operator_for<Op,
            cpo_result_t<compress_t, canonical_type_t<T>, M, T>> &&
        same_as<S, cpo_result_t<compress_t, canonical_type_t<T>, M, T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        S&& src, M&& mask, T&& val, V&& init, Op&& func) {
        auto const pop = dx::popcount(mask);
        auto const last = simd_abi_traits<S>::size() - 1;
        auto const compression_mask =
            dx::bwand(dx::lane_index<S>() != last, mask);
        auto compressed =
            dx::compress(dx::broadcast<T>(__DPL forward<V>(init)),
                compression_mask, __DPL forward<T>(val));
        auto scanned = internal::inclusive_scan(pop,
            dx::rotate_right(__DPL move(compressed), imm<1zu>),
            __DPL forward<Op>(func));
        return dx::select(dx::lane_index<S>() < pop, __DPL move(scanned),
            __DPL forward<S>(src));
    }

    template <fixed_width_vector S, simask_t<S> M, simd_vector T,
        broadcastable_to<T> V, typename Op>
    requires cpo_invocable<compress_t, canonical_type_t<T>, scmask_t<S, M>,
                 T> &&
        scan_operator_for<Op,
            cpo_result_t<compress_t, canonical_type_t<T>, scmask_t<S, M>, T>> &&
        same_as<S,
            cpo_result_t<compress_t, canonical_type_t<T>, scmask_t<S, M>, T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        S&& src, scmask_t<S, M> cmask, T&& val, V&& init, Op&& func) {
        constexpr auto C = (1zu << dx::popcount(cmask)) - 1;
        constexpr auto compression_mask = []() consteval {
            using bitset_t = decltype(dx::to_bitset(scmask_t<S, M>()));
            constexpr auto notlast = ~bitset_t() >> 1zu;
            constexpr auto mask = dx::to_bitset(scmask_t<S, M>());
            return make_const_mask_t<S, (mask & notlast)>{};
        }();

        constexpr make_const_mask_t<T, C> output_mask{};
        constexpr auto pop = imm<dx::popcount(cmask)>;
        auto compressed =
            dx::compress(dx::broadcast<T>(__DPL forward<V>(init)),
                compression_mask, __DPL forward<T>(val));
        auto scanned = internal::inclusive_scan(pop,
            dx::rotate_right(__DPL move(compressed), imm<1zu>),
            __DPL forward<Op>(func));
        return dx::select(
            output_mask, __DPL move(scanned), __DPL forward<S>(src));
    }

    template <simd_vector T, exact_mask_for<T> M, broadcastable_to<T> V,
        typename Op>
    requires cpo_invocable<compress_t, canonical_type_t<T>, M, T> &&
        cpo_invocable<popcount_t, M> &&
        scan_operator_for<Op,
            cpo_result_t<compress_t, canonical_type_t<T>, M, T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        dx::zero_t zero, M&& mask, T&& val, V&& init, Op&& func) {
        auto const pop = dx::popcount(mask);
        auto const last = simd_abi_traits<T>::size() - 1;
        auto const compression_mask =
            dx::bwand(dx::lane_index<T>() != last, mask);
        auto compressed =
            dx::compress(dx::broadcast<T>(__DPL forward<V>(init)),
                compression_mask, __DPL forward<T>(val));
        auto scanned = internal::inclusive_scan(pop,
            dx::rotate_right(__DPL move(compressed), imm<1zu>),
            __DPL forward<Op>(func));
        return dx::select(
            dx::lane_index<T>() < pop, __DPL move(scanned), zero);
    }

    template <simd_vector T, simask_t<T> M, broadcastable_to<T> V, typename Op>
    requires cpo_invocable<compress_t, canonical_type_t<T>, scmask_t<T, M>,
                 T> &&
        scan_operator_for<Op,
            cpo_result_t<compress_t, canonical_type_t<T>, scmask_t<T, M>, T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        dx::zero_t zero, scmask_t<T, M> cmask, T&& val, V&& init, Op&& func) {
        constexpr auto C = (1zu << dx::popcount(cmask)) - 1;
        constexpr auto compression_mask = []() consteval {
            using bitset_t = decltype(dx::to_bitset(scmask_t<T, M>()));
            constexpr auto notlast = ~bitset_t() >> 1zu;
            constexpr auto mask = dx::to_bitset(scmask_t<T, M>());
            return make_const_mask_t<T, (mask & notlast)>{};
        }();

        constexpr make_const_mask_t<T, C> output_mask{};
        constexpr auto pop = imm<dx::popcount(cmask)>;

        auto compressed =
            dx::compress(dx::broadcast<T>(__DPL forward<V>(init)),
                compression_mask, __DPL forward<T>(val));
        auto scanned = internal::inclusive_scan(pop,
            dx::rotate_right(__DPL move(compressed), imm<1zu>),
            __DPL forward<Op>(func));
        return dx::select(output_mask, __DPL move(scanned), zero);
    }
};

} // namespace datapar::internal
namespace datapar {
inline namespace cpo {
inline constexpr internal::exscan_t exscan{};
}
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
