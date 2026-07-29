// Copyright 2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/algorithm/scan/exscan.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/dispatch/maskable/transform.h"
#  include "dpl/core/dispatch/operation/algorithm.h"
#  include "dpl/core/immediate/const_mask.h"
#  include "dpl/core/operations/compare/max.h"
#  include "dpl/std/utility/forward.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::internal {
void exscan_max(...) noexcept = delete;

struct exscan_max_t :
    public exclusive_scan_base<exscan_max_t>,
    public maskable_transform_base<exscan_max_t> {
    using operation_base<exscan_max_t>::operator();
    using maskable_transform_base<exscan_max_t>::operator();
};

template <>
struct operation_signature<exscan_max_t> {
    template <simd_vector T>
    static consteval void operator()(T&&, broadcastable_to<T> auto&&) noexcept {
    }
};

template <typename T, typename V>
concept unqualified_canonical_exscan_max = requires {
    {
        exscan_max(
            internal::abi<T>, internal::declarg<T>(), internal::declarg<V>())
    } -> canonical_vector;
};

template <typename S, typename M, typename T, typename V>
concept unqualified_canonical_mexscan_max = cpo_invocable<exscan_max_t, T, V> &&
    (!simd_type<S> ||
        equivalent_vector_with<S, cpo_result_t<exscan_max_t, T, V>>) &&
    requires {
        {
            exscan_max(internal::abi<conditional_t<simd_type<S>, S, T>>,
                internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>(), internal::declarg<V>())
        } -> equivalent_vector_with<cpo_result_t<exscan_max_t, T, V>>;
    };

template <>
struct canonical_impl<exscan_max_t> {
private:
    template <typename T, typename V>
    using result_t DPL_NODEBUG =
        canonical_type_t<cpo_result_t<exscan_max_t, T, V>>;

    template <typename T, typename V>
    using mask_t DPL_NODEBUG = simd_mask_type_t<result_t<T, V>>;

public:
    template <canonical_vector T, broadcastable_to<T> V>
    requires unqualified_canonical_exscan_max<T, V>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val, V&& init) noexcept {
        return exscan(internal::abi<T>, val, __DPL forward<V>(init));
    }

    template <canonical_vector T, broadcastable_to<T> V>
    requires unqualified_canonical_mexscan_max<result_t<T, V>, mask_t<T, V>, T,
        V>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        result_t<T, V> src, mask_t<T, V> mask, T val, V&& init) noexcept {
        return exscan(
            internal::abi<T>, src, mask, val, __DPL forward<V>(init));
    }

    template <canonical_vector T, broadcastable_to<T> V,
        result_cmask_for<exscan_max_t, T, V> M>
    requires unqualified_canonical_mexscan_max<result_t<T, V>,
        launder_cmask_t<cpo_result_t<exscan_max_t, T, V>, M>, T, V>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        result_t<T, V> src, M cmask, T val, V&& init) noexcept {
        return exscan(internal::abi<T>, src,
            dx::to_const_mask<cpo_result_t<exscan_max_t, T, V>>(cmask), val,
            __DPL forward<V>(init));
    }

    template <canonical_vector T, broadcastable_to<T> V>
    requires unqualified_canonical_mexscan_max<dx::zero_t, mask_t<T, V>, T, V>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, mask_t<T, V> mask, T val, V&& init) noexcept {
        return exscan(
            internal::abi<T>, zero, mask, val, __DPL forward<V>(init));
    }

    template <canonical_vector T, broadcastable_to<T> V,
        result_cmask_for<exscan_max_t, T, V> M>
    requires unqualified_canonical_mexscan_max<dx::zero_t,
        launder_cmask_t<cpo_result_t<exscan_max_t, T, V>, M>, T, V>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, M cmask, T val, V&& init) noexcept {
        return exscan(internal::abi<T>, zero,
            dx::to_const_mask<cpo_result_t<exscan_max_t, T, V>>(cmask), val,
            __DPL forward<V>(init));
    }
};

template <typename T, typename V>
concept unqualified_extended_exscan_max = requires {
    {
        exscan_max(internal::declarg<T>(), internal::declarg<V>())
    } -> equivalent_vector_with<T>;
};

template <typename S, typename M, typename T, typename V>
concept unqualified_extended_mexscan_max = cpo_invocable<exscan_max_t, T, V> &&
    (!simd_type<S> ||
        equivalent_vector_with<S, cpo_result_t<exscan_max_t, T, V>>) &&
    requires {
        {
            exscan_max(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>(), internal::declarg<V>())
        } -> equivalent_vector_with<cpo_result_t<exscan_max_t, T, V>>;
    };

template <>
struct extended_impl<exscan_max_t> {
public:
    template <extended_vector T, broadcastable_to<T> V>
    requires unqualified_extended_exscan_max<T, V>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val, V&& init) {
        return exscan_max(__DPL forward<T>(val), __DPL forward<V>(init));
    }

    template <simd_vector S, exact_mask_for<S> M, common_vector_with<S> T,
        broadcastable_to<T> V>
    requires (extended_vector<S> || extended_vector<T> || extended_mask<M>) &&
        unqualified_extended_mexscan_max<S, M, T, V>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, T&& val, V&& init) {
        return exscan_max(__DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val), __DPL forward<V>(init));
    }

    template <simd_vector S, const_mask_for<S> M, common_vector_with<S> T,
        broadcastable_to<T> V>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_mexscan_max<S, launder_cmask_t<S, M>, T, V>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M cmask, T&& val, V&& init) {
        return exscan_max(__DPL forward<S>(src), dx::to_const_mask<S>(cmask),
            __DPL forward<T>(val), __DPL forward<V>(init));
    }

    template <simd_vector T, broadcastable_to<T> V,
        result_mask_for<exscan_max_t, T, V> M>
    requires (extended_vector<T> || extended_mask<M>) &&
        unqualified_extended_mexscan_max<dx::zero_t, M, T, V>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, T&& val, V&& init) {
        return exscan_max(zero, __DPL forward<M>(mask), __DPL forward<T>(val),
            __DPL forward<V>(init));
    }

    template <extended_vector T, broadcastable_to<T> V,
        result_cmask_for<exscan_max_t, T, V> M>
    requires unqualified_extended_mexscan_max<dx::zero_t,
        launder_cmask_t<cpo_result_t<exscan_max_t, T, V>, M>, T, V>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M cmask, T&& val, V&& init) {
        return exscan_max(zero,
            dx::to_const_mask<cpo_result_t<exscan_max_t, T, V>>(cmask),
            __DPL forward<T>(val), __DPL forward<V>(init));
    }
};

template <>
struct fallback_impl<exscan_max_t> {
public:
    template <simd_vector T, broadcastable_to<T> V>
    requires cpo_invocable<exscan_t, T, V, max_t>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val, V&& init) noexcept {
        return dx::exscan(
            __DPL forward<T>(val), __DPL forward<V>(init), dx::max);
    }

    template <fixed_width_vector S, exact_mask_for<S> M, simd_vector T,
        broadcastable_to<T> V>
    requires cpo_invocable<exscan_t, S, M, T, V, max_t>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        S&& src, M&& mask, T&& val, V&& init) {
        auto const pop = dx::popcount(mask);
        auto const last = simd_abi_traits<S>::size() - 1;
        auto const idx = dx::lane_index<S>();
        auto const compression_mask = dx::bwand(idx != last, mask);
        auto compressed =
            dx::compress(dx::broadcast<T>(__DPL forward<V>(init)),
                compression_mask, __DPL forward<T>(val));
        auto scanned = internal::inclusive_scan(
            dx::rotate_right(__DPL move(compressed), imm<1zu>), dx::max);
        return dx::select(
            idx < pop, __DPL move(scanned), __DPL forward<S>(src));
    }

    template <simd_vector S, const_mask_for<S> M, simd_vector T,
        broadcastable_to<T> V>
    requires cpo_invocable<exscan_t, S, M, T, V, max_t>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        S&& src, M cmask, T&& val, V&& init) {
        return dx::exscan( __DPL forward<S>(src), cmask, __DPL forward<T>(val),
            __DPL forward<V>(init), dx::max);
    }

    template <fixed_width_vector T, exact_mask_for<T> M, broadcastable_to<T> V>
    requires cpo_invocable<exscan_t, dx::zero_t, M, T, V, max_t>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        dx::zero_t zero, M&& mask, T&& val, V&& init) {
        auto const pop = dx::popcount(mask);
        auto const last = simd_abi_traits<T>::size() - 1;
        auto const idx = dx::lane_index<T>();

        auto const compression_mask = dx::bwand(idx != last, mask);
        auto compressed =
            dx::compress(dx::broadcast<T>(__DPL forward<V>(init)),
                compression_mask, __DPL forward<T>(val));
        auto scanned = internal::inclusive_scan(
            dx::rotate_right(__DPL move(compressed), imm<1zu>), dx::max);
        return dx::select(idx < pop, __DPL move(scanned), zero);
    }

    template <simd_vector T, broadcastable_to<T> V,
        result_cmask_for<exscan_t, T, V, max_t> M>
    requires cpo_invocable<exscan_t, dx::zero_t, M, T, V, max_t>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        dx::zero_t zero, M cmask, T&& val, V&& init) {
        return dx::exscan(zero, cmask, __DPL forward<T>(val),
            __DPL forward<V>(init), dx::max);
    }
};

} // namespace datapar::internal
namespace datapar {
inline namespace cpo {
inline constexpr internal::exscan_max_t exscan_max{};
}
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
