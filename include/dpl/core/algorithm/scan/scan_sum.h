// Copyright 2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/algorithm/compress.h"
#include "dpl/core/algorithm/scan/common.h"
#include "dpl/core/algorithm/scan/scan.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/dispatch/maskable/transform.h"
#  include "dpl/core/dispatch/operation/algorithm.h"
#  include "dpl/core/immediate/const_mask.h"
#  include "dpl/core/operations/arithmetic/add.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::internal {
void scan_sum(...) noexcept = delete;

struct scan_sum_t :
    private inclusive_scan_base<scan_sum_t>,
    private maskable_transform_base<scan_sum_t> {
    using operation_base<scan_sum_t>::operator();
    using maskable_transform_base<scan_sum_t>::operator();
};

template <>
struct operation_signature<scan_sum_t> {
    template <simd_vector T>
    static consteval void operator()(T&&) noexcept {}
};

template <typename T>
concept unqualified_canonical_scan_sum = requires {
    { scan_sum(internal::abi<T>, internal::declarg<T>()) } -> canonical_vector;
};

template <typename S, typename M, typename T>
concept unqualified_canonical_mscan_sum = cpo_invocable<scan_sum_t, T> &&
    (!simd_type<S> || equivalent_vector_with<S, cpo_result_t<scan_sum_t, T>>) &&
    requires {
        {
            scan_sum(internal::abi<cpo_result_t<scan_sum_t, T>>,
                internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>())
        } -> equivalent_vector_with<cpo_result_t<scan_sum_t, T>>;
    };

template <>
struct canonical_impl<scan_sum_t> {
private:
    template <typename T>
    using result_t DPL_NODEBUG = canonical_type_t<cpo_result_t<scan_sum_t, T>>;

    template <typename T>
    using mask_t DPL_NODEBUG = basic_mask<simd_element_type_t<result_t<T>>,
        simd_abi_type_t<result_t<T>>>;

public:
    template <canonical_vector T>
    requires unqualified_canonical_scan_sum<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val) noexcept {
        return scan_sum(internal::abi<T>, val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mscan_sum<result_t<T>, mask_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        result_t<T> src, mask_t<T> mask, T val) noexcept {
        return scan_sum(internal::abi<T>, src, mask, val);
    }

    template <canonical_vector T, result_cmask_for<scan_sum_t, T> M>
    requires unqualified_canonical_mscan_sum<result_t<T>,
        launder_cmask_t<cpo_result_t<scan_sum_t, T>, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(result_t<T> src, M cmask, T val) noexcept {
        return scan_sum(internal::abi<T>, src,
            dx::to_const_mask<cpo_result_t<scan_sum_t, T>>(cmask), val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mscan_sum<dx::zero_t, mask_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, mask_t<T> mask, T val) noexcept {
        return scan_sum(internal::abi<T>, zero, mask, val);
    }

    template <canonical_vector T, result_cmask_for<scan_sum_t, T> M>
    requires unqualified_canonical_mscan_sum<dx::zero_t,
        launder_cmask_t<cpo_result_t<scan_sum_t, T>, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(dx::zero_t zero, M cmask, T val) noexcept {
        return scan_sum(internal::abi<T>, zero,
            dx::to_const_mask<cpo_result_t<scan_sum_t, T>>(cmask), val);
    }
};

template <typename T>
concept unqualified_extended_scan_sum = requires {
    { scan_sum(internal::declarg<T>()) } -> equivalent_vector_with<T>;
};

template <typename S, typename M, typename T>
concept unqualified_extended_mscan_sum = cpo_invocable<scan_sum_t, T> &&
    (!simd_type<S> || equivalent_vector_with<S, cpo_result_t<scan_sum_t, T>>) &&
    requires {
        {
            scan_sum(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>())
        } -> equivalent_vector_with<cpo_result_t<scan_sum_t, T>>;
    };

template <>
struct extended_impl<scan_sum_t> {
public:
    template <extended_vector T>
    requires unqualified_extended_scan_sum<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val) {
        return scan_sum(__DPL forward<T>(val));
    }

    template <simd_vector S, exact_mask_for<S> M, common_vector_with<S> T,
        broadcastable_to<T> V>
    requires (extended_vector<S> || extended_vector<T> || extended_mask<M>) &&
        unqualified_extended_mscan_sum<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, T&& val) {
        return scan_sum(__DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val));
    }

    template <simd_vector S, const_mask_for<S> M, common_vector_with<S> T,
        broadcastable_to<T> V>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_mscan_sum<S, launder_cmask_t<S, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M cmask, T&& val) {
        return scan_sum(__DPL forward<S>(src), dx::to_const_mask<S>(cmask),
            __DPL forward<T>(val));
    }

    template <simd_vector T, result_mask_for<scan_sum_t, T> M>
    requires (extended_vector<T> || extended_mask<M>) &&
        unqualified_extended_mscan_sum<dx::zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M&& mask, T&& val) {
        return scan_sum(zero, __DPL forward<M>(mask), __DPL forward<T>(val));
    }

    template <fixed_width_vector T, result_cmask_for<scan_sum_t, T> M>
    requires extended_vector<T> &&
        unqualified_extended_mscan_sum<dx::zero_t,
            launder_cmask_t<cpo_result_t<scan_sum_t, T>, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M cmask, T&& val) {
        return scan_sum(zero,
            dx::to_const_mask<cpo_result_t<scan_sum_t, T>>(cmask),
            __DPL forward<T>(val));
    }
};

template <>
struct fallback_impl<scan_sum_t> {
public:
    template <simd_vector T>
    requires scan_operator_for<add_t, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val) noexcept(canonical_vector<T>) {
        return internal::inclusive_scan( __DPL forward<T>(val), dx::add);
    }

    template <simd_vector S, simd_mask M, simd_vector T>
    requires cpo_invocable<scan_t, S, M, T, add_t>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        S&& src, M&& mask, T&& val) noexcept(canonical_vector<S> &&
        canonical_mask<M> && canonical_vector<T>) {
        auto const pop = dx::popcount(mask);
        return dx::select(dx::lane_index<T>() < pop,
            internal::inclusive_scan(
                dx::compress(mask, __DPL forward<T>(val)), dx::add),
            __DPL forward<S>(src));
    }

    template <simd_vector S, const_mask_for<S> M, simd_vector T>
    requires cpo_invocable<scan_t, S, launder_cmask_t<S, M>, T, add_t>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(S&& src, M cmask,
        T&& val) noexcept(canonical_vector<S> && canonical_vector<T>) {
        return dx::scan(
            __DPL forward<S>(src), cmask, __DPL forward<T>(val), dx::add);
    }

    template <simd_mask M, simd_vector T>
    requires cpo_invocable<scan_t, dx::zero_t, M, T, add_t>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(dx::zero_t zero, M mask,
        T&& val) noexcept(canonical_mask<M> && canonical_vector<T>) {
        auto const pop = dx::popcount(mask);
        return dx::select(dx::lane_index<T>() < pop,
            internal::inclusive_scan(
                dx::compress(mask, __DPL forward<T>(val)), dx::add),
            zero);
    }

    template <simd_vector T, result_cmask_for<scan_t, T, add_t> M>
    requires cpo_invocable<scan_t, dx::zero_t, M, T, add_t>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        dx::zero_t zero, M cmask, T&& val) noexcept(canonical_vector<T>) {
        return dx::scan(zero, cmask, __DPL forward<T>(val), dx::add);
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::scan_sum_t scan_sum{};
}
} // namespace datapar

__DPL_DEFAULT_NAMESPACE_END
