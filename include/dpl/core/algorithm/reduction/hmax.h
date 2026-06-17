// Copyright 2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/algorithm/compress.h"
#include "dpl/core/algorithm/reduction/common.h"
#include "dpl/core/algorithm/reduction/reduce.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/dispatch/maskable/transform.h"
#  include "dpl/core/dispatch/operation/algorithm.h"
#  include "dpl/core/operations/compare/max.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::internal {
void hmax(...) noexcept = delete;

struct hmax_t :
    private inclusive_scan_base<hmax_t>,
    private maskable_transform_base<hmax_t> {
    using operation_base<hmax_t>::operator();
    using maskable_transform_base<hmax_t>::operator();
};

template <>
struct operation_signature<hmax_t> {
    template <simd_vector T>
    static consteval void operator()(T&&) noexcept {}
};

template <typename T>
concept unqualified_canonical_hmax = requires {
    { hmax(internal::abi<T>, internal::declarg<T>()) } -> canonical_vector;
};

template <typename S, typename M, typename T>
concept unqualified_canonical_mhmax =
    (!simd_type<S> || equivalent_vector_with<S, cpo_result_t<hmax_t, T>>) &&
    requires {
        {
            hmax(internal::abi<conditional_t<simd_type<S>, S, T>>,
                internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>())
        } -> equivalent_vector_with<cpo_result_t<hmax_t, T>>;
    };

template <>
struct canonical_impl<hmax_t> {
private:
    template <typename T>
    using source_t DPL_NODEBUG = canonical_type_t<cpo_result_t<hmax_t, T>>;

    template <typename T>
    using mask_t DPL_NODEBUG = basic_mask<simd_element_type_t<source_t<T>>,
        simd_abi_type_t<source_t<T>>>;

    template <typename T>
    using imask_t DPL_NODEBUG =
        mask_value_t<simd_abi_traits<source_t<T>>::size>;

    template <typename T, imask_t<T> M>
    using cmask_t DPL_NODEBUG =
        const_mask<simd_abi_traits<source_t<T>>::size, M>;

public:
    template <canonical_vector T>
    requires unqualified_canonical_hmax<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val) noexcept {
        return hmax(internal::abi<T>, val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mhmax<source_t<T>, mask_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        source_t<T> src, mask_t<T> mask, T val) noexcept {
        return hmax(internal::abi<T>, src, mask, val);
    }

    template <canonical_vector T, imask_t<T> M>
    requires unqualified_canonical_mhmax<source_t<T>, cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        source_t<T> src, cmask_t<T, M> cmask, T val) noexcept {
        return hmax(internal::abi<T>, src, cmask, val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mhmax<dx::zero_t, mask_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, mask_t<T> mask, T val) noexcept {
        return hmax(internal::abi<T>, zero, mask, val);
    }

    template <canonical_vector T, imask_t<T> M>
    requires unqualified_canonical_mhmax<dx::zero_t, cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, cmask_t<T, M> cmask, T val) noexcept {
        return hmax(internal::abi<T>, zero, cmask, val);
    }
};

template <typename T>
concept unqualified_extended_hmax = requires {
    { hmax(internal::declarg<T>()) } -> equivalent_vector_with<T>;
};

template <typename S, typename M, typename T>
concept unqualified_extended_mhmax =
    (!simd_type<S> || equivalent_vector_with<S, cpo_result_t<hmax_t, T>>) &&
    requires {
        {
            hmax(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>())
        } -> equivalent_vector_with<cpo_result_t<hmax_t, T>>;
    };

template <>
struct extended_impl<hmax_t> {
private:
    template <typename S>
    using simask_t DPL_NODEBUG = mask_value_t<simd_abi_traits<S>::size>;

    template <typename S, simask_t<S> M>
    using scmask_t DPL_NODEBUG = const_mask<simd_abi_traits<S>::size, M>;

    template <typename T>
    using mask_t DPL_NODEBUG =
        basic_mask<simd_element_type_t<cpo_result_t<hmax_t, T>>,
            simd_abi_type_t<cpo_result_t<hmax_t, T>>>;

    template <typename T>
    using imask_t DPL_NODEBUG = simask_t<cpo_result_t<hmax_t, T>>;

    template <typename T, imask_t<T> M>
    using cmask_t DPL_NODEBUG = scmask_t<cpo_result_t<hmax_t, T>, M>;

public:
    template <extended_vector T>
    requires unqualified_extended_hmax<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val) {
        return hmax(__DPL forward<T>(val));
    }

    template <simd_vector S, exact_mask_for<S> M, common_vector_with<S> T,
        broadcastable_to<T> V>
    requires (extended_vector<S> || extended_vector<T> || extended_mask<M>) &&
        unqualified_extended_mhmax<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, T&& val) {
        return hmax(__DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val));
    }

    template <simd_vector S, simask_t<S> M, common_vector_with<S> T,
        broadcastable_to<T> V>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_mhmax<S, scmask_t<S, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, scmask_t<S, M> cmask, T&& val) {
        return hmax(__DPL forward<S>(src), cmask, __DPL forward<T>(val));
    }

    template <simd_vector T, common_mask_with<mask_t<T>> M>
    requires (extended_vector<T> || extended_mask<M>) &&
        unqualified_extended_mhmax<dx::zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M&& mask, T&& val) {
        return hmax(zero, __DPL forward<M>(mask), __DPL forward<T>(val));
    }

    template <fixed_width_vector T, imask_t<T> M>
    requires extended_vector<T> &&
        unqualified_extended_mhmax<dx::zero_t, cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, cmask_t<T, M> cmask, T&& val) {
        return hmax(zero, cmask, __DPL forward<T>(val));
    }
};

template <>
struct fallback_impl<hmax_t> {
private:
    template <typename S>
    using simask_t DPL_NODEBUG = mask_value_t<simd_abi_traits<S>::size>;

    template <typename S, simask_t<S> M>
    using scmask_t DPL_NODEBUG = const_mask<simd_abi_traits<S>::size, M>;

public:
    template <simd_vector T>
    requires scan_operator_for<max_t, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val) noexcept(canonical_vector<T>) {
        return internal::reduction( __DPL forward<T>(val), dx::max);
    }

    template <simd_vector S, simd_mask M, simd_vector T>
    requires cpo_invocable<reduce_t, S, M, T, max_t>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        S&& src, M&& mask, T&& val) noexcept(canonical_vector<S> &&
        canonical_mask<M> && canonical_vector<T>) {
        auto const pop = dx::popcount(mask);
        return dx::select(dx::lane_index<T>() < pop,
            internal::reduction(
                dx::compress(mask, __DPL forward<T>(val)), dx::max),
            __DPL forward<S>(src));
    }

    template <simd_vector S, simask_t<S> M, simd_vector T>
    requires cpo_invocable<reduce_t, S, scmask_t<S, M>, T, max_t>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(S&& src,
        scmask_t<S, M> cmask,
        T&& val) noexcept(canonical_vector<S> && canonical_vector<T>) {
        return dx::reduce(
            __DPL forward<S>(src), cmask, __DPL forward<T>(val), dx::max);
    }

    template <simd_mask M, simd_vector T>
    requires cpo_invocable<reduce_t, dx::zero_t, M, T, max_t>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(dx::zero_t zero, M mask,
        T&& val) noexcept(canonical_mask<M> && canonical_vector<T>) {
        auto const pop = dx::popcount(mask);
        return dx::select(dx::lane_index<T>() < pop,
            internal::reduction(
                dx::compress(mask, __DPL forward<T>(val)), dx::max),
            zero);
    }

    template <simd_vector T, simask_t<T> M>
    requires cpo_invocable<reduce_t, dx::zero_t, scmask_t<T, M>, T, max_t>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(dx::zero_t zero,
        scmask_t<T, M> cmask, T&& val) noexcept(canonical_vector<T>) {
        return dx::reduce(zero, cmask, __DPL forward<T>(val), dx::max);
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::hmax_t hmax{};
}
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
