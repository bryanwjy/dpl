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
#  include "dpl/core/operations/compare/min.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::internal {
void hmin(...) noexcept = delete;

struct hmin_t :
    private inclusive_scan_base<hmin_t>,
    private maskable_transform_base<hmin_t> {
    using operation_base<hmin_t>::operator();
    using maskable_transform_base<hmin_t>::operator();
};

template <>
struct operation_signature<hmin_t> {
    template <simd_vector T>
    static consteval void operator()(T&&) noexcept {}
};

template <typename T>
concept unqualified_canonical_hmin = requires {
    { hmin(internal::abi<T>, internal::declarg<T>()) } -> canonical_vector;
};

template <typename S, typename M, typename T>
concept unqualified_canonical_mhmin = cpo_invocable<hmin_t, T> &&
    (!simd_type<S> || equivalent_vector_with<S, cpo_result_t<hmin_t, T>>) &&
    requires {
        {
            hmin(internal::abi<conditional_t<simd_type<S>, S, T>>,
                internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>())
        } -> equivalent_vector_with<cpo_result_t<hmin_t, T>>;
    };

template <>
struct canonical_impl<hmin_t> {
private:
    template <typename T>
    using mask_t DPL_NODEBUG =
        basic_mask<simd_element_type_t<T>, simd_abi_type_t<T>>;

public:
    template <canonical_vector T>
    requires unqualified_canonical_hmin<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val) noexcept {
        return hmin(internal::abi<T>, val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mhmin<type_identity_t<T>, mask_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, mask_t<T> mask, T val) noexcept {
        return hmin(internal::abi<T>, src, mask, val);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mhmin<type_identity_t<T>,
        launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, M cmask, T val) noexcept {
        return hmin(internal::abi<T>, src, dx::to_const_mask<T>(cmask), val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mhmin<dx::zero_t, mask_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, mask_t<T> mask, T val) noexcept {
        return hmin(internal::abi<T>, zero, mask, val);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mhmin<dx::zero_t, launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(dx::zero_t zero, M cmask, T val) noexcept {
        return hmin(internal::abi<T>, zero, dx::to_const_mask<T>(cmask), val);
    }
};

template <typename T>
concept unqualified_extended_hmin = requires {
    { hmin(internal::declarg<T>()) } -> equivalent_vector_with<T>;
};

template <typename S, typename M, typename T>
concept unqualified_extended_mhmin = cpo_invocable<hmin_t, T> &&
    (!simd_type<S> || equivalent_vector_with<S, cpo_result_t<hmin_t, T>>) &&
    requires {
        {
            hmin(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>())
        } -> equivalent_vector_with<cpo_result_t<hmin_t, T>>;
    };

template <>
struct extended_impl<hmin_t> {
public:
    template <extended_vector T>
    requires unqualified_extended_hmin<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val) {
        return hmin(__DPL forward<T>(val));
    }

    template <simd_vector S, exact_mask_for<S> M, common_vector_with<S> T,
        broadcastable_to<T> V>
    requires (extended_vector<S> || extended_vector<T> || extended_mask<M>) &&
        unqualified_extended_mhmin<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, T&& val) {
        return hmin(__DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val));
    }

    template <simd_vector S, const_mask_for<S> M, common_vector_with<S> T,
        broadcastable_to<T> V>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_mhmin<S, launder_cmask_t<S, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M cmask, T&& val) {
        return hmin(__DPL forward<S>(src), dx::to_const_mask<S>(cmask),
            __DPL forward<T>(val));
    }

    template <simd_vector T, result_mask_for<hmin_t, T> M>
    requires (extended_vector<T> || extended_mask<M>) &&
        unqualified_extended_mhmin<dx::zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M&& mask, T&& val) {
        return hmin(zero, __DPL forward<M>(mask), __DPL forward<T>(val));
    }

    template <fixed_width_vector T, result_cmask_for<hmin_t, T> M>
    requires extended_vector<T> &&
        unqualified_extended_mhmin<dx::zero_t,
            launder_cmask_t<cpo_result_t<hmin_t, T>, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M cmask, T&& val) {
        return hmin(zero, dx::to_const_mask<cpo_result_t<hmin_t, T>>(cmask),
            __DPL forward<T>(val));
    }
};

template <>
struct fallback_impl<hmin_t> {
public:
    template <simd_vector T>
    requires scan_operator_for<min_t, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val) noexcept(canonical_vector<T>) {
        return internal::reduction( __DPL forward<T>(val), dx::min);
    }

    template <simd_vector S, simd_mask M, simd_vector T>
    requires cpo_invocable<reduce_t, S, M, T, min_t>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        S&& src, M&& mask, T&& val) noexcept(canonical_vector<S> &&
        canonical_mask<M> && canonical_vector<T>) {
        auto const pop = dx::popcount(mask);
        return dx::select(dx::lane_index<T>() < pop,
            internal::reduction(
                dx::compress(mask, __DPL forward<T>(val)), dx::min),
            __DPL forward<S>(src));
    }

    template <simd_vector S, const_mask_for<S> M, simd_vector T>
    requires cpo_invocable<reduce_t, S, M, T, min_t>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(S&& src, M cmask,
        T&& val) noexcept(canonical_vector<S> && canonical_vector<T>) {
        return dx::reduce(
            __DPL forward<S>(src), cmask, __DPL forward<T>(val), dx::min);
    }

    template <simd_mask M, simd_vector T>
    requires cpo_invocable<reduce_t, dx::zero_t, M, T, min_t>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(dx::zero_t zero, M mask,
        T&& val) noexcept(canonical_mask<M> && canonical_vector<T>) {
        auto const pop = dx::popcount(mask);
        return dx::select(dx::lane_index<T>() < pop,
            internal::reduction(
                dx::compress(mask, __DPL forward<T>(val)), dx::min),
            zero);
    }

    template <simd_vector T, result_cmask_for<reduce_t, T, min_t> M>
    requires cpo_invocable<reduce_t, dx::zero_t, M, T, min_t>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        dx::zero_t zero, M cmask, T&& val) noexcept(canonical_vector<T>) {
        return dx::reduce(zero, cmask, __DPL forward<T>(val), dx::min);
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::hmin_t hmin{};
}
} // namespace datapar

__DPL_DEFAULT_NAMESPACE_END
