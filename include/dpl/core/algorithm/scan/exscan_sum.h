// Copyright 2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/algorithm/scan/exscan.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/dispatch/operation/algorithm.h"
#  include "dpl/core/immediate/const_mask.h"
#  include "dpl/core/operations/arithmetic/add.h"
#  include "dpl/core/type_traits/rebind_simd.h"
#  include "dpl/std/utility/forward.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::internal {
void exscan_sum(...) noexcept = delete;

struct exscan_sum_t : public exclusive_scan_base<exscan_sum_t> {
    using operation_base<exscan_sum_t>::operator();
};

template <>
struct operation_signature<exscan_sum_t> {
    template <simd_vector T, broadcastable_to<T> V>
    static consteval void operator()(T&&, V&&) noexcept {}
    template <simd_vector T, exact_mask_for<T> M, broadcastable_to<T> V>
    static consteval void operator()(T&&, M&&, V&&) noexcept {}
    template <simd_vector T, const_mask_for<T> M, broadcastable_to<T> V>
    static consteval void operator()(T&&, M, V&&) noexcept {}
};

template <typename T, typename V>
concept unqualified_canonical_exscan_sum = requires {
    {
        exscan_sum(
            internal::abi<T>, internal::declarg<T>(), internal::declarg<V>())
    } -> same_as<T>;
};

template <typename T, typename M, typename V>
concept unqualified_canonical_mexscan_sum =
    cpo_invocable<exscan_sum_t, T, V> && requires {
        {
            exscan_sum(internal::abi<T>, internal::declarg<T>(),
                internal::declarg<M>(), internal::declarg<V>())
        } -> same_as<T>;
    };

template <>
struct canonical_impl<exscan_sum_t> {
    template <canonical_vector T, broadcastable_to<T> V>
    requires unqualified_canonical_exscan_sum<T, V>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val, V&& init) noexcept {
        return exscan_sum(internal::abi<T>, val, __DPL forward<V>(init));
    }

    template <canonical_vector T, broadcastable_to<T> V>
    requires unqualified_canonical_mexscan_sum<T, simd_mask_type_t<T>, V>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        T val, simd_mask_type_t<T> mask, V&& init) noexcept {
        return exscan_sum(internal::abi<T>, val, mask, __DPL forward<V>(init));
    }

    template <canonical_vector T, broadcastable_to<T> V, const_mask_for<T> M>
    requires unqualified_canonical_mexscan_sum<T, launder_cmask_t<T, M>, V>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val, M mask, V&& init) noexcept {
        return exscan_sum(internal::abi<T>, val, dx::to_const_mask<T>(mask),
            __DPL forward<V>(init));
    }
};

template <typename T, typename V>
concept unqualified_extended_exscan_sum = requires {
    {
        exscan_sum(internal::declarg<T>(), internal::declarg<V>())
    } -> equivalent_vector_with<T>;
};

template <typename T, typename M, typename V>
concept unqualified_extended_mexscan_sum =
    cpo_invocable<exscan_sum_t, T, V> && requires {
        {
            exscan_sum(internal::declarg<T>(), internal::declarg<M>(),
                internal::declarg<V>())
        } -> equivalent_vector_with<T>;
    };

template <>
struct extended_impl<exscan_sum_t> {
    template <extended_vector T, broadcastable_to<T> V>
    requires unqualified_extended_exscan_sum<T, V>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val, V&& init) {
        return exscan_sum(__DPL forward<T>(val), __DPL forward<V>(init));
    }

    template <simd_vector T, broadcastable_to<T> V, exact_mask_for<T> M>
    requires (extended_vector<T> || extended_mask<M>) &&
        unqualified_extended_mexscan_sum<T, M, V>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val, M&& mask, V&& init) {
        return exscan_sum( __DPL forward<T>(val), __DPL forward<M>(mask),
            __DPL forward<V>(init));
    }

    template <extended_vector T, const_mask_for<T> M, broadcastable_to<T> V>
    requires unqualified_extended_mexscan_sum<T, launder_cmask_t<T, M>, V>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val, M mask, V&& init) {
        return exscan_sum(__DPL forward<T>(val), dx::to_const_mask<T>(mask),
            __DPL forward<V>(init));
    }
};

template <>
struct fallback_impl<exscan_sum_t> {
public:
    template <simd_vector T, broadcastable_to<T> V>
    requires cpo_invocable<exscan_t, T, V, add_t>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val, V&& init) noexcept(
        canonical_vector<T>) {
        return dx::exscan(
            __DPL forward<T>(val), __DPL forward<V>(init), dx::add);
    }

    template <simd_vector T, exact_mask_for<T> M, broadcastable_to<T> V>
    requires cpo_invocable<select_t, M, T, simd_element_type_t<T>> &&
        cpo_invocable<exscan_sum_t,
            cpo_result_t<select_t, M, T, simd_element_type_t<T>>, V>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(T&& val, M&& mask,
        V&& init) noexcept(canonical_vector<T> && canonical_mask<M>) {
        return exscan_sum_t::operator()(dx::select(__DPL forward<M>(mask),
                                            __DPL forward<T>(val), dx::zero),
            __DPL forward<V>(init));
    }

    template <simd_vector T, const_mask_for<T> M, broadcastable_to<T> V>
    requires cpo_invocable<select_t, M, T, simd_element_type_t<T>> &&
        cpo_invocable<exscan_sum_t,
            cpo_result_t<select_t, M, T, simd_element_type_t<T>>, V>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(T&& val, M mask, V&& init) {
        return exscan_sum_t::operator()(
            dx::select(mask, __DPL forward<T>(val), dx::zero),
            __DPL forward<V>(init));
    }
};

} // namespace datapar::internal
namespace datapar {
inline namespace cpo {
inline constexpr internal::exscan_sum_t exscan_sum{};
}
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
