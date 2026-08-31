// Copyright 2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/algorithm/scan/scan.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/dispatch/maskable/fold.h"
#  include "dpl/core/dispatch/operation/algorithm.h"
#  include "dpl/core/immediate/const_mask.h"
#  include "dpl/core/operations/arithmetic/add.h"
#  include "dpl/std/utility/forward.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::internal {
void scan_sum(...) noexcept = delete;

struct scan_sum_t :
    public inclusive_scan_base<scan_sum_t>,
    public maskable_operation_base<scan_sum_t> {
    using operation_base<scan_sum_t>::operator();
};

template <>
struct operation_signature<scan_sum_t> {
    template <simd_vector T>
    static consteval void operator()(T&&) noexcept {}
};

template <typename T>
concept unqualified_canonical_scan_sum = requires {
    { scan_sum(internal::abi<T>, internal::declarg<T>()) } -> same_as<T>;
};

template <typename M, typename T>
concept unqualified_canonical_mscan_sum =
    cpo_invocable<scan_sum_t, T> && requires {
        {
            scan_sum(internal::abi<T>, internal::declarg<M>(),
                internal::declarg<T>())
        } -> same_as<T>;
    };

template <>
struct canonical_impl<scan_sum_t> {
public:
    template <canonical_vector T>
    requires unqualified_canonical_scan_sum<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val) noexcept {
        return scan_sum(internal::abi<T>, val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mscan_sum<simd_mask_type_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(simd_mask_type_t<T> mask, T val) noexcept {
        return scan_sum(internal::abi<T>, mask, val);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mscan_sum<launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(M mask, T val) noexcept {
        return scan_sum(internal::abi<T>, dx::to_const_mask<T>(mask), val);
    }
};

template <typename T>
concept unqualified_extended_scan_sum = requires {
    { scan_sum(internal::declarg<T>()) } -> equivalent_vector_with<T>;
};

template <typename M, typename T>
concept unqualified_extended_mscan_sum =
    cpo_invocable<scan_sum_t, T> && requires {
        {
            scan_sum(internal::declarg<M>(), internal::declarg<T>())
        } -> equivalent_vector_with<T>;
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

    template <simd_vector T, exact_mask_for<T> M>
    requires (extended_vector<T> || extended_mask<M>) &&
        unqualified_extended_mscan_sum<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M&& mask, T&& val) {
        return scan_sum( __DPL forward<M>(mask), __DPL forward<T>(val));
    }

    template <extended_vector T, const_mask_for<T> M>
    requires unqualified_extended_mscan_sum<launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T&& val) {
        return scan_sum(dx::to_const_mask<T>(mask), __DPL forward<T>(val));
    }
};

template <>
struct fallback_impl<scan_sum_t> {
    template <simd_vector T>
    requires cpo_invocable<scan_t, T, add_t>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val) noexcept(canonical_vector<T>) {
        return dx::scan( __DPL forward<T>(val), dx::add);
    }

    template <simd_vector T, exact_mask_for<T> M>
    requires cpo_invocable<select_t, M, T, dx::zero_t> &&
        cpo_invocable<scan_sum_t, cpo_result_t<select_t, M, T, dx::zero_t>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(M&& mask, T&& val) noexcept(
        canonical_vector<T> && canonical_mask<M>) {
        return scan_sum_t::operator()(dx::select(
            __DPL forward<M>(mask), __DPL forward<T>(val), dx::zero));
    }

    template <simd_vector T, const_mask_for<T> M>
    requires cpo_invocable<select_t, M, T, dx::zero_t> &&
        cpo_invocable<scan_sum_t, cpo_result_t<select_t, M, T, dx::zero_t>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(M mask, T&& val) noexcept(
        canonical_vector<T>) {
        return scan_sum_t::operator()(
            dx::select(mask, __DPL forward<T>(val), dx::zero));
    }
};

} // namespace datapar::internal
namespace datapar {
inline namespace cpo {
inline constexpr internal::scan_sum_t scan_sum{};
}
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
