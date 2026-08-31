// Copyright 2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/algorithm/scan/scan.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/dispatch/operation/algorithm.h"
#  include "dpl/core/immediate/const_mask.h"
#  include "dpl/core/operations/compare/min.h"
#  include "dpl/std/utility/forward.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::internal {
void scan_min(...) noexcept = delete;

struct scan_min_t : public inclusive_scan_base<scan_min_t> {
    using operation_base<scan_min_t>::operator();
};

template <>
struct operation_signature<scan_min_t> {
    template <simd_vector T>
    static consteval void operator()(T&&) noexcept {}
    template <simd_vector T, exact_mask_for<T> M>
    static consteval void operator()(T&&, M&&) noexcept {}
    template <simd_vector T, const_mask_for<T> M>
    static consteval void operator()(T&&, M) noexcept {}
};

template <typename T>
concept unqualified_canonical_scan_min = requires {
    { scan_min(internal::abi<T>, internal::declarg<T>()) } -> same_as<T>;
};

template <typename T, typename M>
concept unqualified_canonical_mscan_min =
    cpo_invocable<scan_min_t, T> && requires {
        {
            scan_min(internal::abi<T>, internal::declarg<T>(),
                internal::declarg<M>())
        } -> same_as<T>;
    };

template <>
struct canonical_impl<scan_min_t> {
public:
    template <canonical_vector T>
    requires unqualified_canonical_scan_min<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val) noexcept {
        return scan_min(internal::abi<T>, val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mscan_min<T, simd_mask_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val, simd_mask_type_t<T> mask) noexcept {
        return scan_min(internal::abi<T>, mask, val);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mscan_min<T, launder_cmask_t<T, M>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val, M mask) noexcept {
        return scan_min(internal::abi<T>, dx::to_const_mask<T>(mask), val);
    }
};

template <typename T>
concept unqualified_extended_scan_min = requires {
    { scan_min(internal::declarg<T>()) } -> equivalent_vector_with<T>;
};

template <typename T, typename M>
concept unqualified_extended_mscan_min =
    cpo_invocable<scan_min_t, T> && requires {
        {
            scan_min(internal::declarg<T>(), internal::declarg<M>())
        } -> equivalent_vector_with<T>;
    };

template <>
struct extended_impl<scan_min_t> {
public:
    template <extended_vector T>
    requires unqualified_extended_scan_min<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val) {
        return scan_min(__DPL forward<T>(val));
    }

    template <simd_vector T, exact_mask_for<T> M>
    requires (extended_vector<T> || extended_mask<M>) &&
        unqualified_extended_mscan_min<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val, M&& mask) {
        return scan_min( __DPL forward<T>(val), __DPL forward<M>(mask));
    }

    template <extended_vector T, const_mask_for<T> M>
    requires unqualified_extended_mscan_min<T, launder_cmask_t<T, M>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val, M mask) {
        return scan_min(__DPL forward<T>(val), dx::to_const_mask<T>(mask));
    }
};

template <>
struct fallback_impl<scan_min_t> {
private:
    template <typename E>
    static consteval E identity() noexcept {
        if constexpr (floating_point_like<E>) {
            if constexpr (requires { dx::infinity_v<E>; }) {
                return dx::infinity_v<E>;
            } else {
                return dx::max_value_v<E>;
            }
        } else {
            return dx::max_value_v<E>;
        }
    }

public:
    template <simd_vector T>
    requires cpo_invocable<scan_t, T, min_t>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val) noexcept(canonical_vector<T>) {
        return dx::scan( __DPL forward<T>(val), dx::min);
    }

    template <simd_vector T, exact_mask_for<T> M>
    requires cpo_invocable<select_t, M, T, simd_element_type_t<T>> &&
        cpo_invocable<scan_min_t,
            cpo_result_t<select_t, M, T, simd_element_type_t<T>>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(T&& val, M&& mask) noexcept(
        canonical_vector<T> && canonical_mask<M>) {
        using E = simd_element_type_t<T>;
        return scan_min_t::operator()(dx::select(
            __DPL forward<M>(mask), __DPL forward<T>(val), identity<E>()));
    }

    template <simd_vector T, const_mask_for<T> M>
    requires cpo_invocable<select_t, M, T, simd_element_type_t<T>> &&
        cpo_invocable<scan_min_t,
            cpo_result_t<select_t, M, T, simd_element_type_t<T>>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(T&& val, M mask) noexcept(
        canonical_vector<T>) {
        using E = simd_element_type_t<T>;
        return scan_min_t::operator()(
            dx::select(mask, __DPL forward<T>(val), identity<E>()));
    }
};

} // namespace datapar::internal
namespace datapar {
inline namespace cpo {
inline constexpr internal::scan_min_t scan_min{};
}
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
