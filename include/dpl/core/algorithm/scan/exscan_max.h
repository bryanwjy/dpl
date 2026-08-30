// Copyright 2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/algorithm/scan/exscan.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/dispatch/maskable/fold.h"
#  include "dpl/core/dispatch/operation/algorithm.h"
#  include "dpl/core/immediate/const_mask.h"
#  include "dpl/core/operations/compare/max.h"
#  include "dpl/core/type_traits/rebind_simd.h"
#  include "dpl/std/utility/forward.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::internal {
void exscan_max(...) noexcept = delete;

struct exscan_max_t :
    public exclusive_scan_base<exscan_max_t>,
    public maskable_operation_base<exscan_max_t> {
    using operation_base<exscan_max_t>::operator();
};

template <>
struct operation_signature<exscan_max_t> {
    template <simd_vector T, broadcastable_to<T> V>
    static consteval void operator()(T&&, V&&) noexcept {}
};

template <typename T, typename V>
concept unqualified_canonical_exscan_max = requires {
    {
        exscan_max(
            internal::abi<T>, internal::declarg<T>(), internal::declarg<V>())
    } -> same_as<T>;
};

template <typename M, typename T, typename V>
concept unqualified_canonical_mexscan_max =
    cpo_invocable<exscan_max_t, T, V> && requires {
        {
            exscan_max(internal::abi<T>, internal::declarg<M>(),
                internal::declarg<T>(), internal::declarg<V>())
        } -> same_as<T>;
    };

template <>
struct canonical_impl<exscan_max_t> {
public:
    template <canonical_vector T, broadcastable_to<T> V>
    requires unqualified_canonical_exscan_max<T, V>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val, V&& init) noexcept {
        return exscan_max(internal::abi<T>, val, __DPL forward<V>(init));
    }

    template <canonical_vector T, broadcastable_to<T> V>
    requires unqualified_canonical_mexscan_max<simd_mask_type_t<T>, T, V>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        simd_mask_type_t<T> mask, T val, V&& init) noexcept {
        return exscan_max(internal::abi<T>, mask, val, __DPL forward<V>(init));
    }

    template <canonical_vector T, broadcastable_to<T> V, const_mask_for<T> M>
    requires unqualified_canonical_mexscan_max<launder_cmask_t<T, M>, T, V>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(M mask, T val, V&& init) noexcept {
        return exscan_max(internal::abi<T>, dx::to_const_mask<T>(mask), val,
            __DPL forward<V>(init));
    }
};

template <typename T, typename V>
concept unqualified_extended_exscan_max = requires {
    {
        exscan_max(internal::declarg<T>(), internal::declarg<V>())
    } -> equivalent_vector_with<T>;
};

template <typename M, typename T, typename V>
concept unqualified_extended_mexscan_max =
    cpo_invocable<exscan_max_t, T, V> && requires {
        {
            exscan_max(internal::declarg<M>(), internal::declarg<T>(),
                internal::declarg<V>())
        } -> equivalent_vector_with<T>;
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

    template <simd_vector T, broadcastable_to<T> V, exact_mask_for<T> M>
    requires (extended_vector<T> || extended_mask<M>) &&
        unqualified_extended_mexscan_max<M, T, V>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M&& mask, T&& val, V&& init) {
        return exscan_max( __DPL forward<M>(mask), __DPL forward<T>(val),
            __DPL forward<V>(init));
    }

    template <extended_vector T, broadcastable_to<T> V, const_mask_for<T> M>
    requires unqualified_extended_mexscan_max<launder_cmask_t<T, M>, T, V>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T&& val, V&& init) {
        return exscan_max(dx::to_const_mask<T>(mask), __DPL forward<T>(val),
            __DPL forward<V>(init));
    }
};

template <>
struct fallback_impl<exscan_max_t> {
private:
    template <typename E>
    static consteval E identity() noexcept {
        if constexpr (floating_point_like<E>) {
            if constexpr (requires { dx::infinity_v<E>; }) {
                return -dx::infinity_v<E>;
            } else {
                return -dx::max_value_v<E>;
            }
        } else {
            return dx::min_value_v<E>;
        }
    }

public:
    template <simd_vector T, broadcastable_to<T> V>
    requires cpo_invocable<exscan_t, T, V, max_t>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val, V&& init) noexcept(
        canonical_vector<T>) {
        return dx::exscan(
            __DPL forward<T>(val), __DPL forward<V>(init), dx::max);
    }

    template <simd_vector T, exact_mask_for<T> M, broadcastable_to<T> V>
    requires cpo_invocable<select_t, M, T, simd_element_type_t<T>> &&
        cpo_invocable<exscan_max_t,
            cpo_result_t<select_t, M, T, simd_element_type_t<T>>, V>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(M&& mask, T&& val,
        V&& init) noexcept(canonical_vector<T> && canonical_mask<M>) {
        using E = simd_element_type_t<T>;
        return exscan_max_t::operator()(
            dx::select(__DPL forward<M>(mask), __DPL forward<T>(val),
                identity<E>()),
            __DPL forward<V>(init));
    }

    template <simd_vector T, const_mask_for<T> M, broadcastable_to<T> V>
    requires cpo_invocable<select_t, M, T, simd_element_type_t<T>> &&
        cpo_invocable<exscan_max_t,
            cpo_result_t<select_t, M, T, simd_element_type_t<T>>, V>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(M mask, T&& val, V&& init) {
        using E = simd_element_type_t<T>;
        return exscan_max_t::operator()(
            dx::select(mask, __DPL forward<T>(val), identity<E>()),
            __DPL forward<V>(init));
    }
};

} // namespace datapar::internal
namespace datapar {
inline namespace cpo {
inline constexpr internal::exscan_max_t exscan_max{};
}
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
