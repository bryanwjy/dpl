// Copyright 2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/algorithm/reduction/reduce.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/dispatch/operation/algorithm.h"
#  include "dpl/core/operations/compare/max.h"
#  include "dpl/core/operations/internal/reduction.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::internal {
void hmax(...) noexcept = delete;

struct hmax_t : public reduction_base<hmax_t> {
    using operation_base<hmax_t>::operator();
};

template <>
struct operation_signature<hmax_t> {
    template <simd_vector T>
    static consteval void operator()(T&&) noexcept {}

    template <simd_vector T, exact_mask_for<T> M>
    static consteval void operator()(T&&, M&&) noexcept {}

    template <simd_vector T, const_mask_for<T> M>
    static consteval void operator()(T&&, M&&) noexcept {}
};

template <typename T>
concept unqualified_canonical_hmax = requires {
    {
        hmax(internal::abi<T>, internal::declarg<T>())
    } -> same_as<simd_element_type_t<T>>;
};

template <typename M, typename T>
concept unqualified_canonical_mhmax = cpo_invocable<hmax_t, T> && requires {
    {
        hmax(internal::abi<T>, internal::declarg<T>(), internal::declarg<M>())
    } -> same_as<simd_element_type_t<T>>;
};

template <>
struct canonical_impl<hmax_t> {
public:
    template <canonical_vector T>
    requires unqualified_canonical_hmax<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr simd_element_type_t<T> operator()(T val) noexcept {
        return hmax(internal::abi<T>, val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mhmax<T, simd_mask_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr simd_element_type_t<T> operator()(
        T val, simd_mask_type_t<T> mask) noexcept {
        return hmax(internal::abi<T>, val, mask);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mhmax<T, launder_cmask_t<T, M>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr simd_element_type_t<T> operator()(T val, M mask) noexcept {
        return hmax(internal::abi<T>, val, dx::to_const_mask<T>(mask));
    }
};

template <typename T>
concept unqualified_extended_hmax = requires {
    { hmax(internal::declarg<T>()) } -> convertible_to<simd_element_type_t<T>>;
};

template <typename T, typename M>
concept unqualified_extended_mhmax = cpo_invocable<hmax_t, T> && requires {
    {
        hmax(internal::declarg<T>(), internal::declarg<M>())
    } -> convertible_to<simd_element_type_t<T>>;
};

template <>
struct extended_impl<hmax_t> {
public:
    template <extended_vector T>
    requires unqualified_extended_hmax<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val) {
        return hmax(__DPL forward<T>(val));
    }

    template <simd_vector T, exact_mask_for<T> M>
    requires (extended_vector<T> || extended_mask<M>) &&
        unqualified_extended_mhmax<T, M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val, M&& mask) {
        return hmax(__DPL forward<T>(val), __DPL forward<M>(mask));
    }

    template <extended_vector T, const_mask_for<T> M>
    requires unqualified_extended_mhmax<T, launder_cmask_t<T, M>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val, M mask) {
        return hmax(__DPL forward<T>(val), dx::to_const_mask<T>(mask));
    }
};

template <>
struct fallback_impl<hmax_t> {
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
    template <simd_vector T>
    requires cpo_invocable<reduce_t, T, max_t>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr simd_element_type_t<T> operator()(T&& val) noexcept(
        canonical_vector<T>) {
        return dx::reduce(__DPL forward<T>(val), dx::max);
    }

    template <simd_vector T, exact_mask_for<T> M>
    requires cpo_invocable<select_t, M, T, simd_element_type_t<T>> &&
        cpo_invocable<hmax_t,
            cpo_result_t<select_t, M, T, simd_element_type_t<T>>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr simd_element_type_t<T>
        DPL_VECTORCALL operator()(T&& val, M&& mask) noexcept(
            canonical_mask<M> && canonical_vector<T>) {
        using E = simd_element_type_t<T>;
        return hmax_t::operator()(dx::select(
            __DPL forward<M>(mask), __DPL forward<T>(val), identity<E>()));
    }

    template <simd_vector T, const_mask_for<T> M>
    requires cpo_invocable<select_t, M, T, simd_element_type_t<T>> &&
        cpo_invocable<hmax_t,
            cpo_result_t<select_t, M, T, simd_element_type_t<T>>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr simd_element_type_t<T>
        DPL_VECTORCALL operator()(T&& val, M mask) noexcept(
            canonical_vector<T>) {
        using E = simd_element_type_t<T>;
        return hmax_t::operator()(
            dx::select(mask, __DPL forward<T>(val), identity<E>()));
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::hmax_t hmax{};
}
} // namespace datapar

__DPL_DEFAULT_NAMESPACE_END
