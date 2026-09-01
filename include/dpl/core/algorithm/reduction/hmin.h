// Copyright 2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/algorithm/reduction/reduce.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/dispatch/operation/algorithm.h"
#  include "dpl/core/operations/compare/min.h"
#  include "dpl/core/operations/internal/reduction.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::internal {
void hmin(...) noexcept = delete;

struct hmin_t : public reduction_base<hmin_t> {
    using operation_base<hmin_t>::operator();
};

template <>
struct operation_signature<hmin_t> {
    template <simd_vector T>
    static consteval void operator()(T&&) noexcept {}

    template <simd_vector T, exact_mask_for<T> M>
    static consteval void operator()(T&&, M&&) noexcept {}

    template <simd_vector T, const_mask_for<T> M>
    static consteval void operator()(T&&, M&&) noexcept {}
};

template <typename T>
concept unqualified_canonical_hmin = requires {
    {
        hmin(internal::abi<T>, internal::declarg<T>())
    } -> same_as<simd_element_type_t<T>>;
};

template <typename M, typename T>
concept unqualified_canonical_mhmin = cpo_invocable<hmin_t, T> && requires {
    {
        hmin(internal::abi<T>, internal::declarg<T>(), internal::declarg<M>())
    } -> same_as<simd_element_type_t<T>>;
};

template <>
struct canonical_impl<hmin_t> {
public:
    template <canonical_vector T>
    requires unqualified_canonical_hmin<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr simd_element_type_t<T> operator()(T val) noexcept {
        return hmin(internal::abi<T>, val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mhmin<T, simd_mask_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr simd_element_type_t<T> operator()(
        T val, simd_mask_type_t<T> mask) noexcept {
        return hmin(internal::abi<T>, val, mask);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mhmin<T, launder_cmask_t<T, M>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr simd_element_type_t<T> operator()(T val, M mask) noexcept {
        static_assert(dx::any_of(mask));
        return hmin(internal::abi<T>, val, dx::to_const_mask<T>(mask));
    }
};

template <typename T>
concept unqualified_extended_hmin = requires {
    { hmin(internal::declarg<T>()) } -> convertible_to<simd_element_type_t<T>>;
};

template <typename T, typename M>
concept unqualified_extended_mhmin = cpo_invocable<hmin_t, T> && requires {
    {
        hmin(internal::declarg<T>(), internal::declarg<M>())
    } -> convertible_to<simd_element_type_t<T>>;
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

    template <simd_vector T, exact_mask_for<T> M>
    requires (extended_vector<T> || extended_mask<M>) &&
        unqualified_extended_mhmin<T, M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val, M&& mask) {
        return hmin(__DPL forward<T>(val), __DPL forward<M>(mask));
    }

    template <extended_vector T, const_mask_for<T> M>
    requires unqualified_extended_mhmin<T, launder_cmask_t<T, M>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val, M mask) {
        static_assert(dx::any_of(mask));
        return hmin(__DPL forward<T>(val), dx::to_const_mask<T>(mask));
    }
};

template <>
struct fallback_impl<hmin_t> {
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
    requires cpo_invocable<reduce_t, T, min_t>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr simd_element_type_t<T> operator()(T&& val) noexcept(
        canonical_vector<T>) {
        return dx::reduce(__DPL forward<T>(val), dx::min);
    }

    template <simd_vector T, exact_mask_for<T> M>
    requires cpo_invocable<select_t, M, T, simd_element_type_t<T>> &&
        cpo_invocable<hmin_t,
            cpo_result_t<select_t, M, T, simd_element_type_t<T>>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr simd_element_type_t<T>
        DPL_VECTORCALL operator()(T&& val, M&& mask) noexcept(
            canonical_mask<M> && canonical_vector<T>) {
        using E = simd_element_type_t<T>;
        return hmin_t::operator()(dx::select(
            __DPL forward<M>(mask), __DPL forward<T>(val), identity<E>()));
    }

    template <simd_vector T, const_mask_for<T> M>
    requires cpo_invocable<select_t, M, T, simd_element_type_t<T>> &&
        cpo_invocable<hmin_t,
            cpo_result_t<select_t, M, T, simd_element_type_t<T>>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr simd_element_type_t<T>
        DPL_VECTORCALL operator()(T&& val, M mask) noexcept(
            canonical_vector<T>) {
        static_assert(dx::any_of(mask));
        using E = simd_element_type_t<T>;
        return hmin_t::operator()(
            dx::select(mask, __DPL forward<T>(val), identity<E>()));
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::hmin_t hmin{};
}
} // namespace datapar

__DPL_DEFAULT_NAMESPACE_END
