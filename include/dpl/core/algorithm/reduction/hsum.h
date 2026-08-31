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
void hsum(...) noexcept = delete;

struct hsum_t : public reduction_base<hsum_t> {
    using operation_base<hsum_t>::operator();
};

template <>
struct operation_signature<hsum_t> {
    template <simd_vector T>
    static consteval void operator()(T&&) noexcept {}

    template <simd_vector T, exact_mask_for<T> M>
    static consteval void operator()(T&&, M&&) noexcept {}

    template <simd_vector T, const_mask_for<T> M>
    static consteval void operator()(T&&, M&&) noexcept {}
};

template <typename T>
concept unqualified_canonical_hsum = requires {
    {
        hsum(internal::abi<T>, internal::declarg<T>())
    } -> same_as<simd_element_type_t<T>>;
};

template <typename M, typename T>
concept unqualified_canonical_mhsum = cpo_invocable<hsum_t, T> && requires {
    {
        hsum(internal::abi<T>, internal::declarg<T>(), internal::declarg<M>())
    } -> same_as<simd_element_type_t<T>>;
};

template <>
struct canonical_impl<hsum_t> {
public:
    template <canonical_vector T>
    requires unqualified_canonical_hsum<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr simd_element_type_t<T> operator()(T val) noexcept {
        return hsum(internal::abi<T>, val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mhsum<T, simd_mask_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr simd_element_type_t<T> operator()(
        T val, simd_mask_type_t<T> mask) noexcept {
        return hsum(internal::abi<T>, val, mask);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mhsum<T, launder_cmask_t<T, M>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr simd_element_type_t<T> operator()(T val, M mask) noexcept {
        return hsum(internal::abi<T>, val, dx::to_const_mask<T>(mask));
    }
};

template <typename T>
concept unqualified_extended_hsum = requires {
    { hsum(internal::declarg<T>()) } -> convertible_to<simd_element_type_t<T>>;
};

template <typename T, typename M>
concept unqualified_extended_mhsum = cpo_invocable<hsum_t, T> && requires {
    {
        hsum(internal::declarg<T>(), internal::declarg<M>())
    } -> convertible_to<simd_element_type_t<T>>;
};

template <>
struct extended_impl<hsum_t> {
public:
    template <extended_vector T>
    requires unqualified_extended_hsum<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val) {
        return hsum(__DPL forward<T>(val));
    }

    template <simd_vector T, exact_mask_for<T> M>
    requires (extended_vector<T> || extended_mask<M>) &&
        unqualified_extended_mhsum<T, M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val, M&& mask) {
        return hsum(__DPL forward<T>(val), __DPL forward<M>(mask));
    }

    template <extended_vector T, const_mask_for<T> M>
    requires unqualified_extended_mhsum<T, launder_cmask_t<T, M>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val, M mask) {
        return hsum(__DPL forward<T>(val), dx::to_const_mask<T>(mask));
    }
};

template <>
struct fallback_impl<hsum_t> {
public:
    template <simd_vector T>
    requires cpo_invocable<reduce_t, T, add_t>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr simd_element_type_t<T> operator()(T&& val) noexcept(
        canonical_vector<T>) {
        return dx::reduce(__DPL forward<T>(val), dx::add);
    }

    template <simd_vector T, exact_mask_for<T> M>
    requires cpo_invocable<select_t, M, T, dx::zero_t> &&
        cpo_invocable<hsum_t, cpo_result_t<select_t, M, T, dx::zero_t>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr simd_element_type_t<T>
        DPL_VECTORCALL operator()(T&& val, M&& mask) noexcept(
            canonical_mask<M> && canonical_vector<T>) {
        using E = simd_element_type_t<T>;
        return hsum_t::operator()(dx::select(
            __DPL forward<M>(mask), __DPL forward<T>(val), dx::zero));
    }

    template <simd_vector T, const_mask_for<T> M>
    requires cpo_invocable<select_t, M, T, dx::zero_t> &&
        cpo_invocable<hsum_t, cpo_result_t<select_t, M, T, dx::zero_t>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr simd_element_type_t<T>
        DPL_VECTORCALL operator()(T&& val, M mask) noexcept(
            canonical_vector<T>) {
        using E = simd_element_type_t<T>;
        using Arg = cpo_result_t<select_t, M, T, dx::zero_t>;
        return hsum_t::operator()(
            dx::select(mask, __DPL forward<T>(val), dx::zero));
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::hsum_t hsum{};
}
} // namespace datapar

__DPL_DEFAULT_NAMESPACE_END
