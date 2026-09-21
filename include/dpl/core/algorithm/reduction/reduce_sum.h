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
void reduce_sum(...) noexcept = delete;

struct reduce_sum_t : public reduction_base<reduce_sum_t> {
    using operation_base<reduce_sum_t>::operator();
};

template <>
struct operation_signature<reduce_sum_t> {
    template <simd_vector T>
    static consteval void operator()(T&&) noexcept {}

    template <simd_vector T, exact_mask_for<T> M>
    static consteval void operator()(T&&, M&&) noexcept {}

    template <simd_vector T, const_mask_for<T> M>
    static consteval void operator()(T&&, M) noexcept {}
};

template <typename T>
concept unqualified_canonical_reduce_sum = requires {
    {
        reduce_sum(internal::abi<T>, internal::declarg<T>())
    } -> same_as<simd_element_type_t<T>>;
};

template <typename M, typename T>
concept unqualified_canonical_mreduce_sum =
    cpo_invocable<reduce_sum_t, T> && requires {
        {
            reduce_sum(internal::abi<T>, internal::declarg<T>(),
                internal::declarg<M>())
        } -> same_as<simd_element_type_t<T>>;
    };

template <>
struct canonical_impl<reduce_sum_t> {
public:
    template <canonical_vector T>
    requires unqualified_canonical_reduce_sum<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr simd_element_type_t<T> operator()(T val) noexcept {
        return reduce_sum(internal::abi<T>, val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mreduce_sum<T, simd_mask_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr simd_element_type_t<T> operator()(
        T val, simd_mask_type_t<T> mask) noexcept {
        return reduce_sum(internal::abi<T>, val, mask);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mreduce_sum<T, launder_cmask_t<T, M>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr simd_element_type_t<T> operator()(T val, M mask) noexcept {
        static_assert(dx::any_of(mask));
        return reduce_sum(internal::abi<T>, val, dx::to_const_mask<T>(mask));
    }
};

template <typename T>
concept unqualified_extended_reduce_sum = requires {
    {
        reduce_sum(internal::declarg<T>())
    } -> core_convertible_to<simd_element_type_t<T>>;
};

template <typename T, typename M>
concept unqualified_extended_mreduce_sum =
    cpo_invocable<reduce_sum_t, T> && requires {
        {
            reduce_sum(internal::declarg<T>(), internal::declarg<M>())
        } -> core_convertible_to<simd_element_type_t<T>>;
    };

template <>
struct extended_impl<reduce_sum_t> {
public:
    template <extended_vector T>
    requires unqualified_extended_reduce_sum<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val) {
        return reduce_sum(__DPL forward<T>(val));
    }

    template <simd_vector T, exact_mask_for<T> M>
    requires (extended_vector<T> || extended_mask<M>) &&
        unqualified_extended_mreduce_sum<T, M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val, M&& mask) {
        return reduce_sum(__DPL forward<T>(val), __DPL forward<M>(mask));
    }

    template <extended_vector T, const_mask_for<T> M>
    requires unqualified_extended_mreduce_sum<T, launder_cmask_t<T, M>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val, M mask) {
        static_assert(dx::any_of(mask));
        return reduce_sum(__DPL forward<T>(val), dx::to_const_mask<T>(mask));
    }
};

template <>
struct fallback_impl<reduce_sum_t> {
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
        cpo_invocable<reduce_sum_t, cpo_result_t<select_t, M, T, dx::zero_t>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr simd_element_type_t<T>
        DPL_VECTORCALL operator()(T&& val, M&& mask) noexcept(
            canonical_mask<M> && canonical_vector<T>) {
        using E = simd_element_type_t<T>;
        return reduce_sum_t::operator()(dx::select(
            __DPL forward<M>(mask), __DPL forward<T>(val), dx::zero));
    }

    template <simd_vector T, const_mask_for<T> M>
    requires cpo_invocable<select_t, M, T, dx::zero_t> &&
        cpo_invocable<reduce_sum_t, cpo_result_t<select_t, M, T, dx::zero_t>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr simd_element_type_t<T>
        DPL_VECTORCALL operator()(T&& val, M mask) noexcept(
            canonical_vector<T>) {
        static_assert(dx::any_of(mask));
        using E = simd_element_type_t<T>;
        using Arg = cpo_result_t<select_t, M, T, dx::zero_t>;
        return reduce_sum_t::operator()(
            dx::select(mask, __DPL forward<T>(val), dx::zero));
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::reduce_sum_t reduce_sum{};
}
} // namespace datapar

__DPL_DEFAULT_NAMESPACE_END
