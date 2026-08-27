// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/cpo_invocable.h"
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/operation/algorithm.h"
#  include "dpl/core/operations/permute.h"
#  include "dpl/core/type_traits/details/cpo_result.h"
#  include "dpl/core/type_traits/representation.h"
#  include "dpl/std/type_traits/type_identity.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void lookup(...) noexcept = delete;

struct lookup_t : public algorithm_base<lookup_t> {
    using operation_base<lookup_t>::operator();
};

template <>
struct fallback_impl<lookup_t> {
    template <simd_vector T, vindex_for<T> I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        dx::zero_t zero, T&& arg, I&& idx) noexcept {
        using idx_t = simd_element_type_t<I>;
        using uidx_t = signed_representation_t<idx_t>;
        auto const sizem1 = simd_abi_traits<T>::size() - 1;
        auto const oob = dx::cmpgt(dx::reinterpret<uidx_t>(idx), sizem1);
        return dx::select(oob, zero,
            dx::permute(__DPL forward<T>(arg), dx::bwand(idx, sizem1)));
    }

    template <simd_vector T, vindex_for<T> I, equivalent_vector_with<T> S>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        S&& src, T&& arg, I&& idx) noexcept {
        using idx_t = simd_element_type_t<I>;
        using uidx_t = signed_representation_t<idx_t>;
        auto const sizem1 = simd_abi_traits<T>::size() - 1;
        auto const oob = dx::cmpgt(dx::reinterpret<uidx_t>(idx), sizem1);
        return dx::select(oob, __DPL forward<S>(src),
            dx::permute(__DPL forward<T>(arg), dx::bwand(idx, sizem1)));
    }
};

template <typename S, typename T, typename I>
concept unqualified_canonical_lookup = requires {
    {
        lookup(internal::abi<T>, internal::declarg<S>(), internal::declarg<T>(),
            internal::declarg<I>())
    } -> same_as<T>;
};

template <>
struct canonical_impl<lookup_t> {
public:
    template <canonical_vector T, canonical_vindex_for<T> I>
    requires unqualified_canonical_lookup<dx::zero_t, T, I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(dx::zero_t zero, T val, I idx) noexcept {
        return lookup(internal::abi<T>, zero, val, idx);
    }

    template <canonical_vector T, canonical_vindex_for<T> I>
    requires unqualified_canonical_lookup<T, T, I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, T val, I idx) noexcept {
        return lookup(internal::abi<T>, src, val, idx);
    }
};

template <typename S, typename T, typename I>
concept unqualified_extended_lookup = requires {
    {
        lookup(internal::declarg<S>(), internal::declarg<T>(),
            internal::declarg<I>())
    } -> vector_with_common_abi<simd_abi_type_t<T>>;
};

template <>
struct extended_impl<lookup_t> {
public:
    template <simd_vector T, vindex_for<T> I>
    requires unqualified_extended_lookup<dx::zero_t, T, I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(dx::zero_t zero, T val, I idx) noexcept {
        return lookup(zero, __DPL forward<T>(val), __DPL forward<I>(idx));
    }

    template <simd_vector T, vindex_for<T> I, equivalent_vector_with<T> S>
    requires unqualified_extended_lookup<S, T, I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(S&& src, T&& val, I&& idx) noexcept {
        return lookup(__DPL forward<S>(src), __DPL forward<T>(val),
            __DPL forward<I>(idx));
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::lookup_t lookup{};
} // namespace cpo
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
