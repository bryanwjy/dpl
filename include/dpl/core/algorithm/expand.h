// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/algorithm/scan/exscan_sum.h"

#if !DPL_MODULES
#  include "dpl/core/basic/broadcast.h"
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/maskable/transform.h"
#  include "dpl/core/dispatch/operation/algorithm.h"
#  include "dpl/core/operations/permute.h"
#  include "dpl/core/type_traits/simd_abi_type.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void expand(...) noexcept = delete;

struct expand_t : public algorithm_base<expand_t> {
    using operation_base<expand_t>::operator();
};

template <>
struct operation_signature<expand_t> {
    template <simd_vector T, const_mask_for<T> M, equivalent_vector_with<T> S>
    static consteval void operator()(T&&, M, S&&) noexcept {}

    template <simd_vector T, exact_mask_for<T> M, equivalent_vector_with<T> S>
    static consteval void operator()(T&&, M, S&&) noexcept {}

    template <simd_vector T, const_mask_for<T> M, broadcastable_to<T> S>
    static consteval void operator()(T&&, M, S&&) noexcept {}

    template <simd_vector T, exact_mask_for<T> M, broadcastable_to<T> S>
    static consteval void operator()(T&&, M, S&&) noexcept {}
};

template <typename T, typename M, typename S>
concept unqualified_canonical_expand = requires(T val, M mask, S src) {
    { expand(internal::abi<T>, val, mask, src) } -> same_as<T>;
};
template <typename T, typename M>
concept unqualified_canonical_zexpand = requires(T val, M mask) {
    { expand(internal::abi<T>, val, mask, dx::zero) } -> same_as<T>;
};

template <>
struct canonical_impl<expand_t> {
    template <canonical_vector T>
    requires unqualified_canonical_expand<T, simd_mask_type_t<T>,
        type_identity_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        T val, simd_mask_type_t<T> mask, type_identity_t<T> src) noexcept {
        return expand(internal::abi<T>, val, mask, src);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_expand<T, launder_cmask_t<T, M>,
        type_identity_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        T val, M mask, type_identity_t<T> src) noexcept {
        return expand(internal::abi<T>, val, dx::to_const_mask<T>(mask), src);
    }

    template <canonical_vector T>
    requires unqualified_canonical_zexpand<T, simd_mask_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        T val, simd_mask_type_t<T> mask, dx::zero_t zero) noexcept {
        return expand(internal::abi<T>, val, mask, zero);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_zexpand<T, launder_cmask_t<T, M>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val, M mask, dx::zero_t zero) noexcept {
        return expand(internal::abi<T>, val, dx::to_const_mask<T>(mask), zero);
    }
};

template <typename T, typename M, typename S>
concept unqualified_extended_expand = requires {
    {
        expand(internal::declarg<T>(), internal::declarg<M>(),
            internal::declarg<S>())
    } -> equivalent_vector_with<S>;
};

template <typename T, typename M>
concept unqualified_extended_zexpand = requires {
    {
        expand(internal::declarg<T>(), internal::declarg<M>(), dx::zero)
    } -> equivalent_vector_with<T>;
};

template <>
struct extended_impl<expand_t> {
    template <simd_vector T, exact_mask_for<T> M, equivalent_vector_with<T> S>
    requires (extended_vector<T> || extended_mask<T> || extended_vector<S>) &&
        unqualified_extended_expand<T, M, S>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val, M&& mask, S&& src) {
        return expand(__DPL forward<T>(val), __DPL forward<M>(mask),
            __DPL forward<S>(src));
    }

    template <simd_vector T, const_mask_for<T> M, equivalent_vector_with<T> S>
    requires (extended_vector<T> || extended_vector<S>) &&
        unqualified_extended_expand<T, launder_cmask_t<T, M>, S>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val, M mask, S&& src) {
        return expand(__DPL forward<T>(val), dx::to_const_mask<T>(mask),
            __DPL forward<S>(src));
    }

    template <simd_vector T, exact_mask_for<T> M>
    requires (extended_vector<T> || extended_mask<T>) &&
        unqualified_extended_zexpand<T, M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val, M&& mask, dx::zero_t zero) {
        return expand(__DPL forward<T>(val), __DPL forward<M>(mask), zero);
    }

    template <extended_vector T, const_mask_for<T> M>
    requires unqualified_extended_zexpand<T, M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val, M mask, dx::zero_t zero) {
        return expand( __DPL forward<T>(val), dx::to_const_mask<T>(mask), zero);
    }
};

template <>
struct fallback_impl<expand_t> {

private:
    template <size_t W>
    static consteval size_t prefix_sum(
        bitset<W> const& set, size_t idx) noexcept {
        auto result = 0zu;
        for (auto i = 0zu; i < idx; ++i) {
            result += set.test(i);
        }
        return result;
    }

    template <typename M>
    static consteval auto prefix_sum(M cmask) noexcept {
        return []<size_t... Is>(M mask, index_sequence<Is...>) {
            constexpr auto set = static_cast<bitset<M::width>>(mask);
            return index_sequence<fallback_impl::prefix_sum(set, Is)...>{};
        }(cmask, make_index_sequence<M::width>{});
    }

public:
    template <simd_vector T, exact_mask_for<T> M, equivalent_vector_with<T> S>
    requires cpo_invocable<permute_t, S, M, T, signed_canonical_vector_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        T&& val, M&& mask, S&& src) noexcept(canonical_vector<T> &&
        canonical_vector<S> && canonical_mask<M>) {
        using vidx_t = signed_canonical_vector_t<T>;
        auto const vone = dx::broadcast<vidx_t>(dx::one);
        auto const idx =
            dx::exscan_sum(dx::select(mask, vone, dx::zero), dx::zero);
        return dx::permute(__DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val), idx);
    }

    template <simd_vector S, const_mask_for<S> M, simd_vector T>
    requires cpo_invocable<permute_t, S, M, T,
        decltype(fallback_impl::prefix_sum(internal::declarg<M>()))>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(T&& val, M mask,
        S&& src) noexcept(canonical_vector<T> && canonical_vector<S>) {
        constexpr auto cmask = dx::to_const_mask<S>(mask);
        constexpr auto seq = fallback_impl::prefix_sum(cmask);
        return dx::permute(
            __DPL forward<S>(src), cmask, __DPL forward<T>(val), seq);
    }

    template <simd_vector T, exact_mask_for<T> M, broadcastable_to<T> V>
    requires cpo_invocable<expand_t, T, M, canonical_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr cpo_result_t<expand_t, T, M, canonical_type_t<T>>
    operator()(T&& val, M&& mask, V&& src) noexcept(
        canonical_mask<M> && canonical_vector<T>) {
        return expand_t::operator()(__DPL forward<T>(val),
            __DPL forward<M>(mask),
            dx::broadcast<T>(__DPL forward<V>(src)));
    }

    template <simd_vector T, const_mask_for<T> M, broadcastable_to<T> V>
    requires cpo_invocable<expand_t, T, M, canonical_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr cpo_result_t<expand_t, T, M, canonical_type_t<T>>
    operator()(T&& val, M mask, V&& src) noexcept(canonical_vector<T>) {
        return expand_t::operator()( __DPL forward<T>(val), mask,
            dx::broadcast<T>(__DPL forward<V>(src)));
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::expand_t expand{};
}
} // namespace datapar

__DPL_DEFAULT_NAMESPACE_END
