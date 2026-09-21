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
    static consteval void operator()(M, T&&, S&&) noexcept {}

    template <simd_vector T, exact_mask_for<T> M, equivalent_vector_with<T> S>
    static consteval void operator()(M&&, T&&, S&&) noexcept {}

    template <simd_vector T, const_mask_for<T> M, broadcastable_to<T> S>
    static consteval void operator()(M, T&&, S&&) noexcept {}

    template <simd_vector T, exact_mask_for<T> M, broadcastable_to<T> S>
    static consteval void operator()(M&&, T&&, S&&) noexcept {}

    template <simd_vector S, const_mask_for<S> M, broadcastable_to<S> T>
    static consteval void operator()(M, T&&, S&&) noexcept {}

    template <simd_vector S, exact_mask_for<S> M, broadcastable_to<S> T>
    static consteval void operator()(M&&, T&&, S&&) noexcept {}
};

template <typename M, typename T, typename S>
concept unqualified_canonical_expand = requires(M mask, T val, S src) {
    { expand(internal::abi<T>, mask, val, src) } -> same_as<T>;
};
template <typename M, typename T>
concept unqualified_canonical_zexpand = requires(M mask, T val) {
    { expand(internal::abi<T>, mask, val, dx::zero) } -> same_as<T>;
};

template <>
struct canonical_impl<expand_t> {
    template <canonical_vector T, same_as<T> S>
    requires unqualified_canonical_expand<simd_mask_type_t<T>, T, S>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        simd_mask_type_t<T> mask, T val, S src) noexcept {
        return expand(internal::abi<T>, mask, val, src);
    }

    template <canonical_vector T, const_mask_for<T> M, same_as<T> S>
    requires unqualified_canonical_expand<launder_cmask_t<T, M>, T, S>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(M mask, T val, S src) noexcept {
        return expand(internal::abi<T>, dx::to_const_mask<T>(mask), val, src);
    }

    template <canonical_vector T>
    requires unqualified_canonical_zexpand<simd_mask_type_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        simd_mask_type_t<T> mask, T val, dx::zero_t zero) noexcept {
        return expand(internal::abi<T>, mask, val, zero);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_zexpand<launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(M mask, T val, dx::zero_t zero) noexcept {
        return expand(internal::abi<T>, dx::to_const_mask<T>(mask), val, zero);
    }
};

template <typename M, typename T, typename S>
concept unqualified_extended_expand = requires {
    {
        expand(internal::declarg<M>(), internal::declarg<T>(),
            internal::declarg<S>())
    } -> equivalent_vector_with<S>;
};

template <typename M, typename T>
concept unqualified_extended_zexpand = requires {
    {
        expand(internal::declarg<M>(), internal::declarg<T>(), dx::zero)
    } -> equivalent_vector_with<T>;
};

template <>
struct extended_impl<expand_t> {
    template <simd_vector T, exact_mask_for<T> M, equivalent_vector_with<T> S>
    requires (extended_vector<T> || extended_mask<T> || extended_vector<S>) &&
        unqualified_extended_expand<M, T, S>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M&& mask, T&& val, S&& src) {
        return expand(__DPL forward<M>(mask),__DPL forward<T>(val),
            __DPL forward<S>(src));
    }

    template <simd_vector T, const_mask_for<T> M, equivalent_vector_with<T> S>
    requires (extended_vector<T> || extended_vector<S>) &&
        unqualified_extended_expand<launder_cmask_t<T, M>, T, S>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T&& val, S&& src) {
        return expand(dx::to_const_mask<T>(mask),__DPL forward<T>(val),
            __DPL forward<S>(src));
    }

    template <simd_vector T, exact_mask_for<T> M>
    requires (extended_vector<T> || extended_mask<T>) &&
        unqualified_extended_zexpand<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M&& mask, T&& val, dx::zero_t zero) {
        return expand(__DPL forward<M>(mask), __DPL forward<T>(val), zero);
    }

    template <extended_vector T, const_mask_for<T> M>
    requires unqualified_extended_zexpand<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T&& val, dx::zero_t zero) {
        return expand(dx::to_const_mask<T>(mask), __DPL forward<T>(val), zero);
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
        M&& mask, T&& val, S&& src) noexcept(canonical_vector<T> &&
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
    static constexpr auto DPL_VECTORCALL operator()(M mask, T&& val,
        S&& src) noexcept(canonical_vector<T> && canonical_vector<S>) {
        constexpr auto cmask = dx::to_const_mask<S>(mask);
        constexpr auto seq = fallback_impl::prefix_sum(cmask);
        return dx::permute(
            __DPL forward<S>(src), cmask, __DPL forward<T>(val), seq);
    }

    template <simd_vector T, exact_mask_for<T> M, broadcastable_to<T> V>
    requires cpo_invocable<expand_t, M, T, canonical_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr cpo_result_t<expand_t, M, T, canonical_type_t<T>>
    operator()(M&& mask, T&& val, V&& src) noexcept(
        canonical_mask<M> && canonical_vector<T>) {
        return expand_t::operator()(__DPL forward<M>(mask),
            __DPL forward<T>(val), dx::broadcast<T>(__DPL forward<V>(src)));
    }

    template <simd_vector T, const_mask_for<T> M, broadcastable_to<T> V>
    requires cpo_invocable<expand_t, M, T, canonical_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr cpo_result_t<expand_t, M, T, canonical_type_t<T>>
    operator()(M mask, T&& val, V&& src) noexcept(canonical_vector<T>) {
        return expand_t::operator()(mask,__DPL forward<T>(val),
            dx::broadcast<T>(__DPL forward<V>(src)));
    }

    template <simd_vector S, exact_mask_for<S> M, broadcastable_to<S> V>
    requires cpo_invocable<expand_t, M, canonical_type_t<S>, S>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr cpo_result_t<expand_t, M, canonical_type_t<S>, S>
    operator()(M&& mask, V&& val, S&& src) noexcept(
        canonical_mask<M> && canonical_vector<S>) {
        return expand_t::operator()(__DPL forward<M>(mask),
            dx::broadcast<S>(__DPL forward<V>(val)), __DPL forward<S>(src));
    }

    template <simd_vector S, const_mask_for<S> M, broadcastable_to<S> V>
    requires cpo_invocable<expand_t, M, canonical_type_t<S>, S>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr cpo_result_t<expand_t, M, canonical_type_t<S>, S>
    operator()(M mask, V&& val, S&& src) noexcept(canonical_vector<S>) {
        return expand_t::operator()(mask,
            dx::broadcast<S>(__DPL forward<V>(val)), __DPL forward<S>(src));
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::expand_t expand{};
}
} // namespace datapar

__DPL_DEFAULT_NAMESPACE_END
