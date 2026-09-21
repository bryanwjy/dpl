// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/algorithm/shift.h"
#include "dpl/core/algorithm/slide.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/operation/algorithm.h"
#  include "dpl/core/immediate/immediate.h"
#  include "dpl/core/operations/bit.h"
#  include "dpl/core/operations/select.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void splice(...) noexcept = delete;

struct splice_t : public algorithm_base<splice_t> {
    using operation_base<splice_t>::operator();
};

template <>
struct operation_signature<splice_t> {
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
concept unqualified_canonical_splice = requires(M mask, T val, S src) {
    { splice(internal::abi<T>, mask, val, src) } -> same_as<T>;
};
template <typename M, typename T>
concept unqualified_canonical_zsplice = requires(M mask, T val) {
    { splice(internal::abi<T>, mask, val, dx::zero) } -> same_as<T>;
};

template <>
struct canonical_impl<splice_t> {
    template <canonical_vector T, same_as<T> S>
    requires unqualified_canonical_splice<simd_mask_type_t<T>, T, S>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        simd_mask_type_t<T> mask, T val, S src) noexcept {
        return splice(internal::abi<T>, mask, val, src);
    }

    template <canonical_vector T, const_mask_for<T> M, same_as<T> S>
    requires unqualified_canonical_splice<launder_cmask_t<T, M>, T, S>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(M mask, T val, S src) noexcept {
        return splice(internal::abi<T>, dx::to_const_mask<T>(mask), val, src);
    }

    template <canonical_vector T>
    requires unqualified_canonical_zsplice<simd_mask_type_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        simd_mask_type_t<T> mask, T val, dx::zero_t zero) noexcept {
        return splice(internal::abi<T>, mask, val, zero);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_zsplice<launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(M mask, T val, dx::zero_t zero) noexcept {
        return splice(internal::abi<T>, dx::to_const_mask<T>(mask), val, zero);
    }
};

template <typename M, typename T, typename S>
concept unqualified_extended_splice = requires {
    {
        splice(internal::declarg<M>(), internal::declarg<T>(),
            internal::declarg<S>())
    } -> equivalent_vector_with<S>;
};

template <typename M, typename T>
concept unqualified_extended_zsplice = requires {
    {
        splice(internal::declarg<M>(), internal::declarg<T>(), dx::zero)
    } -> equivalent_vector_with<T>;
};

template <>
struct extended_impl<splice_t> {
    template <simd_vector T, exact_mask_for<T> M, equivalent_vector_with<T> S>
    requires (extended_vector<T> || extended_mask<T> || extended_vector<S>) &&
        unqualified_extended_splice<M, T, S>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M&& mask, T&& val, S&& src) {
        return splice(__DPL forward<M>(mask),__DPL forward<T>(val),
            __DPL forward<S>(src));
    }

    template <simd_vector T, const_mask_for<T> M, equivalent_vector_with<T> S>
    requires (extended_vector<T> || extended_vector<S>) &&
        unqualified_extended_splice<launder_cmask_t<T, M>, T, S>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T&& val, S&& src) {
        return splice(dx::to_const_mask<T>(mask),__DPL forward<T>(val),
            __DPL forward<S>(src));
    }

    template <simd_vector T, exact_mask_for<T> M>
    requires (extended_vector<T> || extended_mask<T>) &&
        unqualified_extended_zsplice<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M&& mask, T&& val, dx::zero_t zero) {
        return splice(__DPL forward<M>(mask), __DPL forward<T>(val), zero);
    }

    template <extended_vector T, const_mask_for<T> M>
    requires unqualified_extended_zsplice<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T&& val, dx::zero_t zero) {
        return splice(dx::to_const_mask<T>(mask), __DPL forward<T>(val), zero);
    }
};

template <>
struct fallback_impl<splice_t> {
public:
    template <simd_vector L, equivalent_vector_with<L> R, exact_mask_for<L> M>
    requires cpo_invocable<select_t, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        M&& mask, L&& lhs, R&& rhs) noexcept(canonical_mask<M> &&
        canonical_vector<L> && canonical_vector<R>) {
        auto const low = dx::countr_zero(mask);
        auto const high = dx::countl_zero(__DPL forward<M>(mask));
        return dx::slide_left(dx::shift_right(__DPL forward<L>(lhs), high),
            __DPL forward<R>(rhs), high + low); // slide will clamp `high+low`
    }

    template <simd_vector L, equivalent_vector_with<L> R, const_mask_for<L> M>
    requires cpo_invocable<select_t, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        M mask, L&& lhs, R&& rhs) noexcept {
        constexpr auto cmask = dx::to_const_mask<L>(mask);
        constexpr auto low = dx::countr_zero(cmask);
        if constexpr (dx::none_of(cmask)) {
            return rhs;
        } else if constexpr (dx::all_of(cmask)) {
            return lhs;
        } else {
            constexpr auto high = dx::countl_zero(cmask);
            return dx::slide_left(
                dx::shift_right(__DPL forward<L>(lhs), imm<high>),
                __DPL forward<R>(rhs), imm<high + low>);
        }
    }

    template <simd_vector T, exact_mask_for<T> M, broadcastable_to<T> V>
    requires cpo_invocable<splice_t, M, T, canonical_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr cpo_result_t<splice_t, M, T, canonical_type_t<T>>
    operator()(M&& mask, T&& val, V&& src) noexcept(
        canonical_mask<M> && canonical_vector<T>) {
        return splice_t::operator()(__DPL forward<M>(mask),
            __DPL forward<T>(val), dx::broadcast<T>(__DPL forward<V>(src)));
    }

    template <simd_vector T, const_mask_for<T> M, broadcastable_to<T> V>
    requires cpo_invocable<splice_t, M, T, canonical_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr cpo_result_t<splice_t, M, T, canonical_type_t<T>>
    operator()(M mask, T&& val, V&& src) noexcept(canonical_vector<T>) {
        return splice_t::operator()(mask,__DPL forward<T>(val),
            dx::broadcast<T>(__DPL forward<V>(src)));
    }

    template <simd_vector S, exact_mask_for<S> M, broadcastable_to<S> V>
    requires cpo_invocable<splice_t, M, canonical_type_t<S>, S>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr cpo_result_t<splice_t, M, canonical_type_t<S>, S>
    operator()(M&& mask, V&& val, S&& src) noexcept(
        canonical_mask<M> && canonical_vector<S>) {
        return splice_t::operator()(__DPL forward<M>(mask),
            dx::broadcast<S>(__DPL forward<V>(val)), __DPL forward<S>(src));
    }

    template <simd_vector S, const_mask_for<S> M, broadcastable_to<S> V>
    requires cpo_invocable<splice_t, M, canonical_type_t<S>, S>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr cpo_result_t<splice_t, M, canonical_type_t<S>, S>
    operator()(M mask, V&& val, S&& src) noexcept(canonical_vector<S>) {
        return splice_t::operator()(mask,
            dx::broadcast<S>(__DPL forward<V>(val)), __DPL forward<S>(src));
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::splice_t splice{};
} // namespace cpo
} // namespace datapar

__DPL_DEFAULT_NAMESPACE_END
