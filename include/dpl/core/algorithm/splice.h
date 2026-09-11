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

struct DPL_EMPTY_BASES splice_t : public algorithm_base<splice_t> {
    using operation_base<splice_t>::operator();
};

template <>
struct operation_signature<splice_t> {
    template <typename L, typename R>
    using vector_t DPL_NODEBUG = common_canonical_simd_t<L, R>;

    template <simd_vector L, equivalent_vector_with<L> R, exact_mask_for<L> M>
    static consteval void operator()(M&&, L&&, R&&) noexcept {}
    template <simd_vector L, equivalent_vector_with<L> R, const_mask_for<L> M>
    static consteval void operator()(M, L&&, R&&) noexcept {}
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

    template <simd_vector R, exact_mask_for<R> M, broadcastable_to<R> L>
    requires cpo_invocable<splice_t, M, R, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr cpo_result<splice_t, M, R, R> operator()(
        M&& mask, L&& lhs, R&& rhs) noexcept {
        return splice_t::operator()(__DPL forward<M>(mask),
            dx::broadcast<R>(__DPL forward<L>(lhs)), __DPL forward<R>(rhs));
    }

    template <simd_vector R, const_mask_for<R> M, broadcastable_to<R> L>
    requires cpo_invocable<splice_t, M, R, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr cpo_result<splice_t, M, R, R> operator()(
        M mask, L&& lhs, R&& rhs) noexcept {
        return splice_t::operator()(mask,
            dx::broadcast<R>(__DPL forward<L>(lhs)), __DPL forward<R>(rhs));
    }

    template <simd_vector L, exact_mask_for<L> M, broadcastable_to<L> R>
    requires cpo_invocable<splice_t, M, L, L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr cpo_result<splice_t, M, L, L> operator()(
        M&& mask, L&& lhs, R&& rhs) noexcept {
        return splice_t::operator()(__DPL forward<M>(mask),
            __DPL forward<L>(lhs), dx::broadcast<L>(__DPL forward<R>(rhs)));
    }

    template <simd_vector L, const_mask_for<L> M, broadcastable_to<L> R>
    requires cpo_invocable<splice_t, M, L, L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr cpo_result<splice_t, M, L, L> operator()(
        M mask, L&& lhs, R&& rhs) noexcept {
        return splice_t::operator()(mask, __DPL forward<L>(lhs),
            dx::broadcast<L>(__DPL forward<R>(rhs)));
    }
};

template <typename M, typename L, typename R = L, typename T = L>
concept unqualified_canonical_splice = requires {
    {
        splice(internal::abi<T>, internal::declarg<M>(), internal::declarg<L>(),
            internal::declarg<R>())
    } -> same_as<T>;
};

template <>
struct canonical_impl<splice_t> {
public:
    template <canonical_vector L, same_as<L> R>
    requires unqualified_canonical_splice<simd_mask_type_t<L>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr L operator()(
        simd_mask_type_t<L> mask, L lhs, R rhs) noexcept {
        return splice(internal::abi<L>, mask, lhs, rhs);
    }

    template <canonical_vector R, broadcastable_to<R> L>
    requires unqualified_canonical_splice<simd_mask_type_t<R>, L, R, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr R operator()(
        simd_mask_type_t<R> mask, L&& lhs, R rhs) noexcept {
        return splice(internal::abi<R>, dx::to_const_mask<R>(mask),
            __DPL forward<L>(lhs), rhs);
    }

    template <canonical_vector L, broadcastable_to<L> R>
    requires unqualified_canonical_splice<simd_mask_type_t<L>, L, R, L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr L operator()(
        simd_mask_type_t<L> mask, L lhs, R&& rhs) noexcept {
        return splice(internal::abi<R>, mask, lhs, __DPL forward<R>(rhs));
    }

    template <canonical_vector L, same_as<L> R, const_mask_for<L> M>
    requires unqualified_canonical_splice<launder_cmask_t<L, M>, L, R, L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr L operator()(M mask, L lhs, R rhs) noexcept {
        return splice(internal::abi<L>, dx::to_const_mask<L>(mask), lhs, rhs);
    }

    template <canonical_vector R, broadcastable_to<R> L, const_mask_for<R> M>
    requires unqualified_canonical_splice<launder_cmask_t<R, M>, L, R, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr R operator()(M mask, L&& lhs, R rhs) noexcept {
        return splice(internal::abi<R>, mask, __DPL forward<L>(lhs), rhs);
    }

    template <canonical_vector L, broadcastable_to<L> R, const_mask_for<L> M>
    requires unqualified_canonical_splice<launder_cmask_t<L, M>, L, R, L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr L operator()(M mask, L lhs, R&& rhs) noexcept {
        return splice(internal::abi<L>, mask, lhs, __DPL forward<R>(rhs));
    }
};

template <typename M, typename L, typename R, typename T = L>
concept unqualified_extended_splice = requires(M mask, L lhs, R rhs) {
    { splice(mask, lhs, rhs) } -> equivalent_vector_with<T>;
};

template <>
struct extended_impl<splice_t> {
public:
    template <simd_vector L, equivalent_vector_with<L> R, exact_mask_for<L> M>
    requires (extended_mask<M> || extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_splice<M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M&& mask, L&& lhs, R&& rhs) {
        return splice(__DPL forward<M>(mask), __DPL forward<L>(lhs),
            __DPL forward<R>(rhs));
    }

    template <simd_vector L, equivalent_vector_with<L> R, const_mask_for<L> M>
    requires (extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_splice<launder_cmask_t<L, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto operator()(M mask, L&& lhs, R&& rhs) {
        return splice(dx::to_const_mask<L>(mask), __DPL forward<L>(lhs),
            __DPL forward<R>(rhs));
    }

    template <simd_vector R, exact_mask_for<R> M, broadcastable_to<R> L>
    requires (extended_vector<R> || extended_mask<M>) &&
        unqualified_extended_splice<M, L, R, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M&& mask, L&& lhs, R&& rhs) noexcept {
        return splice(__DPL forward<M>(mask), __DPL forward<L>(lhs),
            __DPL forward<R>(rhs));
    }

    template <extended_vector R, const_mask_for<R> M, broadcastable_to<R> L>
    requires unqualified_extended_splice<M, L, R, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, L&& lhs, R&& rhs) noexcept {
        return splice(dx::to_const_mask<L>(mask), __DPL forward<L>(lhs),
            __DPL forward<R>(rhs));
    }

    template <simd_vector L, exact_mask_for<L> M, broadcastable_to<L> R>
    requires (extended_vector<L> || extended_mask<M>) &&
        unqualified_extended_splice<M, L, R, L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M&& mask, L&& lhs, R&& rhs) noexcept {
        return splice(__DPL forward<M>(mask), __DPL forward<L>(lhs),
            __DPL forward<R>(rhs));
    }

    template <extended_vector L, const_mask_for<L> M, broadcastable_to<L> R>
    requires unqualified_extended_splice<M, L, R, L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, L&& lhs, R&& rhs) noexcept {
        return splice(dx::to_const_mask<L>(mask), __DPL forward<L>(lhs),
            __DPL forward<R>(rhs));
    }
};

template <auto V>
struct splicei_t {
    template <typename L, typename R>
    requires requires { typename cmask_t<V>; } &&
        cpo_invocable<splice_t, cmask_t<V>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R&& rhs) noexcept {
        return splice_t::operator()(
            cmask_v<V>, __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
template <auto V>
inline constexpr internal::splicei_t<V> splicei{};
inline constexpr internal::splice_t splice{};
} // namespace cpo
} // namespace datapar

__DPL_DEFAULT_NAMESPACE_END
