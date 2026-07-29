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
    template <simd_mask M, simd_vector L, simd_vector R>
    static consteval void operator()(M&&, L&&, R&&) noexcept {}
};

template <>
struct fallback_impl<splice_t> {
private:
    template <typename L, typename R>
    using vector_t DPL_NODEBUG =
        make_canonical_vector_t<simd_element_type_t<L>, common_abi_t<L, R>>;

public:
    template <simd_vector L, common_vector_with<L> R,
        exact_mask_for<vector_t<L, R>> M>
    requires cpo_invocable<select_t, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        M&& mask, L&& lhs, R&& rhs) noexcept(canonical_mask<M> &&
        canonical_vector<L> && canonical_vector<R>) {
        auto const low = dx::countr_zero(mask);
        auto const high = dx::countl_zero(__DPL forward<M>(mask));
        return dx::slide_left(dx::shift_right(__DPL forward<L>(lhs), high),
            __DPL forward<R>(rhs), high + low);
    }

    template <simd_vector L, common_vector_with<L> R,
        const_mask_for<vector_t<L, R>> M>
    requires cpo_invocable<select_t, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        M mask, L lhs, R rhs) noexcept {
        constexpr auto cmask = dx::to_const_mask<vector_t<L, R>>(mask);
        constexpr auto low = dx::countr_zero(cmask);
        constexpr auto high = dx::countr_zero(cmask);
        return dx::slide_left(
            dx::shift_right(lhs, imm<high>), rhs, imm<high + low>);
    }
};

template <>
struct canonical_impl<splice_t> {
private:
    template <typename L, typename R>
    using result_t DPL_NODEBUG =
        make_canonical_vector_t<simd_element_type_t<L>, common_abi_t<L, R>>;

    template <typename L, typename R>
    using mask_t DPL_NODEBUG =
        make_canonical_mask_t<simd_element_type_t<L>, common_abi_t<L, R>>;

public:
    template <canonical_vector L, common_vector_with<L> R>
    requires canonical_vector<R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<L, R> operator()(
        mask_t<L, R> mask, L lhs, R rhs) noexcept
    requires requires {
        splice(internal::abi<common_abi_t<L, R>>, mask, lhs, rhs);
    }
    {
        return splice(internal::abi<common_abi_t<L, R>>, mask, lhs, rhs);
    }

    template <canonical_vector L, common_vector_with<L> R,
        const_mask_for<result_t<L, R>> M>
    requires canonical_vector<R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<L, R> operator()(M cmask, L lhs, R rhs) noexcept
    requires requires {
        splice(internal::abi<common_abi_t<L, R>>, cmask, lhs, rhs);
    }
    {
        return splice(internal::abi<common_abi_t<L, R>>,
            dx::to_const_mask<result_t<L, R>>(cmask), lhs, rhs);
    }
};

template <typename M, typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_extended_splice = requires(M mask, L lhs, R rhs) {
    {
        splice(mask, lhs, rhs)
    } -> equivalent_vector_with<basic_vector<simd_element_type_t<L>, A>>;
};

template <>
struct extended_impl<splice_t> {
private:
    template <typename L, typename R>
    using vector_t DPL_NODEBUG =
        basic_mask<simd_element_type_t<L>, common_abi_t<L, R>>;

    template <typename L, typename R>
    using imask_t DPL_NODEBUG = mask_value_t<
        simd_abi_traits<simd_element_type_t<L>, common_abi_t<L, R>>::size>;

public:
    template <simd_vector L, common_vector_with<L> R,
        exact_mask_for<vector_t<L, R>> M>
    requires (extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_splice<M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M&& mask, L&& lhs, R&& rhs) {
        return splice(mask, __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <simd_vector L, common_vector_with<L> R,
        const_mask_for<vector_t<L, R>> M>
    requires (extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_splice<launder_cmask_t<vector_t<L, R>, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto operator()(M cmask, L&& lhs, R&& rhs) {
        return splice(dx::to_const_mask<vector_t<L, R>>(cmask),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
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
            imm<V>, __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }
};

} // namespace datapar::internal

namespace datapar {
template <auto V>
inline constexpr internal::splicei_t<V> splicei{};
inline constexpr internal::splice_t splice{};
} // namespace datapar

__DPL_DEFAULT_NAMESPACE_END
