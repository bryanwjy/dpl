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

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void splice(...) noexcept = delete;

struct DPL_EMPTY_BASES splice_t : private algorithm_base<splice_t> {
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
    using mask_t DPL_NODEBUG =
        basic_mask<simd_element_type_t<L>, common_abi_t<L, R>>;

    template <typename L, typename R>
    using imask_t DPL_NODEBUG = mask_value_t<
        simd_abi_traits<simd_element_type_t<L>, common_abi_t<L, R>>::size>;

    template <typename L, typename R, imask_t<L, R> M>
    using cmask_t DPL_NODEBUG = const_mask<
        simd_abi_traits<simd_element_type_t<L>, common_abi_t<L, R>>::size, M>;

public:
    template <simd_vector L, common_vector_with<L> R,
        equivalent_mask_with<mask_t<L, R>> M>
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

    template <simd_vector L, common_vector_with<L> R, imask_t<L, R> M>
    requires fixed_width_abi<common_abi_t<L, R>> &&
        cpo_invocable<select_t, cmask_t<L, R, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        cmask_t<L, R, M> cmask, L lhs, R rhs) noexcept {
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
        basic_vector<simd_element_type_t<L>, common_abi_t<L, R>>;

    template <typename L, typename R>
    using mask_t DPL_NODEBUG =
        basic_mask<simd_element_type_t<L>, common_abi_t<L, R>>;

    template <typename L, typename R>
    using imask_t DPL_NODEBUG = mask_value_t<
        simd_abi_traits<simd_element_type_t<L>, common_abi_t<L, R>>::size>;

    template <typename L, typename R, imask_t<L, R> M>
    using cmask_t DPL_NODEBUG = const_mask<
        simd_abi_traits<simd_element_type_t<L>, common_abi_t<L, R>>::size, M>;

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

    template <canonical_vector L, common_vector_with<L> R, imask_t<L, R> M>
    requires canonical_vector<R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<L, R> operator()(
        cmask_t<L, R, M> cmask, L lhs, R rhs) noexcept
    requires requires {
        splice(internal::abi<common_abi_t<L, R>>, cmask, lhs, rhs);
    }
    {
        return splice(internal::abi<common_abi_t<L, R>>, cmask, lhs, rhs);
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
    using mask_t DPL_NODEBUG =
        basic_mask<simd_element_type_t<L>, common_abi_t<L, R>>;

    template <typename L, typename R>
    using imask_t DPL_NODEBUG = mask_value_t<
        simd_abi_traits<simd_element_type_t<L>, common_abi_t<L, R>>::size>;

    template <typename L, typename R, imask_t<L, R> M>
    using cmask_t DPL_NODEBUG = const_mask<
        simd_abi_traits<simd_element_type_t<L>, common_abi_t<L, R>>::size, M>;

public:
    template <simd_vector L, common_vector_with<L> R,
        equivalent_mask_with<mask_t<L, R>> M>
    requires (extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_splice<M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M&& mask, L&& lhs, R&& rhs) {
        return splice(mask, __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <simd_vector L, common_vector_with<L> R, imask_t<L, R> M>
    requires (extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_splice<cmask_t<L, R, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto operator()(cmask_t<L, R, M> cmask, L&& lhs, R&& rhs) {
        return splice(cmask, __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }
};

template <auto V>
struct splicei_t {
    template <typename L, typename R>
    requires cpo_invocable<splice_t, immediate<V>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R&& rhs) noexcept {
        return splice_t::operator()(
            imm<V>, __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }
};

} // namespace datapar::internal

namespace datapar {
DPL_EXPORT template <auto V>
inline constexpr internal::splicei_t<V> splicei{};
DPL_EXPORT inline constexpr internal::splice_t splice{};
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
