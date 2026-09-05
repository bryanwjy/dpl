// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/operations/bitwise/bwxor.h"
#include "dpl/core/operations/internal/broadcasting.h"
#include "dpl/core/operations/internal/transform.h"

#if !DPL_MODULES
#  include "dpl/core/basic/from_bitset.h"
#  include "dpl/core/basic/internal/abi.h"
#  include "dpl/core/basic/to_bitset.h"
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/maskable/predicate.h"
#  include "dpl/core/dispatch/operation/primitive.h"
#  include "dpl/core/type_traits/common_size_type.h"
#  include "dpl/std/concepts/equality_comparable.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void cmpneq(...) noexcept = delete;

struct DPL_EMPTY_BASES cmpneq_t :
    public comparison_base<cmpneq_t>,
    public maskable_predicate_base<cmpneq_t> {
    using operation_base<cmpneq_t>::operator();
    using maskable_predicate_base<cmpneq_t>::operator();
};

template <>
struct operation_signature<cmpneq_t> {
    template <typename L, typename R>
    requires simd_type<L> || simd_type<R>
    static consteval void operator()(L&&, R&&) noexcept {}
};

template <>
struct fallback_impl<cmpneq_t> : binary_canonical_broadcaster<cmpneq_t> {

    template <canonical_vector T>
    requires equality_comparable<simd_element_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr simd_mask_type_t<T>
        DPL_VECTORCALL operator()(T lhs, T rhs) noexcept {
        return internal::transform<simd_mask_type_t<T>>(
            [](auto lhs, auto rhs) -> bool { return lhs != rhs; }, lhs, rhs);
    }

    template <canonical_mask L, canonical_mask R>
    requires cpo_invocable<bwxor_t, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr common_canonical_simd_t<L, R>
        DPL_VECTORCALL operator()(L lhs, R rhs) noexcept {
        return dx::bwxor(lhs, rhs);
    }

    using binary_canonical_broadcaster<cmpneq_t>::operator();
};

template <typename L, typename R, typename T = common_canonical_simd_t<L, R>>
concept unqualified_canonical_cmpneq = requires {
    {
        cmpneq(internal::abi<T>, internal::declarg<L>(), internal::declarg<R>())
    } -> equivalent_mask_with<simd_mask_type_t<T>>;
};

template <typename M, typename L, typename R,
    typename T = simd_mask_type_t<common_canonical_simd_t<L, R>>>
concept unqualified_canonical_mcmpneq =
    cpo_invocable<cmpneq_t, L, R> && requires {
        {
            cmpneq(internal::abi<T>, internal::declarg<M>(),
                internal::declarg<L>(), internal::declarg<R>())
        } -> equivalent_mask_with<T>;
    };

template <>
struct canonical_impl<cmpneq_t> {
private:
    template <typename L, typename R>
    using result_t DPL_NODEBUG = cpo_result_t<cmpneq_t, L, R>;

    template <typename L, typename R>
    using mask_t DPL_NODEBUG = cpo_result_t<cmpneq_t, L, R>;

    template <typename L, typename R>
    using main_result_t DPL_NODEBUG =
        simd_mask_type_t<common_canonical_simd_t<L, R>>;

public:
    template <canonical_mask L, common_mask_with<L> R>
    requires canonical_mask<R> && unqualified_canonical_cmpneq<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr main_result_t<L, R> operator()(L lhs, R rhs) noexcept {
        return cmpneq(internal::abi<common_abi_t<L, R>>, lhs, rhs);
    }

    template <canonical_vector L, common_vector_with<L> R>
    requires canonical_vector<R> && unqualified_canonical_cmpneq<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr main_result_t<L, R> operator()(L lhs, R rhs) noexcept {
        return cmpneq(internal::abi<common_abi_t<L, R>>, lhs, rhs);
    }

    template <canonical_vector L, broadcastable_to<L> R>
    requires unqualified_canonical_cmpneq<L, R, L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr simd_mask_type_t<L> operator()(L lhs, R&& rhs) noexcept {
        return cmpneq(internal::abi<L>, lhs, __DPL forward<R>(rhs));
    }

    template <canonical_vector R, broadcastable_to<R> L>
    requires unqualified_canonical_cmpneq<L, R, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr simd_mask_type_t<R> operator()(L&& lhs, R rhs) noexcept {
        return cmpneq(internal::abi<R>, __DPL forward<L>(lhs), rhs);
    }

    template <unextended_type L, unextended_terminal_of<cmpneq_t, L> R>
    requires unqualified_canonical_mcmpneq<mask_t<L, R>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<L, R> operator()(
        mask_t<L, R> src, L lhs, R rhs) noexcept {
        return cmpneq(internal::abi<common_abi_t<L, R>>, src, lhs, rhs);
    }

    template <unextended_type L, unextended_type R,
        result_cmask_for<cmpneq_t, L, R> M>
    requires unqualified_canonical_mcmpneq<launder_cmask_t<result_t<L, R>, M>,
        L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<L, R> operator()(M mask, L lhs, R rhs) noexcept {
        return cmpneq(internal::abi<common_abi_t<L, R>>,
            dx::to_const_mask<result_t<L, R>>(mask), lhs, rhs);
    }
};

template <typename L, typename R, typename T = common_canonical_simd_t<L, R>>
concept unqualified_extended_cmpneq = requires(L lhs, R rhs) {
    { cmpneq(lhs, rhs) } -> equivalent_mask_with<simd_mask_type_t<T>>;
};

template <typename S, typename L, typename R>
concept unqualified_extended_mcmpneq =
    cpo_invocable<cmpneq_t, L, R> && requires {
        {
            cmpneq(internal::declarg<S>(), internal::declarg<L>(),
                internal::declarg<R>())
        } -> equivalent_mask_with<cpo_result_t<cmpneq_t, L, R>>;
    };

template <>
struct extended_impl<cmpneq_t> {
public:
    template <simd_mask L, common_mask_with<L> R>
    requires (extended_mask<L> || extended_mask<R>) &&
        unqualified_extended_cmpneq<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R&& rhs) {
        return cmpneq(__DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <simd_vector L, common_vector_with<L> R>
    requires (extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_cmpneq<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R&& rhs) {
        return cmpneq(__DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <extended_vector L, broadcastable_to<L> R>
    requires unqualified_extended_cmpneq<L, R, L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R&& rhs) {
        return cmpneq(__DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <extended_vector R, broadcastable_to<R> L>
    requires unqualified_extended_cmpneq<L, R, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R&& rhs) {
        return cmpneq(__DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <simd_mask S, typename L, typename R>
    requires (extended_mask<S> || extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_mcmpneq<S, L, R> &&
        equivalent_mask_with<S, cpo_result_t<cmpneq_t, L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, L&& lhs, R&& rhs) {
        return cmpneq(__DPL forward<S>(src), __DPL forward<L>(lhs),
            __DPL forward<R>(rhs));
    }

    template <typename L, typename R, result_cmask_for<cmpneq_t, L, R> M>
    requires (extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_mcmpneq<
            launder_cmask_t<cpo_result_t<cmpneq_t, L, R>, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, L&& lhs, R&& rhs) {
        return cmpneq(dx::to_const_mask<cpo_result_t<cmpneq_t, L, R>>(mask),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::cmpneq_t cmpneq{};
}
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
