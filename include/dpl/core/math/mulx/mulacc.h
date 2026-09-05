// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/details/fm_broadcast.h"
#include "dpl/core/math/fma/fmacc.h"

#if !DPL_MODULES
#  include "dpl/core/dispatch/concepts/operation.h"
#  include "dpl/core/operations/arithmetic/add.h"
#  include "dpl/core/operations/arithmetic/multiply.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

struct DPL_EMPTY_BASES mulacc_t :
    public math_operation_base<mulacc_t>,
    public maskable_accumulation_base<mulacc_t> {
    using operation_base<mulacc_t>::operator();
    using maskable_accumulation_base<mulacc_t>::operator();
};

template <>
struct operation_signature<mulacc_t> {
    template <typename AT, typename BT, typename CT>
    requires simd_vector<AT> || simd_vector<BT> || simd_vector<CT>
    static consteval void operator()(AT&&, BT&&, CT&&) noexcept {}
};

template <>
struct fallback_impl<mulacc_t> : fm_canonical_broadcaster<mulacc_t> {

    template <simd_vector AT, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& a, BT&& b, CT&& c) noexcept(
        canonical_vector<AT> && canonical_vector<BT> && canonical_vector<CT>) {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmacc)) {
            return dx::fmacc(__DPL forward<AT>(a), __DPL forward<BT>(b),
                __DPL forward<CT>(c));
        } else {
            return dx::add(
                dx::multiply(__DPL forward<BT>(b), __DPL forward<CT>(c)),
                __DPL forward<AT>(a));
        }
    }

    template <unextended_type AT, unextended_type BT, unextended_type CT>
    requires cpo_invocable<mulacc_t, AT, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr cpo_result_t<mulacc_t, AT, BT, CT> operator()(AT a,
        simd_mask_type_t<cpo_result_t<mulacc_t, AT, BT, CT>> mask, BT b,
        CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmacc)) {
            return dx::fmacc(a, mask, b, c);
        } else {
            return dx::add(a, mask, dx::multiply(b, c), a);
        }
    }

    template <typename AT, typename BT, typename CT,
        result_mask_for<mulacc_t, AT, BT, CT> M>
    requires (extended_mask<M> || extended_vector<AT> || extended_vector<BT> ||
        extended_vector<CT>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& a, M&& mask, BT&& b, CT&& c) {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmacc)) {
            return dx::fmacc( __DPL forward<AT>(a), __DPL forward<M>(mask),
                __DPL forward<BT>(b), __DPL forward<CT>(c));
        } else {
            return dx::add(a, __DPL forward<M>(mask),
                dx::multiply(__DPL forward<BT>(b), __DPL forward<CT>(c)), a);
        }
    }

    template <typename AT, typename BT, typename CT,
        result_cmask_for<mulacc_t, AT, BT, CT> M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& a, M mask, BT&& b, CT&& c) noexcept(
        canonical_vector<AT> && canonical_vector<BT> && canonical_vector<CT>) {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmacc)) {
            return dx::fmacc( __DPL forward<AT>(a), mask, __DPL forward<BT>(b),
                __DPL forward<CT>(c));
        } else {
            return dx::add(a, mask,
                dx::multiply(__DPL forward<BT>(b), __DPL forward<CT>(c)), a);
        }
    }

    template <unextended_type AT, unextended_type BT, unextended_type CT>
    requires cpo_invocable<mulacc_t, AT, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr cpo_result_t<mulacc_t, AT, BT, CT> operator()(
        dx::zero_t zero,
        simd_mask_type_t<cpo_result_t<mulacc_t, AT, BT, CT>> mask, AT a, BT b,
        CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmacc)) {
            return dx::fmacc(zero, mask, a, b, c);
        } else {
            return dx::add(zero, mask, dx::multiply(b, c), a);
        }
    }

    template <typename AT, typename BT, typename CT,
        result_mask_for<mulacc_t, AT, BT, CT> M>
    requires (extended_mask<M> || extended_vector<AT> || extended_vector<BT> ||
        extended_vector<CT>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, AT&& a, BT&& b, CT&& c) {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmacc)) {
            return dx::fmacc(zero, __DPL forward<M>(mask), __DPL forward<AT>(a),
                __DPL forward<BT>(b), __DPL forward<CT>(c));
        } else {
            return dx::add(zero, __DPL forward<M>(mask),
                dx::multiply(__DPL forward<BT>(b), __DPL forward<CT>(c)),
                __DPL forward<AT>(a));
        }
    }

    template <typename AT, typename BT, typename CT,
        result_cmask_for<mulacc_t, AT, BT, CT> M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M mask, AT&& a, BT&& b,
        CT&& c) noexcept(canonical_vector<AT> && canonical_vector<BT> &&
        canonical_vector<CT>) {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmacc)) {
            return dx::fmacc(zero, mask, __DPL forward<AT>(a),
                __DPL forward<BT>(b), __DPL forward<CT>(c));
        } else {
            return dx::add(zero, mask,
                dx::multiply(__DPL forward<BT>(b), __DPL forward<CT>(c)),
                __DPL forward<AT>(a));
        }
    }

    using fm_canonical_broadcaster<mulacc_t>::operator();
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::mulacc_t mulacc{};
} // namespace cpo
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
