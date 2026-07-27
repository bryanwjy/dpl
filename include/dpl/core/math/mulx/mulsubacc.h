// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/fma/fmsubacc.h"
#if !DPL_MODULES
#  include "dpl/core/dispatch/concepts/operation.h"
#  include "dpl/core/operations/arithmetic/multiply.h"
#  include "dpl/core/operations/arithmetic/subadd.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

struct DPL_EMPTY_BASES mulsubacc_t :
    public math_operation_base<mulsubacc_t>,
    public maskable_accumulation_base<mulsubacc_t>,
    public ternary_broadcastable_operation<mulsubacc_t> {
    using operation_base<mulsubacc_t>::operator();
    using maskable_accumulation_base<mulsubacc_t>::operator();
    using ternary_broadcastable_operation<mulsubacc_t>::operator();
};

template <>
struct operation_signature<mulsubacc_t> {
    template <typename AT, typename BT, typename CT>
    requires simd_vector<AT> || simd_vector<BT> || simd_vector<CT>
    static consteval void operator()(AT&&, BT&&, CT&&) noexcept {}
};

template <>
struct fallback_impl<mulsubacc_t> : ternary_broadcasting_fallback<mulsubacc_t> {

    template <simd_vector AT, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& a, BT&& b, CT&& c) noexcept(
        canonical_vector<AT> && canonical_vector<BT> && canonical_vector<CT>) {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmsubacc)) {
            return dx::fmsubacc(__DPL forward<AT>(a), __DPL forward<BT>(b),
                __DPL forward<CT>(c));
        } else {
            return dx::subadd(
                dx::multiply(__DPL forward<BT>(b), __DPL forward<CT>(c)),
                __DPL forward<AT>(a));
        }
    }

    template <canonical_vector AT, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    requires canonical_vector<BT> && canonical_vector<CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        AT a, simd_mask_type_t<AT> mask, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmsubacc)) {
            return dx::fmsubacc(a, mask, b, c);
        } else {
            return dx::subadd(a, mask, dx::multiply(b, c), a);
        }
    }

    template <simd_vector AT, const_mask_for<AT> M, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& a, M mask, BT&& b, CT&& c) noexcept(
        canonical_vector<AT> && canonical_vector<BT> && canonical_vector<CT>) {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmsubacc)) {
            return dx::fmsubacc( __DPL forward<AT>(a), mask,
                __DPL forward<BT>(b), __DPL forward<CT>(c));
        } else {
            return dx::subadd(a, mask,
                dx::multiply(__DPL forward<BT>(b), __DPL forward<CT>(c)), a);
        }
    }

    template <simd_vector AT, exact_mask_for<AT> M, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    requires (extended_mask<M> || extended_vector<AT> || extended_vector<BT> ||
        extended_vector<CT>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& a, M&& mask, BT&& b, CT&& c) {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmsubacc)) {
            return dx::fmsubacc( __DPL forward<AT>(a), __DPL forward<M>(mask),
                __DPL forward<BT>(b), __DPL forward<CT>(c));
        } else {
            return dx::subadd(a, __DPL forward<M>(mask),
                dx::multiply(__DPL forward<BT>(b), __DPL forward<CT>(c)), a);
        }
    }

    template <canonical_vector AT, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    requires canonical_vector<BT> && canonical_vector<CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, simd_mask_type_t<AT> mask, AT a, BT b, CT c) noexcept {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmsubacc)) {
            return dx::fmsubacc(zero, mask, a, b, c);
        } else {
            return dx::subadd(zero, mask, dx::multiply(b, c), a);
        }
    }

    template <simd_vector AT, const_mask_for<AT> M, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M mask, AT a, BT b,
        CT c) noexcept(canonical_vector<AT> && canonical_vector<BT> &&
        canonical_vector<CT>) {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmsubacc)) {
            return dx::fmsubacc(zero, mask, __DPL forward<AT>(a),
                __DPL forward<BT>(b), __DPL forward<CT>(c));
        } else {
            return dx::subadd(zero, mask,
                dx::multiply(__DPL forward<BT>(b), __DPL forward<CT>(c)),
                __DPL forward<AT>(a));
        }
    }

    template <simd_vector AT, exact_mask_for<AT> M, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    requires (extended_mask<M> || extended_mask<M> || extended_vector<AT> ||
        extended_vector<BT> || extended_vector<CT>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, AT&& a, BT&& b, CT&& c) {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmsubacc)) {
            return dx::fmsubacc(zero, __DPL forward<M>(mask),
                __DPL forward<AT>(a), __DPL forward<BT>(b),
                __DPL forward<CT>(c));
        } else {
            return dx::subadd(zero, __DPL forward<M>(mask),
                dx::multiply(__DPL forward<BT>(b), __DPL forward<CT>(c)),
                __DPL forward<AT>(a));
        }
    }

    using ternary_broadcasting_fallback<mulsubacc_t>::operator();
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::mulsubacc_t mulsubacc{};
} // namespace cpo
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
