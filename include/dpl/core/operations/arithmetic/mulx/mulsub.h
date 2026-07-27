// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/operations/arithmetic/fma/fmsub.h"
#include "dpl/core/operations/arithmetic/multiply.h"
#include "dpl/core/operations/arithmetic/subtract.h"
#if !DPL_MODULES
#  include "dpl/core/dispatch/concepts/operation.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

struct DPL_EMPTY_BASES mulsub_t :
    public arithmetic_base<mulsub_t>,
    public maskable_accumulation_base<mulsub_t>,
    public ternary_broadcastable_operation<mulsub_t> {
    using operation_base<mulsub_t>::operator();
    using maskable_accumulation_base<mulsub_t>::operator();
    using ternary_broadcastable_operation<mulsub_t>::operator();
};

template <>
struct operation_signature<mulsub_t> {
    template <typename AT, typename BT, typename CT>
    requires simd_vector<AT> || simd_vector<BT> || simd_vector<CT>
    static consteval void operator()(AT&&, BT&&, CT&&) noexcept {}
};

template <>
struct fallback_impl<mulsub_t> : ternary_broadcasting_fallback<mulsub_t> {

    template <simd_vector AT, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(AT a, BT b, CT c) noexcept(
        canonical_vector<AT> && canonical_vector<BT> && canonical_vector<CT>) {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmsub)) {
            return dx::fmsub(a, b, c);
        } else {
            return dx::subtract(dx::multiply(a, b), c);
        }
    }

    template <simd_vector AT, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(AT a,
        simd_mask_type_t<AT> mask, BT b, CT c) noexcept(canonical_vector<AT> &&
        canonical_vector<BT> && canonical_vector<CT>) {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmsub)) {
            return dx::fmsub(a, mask, b, c);
        } else {
            return dx::subtract(a, mask, dx::multiply(a, b), c);
        }
    }

    template <simd_vector AT, const_mask_for<AT> M, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        AT a, M mask, BT b, CT c) noexcept(canonical_vector<AT> &&
        canonical_vector<BT> && canonical_vector<CT>) {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmsub)) {
            return dx::fmsub(a, mask, b, c);
        } else {
            return dx::subtract(a, mask, dx::multiply(a, b), c);
        }
    }

    template <simd_vector AT, exact_mask_for<AT> M, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    requires (extended_mask<M> || extended_vector<AT> || extended_vector<BT> ||
        extended_vector<CT>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(AT a, M mask, BT b, CT c) {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmsub)) {
            return dx::fmsub(a, mask, b, c);
        } else {
            return dx::subtract(a, mask, dx::multiply(a, b), c);
        }
    }

    template <simd_vector AT, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(dx::zero_t zero,
        simd_mask_type_t<AT> mask, AT a, BT b,
        CT c) noexcept(canonical_vector<AT> && canonical_vector<BT> &&
        canonical_vector<CT>) {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmsub)) {
            return dx::fmsub(zero, mask, a, b, c);
        } else {
            return dx::subtract(zero, mask, dx::multiply(a, b), c);
        }
    }

    template <simd_vector AT, const_mask_for<AT> M, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(dx::zero_t zero, M mask,
        AT a, BT b, CT c) noexcept(canonical_vector<AT> &&
        canonical_vector<BT> && canonical_vector<CT>) {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmsub)) {
            return dx::fmsub(zero, mask, a, b, c);
        } else {
            return dx::subtract(zero, mask, dx::multiply(a, b), c);
        }
    }

    template <simd_vector AT, exact_mask_for<AT> M, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    requires (extended_mask<M> || extended_mask<M> || extended_vector<AT> ||
        extended_vector<BT> || extended_vector<CT>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        dx::zero_t zero, M mask, AT a, BT b, CT c) {
        if constexpr (dx::is_simd_invocable<AT, BT, CT>(dx::fmsub)) {
            return dx::fmsub(zero, mask, a, b, c);
        } else {
            return dx::subtract(zero, mask, dx::multiply(a, b), c);
        }
    }

    using ternary_broadcasting_fallback<mulsub_t>::operator();
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::mulsub_t mulsub{};
} // namespace cpo
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
