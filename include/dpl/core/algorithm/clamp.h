// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/decayable.h"
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/operations/arithmetic/result.h"
#  include "dpl/core/operations/evaluate.h"
#  include "dpl/core/operations/internal/operation_base.h"
#  include "dpl/core/operations/minmax.h"
#  include "dpl/std/concepts/totally_ordered.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void clamp(...) noexcept = delete;
template <auto>
void clamp(...) noexcept = delete;

struct clamp_t;

template <typename AT, typename BT, typename CT,
    typename A = common_abi_t<AT, BT, CT>>
concept unqualified_canonical_clamp = requires(AT a, BT b, CT c) {
    { clamp(internal::abi<A>, a, b, c) } -> canonical_fma_result<AT, BT, CT, A>;
};

template <typename AT, typename BT, typename CT,
    typename A = common_abi_t<AT, BT, CT>>
concept unqualified_extended_clamp = requires(AT a, BT b, CT c) {
    { clamp(a, b, c) } -> vector_with_common_abi<A>;
};

template <typename AT, typename BT, typename CT,
    typename A = common_abi_t<AT, BT, CT>>
concept unqualified_clamp = unqualified_canonical_clamp<AT, BT, CT, A> ||
    unqualified_extended_clamp<AT, BT, CT, A> ||
    (decayable_vector_for<AT, operation_category::lane_agnostic> &&
        decayable_vector_for<BT, operation_category::lane_agnostic> &&
        decayable_vector_for<CT, operation_category::lane_agnostic> &&
        regular_invocable<clamp_t, canonical_type_t<AT>, canonical_type_t<BT>,
            canonical_type_t<CT>>);

struct clamp_t : ternary_operation_base<clamp_t> {
private:
    friend ternary_operation_base<clamp_t>;

    template <simd_abi A, typename AT, typename BT, typename CT>
    requires ((canonical_vector<AT> || !simd_vector<AT>) &&
        (canonical_vector<BT> || !simd_vector<BT>) &&
        (canonical_vector<CT> || !simd_vector<CT>))
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, AT a, BT b, CT c) noexcept
    requires requires {
        { clamp(internal::abi<A>, a, b, c) } -> vector_with_common_abi<A>;
    }
    {
        return clamp(internal::abi<A>, a, b, c);
    }

    template <simd_abi A, typename AT, typename BT, typename CT>
    requires ((extended_vector<AT> || !simd_vector<AT>) ||
        (extended_vector<BT> || !simd_vector<BT>) ||
        (extended_vector<CT> || !simd_vector<CT>))
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, AT a, BT b, CT c) noexcept
    requires requires {
        { clamp(a, b, c) } -> vector_with_common_abi<A>;
    }
    {
        return clamp(a, b, c);
    }

    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST NODISCARD) static constexpr auto DPL_VECTORCALL
    fallback(basic_vector<E, A> val, basic_vector<E, A> low,
        basic_vector<E, A> high) noexcept {
        return dx::min(high, dx::max(val, low));
    }

public:
    template <fixed_width_abi A, simd_element_for<A> E>
    requires totally_ordered<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_vector<E, A> a,
        basic_vector<E, A> b, basic_vector<E, A> c) noexcept {
        if constexpr (unqualified_canonical_clamp<basic_vector<E, A>,
                          basic_vector<E, A>, basic_vector<E, A>>) {
            if consteval {
                return fallback(a, b, c);
            } else {
                return clamp(internal::abi<A>, a, b, c);
            }
        } else {
            return fallback(a, b, c);
        }
    }

    template <simd_abi AA, common_abi_with<AA> BA, common_abi_with<BA> CA,
        typename E>
    requires common_abi_with<common_abi_t<AA, BA>, CA> &&
        simd_element_for<E, AA> && simd_element_for<E, BA> &&
        simd_element_for<E, CA> &&
        (scalable_abi<AA> || scalable_abi<BA> || scalable_abi<CA> ||
            different_from<AA, BA> || different_from<AA, CA> ||
            different_from<BA, CA> || !totally_ordered<E>) &&
        unqualified_canonical_clamp<basic_vector<E, AA>, basic_vector<E, BA>,
            basic_vector<E, CA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, common_abi_t<AA, BA, CA>> operator()(
        basic_vector<E, AA> a, basic_vector<E, BA> b,
        basic_vector<E, CA> c) noexcept {
        return clamp(internal::abi<common_abi_t<AA, BA, CA>>, a, b, c);
    }

    template <simd_vector AT, simd_vector BT, simd_vector CT>
    requires (extended_vector<AT> || extended_vector<BT> ||
                 extended_vector<CT>) &&
        unqualified_clamp<AT, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT a, BT b, CT c) noexcept {
        if constexpr (unqualified_extended_clamp<AT, BT, CT>) {
            return clamp(a, b, c);
        } else if constexpr (simd_expression<AT> || simd_expression<BT> ||
            simd_expression<CT>) {
            return operator()(
                dx::evaluate(a), dx::evaluate(b), dx::evaluate(c));
        } else {
            return operator()(
                dx::to_canonical(a), dx::to_canonical(b), dx::to_canonical(c));
        }
    }

    using ternary_operation_base<clamp_t>::operator();
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::clamp_t clamp{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
