// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/cpo_invocable.h"
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/dispatch/broadcastable/ternary.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/maskable/accumulation.h"
#  include "dpl/core/dispatch/operation/math.h"
#  include "dpl/core/math/mulx.h"
#  include "dpl/core/type_traits/details/cpo_result.h"
#  include "dpl/std/concepts/floating_point.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void lerp(...) noexcept = delete;

struct lerp_t;

struct DPL_EMPTY_BASES lerp_t :
    public algorithm_base<lerp_t>,
    public maskable_accumulation_base<lerp_t>,
    public ternary_broadcastable_operation<lerp_t> {
    using operation_base<lerp_t>::operator();
    using maskable_accumulation_base<lerp_t>::operator();
    using ternary_broadcastable_operation<lerp_t>::operator();
};

template <>
struct operation_signature<lerp_t> {
    template <typename AT, typename BT, typename CT>
    requires simd_vector<AT> || simd_vector<BT> || simd_vector<CT>
    static consteval void operator()(AT&&, BT&&, CT&&) noexcept {}
};

template <>
struct fallback_impl<lerp_t> : ternary_broadcasting_fallback<lerp_t> {

    template <canonical_vector AT, canonical_vector BT, canonical_vector CT>
    requires cpo_invocable<subtract_t, BT, AT> &&
        cpo_invocable<mulacc_t, AT, cpo_result_t<subtract_t, BT, AT>, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        AT start, BT end, CT scale) noexcept {
        return dx::mulacc(start, dx::subtract(end, start), scale);
    }

    template <canonical_vector AT, canonical_mask M, canonical_vector BT,
        canonical_vector CT>
    requires cpo_invocable<subtract_t, BT, AT> &&
        cpo_invocable<mulacc_t, AT, M, cpo_result_t<subtract_t, BT, AT>, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        AT start, M mask, BT end, CT scale) noexcept {
        return dx::mulacc(start, mask, dx::subtract(end, start), scale);
    }

    template <canonical_vector AT, const_mask_for<AT> M, canonical_vector BT,
        canonical_vector CT>
    requires cpo_invocable<subtract_t, BT, AT> &&
        cpo_invocable<mulacc_t, AT, M, cpo_result_t<subtract_t, BT, AT>, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        AT start, M mask, BT end, CT scale) noexcept {
        return dx::mulacc(start, mask, dx::subtract(end, start), scale);
    }

    template <canonical_vector AT, canonical_mask M, canonical_vector BT,
        canonical_vector CT>
    requires cpo_invocable<subtract_t, BT, AT> &&
        cpo_invocable<mulacc_t, dx::zero_t, M, AT,
            cpo_result_t<subtract_t, BT, AT>, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        dx::zero_t zero, M mask, AT start, BT end, CT scale) noexcept {
        return dx::mulacc(zero, mask, start, dx::subtract(end, start), scale);
    }

    template <canonical_vector AT, canonical_vector BT, canonical_vector CT,
        typename M>
    requires cpo_invocable<lerp_t, AT, BT, CT> &&
        const_mask_for<M, cpo_result_t<lerp_t, AT, BT, CT>> &&
        cpo_invocable<subtract_t, BT, AT> &&
        cpo_invocable<mulacc_t, dx::zero_t, M, AT,
            cpo_result_t<subtract_t, BT, AT>, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        dx::zero_t zero, M mask, AT start, BT end, CT scale) noexcept {
        return dx::mulacc(zero, mask, start, dx::subtract(end, start), scale);
    }

    using ternary_broadcasting_fallback<lerp_t>::operator();
};

template <typename AT, typename BT>
concept unqualified_canonical_lerp = requires {
    {
        lerp(internal::abi<AT>, internal::declarg<AT>(),
            internal::declarg<AT>(), internal::declarg<BT>())
    } -> same_as<AT>;
};

template <typename AT, typename M, typename BT>
concept unqualified_canonical_mlerp = cpo_invocable<lerp_t, AT, AT, BT> &&
    same_as<AT, cpo_result_t<lerp_t, AT, AT, BT>> && requires {
        {
            lerp(internal::abi<AT>, internal::declarg<AT>(),
                internal::declarg<M>(), internal::declarg<AT>(),
                internal::declarg<BT>())
        } -> equivalent_vector_with<cpo_result_t<lerp_t, AT, AT, BT>>;
    };

template <typename AT, typename M, typename BT>
concept unqualified_canonical_zmlerp =
    cpo_invocable<lerp_t, AT, AT, BT> && requires {
        {
            lerp(internal::abi<AT>, dx::zero, internal::declarg<M>(),
                internal::declarg<AT>(), internal::declarg<AT>(),
                internal::declarg<BT>())
        } -> equivalent_vector_with<cpo_result_t<lerp_t, AT, AT, BT>>;
    };

template <>
struct canonical_impl<lerp_t> {
public:
    template <canonical_vector AT, vector_subsumed_by<AT> BT>
    requires canonical_vector<BT> && unqualified_canonical_lerp<AT, BT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr AT operator()(
        AT start, type_identity_t<AT> end, BT scale) noexcept {
        return lerp(internal::abi<AT>, start, end, scale);
    }

    template <canonical_vector AT, broadcastable_to<AT> BT>
    requires unqualified_canonical_lerp<AT, AT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr AT operator()(
        AT start, type_identity_t<AT> end, BT&& scale) noexcept {
        return lerp(internal::abi<AT>, start, end, __DPL forward<BT>(scale));
    }

    template <canonical_vector RT, different_from<RT> LT, typename BT>
    requires convertible_to<LT, RT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr RT operator()(LT start, RT end, BT scale) noexcept(
        noexcept(static_cast<RT>(start)))
    requires requires { operator()(static_cast<RT>(start), end, scale); }
    {
        return operator()(static_cast<RT>(start), end, scale);
    }

    template <canonical_vector AT, vector_subsumed_by<AT> BT>
    requires canonical_vector<BT> &&
        unqualified_canonical_mlerp<AT, simd_mask_type_t<AT>, BT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr AT operator()(type_identity_t<AT> start,
        simd_mask_type_t<AT> mask, AT end, BT scale) noexcept {
        return lerp(internal::abi<AT>, start, mask, end, scale);
    }

    template <canonical_vector AT, const_mask_for<AT> M,
        vector_subsumed_by<AT> BT>
    requires canonical_vector<BT> &&
        unqualified_canonical_mlerp<AT, launder_cmask_t<AT, M>, BT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr AT operator()(
        type_identity_t<AT> start, M cmask, AT end, BT scale) noexcept {
        return lerp(
            internal::abi<AT>, start, dx::to_const_mask<AT>(cmask), end, scale);
    }

    template <canonical_vector AT, vector_subsumed_by<AT> BT>
    requires canonical_vector<BT> &&
        unqualified_canonical_zmlerp<AT, simd_mask_type_t<AT>, BT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr AT operator()(dx::zero_t zero, simd_mask_type_t<AT> mask,
        AT start, type_identity_t<AT> end, BT scale) noexcept {
        return lerp(internal::abi<AT>, zero, mask, start, end, scale);
    }

    template <canonical_vector AT, const_mask_for<AT> M,
        vector_subsumed_by<AT> BT>
    requires canonical_vector<BT> &&
        unqualified_canonical_zmlerp<AT, launder_cmask_t<AT, M>, BT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr AT operator()(dx::zero_t zero, M cmask, AT start,
        type_identity_t<AT> end, BT scale) noexcept {
        return lerp(internal::abi<AT>, zero, dx::to_const_mask<AT>(cmask),
            start, end, scale);
    }

    template <canonical_vector RT, different_from<RT> LT, typename BT>
    requires convertible_to<LT, RT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr RT operator()(dx::zero_t zero, simd_mask_type_t<RT> mask,
        LT start, RT end, BT scale) noexcept(noexcept(static_cast<RT>(start)))
    requires requires {
        operator()(zero, mask, static_cast<RT>(start), end, scale);
    }
    {
        return operator()(zero, mask, static_cast<RT>(start), end, scale);
    }

    template <canonical_vector RT, different_from<RT> LT, const_mask_for<RT> M,
        typename BT>
    requires convertible_to<LT, RT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr RT operator()(dx::zero_t zero, M cmask, LT start, RT end,
        BT scale) noexcept(noexcept(static_cast<RT>(start)))
    requires requires {
        operator()(zero, dx::to_const_mask<RT>(cmask), static_cast<RT>(start),
            end, scale);
    }
    {
        return operator()(zero, dx::to_const_mask<RT>(cmask),
            static_cast<RT>(start), end, scale);
    }
};

template <typename LT, typename RT, typename BT>
concept unqualified_extended_lerp = requires {
    {
        lerp(internal::declarg<LT>(), internal::declarg<RT>(),
            internal::declarg<BT>())
    } -> vector_with_common_abi<LT>;
};

template <typename LT, typename M, typename RT, typename BT>
concept unqualified_extended_mlerp = cpo_invocable<lerp_t, LT, RT, BT> &&
    equivalent_vector_with<LT, cpo_result_t<lerp_t, LT, RT, BT>> && requires {
        {
            lerp(internal::declarg<LT>(), internal::declarg<M>(),
                internal::declarg<RT>(), internal::declarg<BT>())
        } -> equivalent_vector_with<cpo_result_t<lerp_t, LT, RT, BT>>;
    };

template <typename LT, typename M, typename RT, typename BT>
concept unqualified_extended_zmlerp =
    cpo_invocable<lerp_t, LT, RT, BT> && requires {
        {
            lerp(dx::zero, internal::declarg<M>(), internal::declarg<LT>(),
                internal::declarg<RT>(), internal::declarg<BT>())
        } -> equivalent_vector_with<cpo_result_t<lerp_t, LT, RT, BT>>;
    };

template <>
struct extended_impl<lerp_t> {
public:
    template <simd_vector LT, equivalent_vector_with<LT> RT,
        vector_subsumed_by<LT> BT>
    requires (extended_vector<LT> || extended_vector<RT> ||
                 extended_vector<BT>) &&
        unqualified_extended_lerp<LT, RT, BT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(LT&& start, RT&& end, BT&& scale) {
        return lerp(__DPL forward<LT>(start), __DPL forward<RT>(end),
            __DPL forward<BT>(scale));
    }

    template <simd_vector LT, equivalent_vector_with<LT> RT,
        broadcastable_to<LT> BT>
    requires (extended_vector<LT> || extended_vector<RT>) &&
        unqualified_extended_lerp<LT, RT, BT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(LT&& start, RT&& end, BT&& scale) {
        return lerp(__DPL forward<LT>(start), __DPL forward<RT>(end),
            __DPL forward<BT>(scale));
    }

    template <simd_vector LT, exact_mask_for<LT> M,
        equivalent_vector_with<LT> RT, vector_subsumed_by<LT> BT>
    requires (extended_vector<LT> || extended_mask<M> || extended_vector<RT> ||
                 extended_vector<BT>) &&
        unqualified_extended_mlerp<LT, M, RT, BT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        LT&& start, M&& mask, RT&& end, BT&& scale) {
        return lerp( __DPL forward<LT>(start), mask, __DPL forward<RT>(end),
            __DPL forward<BT>(scale));
    }

    template <simd_vector LT, const_mask_for<LT> M,
        equivalent_vector_with<LT> RT, vector_subsumed_by<LT> BT>
    requires (extended_vector<LT> || extended_vector<RT> ||
                 extended_vector<BT>) &&
        unqualified_extended_mlerp<LT, launder_cmask_t<LT, M>, RT, BT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(LT&& start, M mask, RT&& end, BT&& scale) {
        return lerp(__DPL forward<LT>(start), dx::to_const_mask<LT>(mask),
            __DPL forward<RT>(end), __DPL forward<BT>(scale));
    }

    template <simd_vector LT, exact_mask_for<LT> M,
        equivalent_vector_with<LT> RT, vector_subsumed_by<LT> BT>
    requires (extended_vector<LT> || extended_mask<M> || extended_vector<RT> ||
                 extended_vector<BT>) &&
        unqualified_extended_zmlerp<LT, M, RT, BT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, LT&& start, RT&& end, BT&& scale) {
        return lerp(zero, __DPL forward<LT>(start), mask,
            __DPL forward<RT>(end), __DPL forward<BT>(scale));
    }

    template <simd_vector LT, const_mask_for<LT> M,
        equivalent_vector_with<LT> RT, vector_subsumed_by<LT> BT>
    requires (extended_vector<LT> || extended_vector<RT> ||
                 extended_vector<BT>) &&
        unqualified_extended_zmlerp<LT, launder_cmask_t<LT, M>, RT, BT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M mask, LT&& start, RT&& end, BT&& scale) {
        return lerp(zero, __DPL forward<LT>(start),
            dx::to_const_mask<LT>(mask), __DPL forward<RT>(end),
            __DPL forward<BT>(scale));
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::lerp_t lerp{};
} // namespace cpo
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
