// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/cpo_invocable.h"
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/dispatch/broadcastable/ternary.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/maskable/transform.h"
#  include "dpl/core/dispatch/operation/math.h"
#  include "dpl/core/operations/arithmetic.h"
#  include "dpl/core/type_traits/details/cpo_result.h"
#  include "dpl/std/concepts/floating_point.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void lerp(...) noexcept = delete;

struct lerp_t;

struct DPL_EMPTY_BASES lerp_t :
    private math_operation_base<lerp_t>,
    private maskable_transform_base<lerp_t>,
    private ternary_broadcastable_operation<lerp_t> {
    using operation_base<lerp_t>::operator();
    using maskable_transform_base<lerp_t>::operator();
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
    requires floating_point<simd_element_type_t<AT>> &&
        floating_point<simd_element_type_t<BT>> &&
        floating_point<simd_element_type_t<CT>> &&
        cpo_invocable<subtract_t, BT, AT> &&
        cpo_invocable<fmadd_t, CT, cpo_result_t<subtract_t, BT, AT>, AT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        AT start, BT end, CT scale) noexcept {
        return dx::fmadd(scale, dx::subtract(end, start), start);
    }

    using ternary_broadcasting_fallback<lerp_t>::operator();
};

template <typename AT, typename BT, typename CT,
    typename A = common_abi_t<AT, BT, CT>>
concept unqualified_canonical_lerp = requires {
    {
        lerp(internal::abi<A>, internal::declarg<AT>(), internal::declarg<BT>(),
            internal::declarg<CT>())
    } -> canonical_vector;
};

template <typename S, typename M, typename AT, typename BT, typename CT,
    typename A = common_abi_t<AT, BT, CT>>
concept unqualified_canonical_mlerp = cpo_invocable<lerp_t, AT, BT, CT> &&
    (!simd_type<S> ||
        equivalent_vector_with<S, cpo_result_t<lerp_t, AT, BT, CT>>) &&
    requires {
        {
            lerp(internal::abi<A>, internal::declarg<S>(),
                internal::declarg<M>(), internal::declarg<AT>(),
                internal::declarg<BT>(), internal::declarg<CT>())
        } -> equivalent_vector_with<cpo_result_t<lerp_t, AT, BT, CT>>;
    };

template <>
struct canonical_impl<lerp_t> {
private:
    template <typename AT, typename BT, typename CT>
    using result_t DPL_NODEBUG =
        basic_vector<simd_element_type_t<AT>, common_abi_t<AT, BT, CT>>;

    template <typename AT, typename BT, typename CT>
    using mask_t DPL_NODEBUG =
        basic_mask<simd_element_type_t<AT>, common_abi_t<AT, BT, CT>>;

    template <typename L, typename R>
    using vector_t DPL_NODEBUG =
        basic_vector<simd_element_type_t<L>, common_abi_t<L, R>>;

public:
    template <canonical_vector AT, common_vector_with<AT> BT,
        common_vector_with<BT> CT, typename A = common_abi_t<AT, BT, CT>>
    requires canonical_vector<BT> && canonical_vector<CT> &&
        unqualified_canonical_lerp<AT, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<AT, BT, CT> operator()(
        AT start, BT end, CT scale) noexcept {
        return lerp(internal::abi<A>, start, end, scale);
    }

    template <canonical_vector AT, common_vector_with<AT> BT,
        broadcastable_to<vector_t<AT, BT>> CT,
        typename A = common_abi_t<AT, BT>>
    requires canonical_vector<BT> && unqualified_canonical_lerp<AT, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr vector_t<AT, BT> operator()(
        AT start, BT end, CT&& scale) noexcept {
        return lerp(internal::abi<A>, start, end, __DPL forward<CT>(scale));
    }

    template <canonical_vector AT, common_vector_with<AT> CT,
        broadcastable_to<vector_t<AT, CT>> BT,
        typename A = common_abi_t<AT, CT>>
    requires canonical_vector<CT> && unqualified_canonical_lerp<AT, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr vector_t<AT, CT> operator()(
        AT start, BT&& end, CT scale) noexcept {
        return lerp(internal::abi<A>, start, __DPL forward<BT>(end), scale);
    }

    template <canonical_vector BT, common_vector_with<BT> CT,
        broadcastable_to<vector_t<BT, CT>> AT,
        typename A = common_abi_t<BT, CT>>
    requires canonical_vector<CT> && unqualified_canonical_lerp<AT, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr vector_t<BT, CT> operator()(
        AT&& start, BT end, CT scale) noexcept {
        return lerp(internal::abi<A>, __DPL forward<AT>(start), end, scale);
    }

    template <canonical_vector AT, broadcastable_to<AT> BT,
        broadcastable_to<AT> CT, typename A = simd_abi_type_t<AT>>
    requires unqualified_canonical_lerp<AT, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr AT operator()(AT start, BT&& end, CT&& scale) noexcept {
        return lerp(internal::abi<A>, start, __DPL forward<BT>(end),
            __DPL forward<CT>(scale));
    }

    template <canonical_vector BT, broadcastable_to<BT> AT,
        broadcastable_to<BT> CT, typename A = simd_abi_type_t<BT>>
    requires unqualified_canonical_lerp<AT, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr BT operator()(AT&& start, BT end, CT&& scale) noexcept {
        return lerp(internal::abi<A>, __DPL forward<AT>(start), end,
            __DPL forward<CT>(scale));
    }

    template <canonical_vector CT, broadcastable_to<CT> AT,
        broadcastable_to<CT> BT, typename A = simd_abi_type_t<CT>>
    requires unqualified_canonical_lerp<AT, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr CT operator()(AT&& start, BT&& end, CT scale) noexcept {
        return lerp(internal::abi<A>, __DPL forward<AT>(start),
            __DPL forward<BT>(end), scale);
    }

    template <canonical_vector AT, common_vector_with<AT> BT,
        common_vector_with<BT> CT, typename A = common_abi_t<AT, BT, CT>>
    requires canonical_vector<BT> && canonical_vector<CT> &&
        unqualified_canonical_mlerp<result_t<AT, BT, CT>, mask_t<AT, BT, CT>,
            AT, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<AT, BT, CT> operator()(result_t<AT, BT, CT> src,
        mask_t<AT, BT, CT> mask, AT start, BT end, CT scale) noexcept {
        return lerp(internal::abi<A>, src, mask, start, end, scale);
    }

    template <canonical_vector AT, common_vector_with<AT> BT,
        common_vector_with<BT> CT, const_mask_for<result_t<AT, BT, CT>> M,
        typename A = common_abi_t<AT, BT, CT>>
    requires canonical_vector<BT> && canonical_vector<CT> &&
        unqualified_canonical_mlerp<result_t<AT, BT, CT>,
            launder_cmask_t<result_t<AT, BT, CT>, M>, AT, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<AT, BT, CT> operator()(result_t<AT, BT, CT> src,
        M cmask, AT start, BT end, CT scale) noexcept {
        return lerp(internal::abi<A>, src,
            dx::to_const_mask<result_t<AT, BT, CT>>(cmask), start, end, scale);
    }

    template <canonical_vector AT, common_vector_with<AT> BT,
        common_vector_with<BT> CT, typename A = common_abi_t<AT, BT, CT>>
    requires canonical_vector<BT> && canonical_vector<CT> &&
        unqualified_canonical_mlerp<dx::zero_t, mask_t<AT, BT, CT>, AT, BT, CT,
            A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<AT, BT, CT> operator()(dx::zero_t zero,
        mask_t<AT, BT, CT> mask, AT start, BT end, CT scale) noexcept {
        return lerp(internal::abi<A>, zero, mask, start, end, scale);
    }

    template <canonical_vector AT, common_vector_with<AT> BT,
        common_vector_with<BT> CT, const_mask_for<result_t<AT, BT, CT>> M,
        typename A = common_abi_t<AT, BT, CT>>
    requires canonical_vector<BT> && canonical_vector<CT> &&
        unqualified_canonical_mlerp<dx::zero_t, mask_t<AT, BT, CT>, AT, BT, CT,
            A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<AT, BT, CT> operator()(
        dx::zero_t zero, M cmask, AT start, BT end, CT scale) noexcept {
        return lerp(internal::abi<A>, zero,
            dx::to_const_mask<result_t<AT, BT, CT>>(cmask), start, end, scale);
    }
};

template <typename AT, typename BT, typename CT,
    typename A = common_abi_t<AT, BT, CT>>
concept unqualified_extended_lerp = requires {
    {
        lerp(internal::declarg<AT>(), internal::declarg<BT>(),
            internal::declarg<CT>())
    } -> vector_with_common_abi<A>;
};

template <typename S, typename M, typename AT, typename BT, typename CT>
concept unqualified_extended_mlerp = cpo_invocable<lerp_t, AT, BT, CT> &&
    (!simd_type<S> ||
        equivalent_vector_with<S, cpo_result_t<lerp_t, AT, BT, CT>>) &&
    requires {
        {
            lerp(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<AT>(), internal::declarg<BT>(),
                internal::declarg<CT>())
        } -> equivalent_vector_with<cpo_result_t<lerp_t, AT, BT, CT>>;
    };

template <>
struct extended_impl<lerp_t> {
private:
    template <typename L, typename R>
    using vector_t DPL_NODEBUG =
        basic_vector<simd_element_type_t<L>, common_abi_t<L, R>>;

public:
    template <simd_vector AT, common_vector_with<AT> BT,
        common_vector_with<BT> CT, typename A = common_abi_t<AT, BT, CT>>
    requires (extended_vector<AT> || extended_vector<BT> ||
                 extended_vector<CT>) &&
        unqualified_extended_lerp<AT, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& start, BT&& end, CT&& scale) {
        return lerp(__DPL forward<AT>(start), __DPL forward<BT>(end),
            __DPL forward<CT>(scale));
    }

    template <simd_vector AT, common_vector_with<AT> BT,
        broadcastable_to<vector_t<AT, BT>> CT,
        typename A = common_abi_t<AT, BT>>
    requires (extended_vector<AT> || extended_vector<BT>) &&
        unqualified_extended_lerp<AT, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& start, BT&& end, CT&& scale) {
        return lerp(__DPL forward<AT>(start), __DPL forward<BT>(end),
            __DPL forward<CT>(scale));
    }

    template <simd_vector AT, common_vector_with<AT> CT,
        broadcastable_to<vector_t<AT, CT>> BT,
        typename A = common_abi_t<AT, CT>>
    requires (extended_vector<AT> || extended_vector<CT>) &&
        unqualified_extended_lerp<AT, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& start, BT&& end, CT&& scale) {
        return lerp(__DPL forward<AT>(start), __DPL forward<BT>(end),
            __DPL forward<CT>(scale));
    }

    template <simd_vector BT, common_vector_with<BT> CT,
        broadcastable_to<vector_t<BT, CT>> AT,
        typename A = common_abi_t<BT, CT>>
    requires (extended_vector<BT> || extended_vector<CT>) &&
        unqualified_extended_lerp<AT, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& start, BT&& end, CT&& scale) {
        return lerp(__DPL forward<AT>(start), __DPL forward<BT>(end),
            __DPL forward<CT>(scale));
    }

    template <extended_vector AT, broadcastable_to<AT> BT,
        broadcastable_to<AT> CT, typename A = simd_abi_type_t<AT>>
    requires unqualified_extended_lerp<AT, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& start, BT&& end, CT&& scale) {
        return lerp(__DPL forward<AT>(start), __DPL forward<BT>(end),
            __DPL forward<CT>(scale));
    }

    template <extended_vector BT, broadcastable_to<BT> AT,
        broadcastable_to<BT> CT, typename A = simd_abi_type_t<BT>>
    requires unqualified_extended_lerp<AT, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& start, BT&& end, CT&& scale) {
        return lerp(__DPL forward<AT>(start), __DPL forward<BT>(end),
            __DPL forward<CT>(scale));
    }

    template <extended_vector CT, broadcastable_to<CT> AT,
        broadcastable_to<CT> BT, typename A = simd_abi_type_t<CT>>
    requires unqualified_extended_lerp<AT, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& start, BT&& end, CT&& scale) {
        return lerp(__DPL forward<AT>(start), __DPL forward<BT>(end),
            __DPL forward<CT>(scale));
    }

    template <simd_vector S, exact_mask_for<S> M, simd_vector AT,
        common_vector_with<AT> BT, common_vector_with<BT> CT,
        typename A = common_abi_t<AT, BT, CT>>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<AT> ||
                 extended_vector<BT> || extended_vector<CT>) &&
        unqualified_extended_mlerp<S, M, AT, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        S&& src, M&& mask, AT&& start, BT&& end, CT&& scale) {
        return lerp(__DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<AT>(start), __DPL forward<BT>(end),
            __DPL forward<CT>(scale));
    }

    template <simd_vector S, const_mask_for<S> M, simd_vector AT,
        common_vector_with<AT> BT, common_vector_with<BT> CT>
    requires (extended_vector<S> || extended_vector<AT> ||
                 extended_vector<BT> || extended_vector<CT>) &&
        unqualified_extended_mlerp<S, launder_cmask_t<S, M>, AT, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        S&& src, M cmask, AT&& start, BT&& end, CT&& scale) {
        return lerp(__DPL forward<S>(src), dx::to_const_mask<S>(cmask),
            __DPL forward<AT>(start), __DPL forward<BT>(end),
            __DPL forward<CT>(scale));
    }

    template <simd_vector AT, common_vector_with<AT> BT,
        common_vector_with<BT> CT, result_mask_for<lerp_t, AT, BT, CT> M>
    requires (extended_mask<M> || extended_vector<AT> || extended_vector<BT> ||
                 extended_vector<CT>) &&
        unqualified_extended_mlerp<dx::zero_t, M, AT, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, AT&& start, BT&& end, CT&& scale) {
        return lerp(zero, __DPL forward<M>(mask), __DPL forward<AT>(start),
            __DPL forward<BT>(end), __DPL forward<CT>(scale));
    }

    template <simd_vector AT, common_vector_with<AT> BT,
        common_vector_with<BT> CT, result_cmask_for<lerp_t, AT, BT, CT> M>
    requires (extended_vector<AT> || extended_vector<BT> ||
                 extended_vector<CT>) &&
        unqualified_extended_mlerp<dx::zero_t,
            launder_cmask_t<cpo_result_t<lerp_t, AT, BT, CT>, M>, AT, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M cmask, AT&& start, BT&& end, CT&& scale) {
        return lerp(zero,
            dx::to_const_mask<cpo_result_t<lerp_t, AT, BT, CT>>(cmask),
            __DPL forward<AT>(start), __DPL forward<BT>(end),
            __DPL forward<CT>(scale));
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::lerp_t lerp{};
} // namespace cpo
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
