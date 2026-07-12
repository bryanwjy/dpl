// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/dispatch/broadcastable/ternary.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/maskable/transform.h"
#  include "dpl/core/dispatch/operation/algorithm.h"
#  include "dpl/core/operations/minmax.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void clamp(...) noexcept = delete;

struct clamp_t;

struct DPL_EMPTY_BASES clamp_t :
    private algorithm_base<clamp_t>,
    private maskable_transform_base<clamp_t>,
    private ternary_broadcastable_operation<clamp_t> {
    using operation_base<clamp_t>::operator();
    using maskable_transform_base<clamp_t>::operator();
    using ternary_broadcastable_operation<clamp_t>::operator();
};

template <>
struct operation_signature<clamp_t> {
    template <typename AT, typename BT, typename CT>
    requires simd_vector<AT> || simd_vector<BT> || simd_vector<CT>
    static consteval void operator()(AT&&, BT&&, CT&&) noexcept {}
};

template <>
struct fallback_impl<clamp_t> : ternary_broadcasting_fallback<clamp_t> {

    template <canonical_vector AT, canonical_vector BT, canonical_vector CT>
    requires floating_point<simd_element_type_t<AT>> &&
        floating_point<simd_element_type_t<BT>> &&
        floating_point<simd_element_type_t<CT>> &&
        cpo_invocable<min_t, BT, CT> &&
        cpo_invocable<max_t, CT, cpo_result_t<min_t, BT, AT>, AT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        AT val, BT low, CT high) noexcept {
        return dx::max(low, dx::min(val, high));
    }

    using ternary_broadcasting_fallback<clamp_t>::operator();
};

template <>
struct canonical_impl<clamp_t> {

    template <simd_abi AA, common_abi_with<AA> BA, common_abi_with<BA> CA,
        simd_element_for<AA> E, simd_abi A = common_abi_t<AA, BA, CA>>
    requires simd_element_for<E, BA> && simd_element_for<E, CA> &&
        simd_element_for<E, A>
        DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
        static constexpr basic_vector<E, A> operator()(basic_vector<E, AA> val,
            basic_vector<E, BA> low, basic_vector<E, CA> high) noexcept
    requires requires { clamp(internal::abi<A>, val, low, high); }
    {
        return clamp(internal::abi<A>, val, low, high);
    }

    template <simd_abi AA, common_abi_with<AA> BA, simd_element_for<AA> E,
        typename CT, typename A = common_abi_t<AA, BA>>
    requires simd_element_for<E, BA> &&
        broadcastable_to<CT, basic_vector<E, A>>
        DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
        static constexpr basic_vector<E, A> operator()(basic_vector<E, AA> val,
            basic_vector<E, BA> low, CT&& high) noexcept
    requires requires {
        clamp(internal::abi<A>, val, low, __DPL forward<CT>(high));
    }
    {
        return clamp(internal::abi<A>, val, low, __DPL forward<CT>(high));
    }

    template <simd_abi AA, common_abi_with<AA> CA, simd_element_for<AA> E,
        typename BT, typename A = common_abi_t<AA, CA>>
    requires simd_element_for<E, CA> &&
        broadcastable_to<BT, basic_vector<E, A>>
        DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
        static constexpr basic_vector<E, A> operator()(basic_vector<E, AA> val,
            BT&& low, basic_vector<E, CA> high) noexcept
    requires requires {
        clamp(internal::abi<A>, val, __DPL forward<BT>(low), high);
    }
    {
        return clamp(internal::abi<A>, val, __DPL forward<BT>(low), high);
    }

    template <simd_abi BA, common_abi_with<BA> CA, simd_element_for<BA> E,
        typename AT, typename A = common_abi_t<BA, CA>>
    requires simd_element_for<E, CA> &&
        broadcastable_to<AT, basic_vector<E, A>>
        DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
        static constexpr basic_vector<E, A> operator()(AT&& val,
            basic_vector<E, BA> low, basic_vector<E, CA> high) noexcept
    requires requires {
        clamp(internal::abi<A>, __DPL forward<AT>(val), low, high);
    }
    {
        return clamp(internal::abi<A>, __DPL forward<AT>(val), low, high);
    }

    template <simd_abi A, simd_element_for<A> E,
        broadcastable_to<basic_vector<E, A>> BT,
        broadcastable_to<basic_vector<E, A>> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val, BT&& low, CT&& high) noexcept
    requires requires {
        clamp(internal::abi<A>, val, __DPL forward<BT>(low),
            __DPL forward<CT>(high));
    }
    {
        return clamp(internal::abi<A>, val, __DPL forward<BT>(low),
            __DPL forward<CT>(high));
    }

    template <simd_abi A, simd_element_for<A> E,
        broadcastable_to<basic_vector<E, A>> AT,
        broadcastable_to<basic_vector<E, A>> CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        AT&& val, basic_vector<E, A> low, CT&& high) noexcept
    requires requires {
        clamp(internal::abi<A>, __DPL forward<AT>(val), low,
            __DPL forward<CT>(high));
    }
    {
        return clamp(internal::abi<A>, __DPL forward<AT>(val), low,
            __DPL forward<CT>(high));
    }

    template <simd_abi A, simd_element_for<A> E,
        broadcastable_to<basic_vector<E, A>> AT,
        broadcastable_to<basic_vector<E, A>> BT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        AT&& val, BT&& low, basic_vector<E, A> high) noexcept
    requires requires {
        clamp(internal::abi<A>, __DPL forward<AT>(val),
            __DPL forward<BT>(low), high);
    }
    {
        return clamp(internal::abi<A>, __DPL forward<AT>(val),
            __DPL forward<BT>(low), high);
    }
};

template <typename AT, typename BT, typename CT,
    typename A = common_abi_t<AT, BT, CT>>
concept unqualified_extended_clamp = requires(AT a, BT b, CT c) {
    { clamp(a, b, c) } -> vector_with_common_abi<A>;
};

template <>
struct extended_impl<clamp_t> {
    template <simd_vector AT, simd_vector BT, simd_vector CT>
    requires (extended_vector<AT> || extended_vector<BT> ||
                 extended_vector<CT>) &&
        unqualified_extended_clamp<AT, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& val, BT&& low, CT&& high) {
        if constexpr (unqualified_extended_clamp<AT, BT, CT>) {
            return clamp(__DPL forward<AT>(val), __DPL forward<BT>(low),
                __DPL forward<CT>(high));
        } else {
            return fallback_impl<clamp_t>::operator()(__DPL forward<AT>(val),
                __DPL forward<BT>(low), __DPL forward<CT>(high));
        }
    }

    template <simd_vector AT, common_vector_with<AT> BT, typename CT>
    requires (extended_vector<AT> || extended_vector<BT>) && (!simd_type<CT>) &&
        unqualified_extended_clamp<AT, BT, CT, common_abi_t<AT, BT>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& val, BT&& low, CT&& high) {
        return clamp(__DPL forward<AT>(val), __DPL forward<BT>(low),
            __DPL forward<CT>(high));
    }

    template <simd_vector AT, typename BT, common_vector_with<AT> CT>
    requires (extended_vector<AT> || extended_vector<CT>) && (!simd_type<BT>) &&
        unqualified_extended_clamp<AT, BT, CT, common_abi_t<AT, CT>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& val, BT&& low, CT&& high) {
        return clamp(__DPL forward<AT>(val), __DPL forward<BT>(low),
            __DPL forward<CT>(high));
    }

    template <typename AT, simd_vector BT, common_vector_with<BT> CT>
    requires (extended_vector<BT> || extended_vector<CT>) && (!simd_type<AT>) &&
        unqualified_extended_clamp<AT, BT, CT, common_abi_t<BT, CT>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& val, BT&& low, CT&& high) {
        return clamp(__DPL forward<AT>(val), __DPL forward<BT>(low),
            __DPL forward<CT>(high));
    }

    template <extended_vector AT, broadcastable_to<result_or_decayed_t<AT>> BT,
        broadcastable_to<result_or_decayed_t<AT>> CT>
    requires unqualified_extended_clamp<AT, BT, CT, simd_abi_type_t<AT>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& val, BT&& low, CT&& high) {
        return clamp(__DPL forward<AT>(val), __DPL forward<BT>(low),
            __DPL forward<CT>(high));
    }

    template <extended_vector BT, broadcastable_to<result_or_decayed_t<BT>> AT,
        broadcastable_to<result_or_decayed_t<BT>> CT>
    requires unqualified_extended_clamp<AT, BT, CT, simd_abi_type_t<BT>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& val, BT&& low, CT&& high) {
        return clamp(__DPL forward<AT>(val), __DPL forward<BT>(low),
            __DPL forward<CT>(high));
    }

    template <extended_vector CT, broadcastable_to<result_or_decayed_t<CT>> AT,
        broadcastable_to<result_or_decayed_t<CT>> BT>
    requires unqualified_extended_clamp<AT, BT, CT, simd_abi_type_t<CT>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& val, BT&& low, CT&& high) {
        return clamp(__DPL forward<AT>(val), __DPL forward<BT>(low),
            __DPL forward<CT>(high));
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::clamp_t clamp{};
} // namespace cpo
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
