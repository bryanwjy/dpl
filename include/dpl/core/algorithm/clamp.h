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
    public algorithm_base<clamp_t>,
    public maskable_transform_base<clamp_t>,
    public ternary_broadcastable_operation<clamp_t> {
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

    template <simd_vector T, vector_subsumed_by<T> Lo, vector_subsumed_by<T> Hi>
    requires cpo_invocable<min_t, T, Hi> &&
        cpo_invocable<max_t, Lo, cpo_result_t<min_t, T, Hi>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        T&& val, Lo&& low, Hi&& high) noexcept {
        return dx::max(__DPL forward<Lo>(low),
            dx::min(__DPL forward<T>(val), __DPL forward<Hi>(high)));
    }

    using ternary_broadcasting_fallback<clamp_t>::operator();
};

template <>
struct canonical_impl<clamp_t> {

    template <canonical_vector T, vector_subsumed_by<T> Lo,
        vector_subsumed_by<T> Hi>
    requires (canonical_vector<Lo> && canonical_vector<Hi>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val, Lo low, Hi high) noexcept
    requires requires { clamp(internal::abi<T>, val, low, high); }
    {
        return clamp(internal::abi<T>, val, low, high);
    }

    template <canonical_vector T, vector_subsumed_by<T> Lo,
        broadcastable_to<T> Hi>
    requires canonical_vector<Lo>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val, Lo low, Hi&& high) noexcept
    requires requires {
        clamp(internal::abi<T>, val, low, __DPL forward<Hi>(high));
    }
    {
        return clamp(internal::abi<T>, val, low, __DPL forward<Hi>(high));
    }

    template <canonical_vector T, broadcastable_to<T> Lo,
        vector_subsumed_by<T> Hi>
    requires canonical_vector<Hi>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val, Lo&& low, Hi high) noexcept
    requires requires {
        clamp(internal::abi<T>, val, __DPL forward<Lo>(low), high);
    }
    {
        return clamp(internal::abi<T>, val, __DPL forward<Lo>(low), high);
    }

    template <canonical_vector T, broadcastable_to<T> Lo,
        broadcastable_to<T> Hi>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val, Lo&& low, Hi&& high) noexcept
    requires requires {
        clamp(internal::abi<T>, val, __DPL forward<Lo>(low),
            __DPL forward<Hi>(high));
    }
    {
        return clamp(internal::abi<T>, val, __DPL forward<Lo>(low),
            __DPL forward<Hi>(high));
    }

    template <canonical_vector Lo, common_vector_with<Lo> Hi,
        broadcastable_to<common_canonical_simd_t<Lo, Hi>> T>
    requires canonical_vector<Hi>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val, Lo low, Hi high) noexcept
    requires requires {
        clamp(internal::abi<common_abi_t<Lo, Hi>>, __DPL forward<T>(val), low,
            high);
    }
    {
        return clamp(internal::abi<common_abi_t<Lo, Hi>>,
            __DPL forward<T>(val), low, high);
    }

    template <canonical_vector Lo, broadcastable_to<Lo> T,
        broadcastable_to<Lo> Hi>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr Lo operator()(T&& val, Lo low, Hi&& high) noexcept
    requires requires {
        clamp(internal::abi<Lo>, __DPL forward<T>(val), low,
            __DPL forward<Hi>(high));
    }
    {
        return clamp(internal::abi<Lo>, __DPL forward<T>(val), low,
            __DPL forward<Hi>(high));
    }

    template <canonical_vector Hi, broadcastable_to<Hi> T,
        broadcastable_to<Hi> Lo>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr Hi operator()(T&& val, Lo&& low, Hi high) noexcept
    requires requires {
        clamp(internal::abi<Hi>, __DPL forward<T>(val),
            __DPL forward<Lo>(low), high);
    }
    {
        return clamp(internal::abi<Hi>, __DPL forward<T>(val),
            __DPL forward<Lo>(low), high);
    }
};

template <typename AT, typename BT, typename CT,
    typename A = common_abi_t<AT, BT, CT>>
concept unqualified_extended_clamp = requires(AT a, BT b, CT c) {
    { clamp(a, b, c) } -> vector_with_common_abi<A>;
};

template <>
struct extended_impl<clamp_t> {
    template <simd_vector AT, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    requires (extended_vector<AT> || extended_vector<BT> ||
                 extended_vector<CT>) &&
        unqualified_extended_clamp<AT, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& val, BT&& low, CT&& high) {
        return clamp(__DPL forward<AT>(val), __DPL forward<BT>(low),
            __DPL forward<CT>(high));
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
