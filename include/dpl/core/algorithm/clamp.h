// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/concepts/simd_abi.h"
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
    public maskable_transform_base<clamp_t> {
    using operation_base<clamp_t>::operator();
    using maskable_transform_base<clamp_t>::operator();
};

template <>
struct operation_signature<clamp_t> {
    template <typename AT, typename BT, typename CT>
    requires simd_vector<AT> || simd_vector<BT> || simd_vector<CT>
    static consteval void operator()(AT&&, BT&&, CT&&) noexcept {}
};

template <>
struct fallback_impl<clamp_t> {

    template <simd_vector T, vector_subsumed_by<T> Lo, vector_subsumed_by<T> Hi>
    requires cpo_invocable<min_t, T, Hi> &&
        cpo_invocable<max_t, Lo, cpo_result_t<min_t, T, Hi>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        T&& val, Lo&& low, Hi&& high) noexcept {
        return dx::max(__DPL forward<Lo>(low),
            dx::min(__DPL forward<T>(val), __DPL forward<Hi>(high)));
    }

    template <simd_vector T, broadcastable_to<T> Lo, vector_subsumed_by<T> Hi>
    requires cpo_invocable<clamp_t, T, canonical_type_t<T>, Hi>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        T&& val, Lo&& low, Hi&& high) noexcept {
        return clamp_t::operator()(__DPL forward<T>(val),
            dx::broadcast<T>(__DPL forward<Lo>(low)), __DPL forward<Hi>(high));
    }

    template <simd_vector T, vector_subsumed_by<T> Lo, broadcastable_to<T> Hi>
    requires cpo_invocable<clamp_t, T, Lo, canonical_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        T&& val, Lo&& low, Hi&& high) noexcept {
        return clamp_t::operator()(__DPL forward<T>(val),
            __DPL forward<Lo>(low),
            dx::broadcast<T>(__DPL forward<Hi>(high)));
    }

    template <simd_vector T, broadcastable_to<T> Lo, broadcastable_to<T> Hi>
    requires cpo_invocable<clamp_t, T, canonical_type_t<T>, canonical_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        T&& val, Lo&& low, Hi&& high) noexcept {
        return clamp_t::operator()(__DPL forward<T>(val),
            dx::broadcast<T>(__DPL forward<Lo>(low)),
            dx::broadcast<T>(__DPL forward<Hi>(high)));
    }

    template <simd_vector Lo, common_vector_with<Lo> Hi,
        broadcastable_to<common_canonical_simd_t<Lo, Hi>> T>
    requires cpo_invocable<clamp_t, common_canonical_simd_t<Lo, Hi>, Lo, Hi>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        T&& val, Lo&& low, Hi&& high) noexcept {
        using vec_t = common_canonical_simd_t<Lo, Hi>;
        return clamp_t::operator()(
            dx::broadcast<vec_t>(__DPL forward<T>(val)), __DPL forward<Lo>(low),
            __DPL forward<Hi>(high));
    }

    template <simd_vector Lo, broadcastable_to<Lo> Hi, broadcastable_to<Lo> T>
    requires cpo_invocable<clamp_t, canonical_type_t<Lo>, Lo,
        canonical_type_t<Lo>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        T&& val, Lo&& low, Hi&& high) noexcept {
        return clamp_t::operator()(dx::broadcast<Lo>(__DPL forward<T>(val)),
            __DPL forward<Lo>(low),
            dx::broadcast<Lo>(__DPL forward<Hi>(high)));
    }

    template <simd_vector Hi, broadcastable_to<Hi> Lo, broadcastable_to<Hi> T>
    requires cpo_invocable<clamp_t, canonical_type_t<Hi>, canonical_type_t<Hi>,
        Hi>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        T&& val, Lo&& low, Hi&& high) noexcept {
        return clamp_t::operator()(dx::broadcast<Hi>(__DPL forward<T>(val)),
            dx::broadcast<Hi>(__DPL forward<Lo>(low)), __DPL forward<Hi>(high));
    }

private:
    template <typename V, typename L, typename H>
    using result_t DPL_NODEBUG = cpo_result_t<fallback_impl, V, L, H>;

    template <typename V, typename L, typename H>
    using mask_t DPL_NODEBUG = simd_mask_type_t<result_t<V, L, H>>;

public:
    template <unextended_type T, unextended_type Lo,
        unextended_terminal_of<clamp_t, T, Lo> Hi>
    requires cpo_invocable<fallback_impl, T, Lo, Hi>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<T, Lo, Hi> operator()(result_t<T, Lo, Hi> src,
        mask_t<T, Lo, Hi> mask, T&& val, Lo&& low, Hi&& high) noexcept {
        return dx::max(src, mask, __DPL forward<Lo>(low),
            dx::min(__DPL forward<T>(val), __DPL forward<Hi>(high)));
    }

    template <unextended_type T, unextended_type Lo, unextended_type Hi,
        result_cmask_for<clamp_t, T, Lo, Hi> M>
    requires cpo_invocable<fallback_impl, T, Lo, Hi>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<T, Lo, Hi> operator()(result_t<T, Lo, Hi> src,
        M mask, T&& val, Lo&& low, Hi&& high) noexcept {
        return dx::max(src, mask, __DPL forward<Lo>(low),
            dx::min(__DPL forward<T>(val), __DPL forward<Hi>(high)));
    }

    template <unextended_type T, unextended_type Lo,
        unextended_terminal_of<clamp_t, T, Lo> Hi>
    requires cpo_invocable<fallback_impl, T, Lo, Hi>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<T, Lo, Hi> operator()(dx::zero_t zero,
        mask_t<T, Lo, Hi> mask, T&& val, Lo&& low, Hi&& high) noexcept {
        return dx::max(zero, mask, __DPL forward<Lo>(low),
            dx::min(__DPL forward<T>(val), __DPL forward<Hi>(high)));
    }

    template <unextended_type T, unextended_type Lo, unextended_type Hi,
        result_cmask_for<clamp_t, T, Lo, Hi> M>
    requires cpo_invocable<fallback_impl, T, Lo, Hi>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<T, Lo, Hi> operator()(
        dx::zero_t zero, M mask, T&& val, Lo&& low, Hi&& high) noexcept {
        return dx::max(zero, mask, __DPL forward<Lo>(low),
            dx::min(__DPL forward<T>(val), __DPL forward<Hi>(high)));
    }
};

template <typename V, typename L, typename H, typename T = V>
concept unqualified_canonical_clamp = requires {
    {
        clamp(internal::abi<T>, internal::declarg<V>(), internal::declarg<L>(),
            internal::declarg<H>())
    } -> same_as<T>;
};

template <typename S, typename M, typename V, typename L, typename H>
concept unqualified_canonical_mclamp =
    cpo_invocable<clamp_t, V, L, H> && requires {
        {
            clamp(internal::abi<S>, internal::declarg<S>(),
                internal::declarg<M>(), internal::declarg<V>(),
                internal::declarg<L>(), internal::declarg<H>())
        } -> same_as<S>;
    };

template <>
struct canonical_impl<clamp_t> {
private:
    template <typename V, typename L, typename H>
    using result_t DPL_NODEBUG = cpo_result_t<clamp_t, V, L, H>;
    template <typename V, typename L, typename H>
    using mask_t DPL_NODEBUG = simd_mask_type_t<result_t<V, L, H>>;

public:
    template <canonical_vector T, vector_subsumed_by<T> Lo,
        vector_subsumed_by<T> Hi>
    requires (canonical_vector<Lo> && canonical_vector<Hi>) &&
        unqualified_canonical_clamp<T, Lo, Hi>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val, Lo low, Hi high) noexcept {
        return clamp(internal::abi<T>, val, low, high);
    }

    template <canonical_vector T, vector_subsumed_by<T> Lo,
        broadcastable_to<T> Hi>
    requires canonical_vector<Lo> && unqualified_canonical_clamp<T, Lo, Hi>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val, Lo low, Hi&& high) noexcept {
        return clamp(internal::abi<T>, val, low, __DPL forward<Hi>(high));
    }

    template <canonical_vector T, broadcastable_to<T> Lo,
        vector_subsumed_by<T> Hi>
    requires canonical_vector<Hi> && unqualified_canonical_clamp<T, Lo, Hi>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val, Lo&& low, Hi high) noexcept {
        return clamp(internal::abi<T>, val, __DPL forward<Lo>(low), high);
    }

    template <canonical_vector T, broadcastable_to<T> Lo,
        broadcastable_to<T> Hi>
    requires unqualified_canonical_clamp<T, Lo, Hi>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val, Lo&& low, Hi&& high) noexcept {
        return clamp(internal::abi<T>, val, __DPL forward<Lo>(low),
            __DPL forward<Hi>(high));
    }

    template <canonical_vector Lo, common_vector_with<Lo> Hi,
        broadcastable_to<common_canonical_simd_t<Lo, Hi>> T>
    requires canonical_vector<Hi> &&
        unqualified_canonical_clamp<T, Lo, Hi, common_canonical_simd_t<Lo, Hi>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr common_canonical_simd_t<Lo, Hi> operator()(
        T&& val, Lo low, Hi high) noexcept {
        return clamp(internal::abi<common_abi_t<Lo, Hi>>,
            __DPL forward<T>(val), low, high);
    }

    template <canonical_vector Lo, broadcastable_to<Lo> T,
        broadcastable_to<Lo> Hi>
    requires unqualified_canonical_clamp<T, Lo, Hi, Lo>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr Lo operator()(T&& val, Lo low, Hi&& high) noexcept {
        return clamp(internal::abi<Lo>, __DPL forward<T>(val), low,
            __DPL forward<Hi>(high));
    }

    template <canonical_vector Hi, broadcastable_to<Hi> T,
        broadcastable_to<Hi> Lo>
    requires unqualified_canonical_clamp<T, Lo, Hi, Hi>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr Hi operator()(T&& val, Lo&& low, Hi high) noexcept {
        return clamp(internal::abi<Hi>, __DPL forward<T>(val),
            __DPL forward<Lo>(low), high);
    }

    ///

    template <unextended_type T, unextended_type Lo,
        unextended_terminal_of<clamp_t, T, Lo> Hi>
    requires unqualified_canonical_mclamp<result_t<T, Lo, Hi>,
        mask_t<T, Lo, Hi>, T, Lo, Hi>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<T, Lo, Hi> operator()(result_t<T, Lo, Hi> src,
        mask_t<T, Lo, Hi> mask, T&& val, Lo&& low, Hi&& high) noexcept {
        return clamp(internal::abi<result_t<T, Lo, Hi>>, src, mask,
            __DPL forward<T>(val), __DPL forward<Lo>(low),
            __DPL forward<Hi>(high));
    }

    template <unextended_type T, unextended_type Lo, unextended_type Hi,
        result_cmask_for<clamp_t, T, Lo, Hi> M>
    requires unqualified_canonical_mclamp<result_t<T, Lo, Hi>,
        launder_cmask_t<result_t<T, Lo, Hi>, M>, T, Lo, Hi>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<T, Lo, Hi> operator()(result_t<T, Lo, Hi> src,
        M mask, T&& val, Lo&& low, Hi&& high) noexcept {
        return clamp(internal::abi<result_t<T, Lo, Hi>>, src,
            dx::to_const_mask<result_t<T, Lo, Hi>>(mask), __DPL forward<T>(val),
            __DPL forward<Lo>(low), __DPL forward<Hi>(high));
    }

    template <unextended_type T, unextended_type Lo,
        unextended_terminal_of<clamp_t, T, Lo> Hi>
    requires unqualified_canonical_mclamp<dx::zero_t, mask_t<T, Lo, Hi>, T, Lo,
        Hi>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<T, Lo, Hi> operator()(dx::zero_t zero,
        mask_t<T, Lo, Hi> mask, T&& val, Lo&& low, Hi&& high) noexcept {
        return clamp(internal::abi<result_t<T, Lo, Hi>>, zero, mask,
            __DPL forward<T>(val), __DPL forward<Lo>(low),
            __DPL forward<Hi>(high));
    }

    template <unextended_type T, unextended_type Lo, unextended_type Hi,
        result_cmask_for<clamp_t, T, Lo, Hi> M>
    requires unqualified_canonical_mclamp<dx::zero_t,
        launder_cmask_t<result_t<T, Lo, Hi>, M>, T, Lo, Hi>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<T, Lo, Hi> operator()(
        dx::zero_t zero, M mask, T&& val, Lo&& low, Hi&& high) noexcept {
        return clamp(internal::abi<result_t<T, Lo, Hi>>, zero,
            dx::to_const_mask<result_t<T, Lo, Hi>>(mask), __DPL forward<T>(val),
            __DPL forward<Lo>(low), __DPL forward<Hi>(high));
    }
};

template <typename V, typename L, typename H,
    typename T = common_canonical_simd_t<L, H>>
concept unqualified_extended_clamp = requires {
    {
        clamp(internal::declarg<V>(), internal::declarg<L>(),
            internal::declarg<H>())
    } -> equivalent_vector_with<T>;
};

template <typename S, typename M, typename V, typename L, typename H>
concept unqualified_extended_mclamp_base =
    cpo_invocable<clamp_t, V, L, H> && requires {
        {
            clamp(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<V>(), internal::declarg<L>(),
                internal::declarg<H>())
        } -> equivalent_vector_with<cpo_result_t<clamp_t, V, L, H>>;
    };

template <typename S, typename M, typename V, typename L, typename H>
concept unqualified_extended_mclamp =
    unqualified_extended_mclamp_base<S, M, V, L, H> &&
    equivalent_vector_with<S, cpo_result_t<clamp_t, V, L, H>>;

template <typename M, typename V, typename L, typename H>
concept unqualified_extended_zmclamp =
    unqualified_extended_mclamp_base<dx::zero_t, M, V, L, H>;

template <>
struct extended_impl<clamp_t> {
    //
    template <simd_vector T, vector_subsumed_by<T> Lo, vector_subsumed_by<T> Hi>
    requires (extended_vector<T> || extended_vector<Lo> ||
                 extended_vector<Hi>) &&
        unqualified_extended_clamp<T, Lo, Hi>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val, Lo&& low, Hi&& high) {
        return clamp(__DPL forward<T>(val), __DPL forward<Lo>(low),
            __DPL forward<Hi>(high));
    }

    template <simd_vector T, broadcastable_to<T> Lo, vector_subsumed_by<T> Hi>
    requires (extended_vector<T> || extended_vector<Hi>) &&
        unqualified_extended_clamp<T, Lo, Hi>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val, Lo&& low, Hi&& high) {
        return clamp(__DPL forward<T>(val), __DPL forward<Lo>(low),
            __DPL forward<Hi>(high));
    }

    template <simd_vector T, vector_subsumed_by<T> Lo, broadcastable_to<T> Hi>
    requires (extended_vector<T> || extended_vector<Lo>) &&
        unqualified_extended_clamp<T, Lo, Hi>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val, Lo&& low, Hi&& high) {
        return clamp(__DPL forward<T>(val), __DPL forward<Lo>(low),
            __DPL forward<Hi>(high));
    }

    template <extended_vector T, broadcastable_to<T> Lo, broadcastable_to<T> Hi>
    requires unqualified_extended_clamp<T, Lo, Hi>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val, Lo&& low, Hi&& high) {
        return clamp(__DPL forward<T>(val), __DPL forward<Lo>(low),
            __DPL forward<Hi>(high));
    }

    template <simd_vector Lo, common_vector_with<Lo> Hi,
        broadcastable_to<common_canonical_simd_t<Lo, Hi>> T>
    requires (extended_vector<Lo> || extended_vector<Hi>) &&
        unqualified_extended_clamp<T, Lo, Hi, common_canonical_simd_t<Lo, Hi>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val, Lo&& low, Hi&& high) {
        return clamp(__DPL forward<T>(val), __DPL forward<Lo>(low),
            __DPL forward<Hi>(high));
    }

    template <extended_vector Lo, broadcastable_to<Lo> Hi,
        broadcastable_to<Lo> T>
    requires unqualified_extended_clamp<T, Lo, Hi, Lo>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val, Lo&& low, Hi&& high) {
        return clamp(__DPL forward<T>(val), __DPL forward<Lo>(low),
            __DPL forward<Hi>(high));
    }

    template <extended_vector Hi, broadcastable_to<Hi> Lo,
        broadcastable_to<Hi> T>
    requires unqualified_extended_clamp<T, Lo, Hi, Hi>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val, Lo&& low, Hi&& high) {
        return clamp(__DPL forward<T>(val), __DPL forward<Lo>(low),
            __DPL forward<Hi>(high));
    }
    //

    template <simd_vector S, exact_mask_for<S> M, typename T, typename Lo,
        typename Hi>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T> ||
                 extended_vector<Lo> || extended_vector<Hi>) &&
        unqualified_extended_mclamp<S, M, T, Lo, Hi>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        S&& src, M&& mask, T&& val, Lo&& low, Hi&& high) {
        return clamp(__DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val), __DPL forward<Lo>(low),
            __DPL forward<Hi>(high));
    }

    template <simd_vector S, exact_mask_for<S> M, typename T, typename Lo,
        typename Hi>
    requires (extended_vector<S> || extended_vector<T> || extended_vector<Lo> ||
                 extended_vector<Hi>) &&
        unqualified_extended_mclamp<S, launder_cmask_t<S, M>, T, Lo, Hi>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        S&& src, M mask, T&& val, Lo&& low, Hi&& high) {
        return clamp( __DPL forward<S>(src), dx::to_const_mask<S>(mask),
            __DPL forward<T>(val),__DPL forward<Lo>(low),
            __DPL forward<Hi>(high));
    }

    template <typename T, typename Lo, typename Hi,
        result_mask_for<clamp_t, T, Lo, Hi> M>
    requires (extended_mask<M> || extended_vector<T> || extended_vector<Lo> ||
                 extended_vector<Hi>) &&
        unqualified_extended_zmclamp<M, T, Lo, Hi>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, T&& val, Lo&& low, Hi&& high) {
        return clamp(zero, __DPL forward<M>(mask), __DPL forward<T>(val),
            __DPL forward<Lo>(low), __DPL forward<Hi>(high));
    }

    template <typename T, typename Lo, typename Hi,
        result_cmask_for<clamp_t, T, Lo, Hi> M>
    requires (extended_vector<T> || extended_vector<Lo> ||
                 extended_vector<Hi>) &&
        unqualified_extended_zmclamp<
            launder_cmask_t<cpo_result_t<clamp_t, T, Lo, Hi>, M>, T, Lo, Hi>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M mask, T&& val, Lo&& low, Hi&& high) {
        return clamp(zero,
            dx::to_const_mask<cpo_result_t<clamp_t, T, Lo, Hi>>(mask),
            __DPL forward<T>(val),__DPL forward<Lo>(low),
            __DPL forward<Hi>(high));
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::clamp_t clamp{};
} // namespace cpo
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
