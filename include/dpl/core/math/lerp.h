// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/mulx.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/cpo_invocable.h"
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/maskable/transform.h"
#  include "dpl/core/dispatch/operation/math.h"
#  include "dpl/core/operations/select.h"
#  include "dpl/core/type_traits/details/cpo_result.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void lerp(...) noexcept = delete;

struct lerp_t;

struct DPL_EMPTY_BASES lerp_t :
    public math_operation_base<lerp_t>,
    public maskable_transform_base<lerp_t> {
    using operation_base<lerp_t>::operator();
    using maskable_transform_base<lerp_t>::operator();
};

template <>
struct operation_signature<lerp_t> {
    template <typename AT, typename BT, typename CT>
    requires simd_vector<AT> || simd_vector<BT> || simd_vector<CT>
    static consteval void operator()(AT&&, BT&&, CT&&) noexcept {}
};

template <>
struct fallback_impl<lerp_t> {
private:
    template <typename A, typename B, typename C>
    using result_t DPL_NODEBUG = cpo_result_t<lerp_t, A const&, B, C>;
    template <typename A, typename B, typename C>
    using mask_t DPL_NODEBUG = simd_mask_type_t<result_t<A, B, C>>;
    template <typename A, typename B, typename C>
    using main_result_t DPL_NODEBUG = cpo_result_t<mulacc_t, A const&,
        cpo_result_t<subtract_t, B, A const&>, C>;

public:
    template <unextended_type AT, unextended_type BT, unextended_type CT>
    requires cpo_invocable<subtract_t, BT, AT const&> &&
        cpo_invocable<mulacc_t, AT const&,
            cpo_result_t<subtract_t, BT, AT const&>, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr main_result_t<AT, BT, CT>
        DPL_VECTORCALL operator()(
            AT const& start, BT&& end, CT&& scale) noexcept {
        return dx::mulacc(start, dx::subtract(__DPL forward<BT>(end), start),
            __DPL forward<CT>(scale));
    }

    template <unextended_type AT, unextended_type BT, unextended_type CT>
    requires cpo_invocable<lerp_t, AT, BT, CT> &&
        cpo_invocable<mulacc_t, result_t<AT, BT, CT>, mask_t<AT, BT, CT>,
            AT const&, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr result_t<AT, BT, CT>
        DPL_VECTORCALL operator()(result_t<AT, BT, CT> src,
            mask_t<AT, BT, CT> mask, AT const& start, BT&& end,
            CT&& scale) noexcept {
        return dx::mulacc(src, mask, start,
            dx::subtract(__DPL forward<BT>(end), start),
            __DPL forward<CT>(scale));
    }

    template <unextended_type AT, unextended_type BT, unextended_type CT,
        result_cmask_for<lerp_t, AT, BT, CT> M>
    requires cpo_invocable<mulacc_t, result_t<AT, BT, CT>, M, AT const&, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr result_t<AT, BT, CT>
        DPL_VECTORCALL operator()(result_t<AT, BT, CT> src, M mask,
            AT const& start, BT&& end, CT&& scale) noexcept {
        return dx::mulacc(src, mask, start,
            dx::subtract(__DPL forward<BT>(end), start),
            __DPL forward<CT>(scale));
    }

    template <unextended_type AT, unextended_type BT, unextended_type CT>
    requires cpo_invocable<lerp_t, AT, BT, CT> &&
        cpo_invocable<mulacc_t, dx::zero_t, mask_t<AT, BT, CT>, AT const&, BT,
            CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr result_t<AT, BT, CT>
        DPL_VECTORCALL operator()(dx::zero_t zero, mask_t<AT, BT, CT> mask,
            AT const& start, BT&& end, CT&& scale) noexcept {
        return dx::mulacc(zero, mask, start,
            dx::subtract(__DPL forward<BT>(end), start),
            __DPL forward<CT>(scale));
    }

    template <unextended_type AT, unextended_type BT, unextended_type CT,
        result_cmask_for<lerp_t, AT const&, BT, CT> M>
    requires cpo_invocable<lerp_t, AT const&, BT, CT> &&
        cpo_invocable<mulacc_t, dx::zero_t, M, AT const&, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr result_t<AT, BT, CT>
        DPL_VECTORCALL operator()(dx::zero_t zero, M mask, AT const& start,
            BT&& end, CT&& scale) noexcept {
        return dx::mulacc(zero, mask, start,
            dx::subtract(__DPL forward<BT>(end), start),
            __DPL forward<CT>(scale));
    }
};

template <typename AT, typename BT, typename CT,
    typename T = common_canonical_simd_t<AT, BT, CT>>
concept unqualified_canonical_lerp = requires {
    {
        lerp(internal::abi<T>, internal::declarg<AT>(), internal::declarg<BT>(),
            internal::declarg<CT>())
    } -> same_as<T>;
};

template <typename M, typename AT, typename BT, typename CT,
    typename T = common_canonical_simd_t<AT, BT, CT>>
concept unqualified_canonical_mlerp =
    cpo_invocable<lerp_t, AT, BT, CT> && requires {
        {
            lerp(internal::abi<T>, internal::declarg<T>(),
                internal::declarg<M>(), internal::declarg<AT>(),
                internal::declarg<BT>(), internal::declarg<CT>())
        } -> same_as<T>;
    };

template <typename M, typename AT, typename BT, typename CT,
    typename T = common_canonical_simd_t<AT, BT, CT>>
concept unqualified_canonical_zmlerp =
    cpo_invocable<lerp_t, AT, BT, CT> && requires {
        {
            lerp(internal::abi<T>, dx::zero, internal::declarg<M>(),
                internal::declarg<AT>(), internal::declarg<BT>(),
                internal::declarg<CT>())
        } -> same_as<T>;
    };

template <>
struct canonical_impl<lerp_t> {
private:
    template <typename A, typename B, typename C>
    using result_t DPL_NODEBUG = cpo_result_t<canonical_impl, A const&, B, C>;
    template <typename A, typename B, typename C>
    using mask_t DPL_NODEBUG = simd_mask_type_t<result_t<A, B, C>>;

public:
    template <canonical_vector AT, common_vector_with<AT> BT>
    requires canonical_vector<BT> && unqualified_canonical_lerp<AT, AT, BT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr common_canonical_simd_t<AT, BT> operator()(
        AT start, AT end, BT scale) noexcept {
        return lerp(internal::abi<common_abi_t<AT, BT>>, start, end, scale);
    }

    template <canonical_vector AT, broadcastable_to<AT> BT>
    requires unqualified_canonical_lerp<AT, AT, BT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr AT operator()(AT start, AT end, BT&& scale) noexcept {
        return lerp(internal::abi<AT>, start, end, __DPL forward<BT>(scale));
    }

    template <canonical_vector AT, broadcastable_to<AT> BT,
        common_vector_with<AT> CT>
    requires canonical_vector<CT> && unqualified_canonical_lerp<AT, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr common_canonical_simd_t<AT, CT> operator()(
        AT start, BT&& end, CT scale) noexcept {
        return lerp(internal::abi<common_abi_t<AT, CT>>, start,
            __DPL forward<BT>(end), scale);
    }

    template <canonical_vector BT, broadcastable_to<BT> AT,
        common_vector_with<BT> CT>
    requires canonical_vector<CT> && unqualified_canonical_lerp<AT, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr common_canonical_simd_t<BT, CT> operator()(
        AT&& start, BT end, CT scale) noexcept {
        return lerp(internal::abi<common_abi_t<BT, CT>>,
            __DPL forward<AT>(start), end, scale);
    }

    template <canonical_vector AT, broadcastable_to<AT> BT,
        broadcastable_to<AT> CT>
    requires unqualified_canonical_lerp<AT, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr AT operator()(AT start, BT&& end, CT&& scale) noexcept {
        return lerp(internal::abi<AT>, start, __DPL forward<BT>(end),
            __DPL forward<CT>(scale));
    }

    template <canonical_vector BT, broadcastable_to<BT> AT,
        broadcastable_to<BT> CT>
    requires unqualified_canonical_lerp<AT, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr BT operator()(AT&& start, BT end, CT&& scale) noexcept {
        return lerp(internal::abi<BT>, __DPL forward<AT>(start), end,
            __DPL forward<CT>(scale));
    }

    template <unextended_type AT, unextended_type BT, unextended_type CT>
    requires unqualified_canonical_mlerp<mask_t<AT, BT, CT>, AT, BT, CT,
        result_t<AT, BT, CT>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<AT, BT, CT> operator()(result_t<AT, BT, CT> src,
        mask_t<AT, BT, CT> mask, AT&& start, BT&& end, CT&& scale) noexcept {
        return lerp(internal::abi<result_t<AT, BT, CT>>, src, mask,
            __DPL forward<AT>(start), __DPL forward<BT>(end),
            __DPL forward<CT>(scale));
    }

    template <unextended_type AT, unextended_type BT, unextended_type CT,
        result_cmask_for<AT, BT, CT> M>
    requires unqualified_canonical_mlerp<
        launder_cmask_t<result_t<AT, BT, CT>, M>, AT, BT, CT,
        result_t<AT, BT, CT>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<AT, BT, CT> operator()(result_t<AT, BT, CT> src,
        M mask, AT&& start, BT&& end, CT&& scale) noexcept {
        using T = result_t<AT, BT, CT>;
        return lerp(internal::abi<T>, src, dx::to_const_mask<T>(mask),
            __DPL forward<AT>(start), __DPL forward<BT>(end),
            __DPL forward<CT>(scale));
    }

    template <unextended_type AT, unextended_type BT, unextended_type CT>
    requires unqualified_canonical_zmlerp<mask_t<AT, BT, CT>, AT, BT, CT,
        result_t<AT, BT, CT>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<AT, BT, CT> operator()(dx::zero_t zero,
        mask_t<AT, BT, CT> mask, AT&& start, BT&& end, CT&& scale) noexcept {
        return lerp(internal::abi<result_t<AT, BT, CT>>, zero, mask,
            __DPL forward<AT>(start), __DPL forward<BT>(end),
            __DPL forward<CT>(scale));
    }

    template <unextended_type AT, unextended_type BT, unextended_type CT,
        result_cmask_for<AT, BT, CT> M>
    requires unqualified_canonical_zmlerp<
        launder_cmask_t<result_t<AT, BT, CT>, M>, AT, BT, CT,
        result_t<AT, BT, CT>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<AT, BT, CT> operator()(
        dx::zero_t zero, M mask, AT&& start, BT&& end, CT&& scale) noexcept {
        using T = result_t<AT, BT, CT>;
        return lerp(internal::abi<T>, zero, dx::to_const_mask<T>(mask),
            __DPL forward<AT>(start), __DPL forward<BT>(end),
            __DPL forward<CT>(scale));
    }
};

template <typename AT, typename BT, typename CT, typename T>
concept unqualified_extended_lerp = requires {
    {
        lerp(internal::declarg<AT>(), internal::declarg<BT>(),
            internal::declarg<CT>())
    } -> equivalent_vector_with<T>;
};

template <typename S, typename M, typename AT, typename BT, typename CT>
concept unqualified_extended_mlerp0 =
    cpo_invocable<lerp_t, AT, BT, CT> && requires {
        {
            lerp(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<AT>(), internal::declarg<BT>(),
                internal::declarg<CT>())
        } -> equivalent_vector_with<cpo_result_t<lerp_t, AT, BT, CT>>;
    };

template <typename S, typename M, typename AT, typename BT, typename CT>
concept unqualified_extended_mlerp =
    unqualified_extended_mlerp0<S, M, AT, BT, CT> &&
    equivalent_vector_with<S, cpo_result_t<lerp_t, AT, BT, CT>>;

template <typename M, typename AT, typename BT, typename CT>
concept unqualified_extended_zmlerp =
    unqualified_extended_mlerp0<dx::zero_t, M, AT, BT, CT>;

template <>
struct extended_impl<lerp_t> {
public:
    template <simd_vector AT, equivalent_vector_with<AT> BT,
        common_vector_with<AT> CT>
    requires (extended_vector<AT> || extended_vector<BT> ||
                 extended_vector<CT>) &&
        unqualified_extended_lerp<AT, BT, CT, common_canonical_simd_t<AT, CT>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        AT&& start, BT&& end, CT&& scale) noexcept {
        return lerp(__DPL forward<AT>(start), __DPL forward<BT>(end),
            __DPL forward<CT>(scale));
    }

    template <simd_vector AT, equivalent_vector_with<AT> BT,
        broadcastable_to<AT> CT>
    requires (extended_vector<AT> || extended_vector<BT>) &&
        unqualified_extended_lerp<AT, BT, CT, AT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        AT&& start, BT&& end, CT&& scale) noexcept {
        return lerp(__DPL forward<AT>(start), __DPL forward<BT>(end),
            __DPL forward<CT>(scale));
    }

    template <simd_vector AT, broadcastable_to<AT> BT,
        common_vector_with<AT> CT>
    requires (extended_vector<AT> || extended_vector<CT>) &&
        unqualified_extended_lerp<AT, BT, CT, common_canonical_simd_t<AT, CT>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        AT&& start, BT&& end, CT&& scale) noexcept {
        return lerp(__DPL forward<AT>(start), __DPL forward<BT>(end),
            __DPL forward<CT>(scale));
    }

    template <simd_vector BT, broadcastable_to<BT> AT,
        common_vector_with<BT> CT>
    requires (extended_vector<BT> || extended_vector<CT>) &&
        unqualified_extended_lerp<AT, BT, CT, common_canonical_simd_t<BT, CT>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        AT&& start, BT&& end, CT&& scale) noexcept {
        return lerp(__DPL forward<AT>(start), __DPL forward<BT>(end),
            __DPL forward<CT>(scale));
    }

    template <extended_vector AT, broadcastable_to<AT> BT,
        broadcastable_to<AT> CT>
    requires unqualified_extended_lerp<AT, BT, CT, AT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        AT&& start, BT&& end, CT&& scale) noexcept {
        return lerp(__DPL forward<AT>(start), __DPL forward<BT>(end),
            __DPL forward<CT>(scale));
    }

    template <extended_vector BT, broadcastable_to<BT> AT,
        broadcastable_to<BT> CT>
    requires unqualified_extended_lerp<AT, BT, CT, BT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& start, BT&& end, CT&& scale) {
        return lerp(__DPL forward<AT>(start), __DPL forward<BT>(end),
            __DPL forward<CT>(scale));
    }

    template <simd_vector S, exact_mask_for<S> M, typename AT, typename BT,
        typename CT>
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

    template <simd_vector S, const_mask_for<S> M, typename AT, typename BT,
        typename CT>
    requires (extended_vector<S> || extended_vector<AT> ||
                 extended_vector<BT> || extended_vector<CT>) &&
        unqualified_extended_mlerp<S, launder_cmask_t<S, M>, AT, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        S&& src, M mask, AT&& start, BT&& end, CT&& scale) {
        return lerp(__DPL forward<S>(src), dx::to_const_mask<S>(mask),
            __DPL forward<AT>(start), __DPL forward<BT>(end),
            __DPL forward<CT>(scale));
    }

    template <typename AT, typename BT, typename CT,
        result_mask_for<lerp_t, AT, BT, CT> M>
    requires (extended_mask<M> || extended_vector<AT> || extended_vector<BT> ||
                 extended_vector<CT>) &&
        unqualified_extended_zmlerp<M, AT, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, AT&& start, BT&& end, CT&& scale) {
        return lerp(zero, __DPL forward<M>(mask), __DPL forward<AT>(start),
            __DPL forward<BT>(end), __DPL forward<CT>(scale));
    }

    template <typename AT, typename BT, typename CT,
        result_cmask_for<lerp_t, AT, BT, CT> M>
    requires (extended_vector<AT> || extended_vector<BT> ||
                 extended_vector<CT>) &&
        unqualified_extended_zmlerp<
            launder_cmask_t<cpo_result_t<lerp_t, AT, BT, CT>, M>, AT, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M mask, AT&& start, BT&& end, CT&& scale) {
        return lerp(zero,
            dx::to_const_mask<cpo_result_t<lerp_t, AT, BT, CT>>(mask),
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
