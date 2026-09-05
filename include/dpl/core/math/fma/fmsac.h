// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/details/fm_broadcast.h"
#include "dpl/core/math/fma/fmacc.h"
#include "dpl/core/math/fma/fmsub.h"

#if !DPL_MODULES
#  include "dpl/core/basic/internal/abi.h"
#  include "dpl/core/concepts/cpo_invocable.h"
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/dispatch/concepts/operation.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/maskable/accumulation.h"
#  include "dpl/core/dispatch/operation/primitive.h"
#  include "dpl/core/operations/arithmetic/negate.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
struct fmsac_t;

struct DPL_EMPTY_BASES fmsac_t :
    public math_operation_base<fmsac_t>,
    public maskable_accumulation_base<fmsac_t> {
    using operation_base<fmsac_t>::operator();
    using maskable_accumulation_base<fmsac_t>::operator();
};

template <>
struct operation_signature<fmsac_t> {
    template <typename AT, typename BT, typename CT>
    requires simd_vector<AT> || simd_vector<BT> || simd_vector<CT>
    static consteval void operator()(AT&&, BT&&, CT&&) noexcept {}
};

template <>
struct fallback_impl<fmsac_t> : fm_canonical_broadcaster<fmsac_t> {
    template <canonical_vector CT, vector_subsumed_by<CT> AT,
        vector_subsumed_by<CT> BT>
    requires floating_point_like<simd_element_type_t<CT>> &&
        canonical_vector<AT> && canonical_vector<BT> &&
        cpo_invocable<fmsub_t, CT, AT, BT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        CT cval, AT aval, BT bval) noexcept {
        return dx::fmsub(aval, bval, cval);
    }

    using fm_canonical_broadcaster<fmsac_t>::operator();
};

template <typename AT, typename BT, typename CT, typename T>
concept unqualified_canonical_fmsac =
    unqualified_canonical_fmsub<BT, CT, AT, T>;

template <typename AT, typename M, typename BT, typename CT>
concept unqualified_canonical_mfmsac =
    cpo_invocable<fmsac_t, AT, BT, CT> && requires {
        {
            fmsub(internal::abi<cpo_result_t<fmsac_t, AT, BT, CT>>,
                internal::declarg<BT>(), internal::declarg<CT>(),
                internal::declarg<AT>(), internal::declarg<M>())
        } -> same_as<cpo_result_t<fmsac_t, AT, BT, CT>>;
    };

template <typename M, typename AT, typename BT, typename CT>
concept unqualified_canonical_zmfmsac =
    cpo_invocable<fmsac_t, AT, BT, CT> && requires {
        {
            fmsub(internal::abi<cpo_result_t<fmsac_t, AT, BT, CT>>, dx::zero,
                internal::declarg<M>(), internal::declarg<BT>(),
                internal::declarg<CT>(), internal::declarg<AT>())
        } -> same_as<cpo_result_t<fmsac_t, AT, BT, CT>>;
    };

template <>
struct canonical_impl<fmsac_t> {
private:
    template <typename AT, typename BT, typename CT>
    using result_t DPL_NODEBUG = cpo_result_t<fmsac_t, AT, BT, CT>;

    template <typename AT, typename BT, typename CT>
    using mask_t DPL_NODEBUG = simd_mask_type_t<result_t<AT, BT, CT>>;

    template <typename L, typename R>
    using vector_t DPL_NODEBUG = common_canonical_simd_t<L, R>;

public:
    template <canonical_vector AT, vector_subsumed_by<AT> BT,
        vector_subsumed_by<BT> CT>
    requires canonical_vector<BT> && canonical_vector<CT> &&
        unqualified_canonical_fmsac<AT, BT, CT, AT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr AT operator()(AT aval, BT bval, CT cval) noexcept {
        return fmsub(internal::abi<AT>, bval, cval, aval);
    }

    template <canonical_vector AT, vector_subsumed_by<AT> BT,
        broadcastable_to<AT> CT>
    requires canonical_vector<BT> && unqualified_canonical_fmsac<AT, BT, CT, AT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr AT operator()(AT aval, BT bval, CT&& cval) noexcept {
        return fmsub(internal::abi<AT>, bval, __DPL forward<CT>(cval), aval);
    }

    template <canonical_vector AT, vector_subsumed_by<AT> CT,
        broadcastable_to<AT> BT>
    requires canonical_vector<CT> && unqualified_canonical_fmsac<AT, BT, CT, AT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr AT operator()(AT aval, BT&& bval, CT cval) noexcept {
        return fmsub(internal::abi<AT>, __DPL forward<BT>(bval), cval, aval);
    }

    template <canonical_vector BT, common_vector_with<BT> CT,
        broadcastable_to<vector_t<BT, CT>> AT>
    requires canonical_vector<CT> &&
        unqualified_canonical_fmsac<AT, BT, CT, common_canonical_simd_t<BT, CT>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr vector_t<BT, CT> operator()(
        AT&& aval, BT bval, CT cval) noexcept {
        return fmsub(internal::abi<common_abi_t<BT, CT>>, bval, cval,
            __DPL forward<AT>(aval));
    }

    template <canonical_vector AT, broadcastable_to<AT> BT,
        broadcastable_to<AT> CT>
    requires unqualified_canonical_fmsac<AT, BT, CT, AT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr AT operator()(AT aval, BT&& bval, CT&& cval) noexcept {
        return fmsub(internal::abi<AT>, __DPL forward<BT>(bval),
            __DPL forward<CT>(cval), aval);
    }

    template <canonical_vector BT, broadcastable_to<BT> AT,
        broadcastable_to<BT> CT>
    requires unqualified_canonical_fmsac<AT, BT, CT, BT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr BT operator()(AT&& aval, BT bval, CT&& cval) noexcept {
        return fmsub(internal::abi<BT>, bval, __DPL forward<CT>(cval),
            __DPL forward<AT>(aval));
    }

    template <canonical_vector CT, broadcastable_to<CT> AT,
        broadcastable_to<CT> BT>
    requires unqualified_canonical_fmsac<AT, BT, CT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr CT operator()(AT&& aval, BT&& bval, CT cval) noexcept {
        return fmsub(internal::abi<CT>, __DPL forward<BT>(bval), cval,
            __DPL forward<AT>(aval));
    }

    template <unextended_type AT, unextended_type BT, unextended_type CT>
    requires unqualified_canonical_mfmsac<AT, mask_t<AT, BT, CT>, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<AT, BT, CT> operator()(
        AT aval, mask_t<AT, BT, CT> mask, BT bval, CT cval) noexcept {
        return fmsub(
            internal::abi<result_t<AT, BT, CT>>, bval, cval, aval, mask);
    }

    template <unextended_type AT, unextended_type BT, unextended_type CT,
        result_cmask_for<fmsac_t, AT, BT, CT> M>
    requires unqualified_canonical_mfmsac<AT,
        launder_cmask_t<result_t<AT, BT, CT>, M>, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<AT, BT, CT> operator()(
        AT aval, M mask, BT bval, CT cval) noexcept {
        return fmsub(internal::abi<result_t<AT, BT, CT>>, bval, cval, aval,
            dx::to_const_mask<result_t<AT, BT, CT>>(mask));
    }

    template <unextended_type AT, unextended_type BT, unextended_type CT>
    requires unqualified_canonical_zmfmsac<mask_t<AT, BT, CT>, AT, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<AT, BT, CT> operator()(dx::zero_t zero,
        mask_t<AT, BT, CT> mask, AT aval, BT bval, CT cval) noexcept {
        return fmsub(
            internal::abi<result_t<AT, BT, CT>>, zero, mask, bval, cval, aval);
    }

    template <unextended_type AT, unextended_type BT, unextended_type CT,
        result_cmask_for<fmsac_t, AT, BT, CT> M>
    requires unqualified_canonical_zmfmsac<
        launder_cmask_t<result_t<AT, BT, CT>, M>, AT, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<AT, BT, CT> operator()(
        dx::zero_t zero, M mask, AT aval, BT bval, CT cval) noexcept {
        return fmsub(internal::abi<result_t<AT, BT, CT>>, zero,
            dx::to_const_mask<result_t<AT, BT, CT>>(mask), bval, cval, aval);
    }
};

template <typename AT, typename BT, typename CT, typename T>
concept unqualified_extended_fmsac = unqualified_extended_fmsub<BT, CT, AT, T>;

template <typename AT, typename M, typename BT, typename CT>
concept unqualified_extended_mfmsac =
    cpo_invocable<fmsac_t, AT, BT, CT> && requires {
        {
            fmsub(internal::declarg<BT>(), internal::declarg<CT>(),
                internal::declarg<AT>(), internal::declarg<M>())
        } -> equivalent_vector_with<cpo_result_t<fmsac_t, AT, BT, CT>>;
    };

template <typename M, typename AT, typename BT, typename CT>
concept unqualified_extended_zmfmsac =
    cpo_invocable<fmsac_t, AT, BT, CT> && requires {
        {
            fmsub(dx::zero, internal::declarg<M>(), internal::declarg<BT>(),
                internal::declarg<CT>(), internal::declarg<AT>())
        } -> equivalent_vector_with<cpo_result_t<fmsac_t, AT, BT, CT>>;
    };

template <>
struct extended_impl<fmsac_t> {
private:
    template <typename L, typename R>
    using vector_t DPL_NODEBUG = common_canonical_simd_t<L, R>;

public:
    template <simd_vector AT, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    requires (extended_vector<AT> || extended_vector<BT> ||
                 extended_vector<CT>) &&
        unqualified_extended_fmsac<AT, BT, CT, AT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& aval, BT&& bval, CT&& cval) {
        return fmsub(__DPL forward<BT>(bval), __DPL forward<CT>(cval),
            __DPL forward<AT>(aval));
    }

    ///
    template <simd_vector AT, vector_subsumed_by<AT> BT,
        broadcastable_to<AT> CT>
    requires (extended_vector<AT> || extended_vector<BT>) &&
        unqualified_extended_fmsac<AT, BT, CT, AT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& aval, BT&& bval, CT&& cval) {
        return fmsub(__DPL forward<BT>(bval), __DPL forward<CT>(cval),
            __DPL forward<AT>(aval));
    }

    template <simd_vector AT, vector_subsumed_by<AT> CT,
        broadcastable_to<AT> BT>
    requires (extended_vector<AT> || extended_vector<CT>) &&
        unqualified_extended_fmsac<AT, BT, CT, AT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& aval, BT&& bval, CT&& cval) {
        return fmsub(__DPL forward<BT>(bval), __DPL forward<CT>(cval),
            __DPL forward<AT>(aval));
    }

    template <simd_vector BT, common_vector_with<BT> CT,
        broadcastable_to<vector_t<BT, CT>> AT>
    requires (extended_vector<BT> || extended_vector<CT>) &&
        unqualified_extended_fmsac<AT, BT, CT, vector_t<BT, CT>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& aval, BT&& bval, CT&& cval) {
        return fmsub(__DPL forward<BT>(bval), __DPL forward<CT>(cval),
            __DPL forward<AT>(aval));
    }

    template <extended_vector AT, broadcastable_to<AT> BT,
        broadcastable_to<AT> CT>
    requires unqualified_extended_fmsac<AT, BT, CT, AT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& aval, BT&& bval, CT&& cval) {
        return fmsub(__DPL forward<BT>(bval), __DPL forward<CT>(cval),
            __DPL forward<AT>(aval));
    }

    template <extended_vector BT, broadcastable_to<BT> AT,
        broadcastable_to<BT> CT>
    requires unqualified_extended_fmsac<AT, BT, CT, BT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& aval, BT&& bval, CT&& cval) {
        return fmsub(__DPL forward<BT>(bval), __DPL forward<CT>(cval),
            __DPL forward<AT>(aval));
    }

    template <extended_vector CT, broadcastable_to<CT> AT,
        broadcastable_to<CT> BT>
    requires unqualified_extended_fmsac<AT, BT, CT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& aval, BT&& bval, CT&& cval) {
        return fmsub(__DPL forward<BT>(bval), __DPL forward<CT>(cval),
            __DPL forward<AT>(aval));
    }
    ///

    template <typename AT, typename BT, typename CT,
        result_mask_for<fmsac_t, AT, BT, CT> M>
    requires (extended_mask<M> || extended_vector<AT> || extended_vector<BT> ||
                 extended_vector<CT>) &&
        unqualified_extended_mfmsac<AT, M, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        AT&& aval, M&& mask, BT&& bval, CT&& cval) {
        return fmsub( __DPL forward<BT>(bval), __DPL forward<CT>(cval),
            __DPL forward<AT>(aval), __DPL forward<M>(mask));
    }

    template <typename AT, typename BT, typename CT,
        result_cmask_for<fmsac_t, AT, BT, CT> M>
    requires (extended_vector<AT> || extended_vector<BT> ||
                 extended_vector<CT>) &&
        unqualified_extended_mfmsac<AT,
            launder_cmask_t<cpo_result_t<fmsac_t, AT, BT, CT>, M>, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& aval, M mask, BT&& bval, CT&& cval) {
        return fmsub( __DPL forward<BT>(bval), __DPL forward<CT>(cval),
            __DPL forward<AT>(aval),
            dx::to_const_mask<cpo_result_t<fmsac_t, AT, BT, CT>>(mask));
    }

    template <typename AT, typename BT, typename CT,
        result_mask_for<fmsac_t, AT, BT, CT> M>
    requires (extended_mask<M> || extended_vector<AT> || extended_vector<BT> ||
                 extended_vector<CT>) &&
        unqualified_extended_zmfmsac<M, AT, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, AT&& aval, BT&& bval, CT&& cval) {
        return fmsub(zero, __DPL forward<M>(mask), __DPL forward<BT>(bval),
            __DPL forward<CT>(cval), __DPL forward<AT>(aval));
    }

    template <typename AT, typename BT, typename CT,
        result_cmask_for<fmsac_t, AT, BT, CT> M>
    requires (extended_vector<AT> || extended_vector<BT> ||
                 extended_vector<CT>) &&
        unqualified_extended_zmfmsac<
            launder_cmask_t<cpo_result_t<fmsac_t, AT, BT, CT>, M>, AT, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M mask, AT&& aval, BT&& bval, CT&& cval) {
        return fmsub(zero,
            dx::to_const_mask<cpo_result_t<fmsac_t, AT, BT, CT>>(mask),
            __DPL forward<BT>(bval), __DPL forward<CT>(cval),
            __DPL forward<AT>(aval));
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::fmsac_t fmsac{};
} // namespace cpo
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
