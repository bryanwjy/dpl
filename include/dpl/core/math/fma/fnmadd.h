// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/details/fm_broadcast.h"
#include "dpl/core/math/fma/fmadd.h"

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
void fnmadd(...) noexcept = delete;

struct fnmadd_t;

struct DPL_EMPTY_BASES fnmadd_t :
    public math_operation_base<fnmadd_t>,
    public maskable_accumulation_base<fnmadd_t> {
    using operation_base<fnmadd_t>::operator();
    using maskable_accumulation_base<fnmadd_t>::operator();
};

template <>
struct operation_signature<fnmadd_t> {
    template <typename AT, typename BT, typename CT>
    requires simd_vector<AT> || simd_vector<BT> || simd_vector<CT>
    static consteval void operator()(AT&&, BT&&, CT&&) noexcept {}
};

template <>
struct fallback_impl<fnmadd_t> : fm_canonical_broadcaster<fnmadd_t> {

    template <canonical_vector AT, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    requires floating_point_like<simd_element_type_t<AT>> &&
        canonical_vector<BT> && canonical_vector<CT> &&
        cpo_invocable<fmadd_t, AT, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr AT DPL_VECTORCALL operator()(
        AT aval, BT bval, CT cval) noexcept {
        return dx::negate(dx::fmadd(aval, bval, cval));
    }

    using fm_canonical_broadcaster<fnmadd_t>::operator();
};

template <typename AT, typename BT, typename CT, typename T>
concept unqualified_canonical_fnmadd = requires {
    {
        fnmadd(internal::abi<T>, internal::declarg<AT>(),
            internal::declarg<BT>(), internal::declarg<CT>())
    } -> same_as<T>;
};

template <typename AT, typename M, typename BT, typename CT>
concept unqualified_canonical_mfnmadd =
    cpo_invocable<fnmadd_t, AT, BT, CT> && requires {
        {
            fnmadd(internal::abi<cpo_result_t<fnmadd_t, AT, BT, CT>>,
                internal::declarg<AT>(), internal::declarg<M>(),
                internal::declarg<BT>(), internal::declarg<CT>())
        } -> same_as<cpo_result_t<fnmadd_t, AT, BT, CT>>;
    };

template <typename M, typename AT, typename BT, typename CT>
concept unqualified_canonical_zmfnmadd =
    cpo_invocable<fnmadd_t, AT, BT, CT> && requires {
        {
            fnmadd(internal::abi<cpo_result_t<fnmadd_t, AT, BT, CT>>, dx::zero,
                internal::declarg<M>(), internal::declarg<AT>(),
                internal::declarg<BT>(), internal::declarg<CT>())
        } -> same_as<cpo_result_t<fnmadd_t, AT, BT, CT>>;
    };

template <>
struct canonical_impl<fnmadd_t> {
private:
    template <typename AT, typename BT, typename CT>
    using result_t DPL_NODEBUG = cpo_result_t<fnmadd_t, AT, BT, CT>;

    template <typename AT, typename BT, typename CT>
    using mask_t DPL_NODEBUG = simd_mask_type_t<result_t<AT, BT, CT>>;

    template <typename L, typename R>
    using vector_t DPL_NODEBUG = common_canonical_simd_t<L, R>;

public:
    template <canonical_vector AT, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    requires canonical_vector<BT> && canonical_vector<CT> &&
        unqualified_canonical_fnmadd<AT, BT, CT, AT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr AT operator()(AT aval, BT bval, CT cval) noexcept {
        return fnmadd(internal::abi<AT>, aval, bval, cval);
    }

    template <canonical_vector AT, vector_subsumed_by<AT> BT,
        broadcastable_to<AT> CT>
    requires canonical_vector<BT> &&
        unqualified_canonical_fnmadd<AT, BT, CT, AT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr AT operator()(AT aval, BT bval, CT&& cval) noexcept {
        return fnmadd(internal::abi<AT>, aval, bval, __DPL forward<CT>(cval));
    }

    template <canonical_vector AT, vector_subsumed_by<AT> CT,
        broadcastable_to<AT> BT>
    requires canonical_vector<CT> &&
        unqualified_canonical_fnmadd<AT, BT, CT, AT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr AT operator()(AT aval, BT&& bval, CT cval) noexcept {
        return fnmadd(internal::abi<AT>, aval, __DPL forward<BT>(bval), cval);
    }

    template <canonical_vector BT, common_vector_with<BT> CT,
        broadcastable_to<vector_t<BT, CT>> AT>
    requires canonical_vector<CT> &&
        unqualified_canonical_fnmadd<AT, BT, CT, vector_t<BT, CT>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr vector_t<BT, CT> operator()(
        AT&& aval, BT bval, CT cval) noexcept {
        return fnmadd(internal::abi<common_abi_t<BT, CT>>,
            __DPL forward<AT>(aval), bval, cval);
    }

    template <canonical_vector AT, broadcastable_to<AT> BT,
        broadcastable_to<AT> CT>
    requires unqualified_canonical_fnmadd<AT, BT, CT, AT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr AT operator()(AT aval, BT&& bval, CT&& cval) noexcept {
        return fnmadd(internal::abi<AT>, aval, __DPL forward<BT>(bval),
            __DPL forward<CT>(cval));
    }

    template <canonical_vector BT, broadcastable_to<BT> AT,
        broadcastable_to<BT> CT>
    requires unqualified_canonical_fnmadd<AT, BT, CT, BT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr BT operator()(AT&& aval, BT bval, CT&& cval) noexcept {
        return fnmadd(internal::abi<BT>, __DPL forward<AT>(aval), bval,
            __DPL forward<CT>(cval));
    }

    template <unextended_type AT, unextended_type BT, unextended_type CT>
    requires unqualified_canonical_mfnmadd<AT, mask_t<AT, BT, CT>, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<AT, BT, CT> operator()(
        AT aval, mask_t<AT, BT, CT> mask, BT bval, CT cval) noexcept {
        return fnmadd(
            internal::abi<result_t<AT, BT, CT>>, aval, mask, bval, cval);
    }

    template <unextended_type AT, unextended_type BT, unextended_type CT,
        result_cmask_for<fnmadd_t, AT, BT, CT> M>
    requires unqualified_canonical_mfnmadd<AT,
        launder_cmask_t<result_t<AT, BT, CT>, M>, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<AT, BT, CT> operator()(
        AT aval, M mask, BT bval, CT cval) noexcept {
        return fnmadd(internal::abi<result_t<AT, BT, CT>>, aval,
            dx::to_const_mask<result_t<AT, BT, CT>>(mask), bval, cval);
    }

    template <unextended_type AT, unextended_type BT, unextended_type CT>
    requires unqualified_canonical_zmfnmadd<mask_t<AT, BT, CT>, AT, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<AT, BT, CT> operator()(dx::zero_t zero,
        mask_t<AT, BT, CT> mask, AT aval, BT bval, CT cval) noexcept {
        return fnmadd(
            internal::abi<result_t<AT, BT, CT>>, zero, mask, aval, bval, cval);
    }

    template <unextended_type AT, unextended_type BT, unextended_type CT,
        result_cmask_for<fnmadd_t, AT, BT, CT> M>
    requires unqualified_canonical_zmfnmadd<
        launder_cmask_t<result_t<AT, BT, CT>, M>, AT, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<AT, BT, CT> operator()(
        dx::zero_t zero, M mask, AT aval, BT bval, CT cval) noexcept {
        return fnmadd(internal::abi<result_t<AT, BT, CT>>, zero,
            dx::to_const_mask<result_t<AT, BT, CT>>(mask), aval, bval, cval);
    }
};

template <typename AT, typename BT, typename CT, typename T>
concept unqualified_extended_fnmadd = requires {
    {
        fnmadd(internal::declarg<AT>(), internal::declarg<BT>(),
            internal::declarg<CT>())
    } -> equivalent_vector_with<T>;
};

template <typename AT, typename M, typename BT, typename CT>
concept unqualified_extended_mfnmadd =
    cpo_invocable<fnmadd_t, AT, BT, CT> && requires {
        {
            fnmadd(internal::declarg<AT>(), internal::declarg<M>(),
                internal::declarg<BT>(), internal::declarg<CT>())
        } -> equivalent_vector_with<cpo_result_t<fnmadd_t, AT, BT, CT>>;
    };

template <typename M, typename AT, typename BT, typename CT>
concept unqualified_extended_zmfnmadd =
    cpo_invocable<fnmadd_t, AT, BT, CT> && requires {
        {
            fnmadd(dx::zero, internal::declarg<M>(), internal::declarg<AT>(),
                internal::declarg<BT>(), internal::declarg<CT>())
        } -> equivalent_vector_with<cpo_result_t<fnmadd_t, AT, BT, CT>>;
    };

template <>
struct extended_impl<fnmadd_t> {
private:
    template <typename L, typename R>
    using vector_t DPL_NODEBUG = common_canonical_simd_t<L, R>;

public:
    template <simd_vector AT, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    requires (extended_vector<AT> || extended_vector<BT> ||
                 extended_vector<CT>) &&
        unqualified_extended_fnmadd<AT, BT, CT, AT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& aval, BT&& bval, CT&& cval) {
        return fnmadd(__DPL forward<AT>(aval), __DPL forward<BT>(bval),
            __DPL forward<CT>(cval));
    }

    ///
    template <simd_vector AT, vector_subsumed_by<AT> BT,
        broadcastable_to<AT> CT>
    requires (extended_vector<AT> || extended_vector<BT>) &&
        unqualified_extended_fnmadd<AT, BT, CT, AT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& aval, BT&& bval, CT&& cval) {
        return fnmadd(__DPL forward<AT>(aval), __DPL forward<BT>(bval),
            __DPL forward<CT>(cval));
    }

    template <simd_vector AT, vector_subsumed_by<AT> CT,
        broadcastable_to<AT> BT>
    requires (extended_vector<AT> || extended_vector<CT>) &&
        unqualified_extended_fnmadd<AT, BT, CT, AT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& aval, BT&& bval, CT&& cval) {
        return fnmadd(__DPL forward<AT>(aval), __DPL forward<BT>(bval),
            __DPL forward<CT>(cval));
    }

    template <simd_vector BT, common_vector_with<BT> CT,
        broadcastable_to<vector_t<BT, CT>> AT>
    requires (extended_vector<BT> || extended_vector<CT>) &&
        unqualified_extended_fnmadd<AT, BT, CT, vector_t<BT, CT>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& aval, BT&& bval, CT&& cval) {
        return fnmadd(__DPL forward<AT>(aval), __DPL forward<BT>(bval),
            __DPL forward<CT>(cval));
    }

    template <extended_vector AT, broadcastable_to<AT> BT,
        broadcastable_to<AT> CT>
    requires unqualified_extended_fnmadd<AT, BT, CT, AT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& aval, BT&& bval, CT&& cval) {
        return fnmadd(__DPL forward<AT>(aval), __DPL forward<BT>(bval),
            __DPL forward<CT>(cval));
    }

    template <extended_vector BT, broadcastable_to<BT> AT,
        broadcastable_to<BT> CT>
    requires unqualified_extended_fnmadd<AT, BT, CT, BT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& aval, BT&& bval, CT&& cval) {
        return fnmadd(__DPL forward<AT>(aval), __DPL forward<BT>(bval),
            __DPL forward<CT>(cval));
    }

    template <extended_vector CT, broadcastable_to<CT> AT,
        broadcastable_to<CT> BT>
    requires unqualified_extended_fnmadd<AT, BT, CT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& aval, BT&& bval, CT&& cval) {
        return fnmadd(__DPL forward<AT>(aval), __DPL forward<BT>(bval),
            __DPL forward<CT>(cval));
    }
    ///

    template <typename AT, typename BT, typename CT,
        result_mask_for<fnmadd_t, AT, BT, CT> M>
    requires (extended_mask<M> || extended_vector<AT> || extended_vector<BT> ||
                 extended_vector<CT>) &&
        unqualified_extended_mfnmadd<AT, M, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        AT&& aval, M&& mask, BT&& bval, CT&& cval) {
        return fnmadd( __DPL forward<AT>(aval), __DPL forward<M>(mask),
            __DPL forward<BT>(bval), __DPL forward<CT>(cval));
    }

    template <typename AT, typename BT, typename CT,
        result_cmask_for<fnmadd_t, AT, BT, CT> M>
    requires (extended_vector<AT> || extended_vector<BT> ||
                 extended_vector<CT>) &&
        unqualified_extended_mfnmadd<AT,
            launder_cmask_t<cpo_result_t<fnmadd_t, AT, BT, CT>, M>, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& aval, M mask, BT&& bval, CT&& cval) {
        return fnmadd(__DPL forward<AT>(aval),
            dx::to_const_mask<cpo_result_t<fnmadd_t, AT, BT, CT>>(mask),
            __DPL forward<BT>(bval), __DPL forward<CT>(cval));
    }

    template <typename AT, typename BT, typename CT,
        result_mask_for<fnmadd_t, AT, BT, CT> M>
    requires (extended_mask<M> || extended_vector<AT> || extended_vector<BT> ||
                 extended_vector<CT>) &&
        unqualified_extended_zmfnmadd<M, AT, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, AT&& aval, BT&& bval, CT&& cval) {
        return fnmadd(zero, __DPL forward<M>(mask), __DPL forward<AT>(aval),
            __DPL forward<BT>(bval), __DPL forward<CT>(cval));
    }

    template <typename AT, typename BT, typename CT,
        result_cmask_for<fnmadd_t, AT, BT, CT> M>
    requires (extended_vector<AT> || extended_vector<BT> ||
                 extended_vector<CT>) &&
        unqualified_extended_zmfnmadd<
            launder_cmask_t<cpo_result_t<fnmadd_t, AT, BT, CT>, M>, AT, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M mask, AT&& aval, BT&& bval, CT&& cval) {
        return fnmadd(zero,
            dx::to_const_mask<cpo_result_t<fnmadd_t, AT, BT, CT>>(mask),
            __DPL forward<AT>(aval), __DPL forward<BT>(bval),
            __DPL forward<CT>(cval));
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::fnmadd_t fnmadd{};
} // namespace cpo
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
