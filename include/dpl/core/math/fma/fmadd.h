// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/details/fm_broadcast.h"

#if !DPL_MODULES
#  include "dpl/core/basic/internal/abi.h"
#  include "dpl/core/concepts/cpo_invocable.h"
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/maskable/accumulation.h"
#  include "dpl/core/dispatch/operation/primitive.h"
#  include "dpl/core/numbers/binary_layout_floating_point.h"
#  include "dpl/core/operations/internal/transform.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void fmadd(...) noexcept = delete;

struct DPL_EMPTY_BASES fmadd_t :
    public math_operation_base<fmadd_t>,
    public maskable_accumulation_base<fmadd_t> {
    using operation_base<fmadd_t>::operator();
    using maskable_accumulation_base<fmadd_t>::operator();
};

template <>
struct operation_signature<fmadd_t> {
    template <typename AT, typename BT, typename CT>
    requires simd_vector<AT> || simd_vector<BT> || simd_vector<CT>
    static consteval void operator()(AT&&, BT&&, CT&&) noexcept {}
};

template <>
struct fallback_impl<fmadd_t> : fm_canonical_broadcaster<fmadd_t> {
public:
    template <canonical_vector T>
    requires cpo_invocable<canonical_impl<fmadd_t>, T, T, T> &&
        fixed_width_abi<simd_abi_type_t<T>> &&
        binary_layout_floating_point<simd_element_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static consteval T DPL_VECTORCALL operator()(T lhs, T mid, T rhs) noexcept {
        return internal::transform<T>(
            [](double lhs, double mid, double rhs) -> simd_element_type_t<T> {
#if DPL_HAS_CONSTEXPR_BUILTIN(__builtin_fma)
                return __builtin_fma(lhs, mid, rhs);
#else
                // TODO: Do this properly for MSVC
                return lhs * mid + rhs;
#endif
            },
            lhs, mid, rhs);
    }

    using fm_canonical_broadcaster<fmadd_t>::operator();
};

template <typename AT, typename BT, typename CT, typename T>
concept unqualified_canonical_fmadd = requires {
    {
        fmadd(internal::abi<T>, internal::declarg<AT>(),
            internal::declarg<BT>(), internal::declarg<CT>())
    } -> same_as<T>;
};

template <typename AT, typename M, typename BT, typename CT>
concept unqualified_canonical_mfmadd =
    cpo_invocable<fmadd_t, AT, BT, CT> && requires {
        {
            fmadd(internal::abi<cpo_result_t<fmadd_t, AT, BT, CT>>,
                internal::declarg<AT>(), internal::declarg<M>(),
                internal::declarg<BT>(), internal::declarg<CT>())
        } -> same_as<cpo_result_t<fmadd_t, AT, BT, CT>>;
    };

template <typename M, typename AT, typename BT, typename CT>
concept unqualified_canonical_zmfmadd =
    cpo_invocable<fmadd_t, AT, BT, CT> && requires {
        {
            fmadd(internal::abi<cpo_result_t<fmadd_t, AT, BT, CT>>, dx::zero,
                internal::declarg<M>(), internal::declarg<AT>(),
                internal::declarg<BT>(), internal::declarg<CT>())
        } -> same_as<cpo_result_t<fmadd_t, AT, BT, CT>>;
    };

template <>
struct canonical_impl<fmadd_t> {
private:
    template <typename AT, typename BT, typename CT>
    using result_t DPL_NODEBUG = cpo_result_t<fmadd_t, AT, BT, CT>;

    template <typename AT, typename BT, typename CT>
    using mask_t DPL_NODEBUG = simd_mask_type_t<result_t<AT, BT, CT>>;

    template <typename L, typename R>
    using vector_t DPL_NODEBUG = common_canonical_simd_t<L, R>;

public:
    template <canonical_vector AT, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    requires canonical_vector<BT> && canonical_vector<CT> &&
        unqualified_canonical_fmadd<AT, BT, CT, AT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr AT operator()(AT aval, BT bval, CT cval) noexcept {
        return fmadd(internal::abi<AT>, aval, bval, cval);
    }

    template <canonical_vector AT, vector_subsumed_by<AT> BT,
        broadcastable_to<AT> CT>
    requires canonical_vector<BT> && unqualified_canonical_fmadd<AT, BT, CT, AT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr AT operator()(AT aval, BT bval, CT&& cval) noexcept {
        return fmadd(internal::abi<AT>, aval, bval, __DPL forward<CT>(cval));
    }

    template <canonical_vector AT, vector_subsumed_by<AT> CT,
        broadcastable_to<AT> BT>
    requires canonical_vector<CT> && unqualified_canonical_fmadd<AT, BT, CT, AT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr AT operator()(AT aval, BT&& bval, CT cval) noexcept {
        return fmadd(internal::abi<AT>, aval, __DPL forward<BT>(bval), cval);
    }

    template <canonical_vector BT, common_vector_with<BT> CT,
        broadcastable_to<vector_t<BT, CT>> AT>
    requires canonical_vector<CT> &&
        unqualified_canonical_fmadd<AT, BT, CT, vector_t<BT, CT>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr vector_t<BT, CT> operator()(
        AT&& aval, BT bval, CT cval) noexcept {
        return fmadd(internal::abi<common_abi_t<BT, CT>>,
            __DPL forward<AT>(aval), bval, cval);
    }

    template <canonical_vector AT, broadcastable_to<AT> BT,
        broadcastable_to<AT> CT>
    requires unqualified_canonical_fmadd<AT, BT, CT, AT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr AT operator()(AT aval, BT&& bval, CT&& cval) noexcept {
        return fmadd(internal::abi<AT>, aval, __DPL forward<BT>(bval),
            __DPL forward<CT>(cval));
    }

    template <canonical_vector BT, broadcastable_to<BT> AT,
        broadcastable_to<BT> CT>
    requires unqualified_canonical_fmadd<AT, BT, CT, BT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr BT operator()(AT&& aval, BT bval, CT&& cval) noexcept {
        return fmadd(internal::abi<BT>, __DPL forward<AT>(aval), bval,
            __DPL forward<CT>(cval));
    }

    template <unextended_type AT, unextended_type BT, unextended_type CT>
    requires unqualified_canonical_mfmadd<AT, mask_t<AT, BT, CT>, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<AT, BT, CT> operator()(
        AT aval, mask_t<AT, BT, CT> mask, BT bval, CT cval) noexcept {
        return fmadd(
            internal::abi<result_t<AT, BT, CT>>, aval, mask, bval, cval);
    }

    template <unextended_type AT, unextended_type BT, unextended_type CT,
        result_cmask_for<fmadd_t, AT, BT, CT> M>
    requires unqualified_canonical_mfmadd<AT,
        launder_cmask_t<result_t<AT, BT, CT>, M>, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<AT, BT, CT> operator()(
        AT aval, M mask, BT bval, CT cval) noexcept {
        return fmadd(internal::abi<result_t<AT, BT, CT>>, aval,
            dx::to_const_mask<result_t<AT, BT, CT>>(mask), bval, cval);
    }

    template <unextended_type AT, unextended_type BT, unextended_type CT>
    requires unqualified_canonical_zmfmadd<mask_t<AT, BT, CT>, AT, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<AT, BT, CT> operator()(dx::zero_t zero,
        mask_t<AT, BT, CT> mask, AT aval, BT bval, CT cval) noexcept {
        return fmadd(
            internal::abi<result_t<AT, BT, CT>>, zero, mask, aval, bval, cval);
    }

    template <unextended_type AT, unextended_type BT, unextended_type CT,
        result_cmask_for<fmadd_t, AT, BT, CT> M>
    requires unqualified_canonical_zmfmadd<
        launder_cmask_t<result_t<AT, BT, CT>, M>, AT, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<AT, BT, CT> operator()(
        dx::zero_t zero, M mask, AT aval, BT bval, CT cval) noexcept {
        return fmadd(internal::abi<result_t<AT, BT, CT>>, zero,
            dx::to_const_mask<result_t<AT, BT, CT>>(mask), aval, bval, cval);
    }
};

template <typename AT, typename BT, typename CT, typename T>
concept unqualified_extended_fmadd = requires {
    {
        fmadd(internal::declarg<AT>(), internal::declarg<BT>(),
            internal::declarg<CT>())
    } -> equivalent_vector_with<T>;
};

template <typename AT, typename M, typename BT, typename CT>
concept unqualified_extended_mfmadd =
    cpo_invocable<fmadd_t, AT, BT, CT> && requires {
        {
            fmadd(internal::declarg<AT>(), internal::declarg<M>(),
                internal::declarg<BT>(), internal::declarg<CT>())
        } -> equivalent_vector_with<cpo_result_t<fmadd_t, AT, BT, CT>>;
    };

template <typename M, typename AT, typename BT, typename CT>
concept unqualified_extended_zmfmadd =
    cpo_invocable<fmadd_t, AT, BT, CT> && requires {
        {
            fmadd(dx::zero, internal::declarg<M>(), internal::declarg<AT>(),
                internal::declarg<BT>(), internal::declarg<CT>())
        } -> equivalent_vector_with<cpo_result_t<fmadd_t, AT, BT, CT>>;
    };

template <>
struct extended_impl<fmadd_t> {
private:
    template <typename L, typename R>
    using vector_t DPL_NODEBUG = common_canonical_simd_t<L, R>;

public:
    template <simd_vector AT, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    requires (extended_vector<AT> || extended_vector<BT> ||
                 extended_vector<CT>) &&
        unqualified_extended_fmadd<AT, BT, CT, AT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& aval, BT&& bval, CT&& cval) {
        return fmadd(__DPL forward<AT>(aval), __DPL forward<BT>(bval),
            __DPL forward<CT>(cval));
    }

    ///
    template <simd_vector AT, vector_subsumed_by<AT> BT,
        broadcastable_to<AT> CT>
    requires (extended_vector<AT> || extended_vector<BT>) &&
        unqualified_extended_fmadd<AT, BT, CT, AT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& aval, BT&& bval, CT&& cval) {
        return fmadd(__DPL forward<AT>(aval), __DPL forward<BT>(bval),
            __DPL forward<CT>(cval));
    }

    template <simd_vector AT, vector_subsumed_by<AT> CT,
        broadcastable_to<AT> BT>
    requires (extended_vector<AT> || extended_vector<CT>) &&
        unqualified_extended_fmadd<AT, BT, CT, AT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& aval, BT&& bval, CT&& cval) {
        return fmadd(__DPL forward<AT>(aval), __DPL forward<BT>(bval),
            __DPL forward<CT>(cval));
    }

    template <simd_vector BT, common_vector_with<BT> CT,
        broadcastable_to<vector_t<BT, CT>> AT>
    requires (extended_vector<BT> || extended_vector<CT>) &&
        unqualified_extended_fmadd<AT, BT, CT, vector_t<BT, CT>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& aval, BT&& bval, CT&& cval) {
        return fmadd(__DPL forward<AT>(aval), __DPL forward<BT>(bval),
            __DPL forward<CT>(cval));
    }

    template <extended_vector AT, broadcastable_to<AT> BT,
        broadcastable_to<AT> CT>
    requires unqualified_extended_fmadd<AT, BT, CT, AT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& aval, BT&& bval, CT&& cval) {
        return fmadd(__DPL forward<AT>(aval), __DPL forward<BT>(bval),
            __DPL forward<CT>(cval));
    }

    template <extended_vector BT, broadcastable_to<BT> AT,
        broadcastable_to<BT> CT>
    requires unqualified_extended_fmadd<AT, BT, CT, BT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& aval, BT&& bval, CT&& cval) {
        return fmadd(__DPL forward<AT>(aval), __DPL forward<BT>(bval),
            __DPL forward<CT>(cval));
    }

    template <extended_vector CT, broadcastable_to<CT> AT,
        broadcastable_to<CT> BT>
    requires unqualified_extended_fmadd<AT, BT, CT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& aval, BT&& bval, CT&& cval) {
        return fmadd(__DPL forward<AT>(aval), __DPL forward<BT>(bval),
            __DPL forward<CT>(cval));
    }
    ///

    template <typename AT, typename BT, typename CT,
        result_mask_for<fmadd_t, AT, BT, CT> M>
    requires (extended_mask<M> || extended_vector<AT> || extended_vector<BT> ||
                 extended_vector<CT>) &&
        unqualified_extended_mfmadd<AT, M, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        AT&& aval, M&& mask, BT&& bval, CT&& cval) {
        return fmadd( __DPL forward<AT>(aval), __DPL forward<M>(mask),
            __DPL forward<BT>(bval), __DPL forward<CT>(cval));
    }

    template <typename AT, typename BT, typename CT,
        result_cmask_for<fmadd_t, AT, BT, CT> M>
    requires (extended_vector<AT> || extended_vector<BT> ||
                 extended_vector<CT>) &&
        unqualified_extended_mfmadd<AT,
            launder_cmask_t<cpo_result_t<fmadd_t, AT, BT, CT>, M>, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& aval, M mask, BT&& bval, CT&& cval) {
        return fmadd(__DPL forward<AT>(aval),
            dx::to_const_mask<cpo_result_t<fmadd_t, AT, BT, CT>>(mask),
            __DPL forward<BT>(bval), __DPL forward<CT>(cval));
    }

    template <typename AT, typename BT, typename CT,
        result_mask_for<fmadd_t, AT, BT, CT> M>
    requires (extended_mask<M> || extended_vector<AT> || extended_vector<BT> ||
                 extended_vector<CT>) &&
        unqualified_extended_zmfmadd<M, AT, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, AT&& aval, BT&& bval, CT&& cval) {
        return fmadd(zero, __DPL forward<M>(mask), __DPL forward<AT>(aval),
            __DPL forward<BT>(bval), __DPL forward<CT>(cval));
    }

    template <typename AT, typename BT, typename CT,
        result_cmask_for<fmadd_t, AT, BT, CT> M>
    requires (extended_vector<AT> || extended_vector<BT> ||
                 extended_vector<CT>) &&
        unqualified_extended_zmfmadd<
            launder_cmask_t<cpo_result_t<fmadd_t, AT, BT, CT>, M>, AT, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M mask, AT&& aval, BT&& bval, CT&& cval) {
        return fmadd(zero,
            dx::to_const_mask<cpo_result_t<fmadd_t, AT, BT, CT>>(mask),
            __DPL forward<AT>(aval), __DPL forward<BT>(bval),
            __DPL forward<CT>(cval));
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::fmadd_t fmadd{};
} // namespace cpo
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
