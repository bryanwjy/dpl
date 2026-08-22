// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/fma/fmadd.h"

#if !DPL_MODULES
#  include "dpl/core/basic/internal/abi.h"
#  include "dpl/core/concepts/cpo_invocable.h"
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/dispatch/broadcastable/ternary.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/maskable/accumulation.h"
#  include "dpl/core/dispatch/operation/primitive.h"
#  include "dpl/core/numbers/binary_layout_floating_point.h"
#  include "dpl/core/operations/internal/transform.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
struct fmacc_t;

struct DPL_EMPTY_BASES fmacc_t :
    public math_operation_base<fmacc_t>,
    public maskable_accumulation_base<fmacc_t>,
    public ternary_broadcastable_operation<fmacc_t> {
    using operation_base<fmacc_t>::operator();
    using maskable_accumulation_base<fmacc_t>::operator();
    using ternary_broadcastable_operation<fmacc_t>::operator();
};

template <>
struct operation_signature<fmacc_t> {
    template <typename AT, typename BT, typename CT>
    requires simd_vector<AT> || simd_vector<BT> || simd_vector<CT>
    static consteval void operator()(AT&&, BT&&, CT&&) noexcept {}
};

template <>
struct fallback_impl<fmacc_t> : ternary_broadcasting_fallback<fmacc_t> {
    template <__DPL binary_layout_floating_point E, fixed_width_abi A>
    requires cpo_invocable<canonical_impl<fmadd_t>, basic_vector<E, A>,
        basic_vector<E, A>, basic_vector<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static consteval basic_vector<E, A>
        DPL_VECTORCALL operator()(basic_vector<E, A> lhs,
            basic_vector<E, A> mid, basic_vector<E, A> rhs) noexcept {
        return internal::transform<basic_vector<E, A>>(
            [](double lhs, double mid, double rhs) -> E {
#if DPL_HAS_CONSTEXPR_BUILTIN(__builtin_fma)
                return __builtin_fma(mid, rhs, lhs);
#else
                // TODO: Do this properly for MSVC
                return lhs + mid * rhs;
#endif
            },
            lhs, mid, rhs);
    }

    using ternary_broadcasting_fallback<fmacc_t>::operator();
};

template <typename AT, typename BT, typename CT,
    typename A = simd_abi_type_t<AT>>
concept unqualified_canonical_fmacc =
    unqualified_canonical_fmadd<BT, CT, AT, A>;

template <typename M, typename AT, typename BT, typename CT,
    typename A = simd_abi_type_t<AT>>
concept unqualified_canonical_zmfmacc =
    unqualified_canonical_zmfmadd<M, BT, CT, AT, A>;

template <typename AT, typename M, typename BT, typename CT,
    typename A = simd_abi_type_t<AT>>
concept unqualified_canonical_mfmacc = cpo_invocable<fmacc_t, AT, BT, CT> &&
    equivalent_vector_with<AT, cpo_result_t<fmacc_t, AT, BT, CT>> && requires {
        {
            fmadd(internal::abi<A>, internal::declarg<BT>(),
                internal::declarg<CT>(), internal::declarg<AT>(),
                internal::declarg<M>())
        } -> equivalent_vector_with<cpo_result_t<fmacc_t, AT, BT, CT>>;
    };

template <>
struct canonical_impl<fmacc_t> {
private:
    template <typename AT, typename BT, typename CT>
    using result_t DPL_NODEBUG =
        make_canonical_vector_t<simd_element_type_t<AT>,
            common_abi_t<BT, CT, AT>>;

    template <typename AT, typename BT = AT, typename CT = AT>
    using mask_t DPL_NODEBUG = simd_mask_type_t<result_t<AT, BT, CT>>;

    template <typename L, typename R>
    using vector_t DPL_NODEBUG =
        make_canonical_vector_t<simd_element_type_t<L>, common_abi_t<L, R>>;

public:
    template <canonical_vector AT, vector_subsumed_by<AT> BT,
        vector_subsumed_by<BT> CT, typename A = simd_abi_type_t<AT>>
    requires canonical_vector<BT> && canonical_vector<CT> &&
        unqualified_canonical_fmacc<AT, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr AT operator()(AT aval, BT bval, CT cval) noexcept {
        return fmadd(internal::abi<A>, bval, cval, aval);
    }

    template <canonical_vector AT, vector_subsumed_by<AT> BT,
        broadcastable_to<AT> CT, typename A = simd_abi_type_t<AT>>
    requires canonical_vector<BT> && unqualified_canonical_fmacc<AT, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr AT operator()(AT aval, BT bval, CT&& cval) noexcept {
        return fmadd(internal::abi<A>, bval, __DPL forward<CT>(cval), aval);
    }

    template <canonical_vector AT, vector_subsumed_by<AT> CT,
        broadcastable_to<AT> BT, typename A = simd_abi_type_t<AT>>
    requires canonical_vector<CT> && unqualified_canonical_fmacc<AT, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr AT operator()(AT aval, BT&& bval, CT cval) noexcept {
        return fmadd(internal::abi<A>, __DPL forward<BT>(bval), cval, aval);
    }

    template <canonical_vector BT, common_vector_with<BT> CT,
        broadcastable_to<vector_t<BT, CT>> AT,
        typename A = common_abi_t<BT, CT>>
    requires canonical_vector<CT> && unqualified_canonical_fmacc<AT, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr vector_t<BT, CT> operator()(
        AT&& aval, BT bval, CT cval) noexcept {
        return fmadd(internal::abi<A>, bval, cval, __DPL forward<AT>(aval));
    }

    template <canonical_vector AT, broadcastable_to<AT> BT,
        broadcastable_to<AT> CT, typename A = simd_abi_type_t<AT>>
    requires unqualified_canonical_fmacc<AT, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr AT operator()(AT aval, BT&& bval, CT&& cval) noexcept {
        return fmadd(internal::abi<A>, __DPL forward<BT>(bval),
            __DPL forward<CT>(cval), aval);
    }

    template <canonical_vector BT, broadcastable_to<BT> AT,
        broadcastable_to<BT> CT, typename A = simd_abi_type_t<BT>>
    requires unqualified_canonical_fmacc<AT, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr BT operator()(AT&& aval, BT bval, CT&& cval) noexcept {
        return fmadd(internal::abi<A>, bval, __DPL forward<CT>(cval),
            __DPL forward<AT>(aval));
    }

    template <canonical_vector CT, broadcastable_to<CT> AT,
        broadcastable_to<CT> BT, typename A = simd_abi_type_t<CT>>
    requires unqualified_canonical_fmacc<AT, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr CT operator()(AT&& aval, BT&& bval, CT cval) noexcept {
        return fmadd(internal::abi<A>, __DPL forward<BT>(bval), cval,
            __DPL forward<AT>(aval));
    }

    template <canonical_vector AT, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT, typename A = simd_abi_type_t<AT>>
    requires canonical_vector<BT> && canonical_vector<CT> &&
        unqualified_canonical_mfmacc<AT, mask_t<AT>, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr AT operator()(
        AT aval, mask_t<AT> mask, BT bval, CT cval) noexcept {
        return fmadd(internal::abi<A>, bval, cval, aval, mask);
    }

    template <canonical_vector AT, const_mask_for<AT> M,
        vector_subsumed_by<AT> BT, vector_subsumed_by<AT> CT,
        typename A = simd_abi_type_t<AT>>
    requires canonical_vector<BT> && canonical_vector<CT> &&
        unqualified_canonical_mfmacc<AT, launder_cmask_t<AT, M>, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr AT operator()(
        AT aval, M cmask, BT bval, CT cval) noexcept {
        return fmadd(
            internal::abi<A>, bval, cval, aval, dx::to_const_mask<AT>(cmask));
    }

    template <canonical_vector AT, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT, typename A = simd_abi_type_t<AT>>
    requires canonical_vector<BT> && canonical_vector<CT> &&
        unqualified_canonical_zmfmacc<mask_t<AT, BT, CT>, AT, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr AT operator()(dx::zero_t zero, mask_t<AT, BT, CT> mask,
        AT aval, BT bval, CT cval) noexcept {
        return fmadd(internal::abi<A>, zero, mask, bval, cval, aval);
    }

    template <canonical_vector AT, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT, const_mask_for<AT> M,
        typename A = simd_abi_type_t<AT>>
    requires canonical_vector<BT> && canonical_vector<CT> &&
        unqualified_canonical_zmfmacc<launder_cmask_t<AT, M>, AT, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr AT operator()(
        dx::zero_t zero, M cmask, AT aval, BT bval, CT cval) noexcept {
        return fmadd(internal::abi<A>, zero,
            dx::to_const_mask<result_t<AT, BT, CT>>(cmask), bval, cval, aval);
    }
};

template <typename AT, typename BT, typename CT,
    typename A = common_abi_t<BT, CT, AT>>
concept unqualified_extended_fmacc = unqualified_extended_fmadd<BT, CT, AT, A>;

template <typename AT, typename M, typename BT, typename CT>
concept unqualified_extended_mfmacc = cpo_invocable<fmacc_t, AT, BT, CT> &&
    equivalent_vector_with<AT, cpo_result_t<fmacc_t, AT, BT, CT>> && requires {
        {
            fmadd(internal::declarg<BT>(), internal::declarg<CT>(),
                internal::declarg<AT>(), internal::declarg<M>())
        } -> equivalent_vector_with<cpo_result_t<fmacc_t, AT, BT, CT>>;
    };

template <typename M, typename AT, typename BT, typename CT>
concept unqualified_extended_zmfmacc =
    unqualified_extended_zmfmadd<M, BT, CT, AT>;

template <>
struct extended_impl<fmacc_t> {
private:
    template <typename L, typename R>
    using vector_t DPL_NODEBUG =
        make_canonical_vector_t<simd_element_type_t<L>, common_abi_t<L, R>>;

public:
    template <simd_vector AT, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT, typename A = simd_abi_type_t<AT>>
    requires (extended_vector<AT> || extended_vector<BT> ||
                 extended_vector<CT>) &&
        unqualified_extended_fmacc<AT, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& aval, BT&& bval, CT&& cval) {
        return fmadd(__DPL forward<BT>(bval), __DPL forward<CT>(cval),
            __DPL forward<AT>(aval));
    }

    template <simd_vector AT, vector_subsumed_by<AT> BT,
        broadcastable_to<AT> CT, typename A = simd_abi_type_t<AT>>
    requires (extended_vector<AT> || extended_vector<BT>) &&
        unqualified_extended_fmacc<AT, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& aval, BT&& bval, CT&& cval) {
        return fmadd(__DPL forward<BT>(bval), __DPL forward<CT>(cval),
            __DPL forward<AT>(aval));
    }

    template <simd_vector AT, vector_subsumed_by<AT> CT,
        broadcastable_to<AT> BT, typename A = simd_abi_type_t<AT>>
    requires (extended_vector<AT> || extended_vector<CT>) &&
        unqualified_extended_fmacc<AT, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& aval, BT&& bval, CT&& cval) {
        return fmadd(__DPL forward<BT>(bval), __DPL forward<CT>(cval),
            __DPL forward<AT>(aval));
    }

    template <simd_vector BT, common_vector_with<BT> CT,
        broadcastable_to<vector_t<BT, CT>> AT,
        typename A = common_abi_t<BT, CT>>
    requires (extended_vector<BT> || extended_vector<CT>) &&
        unqualified_extended_fmacc<AT, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& aval, BT&& bval, CT&& cval) {
        return fmadd(__DPL forward<BT>(bval), __DPL forward<CT>(cval),
            __DPL forward<AT>(aval));
    }

    template <extended_vector AT, broadcastable_to<AT> BT,
        broadcastable_to<AT> CT, typename A = simd_abi_type_t<AT>>
    requires unqualified_extended_fmacc<AT, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& aval, BT&& bval, CT&& cval) {
        return fmadd(__DPL forward<BT>(bval), __DPL forward<CT>(cval),
            __DPL forward<AT>(aval));
    }

    template <extended_vector BT, broadcastable_to<BT> AT,
        broadcastable_to<BT> CT, typename A = simd_abi_type_t<BT>>
    requires unqualified_extended_fmacc<AT, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& aval, BT&& bval, CT&& cval) {
        return fmadd(__DPL forward<BT>(bval), __DPL forward<CT>(cval),
            __DPL forward<AT>(aval));
    }

    template <extended_vector CT, broadcastable_to<CT> AT,
        broadcastable_to<CT> BT, typename A = simd_abi_type_t<CT>>
    requires unqualified_extended_fmacc<AT, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& aval, BT&& bval, CT&& cval) {
        return fmadd(__DPL forward<BT>(bval), __DPL forward<CT>(cval),
            __DPL forward<AT>(aval));
    }

    template <simd_vector AT, exact_mask_for<AT> M, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT, typename A = common_abi_t<AT, BT, CT>>
    requires (extended_mask<M> || extended_vector<AT> || extended_vector<BT> ||
                 extended_vector<CT>) &&
        unqualified_extended_mfmacc<AT, M, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        AT&& aval, M&& mask, BT&& bval, CT&& cval) {
        return fmadd( __DPL forward<BT>(bval), __DPL forward<CT>(cval),
            __DPL forward<AT>(aval), __DPL forward<M>(mask));
    }

    template <simd_vector AT, const_mask_for<AT> M, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    requires (extended_vector<AT> || extended_vector<BT> ||
                 extended_vector<CT>) &&
        unqualified_extended_mfmacc<AT, launder_cmask_t<AT, M>, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& aval, M cmask, BT&& bval, CT&& cval) {
        return fmadd( __DPL forward<BT>(bval), __DPL forward<CT>(cval),
            __DPL forward<AT>(aval), dx::to_const_mask<AT>(cmask));
    }

    template <simd_vector AT, exact_mask_for<AT> M, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    requires (extended_mask<M> || extended_vector<AT> || extended_vector<BT> ||
                 extended_vector<CT>) &&
        unqualified_extended_zmfmacc<M, AT, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, AT&& aval, BT&& bval, CT&& cval) {
        return fmadd(zero, __DPL forward<M>(mask), __DPL forward<BT>(bval),
            __DPL forward<CT>(cval), __DPL forward<AT>(aval));
    }

    template <simd_vector AT, const_mask_for<AT> M, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    requires (extended_vector<AT> || extended_vector<BT> ||
                 extended_vector<CT>) &&
        unqualified_extended_zmfmacc<launder_cmask_t<AT, M>, AT, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M cmask, AT&& aval, BT&& bval, CT&& cval) {
        return fmadd(zero, dx::to_const_mask<AT>(cmask),
            __DPL forward<BT>(bval), __DPL forward<CT>(cval),
            __DPL forward<AT>(aval));
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::fmacc_t fmacc{};
} // namespace cpo
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
