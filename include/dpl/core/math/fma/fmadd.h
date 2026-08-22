// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

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
void fmadd(...) noexcept = delete;

struct fmadd_t;

struct DPL_EMPTY_BASES fmadd_t :
    public math_operation_base<fmadd_t>,
    public maskable_accumulation_base<fmadd_t>,
    public ternary_broadcastable_operation<fmadd_t> {
    using operation_base<fmadd_t>::operator();
    using maskable_accumulation_base<fmadd_t>::operator();
    using ternary_broadcastable_operation<fmadd_t>::operator();
};

template <>
struct operation_signature<fmadd_t> {
    template <typename AT, typename BT, typename CT>
    requires simd_vector<AT> || simd_vector<BT> || simd_vector<CT>
    static consteval void operator()(AT&&, BT&&, CT&&) noexcept {}
};

template <>
struct fallback_impl<fmadd_t> : ternary_broadcasting_fallback<fmadd_t> {
public:
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
                return __builtin_fma(lhs, mid, rhs);
#else
                // TODO: Do this properly for MSVC
                return lhs * mid + rhs;
#endif
            },
            lhs, mid, rhs);
    }

    using ternary_broadcasting_fallback<fmadd_t>::operator();
};

template <typename AT, typename BT, typename CT,
    typename A = simd_abi_type_t<AT>>
concept unqualified_canonical_fmadd = requires {
    {
        fmadd(internal::abi<A>, internal::declarg<AT>(),
            internal::declarg<BT>(), internal::declarg<CT>())
    } -> canonical_vector;
};

template <typename AT, typename M, typename BT, typename CT,
    typename A = simd_abi_type_t<AT>>
concept unqualified_canonical_mfmadd = cpo_invocable<fmadd_t, AT, BT, CT> &&
    equivalent_vector_with<AT, cpo_result_t<fmadd_t, AT, BT, CT>> && requires {
        {
            fmadd(internal::abi<A>, internal::declarg<AT>(),
                internal::declarg<M>(), internal::declarg<BT>(),
                internal::declarg<CT>())
        } -> equivalent_vector_with<cpo_result_t<fmadd_t, AT, BT, CT>>;
    };

template <typename M, typename AT, typename BT, typename CT,
    typename A = simd_abi_type_t<AT>>
concept unqualified_canonical_zmfmadd =
    cpo_invocable<fmadd_t, AT, BT, CT> && requires {
        {
            fmadd(internal::abi<A>, dx::zero, internal::declarg<M>(),
                internal::declarg<AT>(), internal::declarg<BT>(),
                internal::declarg<CT>())
        } -> equivalent_vector_with<cpo_result_t<fmadd_t, AT, BT, CT>>;
    };

template <>
struct canonical_impl<fmadd_t> {
private:
    template <typename AT>
    using mask_t DPL_NODEBUG = simd_mask_type_t<AT>;

    template <typename L, typename R>
    using vector_t DPL_NODEBUG =
        make_canonical_vector_t<simd_element_type_t<L>, common_abi_t<L, R>>;

public:
    template <canonical_vector AT, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT, typename A = simd_abi_type_t<AT>>
    requires canonical_vector<BT> && canonical_vector<CT> &&
        unqualified_canonical_fmadd<AT, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr AT operator()(AT aval, BT bval, CT cval) noexcept {
        return fmadd(internal::abi<A>, aval, bval, cval);
    }

    template <canonical_vector AT, vector_subsumed_by<AT> BT,
        broadcastable_to<AT> CT, typename A = simd_abi_type_t<AT>>
    requires canonical_vector<BT> && unqualified_canonical_fmadd<AT, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr AT operator()(AT aval, BT bval, CT&& cval) noexcept {
        return fmadd(internal::abi<A>, aval, bval, __DPL forward<CT>(cval));
    }

    template <canonical_vector AT, vector_subsumed_by<AT> CT,
        broadcastable_to<AT> BT, typename A = simd_abi_type_t<AT>>
    requires canonical_vector<CT> && unqualified_canonical_fmadd<AT, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr AT operator()(AT aval, BT&& bval, CT cval) noexcept {
        return fmadd(internal::abi<A>, aval, __DPL forward<BT>(bval), cval);
    }

    template <canonical_vector BT, common_vector_with<BT> CT,
        broadcastable_to<vector_t<BT, CT>> AT,
        typename A = common_abi_t<BT, CT>>
    requires canonical_vector<CT> && unqualified_canonical_fmadd<AT, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr vector_t<BT, CT> operator()(
        AT&& aval, BT bval, CT cval) noexcept {
        return fmadd(internal::abi<A>, __DPL forward<AT>(aval), bval, cval);
    }

    template <canonical_vector AT, broadcastable_to<AT> BT,
        broadcastable_to<AT> CT, typename A = simd_abi_type_t<AT>>
    requires unqualified_canonical_fmadd<AT, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr AT operator()(AT aval, BT&& bval, CT&& cval) noexcept {
        return fmadd(internal::abi<A>, aval, __DPL forward<BT>(bval),
            __DPL forward<CT>(cval));
    }

    template <canonical_vector BT, broadcastable_to<BT> AT,
        broadcastable_to<BT> CT, typename A = simd_abi_type_t<BT>>
    requires unqualified_canonical_fmadd<AT, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr BT operator()(AT&& aval, BT bval, CT&& cval) noexcept {
        return fmadd(internal::abi<A>, __DPL forward<AT>(aval), bval,
            __DPL forward<CT>(cval));
    }

    template <canonical_vector CT, broadcastable_to<CT> AT,
        broadcastable_to<CT> BT, typename A = simd_abi_type_t<CT>>
    requires unqualified_canonical_fmadd<AT, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr CT operator()(AT&& aval, BT&& bval, CT cval) noexcept {
        return fmadd(internal::abi<A>, __DPL forward<AT>(aval),
            __DPL forward<BT>(bval), cval);
    }

    template <canonical_vector AT, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT, typename A = simd_abi_type_t<AT>>
    requires canonical_vector<BT> && canonical_vector<CT> &&
        unqualified_canonical_mfmadd<AT, mask_t<AT>, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr AT operator()(
        AT aval, mask_t<AT> mask, BT bval, CT cval) noexcept {
        return fmadd(internal::abi<A>, aval, mask, bval, cval);
    }

    template <canonical_vector AT, const_mask_for<AT> M,
        vector_subsumed_by<AT> BT, vector_subsumed_by<BT> CT,
        typename A = simd_abi_type_t<AT>>
    requires canonical_vector<BT> && canonical_vector<CT> &&
        unqualified_canonical_mfmadd<AT, launder_cmask_t<AT, M>, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr AT operator()(
        AT aval, M cmask, BT bval, CT cval) noexcept {
        return fmadd(
            internal::abi<A>, aval, dx::to_const_mask<AT>(cmask), bval, cval);
    }

    template <canonical_vector AT, vector_subsumed_by<AT> BT,
        vector_subsumed_by<BT> CT, typename A = simd_abi_type_t<AT>>
    requires canonical_vector<BT> && canonical_vector<CT> &&
        unqualified_canonical_zmfmadd<mask_t<AT>, AT, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr AT operator()(
        dx::zero_t zero, mask_t<AT> mask, AT aval, BT bval, CT cval) noexcept {
        return fmadd(internal::abi<A>, zero, mask, aval, bval, cval);
    }

    template <canonical_vector AT, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT, const_mask_for<AT> M,
        typename A = simd_abi_type_t<AT>>
    requires canonical_vector<BT> && canonical_vector<CT> &&
        unqualified_canonical_zmfmadd<launder_cmask_t<AT, M>, AT, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr AT operator()(
        dx::zero_t zero, M cmask, AT aval, BT bval, CT cval) noexcept {
        return fmadd(internal::abi<A>, zero, dx::to_const_mask<AT>(cmask), aval,
            bval, cval);
    }
};

template <typename AT, typename BT, typename CT,
    typename A = common_abi_t<AT, BT, CT>>
concept unqualified_extended_fmadd = requires {
    {
        fmadd(internal::declarg<AT>(), internal::declarg<BT>(),
            internal::declarg<CT>())
    } -> vector_with_common_abi<A>;
};

template <typename AT, typename M, typename BT, typename CT>
concept unqualified_extended_mfmadd = cpo_invocable<fmadd_t, AT, BT, CT> &&
    equivalent_vector_with<AT, cpo_result_t<fmadd_t, AT, BT, CT>> && requires {
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
    using vector_t DPL_NODEBUG =
        make_canonical_vector_t<simd_element_type_t<L>, common_abi_t<L, R>>;

public:
    template <simd_vector AT, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT, typename A = common_abi_t<AT, BT, CT>>
    requires (extended_vector<AT> || extended_vector<BT> ||
                 extended_vector<CT>) &&
        unqualified_extended_fmadd<AT, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& aval, BT&& bval, CT&& cval) {
        return fmadd(__DPL forward<AT>(aval), __DPL forward<BT>(bval),
            __DPL forward<CT>(cval));
    }

    template <simd_vector AT, vector_subsumed_by<AT> BT,
        broadcastable_to<AT> CT, typename A = simd_abi_type_t<AT>>
    requires (extended_vector<AT> || extended_vector<BT>) &&
        unqualified_extended_fmadd<AT, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& aval, BT&& bval, CT&& cval) {
        return fmadd(__DPL forward<AT>(aval), __DPL forward<BT>(bval),
            __DPL forward<CT>(cval));
    }

    template <simd_vector AT, vector_subsumed_by<AT> CT,
        broadcastable_to<AT> BT, typename A = simd_abi_type_t<AT>>
    requires (extended_vector<AT> || extended_vector<CT>) &&
        unqualified_extended_fmadd<AT, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& aval, BT&& bval, CT&& cval) {
        return fmadd(__DPL forward<AT>(aval), __DPL forward<BT>(bval),
            __DPL forward<CT>(cval));
    }

    template <simd_vector BT, common_vector_with<BT> CT,
        broadcastable_to<vector_t<BT, CT>> AT,
        typename A = common_abi_t<BT, CT>>
    requires (extended_vector<BT> || extended_vector<CT>) &&
        unqualified_extended_fmadd<AT, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& aval, BT&& bval, CT&& cval) {
        return fmadd(__DPL forward<AT>(aval), __DPL forward<BT>(bval),
            __DPL forward<CT>(cval));
    }

    template <extended_vector AT, broadcastable_to<AT> BT,
        broadcastable_to<AT> CT, typename A = simd_abi_type_t<AT>>
    requires unqualified_extended_fmadd<AT, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& aval, BT&& bval, CT&& cval) {
        return fmadd(__DPL forward<AT>(aval), __DPL forward<BT>(bval),
            __DPL forward<CT>(cval));
    }

    template <extended_vector BT, broadcastable_to<BT> AT,
        broadcastable_to<BT> CT, typename A = simd_abi_type_t<BT>>
    requires unqualified_extended_fmadd<AT, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& aval, BT&& bval, CT&& cval) {
        return fmadd(__DPL forward<AT>(aval), __DPL forward<BT>(bval),
            __DPL forward<CT>(cval));
    }

    template <extended_vector CT, broadcastable_to<CT> AT,
        broadcastable_to<CT> BT, typename A = simd_abi_type_t<CT>>
    requires unqualified_extended_fmadd<AT, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& aval, BT&& bval, CT&& cval) {
        return fmadd(__DPL forward<AT>(aval), __DPL forward<BT>(bval),
            __DPL forward<CT>(cval));
    }

    template <simd_vector AT, exact_mask_for<AT> M, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT, typename A = common_abi_t<AT, BT, CT>>
    requires (extended_mask<M> || extended_vector<AT> || extended_vector<BT> ||
                 extended_vector<CT>) &&
        unqualified_extended_mfmadd<AT, M, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        AT&& aval, M&& mask, BT&& bval, CT&& cval) {
        return fmadd(__DPL forward<AT>(aval), __DPL forward<M>(mask),
            __DPL forward<BT>(bval), __DPL forward<CT>(cval));
    }

    template <simd_vector AT, const_mask_for<AT> M, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    requires (extended_vector<AT> || extended_vector<BT> ||
                 extended_vector<CT>) &&
        unqualified_extended_mfmadd<AT, launder_cmask_t<AT, M>, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& aval, M cmask, BT&& bval, CT&& cval) {
        return fmadd(__DPL forward<AT>(aval), dx::to_const_mask<AT>(cmask),
            __DPL forward<BT>(bval), __DPL forward<CT>(cval));
    }

    template <simd_vector AT, exact_mask_for<AT> M, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    requires (extended_mask<M> || extended_vector<AT> || extended_vector<BT> ||
                 extended_vector<CT>) &&
        unqualified_extended_zmfmadd<M, AT, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, AT&& aval, BT&& bval, CT&& cval) {
        return fmadd(zero, __DPL forward<M>(mask), __DPL forward<AT>(aval),
            __DPL forward<BT>(bval), __DPL forward<CT>(cval));
    }

    template <simd_vector AT, const_mask_for<AT> M, vector_subsumed_by<AT> BT,
        vector_subsumed_by<AT> CT>
    requires (extended_vector<AT> || extended_vector<BT> ||
                 extended_vector<CT>) &&
        unqualified_extended_zmfmadd<launder_cmask_t<AT, M>, AT, BT, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M cmask, AT&& aval, BT&& bval, CT&& cval) {
        return fmadd(zero, dx::to_const_mask<AT>(cmask),
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
