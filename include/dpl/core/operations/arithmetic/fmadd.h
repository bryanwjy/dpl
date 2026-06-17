// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

// IWYU pragma: always_keep
#include "dpl/core/operations/arithmetic/add.h"
#include "dpl/core/operations/arithmetic/multiply.h"

#if !DPL_MODULES
#  include "dpl/core/basic/internal/abi.h"
#  include "dpl/core/concepts/cpo_invocable.h"
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/dispatch/broadcastable/ternary.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/maskable/transform.h"
#  include "dpl/core/dispatch/operation/primitive.h"
#  include "dpl/core/type_traits/cpo_result.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void fmadd(...) noexcept = delete;

struct fmadd_t;

struct DPL_EMPTY_BASES fmadd_t :
    private arithmetic_base<fmadd_t>,
    private maskable_transform_base<fmadd_t>,
    private ternary_broadcastable_operation<fmadd_t> {
    using operation_base<fmadd_t>::operator();
    using maskable_transform_base<fmadd_t>::operator();
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

    template <canonical_vector AT, canonical_vector BT, canonical_vector CT>
    requires floating_point<simd_element_type_t<AT>> &&
        floating_point<simd_element_type_t<BT>> &&
        floating_point<simd_element_type_t<CT>> &&
        cpo_invocable<multiply_t, AT, BT> &&
        cpo_invocable<add_t, cpo_result_t<multiply_t, AT, BT>, CT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        AT aval, BT bval, CT cval) noexcept {
        return dx::add(dx::multiply(aval, bval), cval);
    }

    using ternary_broadcasting_fallback<fmadd_t>::operator();
};

template <typename AT, typename BT, typename CT,
    typename A = common_abi_t<AT, BT, CT>>
concept unqualified_canonical_fmadd = requires {
    {
        fmadd(internal::abi<A>, internal::declarg<AT>(),
            internal::declarg<BT>(), internal::declarg<CT>())
    } -> canonical_vector;
};

template <typename S, typename M, typename AT, typename BT, typename CT,
    typename A = common_abi_t<AT, BT, CT>>
concept unqualified_canonical_mfmadd =
    (!simd_type<S> ||
        equivalent_vector_with<S, cpo_result_t<fmadd_t, AT, BT, CT>>) &&
    requires {
        {
            fmadd(internal::abi<A>, internal::declarg<S>(),
                internal::declarg<M>(), internal::declarg<AT>(),
                internal::declarg<BT>(), internal::declarg<CT>())
        } -> equivalent_vector_with<cpo_result_t<fmadd_t, AT, BT, CT>>;
    };

template <>
struct canonical_impl<fmadd_t> {
private:
    template <typename AT, typename BT, typename CT>
    using source_t DPL_NODEBUG =
        basic_vector<simd_element_type_t<AT>, common_abi_t<AT, BT, CT>>;

    template <typename AT, typename BT, typename CT>
    using mask_t DPL_NODEBUG =
        basic_mask<simd_element_type_t<AT>, common_abi_t<AT, BT, CT>>;

    template <typename AT, typename BT, typename CT>
    using imask_t DPL_NODEBUG = mask_value_t<common_abi_t<AT, BT, CT>::size>;

    template <typename AT, typename BT, typename CT, imask_t<AT, BT, CT> V>
    using cmask_t DPL_NODEBUG = const_mask<common_abi_t<AT, BT, CT>::size, V>;

    template <typename L, typename R>
    using vector_t DPL_NODEBUG =
        basic_vector<simd_element_type_t<L>, common_abi_t<L, R>>;

public:
    template <canonical_vector AT, common_vector_with<AT> BT,
        common_vector_with<BT> CT, typename A = common_abi_t<AT, BT, CT>>
    requires canonical_vector<BT> && canonical_vector<CT> &&
        unqualified_canonical_fmadd<AT, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT aval, BT bval, CT cval) noexcept {
        return fmadd(internal::abi<A>, aval, bval, cval);
    }

    template <canonical_vector AT, common_vector_with<AT> BT,
        broadcastable_to<vector_t<AT, BT>> CT,
        typename A = common_abi_t<AT, BT>>
    requires canonical_vector<BT> && unqualified_canonical_fmadd<AT, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT aval, BT bval, CT&& cval) noexcept {
        return fmadd(internal::abi<A>, aval, bval, __DPL forward<CT>(cval));
    }

    template <canonical_vector AT, common_vector_with<AT> CT,
        broadcastable_to<vector_t<AT, CT>> BT,
        typename A = common_abi_t<AT, CT>>
    requires canonical_vector<CT> && unqualified_canonical_fmadd<AT, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT aval, BT&& bval, CT cval) noexcept {
        return fmadd(internal::abi<A>, aval, __DPL forward<BT>(bval), cval);
    }

    template <canonical_vector BT, common_vector_with<BT> CT,
        broadcastable_to<vector_t<BT, CT>> AT,
        typename A = common_abi_t<BT, CT>>
    requires canonical_vector<CT> && unqualified_canonical_fmadd<AT, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& aval, BT bval, CT cval) noexcept {
        return fmadd(internal::abi<A>, __DPL forward<AT>(aval), bval, cval);
    }

    template <canonical_vector AT, broadcastable_to<AT> BT,
        broadcastable_to<AT> CT, typename A = simd_abi_type_t<AT>>
    requires unqualified_canonical_fmadd<AT, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT aval, BT&& bval, CT&& cval) noexcept {
        return fmadd(internal::abi<A>, aval, __DPL forward<BT>(bval),
            __DPL forward<CT>(cval));
    }

    template <canonical_vector BT, broadcastable_to<BT> AT,
        broadcastable_to<BT> CT, typename A = simd_abi_type_t<BT>>
    requires unqualified_canonical_fmadd<AT, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& aval, BT bval, CT&& cval) noexcept {
        return fmadd(internal::abi<A>, __DPL forward<AT>(aval), bval,
            __DPL forward<CT>(cval));
    }

    template <canonical_vector CT, broadcastable_to<CT> AT,
        broadcastable_to<CT> BT, typename A = simd_abi_type_t<CT>>
    requires unqualified_canonical_fmadd<AT, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& aval, BT&& bval, CT cval) noexcept {
        return fmadd(internal::abi<A>, __DPL forward<AT>(aval),
            __DPL forward<BT>(bval), cval);
    }

    template <canonical_vector AT, common_vector_with<AT> BT,
        common_vector_with<BT> CT, typename A = common_abi_t<AT, BT, CT>>
    requires canonical_vector<BT> && canonical_vector<CT> &&
        unqualified_canonical_fmadd<AT, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(source_t<AT, BT, CT> src,
        mask_t<AT, BT, CT> mask, AT aval, BT bval, CT cval) noexcept {
        return fmadd(internal::abi<A>, src, mask, aval, bval, cval);
    }

    template <canonical_vector AT, common_vector_with<AT> BT,
        common_vector_with<BT> CT, imask_t<AT, BT, CT> M,
        typename A = common_abi_t<AT, BT, CT>>
    requires canonical_vector<BT> && canonical_vector<CT> &&
        unqualified_canonical_fmadd<AT, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(source_t<AT, BT, CT> src,
        cmask_t<AT, BT, CT, M> cmask, AT aval, BT bval, CT cval) noexcept {
        return fmadd(internal::abi<A>, src, cmask, aval, bval, cval);
    }

    template <canonical_vector AT, common_vector_with<AT> BT,
        common_vector_with<BT> CT, typename A = common_abi_t<AT, BT, CT>>
    requires canonical_vector<BT> && canonical_vector<CT> &&
        unqualified_canonical_fmadd<AT, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, mask_t<AT, BT, CT> mask,
        AT aval, BT bval, CT cval) noexcept {
        return fmadd(internal::abi<A>, zero, mask, aval, bval, cval);
    }

    template <canonical_vector AT, common_vector_with<AT> BT,
        common_vector_with<BT> CT, imask_t<AT, BT, CT> M,
        typename A = common_abi_t<AT, BT, CT>>
    requires canonical_vector<BT> && canonical_vector<CT> &&
        unqualified_canonical_fmadd<AT, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero,
        cmask_t<AT, BT, CT, M> cmask, AT aval, BT bval, CT cval) noexcept {
        return fmadd(internal::abi<A>, zero, cmask, aval, bval, cval);
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

template <typename S, typename M, typename AT, typename BT, typename CT,
    typename A = common_abi_t<AT, BT, CT>>
concept unqualified_extended_mfmadd =
    (!simd_type<S> ||
        equivalent_vector_with<S, cpo_result_t<fmadd_t, AT, BT, CT>>) &&
    requires {
        {
            fmadd(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<AT>(), internal::declarg<BT>(),
                internal::declarg<CT>())
        } -> equivalent_vector_with<cpo_result_t<fmadd_t, AT, BT, CT>>;
    };

template <>
struct extended_impl<fmadd_t> {
private:
    template <typename S>
    using simask_t DPL_NODEBUG = mask_value_t<simd_abi_type_t<S>::size>;

    template <typename S, simask_t<S> V>
    using scmask_t DPL_NODEBUG = const_mask<simd_abi_type_t<S>::size, V>;

    template <typename AT, typename BT, typename CT>
    using mask_t DPL_NODEBUG =
        basic_mask<simd_element_type_t<AT>, common_abi_t<AT, BT, CT>>;

    template <typename AT, typename BT, typename CT>
    using imask_t DPL_NODEBUG = mask_value_t<common_abi_t<AT, BT, CT>::size>;

    template <typename AT, typename BT, typename CT, imask_t<AT, BT, CT> V>
    using cmask_t DPL_NODEBUG = const_mask<common_abi_t<AT, BT, CT>::size, V>;

    template <typename L, typename R>
    using vector_t DPL_NODEBUG =
        basic_vector<simd_element_type_t<L>, common_abi_t<L, R>>;

public:
    template <simd_vector AT, common_vector_with<AT> BT,
        common_vector_with<BT> CT, typename A = common_abi_t<AT, BT, CT>>
    requires (extended_vector<AT> || extended_vector<BT> ||
                 extended_vector<CT>) &&
        unqualified_extended_fmadd<AT, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& aval, BT&& bval, CT&& cval) {
        return fmadd(__DPL forward<AT>(aval), __DPL forward<BT>(bval),
            __DPL forward<CT>(cval));
    }

    template <simd_vector AT, common_vector_with<AT> BT,
        broadcastable_to<vector_t<AT, BT>> CT,
        typename A = common_abi_t<AT, BT>>
    requires (extended_vector<AT> || extended_vector<BT>) &&
        unqualified_extended_fmadd<AT, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(AT&& aval, BT&& bval, CT&& cval) {
        return fmadd(__DPL forward<AT>(aval), __DPL forward<BT>(bval),
            __DPL forward<CT>(cval));
    }

    template <simd_vector AT, common_vector_with<AT> CT,
        broadcastable_to<vector_t<AT, CT>> BT,
        typename A = common_abi_t<AT, CT>>
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

    template <simd_vector S, exact_mask_for<S> M, simd_vector AT,
        common_vector_with<AT> BT, common_vector_with<BT> CT,
        typename A = common_abi_t<AT, BT, CT>>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<AT> ||
                 extended_vector<BT> || extended_vector<CT>) &&
        unqualified_extended_mfmadd<S, M, AT, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        S&& src, M&& mask, AT&& aval, BT&& bval, CT&& cval) {
        return fmadd(__DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<AT>(aval), __DPL forward<BT>(bval),
            __DPL forward<CT>(cval));
    }

    template <fixed_width_vector S, simask_t<S> M, simd_vector AT,
        common_vector_with<AT> BT, common_vector_with<BT> CT,
        typename A = common_abi_t<AT, BT, CT>>
    requires (extended_vector<S> || extended_vector<AT> ||
                 extended_vector<BT> || extended_vector<CT>) &&
        unqualified_extended_mfmadd<S, scmask_t<S, M>, AT, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        S&& src, scmask_t<S, M> cmask, AT&& aval, BT&& bval, CT&& cval) {
        return fmadd(__DPL forward<S>(src), cmask, __DPL forward<AT>(aval),
            __DPL forward<BT>(bval), __DPL forward<CT>(cval));
    }

    template <simd_vector AT, common_vector_with<AT> BT,
        common_vector_with<BT> CT, common_mask_with<mask_t<AT, BT, CT>> M,
        typename A = common_abi_t<AT, BT, CT>>
    requires (extended_mask<M> || extended_vector<AT> || extended_vector<BT> ||
                 extended_vector<CT>) &&
        unqualified_extended_mfmadd<dx::zero_t, M, AT, BT, CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, AT&& aval, BT&& bval, CT&& cval) {
        return fmadd(zero, __DPL forward<M>(mask), __DPL forward<AT>(aval),
            __DPL forward<BT>(bval), __DPL forward<CT>(cval));
    }

    template <simd_vector AT, common_vector_with<AT> BT,
        common_vector_with<BT> CT, imask_t<AT, BT, CT> M,
        typename A = common_abi_t<AT, BT, CT>>
    requires (extended_vector<AT> || extended_vector<BT> ||
                 extended_vector<CT>) &&
        unqualified_extended_mfmadd<dx::zero_t, cmask_t<AT, BT, CT, M>, AT, BT,
            CT, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero,
        cmask_t<AT, BT, CT, M> cmask, AT&& aval, BT&& bval, CT&& cval) {
        return fmadd(zero, cmask, __DPL forward<AT>(aval),
            __DPL forward<BT>(bval), __DPL forward<CT>(cval));
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::fmadd_t fmadd{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
