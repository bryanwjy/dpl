// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/dispatch/private/fwd.h"

#include "dpl/core/dispatch/maskable/base.h"
#include "dpl/core/dispatch/operation/basic.h" // IWYU pragma: keep
#include "dpl/core/dispatch/private/concepts.h"
#include "dpl/core/dispatch/private/cpo.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/core/concepts/cpo_invocable.h"
#  include "dpl/core/concepts/mask_compatibility.h"
#  include "dpl/core/immediate/const_mask.h"
#  include "dpl/core/immediate/constants/zero.h"
#  include "dpl/core/type_traits/simd_abi_type.h"
#  include "dpl/core/type_traits/simd_element_type.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::internal {

template <typename D, typename R, typename M, typename... Ts>
concept masked_foldable = canonical_cpo_invocable_r<D, R, M, Ts...> ||
    (!inherits_from<D, basic_operation_base<D>> &&
        fallback_cpo_invocable_r<D, R, M, Ts...>);

template <typename D>
class maskable_fold_base : public maskable_operation_base<D> {

    template <typename... Ts>
    using result_t DPL_NODEBUG = cpo_result_t<D, Ts...>;
    template <typename... Ts>
    using canonical_mask_t DPL_NODEBUG = simd_mask_type_t<result_t<Ts...>>;

protected:
    template <canonical_ornot_simd... Ts>
    requires signature_compatible<D, Ts...> && cpo_invocable<D, Ts...> &&
        canonical_vector<result_t<Ts...>> &&
        masked_foldable<D, result_t<Ts...>, canonical_mask_t<Ts...>, Ts...>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<Ts...> operator()(
        canonical_mask_t<Ts...> mask, Ts... args) noexcept {
        using R = result_t<Ts...>;
        using M = canonical_mask_t<Ts...>;
        if constexpr (canonical_cpo_invocable_r<D, R, M, Ts...>) {
            if constexpr (all_same_abi<R, M, Ts...> &&
                fallback_cpo_invocable_r<D, R, M, Ts...>) {
                if consteval {
                    return impl::fallback<D>(mask, args...);
                } else {
                    return impl::canonical<D>(mask, args...);
                }
            } else {
                return impl::canonical<D>(mask, args...);
            }
        } else {
            return impl::fallback<D>(mask, args...);
        }
    }

    template <typename M, canonical_ornot_simd... Ts>
    requires signature_compatible<D, Ts...> && cpo_invocable<D, Ts...> &&
        const_mask_for<M, result_t<Ts...>> &&
        canonical_vector<result_t<Ts...>> &&
        masked_foldable<D, result_t<Ts...>, M, Ts...>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<Ts...> operator()(
        result_t<Ts...> src, M mask, Ts... args) noexcept {
        using R = result_t<Ts...>;
        if constexpr (canonical_cpo_invocable_r<D, R, M, Ts...>) {
            if constexpr (all_same_abi<R, Ts...> &&
                fallback_cpo_invocable_r<D, R, M, Ts...>) {
                if consteval {
                    return impl::fallback<D>(mask, args...);
                } else {
                    return impl::canonical<D>(mask, args...);
                }
            } else {
                return impl::canonical<D>(mask, args...);
            }
        } else {
            return impl::fallback<D>(src, mask, args...);
        }
    }

    template <simd_mask M, typename... Ts>
    requires signature_compatible<D, Ts...> && extended_arguments<M, Ts...> &&
        cpo_invocable<D, Ts...> && exact_mask_for<M, result_t<Ts...>> &&
        simd_vector<result_t<Ts...>> &&
        (extendable_operation_r<D, result_t<Ts...>, M, Ts...> ||
            requires {
                requires (simd_expression<M> || ... || simd_expression<Ts>);
                operator()(internal::declarg<result_or_identity_t<M>>(),
                    internal::declarg<result_or_identity_t<Ts>>()...);
            } || fallback_cpo_invocable_r<D, result_t<Ts...>, M, Ts...>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M&& mask, Ts&&... args) {
        using R DPL_NODEBUG = result_t<Ts...>;
        if constexpr (extended_cpo_invocable_r<D, R, M, Ts...>) {
            return impl::extended<D>(
                __DPL forward<M>(mask), __DPL forward<Ts>(args)...);
        } else if constexpr (extended_nttp_invocable_r<D, R, M, Ts...>) {
            constexpr D Op;
            return extended<Op>(
                __DPL forward<M>(mask), __DPL forward<Ts>(args)...);
        } else if constexpr ((simd_expression<M> || ... ||
                                 simd_expression<Ts>)) {
            return operator()(
                internal::forward_or_eval(__DPL forward<M>(mask)),
                internal::forward_or_eval(__DPL forward<Ts>(args))...);
        } else {
            return impl::fallback<D>(
                __DPL forward<M>(mask), __DPL forward<Ts>(args)...);
        }
    }

    template <typename M, typename... Ts>
    requires signature_compatible<D, Ts...> && extended_arguments<Ts...> &&
        cpo_invocable<D, Ts...> && const_mask_for<M, result_t<Ts...>> &&
        simd_vector<result_t<Ts...>> &&
        (extendable_operation_r<D, result_t<Ts...>, M, Ts...> ||
            requires {
                requires (... || simd_expression<Ts>);
                operator()(internal::declarg<M>(),
                    internal::declarg<result_or_identity_t<Ts>>()...);
            } || fallback_cpo_invocable_r<D, result_t<Ts...>, M, Ts...>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M mask, Ts&&... args) {
        using R DPL_NODEBUG = result_t<Ts...>;
        if constexpr (extended_cpo_invocable_r<D, R, M, Ts...>) {
            return impl::extended<D>(mask, __DPL forward<Ts>(args)...);
        } else if constexpr (extended_nttp_invocable_r<D, R,
                                 launder_cmask_t<R, M>, Ts...>) {
            constexpr D Op;
            return extended<Op>(
                dx::to_const_mask<R>(mask), __DPL forward<Ts>(args)...);
        } else if constexpr ((... || simd_expression<Ts>)) {
            return operator()(
                mask, internal::forward_or_eval(__DPL forward<Ts>(args))...);
        } else {
            return impl::fallback<D>(mask, __DPL forward<Ts>(args)...);
        }
    }
};
} // namespace datapar::internal

__DPL_DEFAULT_NAMESPACE_END
