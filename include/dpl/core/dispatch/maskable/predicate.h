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
#  include "dpl/core/concepts/simd_mask.h"
#  include "dpl/core/immediate/const_mask.h"
#  include "dpl/std/utility/forward.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::internal {

/**
 * @brief A maskable operation that computes a mask from its arguments.
 *
 * The result is always a @ref simd_mask. In the masked variant,
 * inactive lanes take their value from the input mask.
 *
 * @note At least one argument must be a simd_vector
 *
 * @par Signatures
 * @code
 * op(args...)        // unmasked
 * op(mask, args...)  // masked
 * @endcode
 *
 * @see maskable_transform_base
 * @see maskable_accumulation_base
 */
template <typename D>
class maskable_predicate_base : public maskable_operation_base<D> {

    template <typename... Ts>
    using result_t DPL_NODEBUG = cpo_result_t<D, Ts...>;
    template <typename... Ts>
    using basic_mask_t DPL_NODEBUG = result_t<Ts...>;

protected:
    template <canonical_ornot_simd... Ts>
    requires signature_compatible<D, Ts...> && has_simd_vector<Ts...> &&
        cpo_invocable<D, Ts...> && simd_mask<result_t<Ts...>> &&
        cpo_invocable<bwand_t, basic_mask_t<Ts...>, result_t<Ts...>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<Ts...> operator()(
        basic_mask_t<Ts...> mask, Ts... args) noexcept {
        using M = basic_mask_t<Ts...>;
        if constexpr (canonical_cpo_invocable_r<D, M, M, Ts...>) {
            if constexpr (all_same_abi<basic_mask_t<Ts...>, Ts...>) {
                if consteval {
                    if constexpr (fallback_cpo_invocable_r<D, M, M, Ts...>) {
                        return impl::fallback<D>(mask, args...);
                    } else {
                        return fwd::bwand(mask, D::operator()(args...));
                    }
                } else {
                    return impl::canonical<D>(mask, args...);
                }
            } else {
                return impl::canonical<D>(mask, args...);
            }
        } else if constexpr (fallback_cpo_invocable_r<D, M, M, Ts...>) {
            return impl::fallback<D>(mask, args...);
        } else {
            return fwd::bwand(mask, D::operator()(args...));
        }
    }

    template <typename M, canonical_ornot_simd... Ts>
    requires signature_compatible<D, Ts...> && has_simd_vector<Ts...> &&
        cpo_invocable<D, Ts...> && simd_mask<result_t<Ts...>> &&
        const_mask_for<M, result_t<Ts...>> &&
        cpo_invocable<select_t, M, result_t<Ts...>, dx::zero_t>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<Ts...> operator()(M mask, Ts... args) noexcept {
        using R = result_t<Ts...>;
        if constexpr (canonical_cpo_invocable_r<D, R, M, Ts...>) {
            if constexpr (all_same_abi<Ts...>) {
                if consteval {
                    if constexpr (fallback_cpo_invocable_r<D, R, M, Ts...>) {
                        return impl::fallback<D>(mask, args...);
                    } else {
                        return fwd::select(
                            mask, D::operator()(args...), dx::zero);
                    }
                } else {
                    return impl::canonical<D>(mask, args...);
                }
            } else {
                return impl::canonical<D>(mask, args...);
            }
        } else if constexpr (fallback_cpo_invocable_r<D, R, M, Ts...>) {
            return impl::fallback<D>(mask, args...);
        } else {
            return fwd::select(mask, D::operator()(args...), dx::zero);
        }
    }

    template <simd_mask M, typename... Ts>
    requires signature_compatible<D, Ts...> && has_simd_vector<Ts...> &&
        extended_arguments<M, Ts...> && cpo_invocable<D, Ts...> &&
        equivalent_mask_with<M, result_t<Ts...>> &&
        (extendable_operation_r<D, result_t<Ts...>, M, Ts...> ||
            requires {
                requires (simd_expression<M> || ... || simd_expression<Ts>);
                operator()(internal::declarg<result_or_identity_t<M>>(),
                    internal::declarg<result_or_identity_t<Ts>>()...);
            } || fallback_cpo_invocable_r<D, result_t<Ts...>, M, Ts...>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M&& mask, Ts&&... args) {
        if constexpr (extended_cpo_invocable_r<D, result_t<Ts...>, M, Ts...>) {
            return impl::extended<D>(
                __DPL forward<M>(mask), __DPL forward<Ts>(args)...);
        } else if constexpr (extended_nttp_invocable_r<D, result_t<Ts...>, M,
                                 Ts...>) {
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
    requires signature_compatible<D, Ts...> && has_simd_vector<Ts...> &&
        extended_arguments<Ts...> && cpo_invocable<D, Ts...> &&
        const_mask_for<M, result_t<Ts...>> && simd_mask<result_t<Ts...>> &&
        (extendable_operation_r<D, result_t<Ts...>,
             launder_cmask_t<result_t<Ts...>, M>, Ts...> ||
            requires {
                requires (... || simd_expression<Ts>);
                operator()(internal::declarg<M>(),
                    internal::declarg<result_or_identity_t<Ts>>()...);
            } || fallback_cpo_invocable_r<D, result_t<Ts...>, M, Ts...>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, Ts&&... args) {
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

DPL_DEFAULT_NAMESPACE_END
