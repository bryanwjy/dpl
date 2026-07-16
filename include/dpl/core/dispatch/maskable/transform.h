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

template <typename D, typename R, typename S, typename M, typename... Ts>
concept masked_transformable = (!inherits_from<D, basic_operation_base<D>> &&
                                   cpo_invocable<select_t, M, R, S>) ||
    canonical_cpo_invocable_r<D, R, S, M, Ts...>;

/**
 * @brief A maskable operation that computes a new value from its arguments.
 *
 * The result type is determined by the operation and its arguments.
 * In the masked variant, inactive lanes take their value from `src`.
 * In the zeroing masked variant, inactive lanes are zeroed.
 *
 * @par Signatures
 * @code
 * op(args...)             // unmasked
 * op(src, mask, args...)  // masked
 * op(mask, args...)       // zeroing masked
 * @endcode
 *
 * @see maskable_predicate_base
 * @see maskable_accumulation_base
 */
template <typename D>
class maskable_transform_base : public maskable_operation_base<D> {

    template <typename... Ts>
    using result_t DPL_NODEBUG = cpo_result_t<D, Ts...>;
    template <typename... Ts>
    using basic_mask_t DPL_NODEBUG =
        basic_mask<simd_element_type_t<result_t<Ts...>>,
            simd_abi_type_t<result_t<Ts...>>>;

protected:
    template <canonical_ornot_simd... Ts>
    requires signature_compatible<D, Ts...> && cpo_invocable<D, Ts...> &&
        simd_vector<result_t<Ts...>> &&
        masked_transformable<D, result_t<Ts...>, result_t<Ts...>,
            basic_mask_t<Ts...>, Ts...>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<Ts...> operator()(
        result_t<Ts...> src, basic_mask_t<Ts...> mask, Ts... args) noexcept {
        using R = result_t<Ts...>;
        using M = basic_mask_t<Ts...>;
        if constexpr (canonical_cpo_invocable_r<D, R, R, M, Ts...>) {
            if constexpr (all_same_abi<R, M, Ts...> &&
                !inherits_from<D, basic_operation_base<D>>) {
                if consteval {
                    if constexpr (fallback_cpo_invocable_r<D, R, R, M, Ts...>) {
                        return impl::fallback<D>(src, mask, args...);
                    } else {
                        return fwd::select(mask, D::operator()(args...), src);
                    }
                } else {
                    return impl::canonical<D>(src, mask, args...);
                }
            } else {
                return impl::canonical<D>(src, mask, args...);
            }
        } else if constexpr (fallback_cpo_invocable_r<D, R, R, M, Ts...>) {
            return impl::fallback<D>(src, mask, args...);
        } else {
            return fwd::select(mask, D::operator()(args...), src);
        }
    }

    template <canonical_ornot_simd... Ts>
    requires signature_compatible<D, Ts...> && cpo_invocable<D, Ts...> &&
        simd_vector<result_t<Ts...>> &&
        masked_transformable<D, result_t<Ts...>, dx::zero_t,
            basic_mask_t<Ts...>, Ts...>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<Ts...> operator()(
        dx::zero_t zero, basic_mask_t<Ts...> mask, Ts... args) noexcept {
        using R = result_t<Ts...>;
        using M = basic_mask_t<Ts...>;
        if constexpr (canonical_cpo_invocable_r<D, R, R, M, Ts...>) {
            if constexpr (all_same_abi<R, M, Ts...>) {
                if consteval {
                    if constexpr (fallback_cpo_invocable_r<D, R, dx::zero_t, M,
                                      Ts...> &&
                        !inherits_from<D, basic_operation_base<D>>) {
                        return impl::fallback<D>(zero, mask, args...);
                    } else {
                        return fwd::select(mask, D::operator()(args...), zero);
                    }
                } else {
                    return impl::canonical<D>(zero, mask, args...);
                }
            } else {
                return impl::canonical<D>(zero, mask, args...);
            }
        } else if constexpr (fallback_cpo_invocable_r<D, R, dx::zero_t, M,
                                 Ts...>) {
            return impl::fallback<D>(dx::zero, mask, args...);
        } else {
            return fwd::select(mask, D::operator()(args...), zero);
        }
    }

    template <typename M, canonical_ornot_simd... Ts>
    requires signature_compatible<D, Ts...> && cpo_invocable<D, Ts...> &&
        const_mask_for<M, result_t<Ts...>> && simd_vector<result_t<Ts...>> &&
        masked_transformable<D, result_t<Ts...>, result_t<Ts...>, M, Ts...>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<Ts...> operator()(
        result_t<Ts...> src, M mask, Ts... args) noexcept {
        using R = result_t<Ts...>;
        if constexpr (canonical_cpo_invocable_r<D, R, R, M, Ts...>) {
            if constexpr (all_same_abi<R, Ts...> &&
                !inherits_from<D, basic_operation_base<D>>) {
                if consteval {
                    if constexpr (fallback_cpo_invocable_r<D, R, R, M, Ts...>) {
                        return impl::fallback<D>(src, mask, args...);
                    } else {
                        return fwd::select(mask, D::operator()(args...), src);
                    }
                } else {
                    return impl::canonical<D>(src, mask, args...);
                }
            } else {
                return impl::canonical<D>(src, mask, args...);
            }
        } else if constexpr (fallback_cpo_invocable_r<D, R, R, M, Ts...>) {
            return impl::fallback<D>(src, mask, args...);
        } else {
            return fwd::select(mask, D::operator()(args...), src);
        }
    }

    template <typename M, canonical_ornot_simd... Ts>
    requires signature_compatible<D, Ts...> && cpo_invocable<D, Ts...> &&
        const_mask_for<M, result_t<Ts...>> && simd_vector<result_t<Ts...>> &&
        masked_transformable<D, result_t<Ts...>, dx::zero_t, M, Ts...>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<Ts...> operator()(
        dx::zero_t zero, M mask, Ts... args) noexcept {
        using R = result_t<Ts...>;
        if constexpr (canonical_cpo_invocable_r<D, R, dx::zero_t, M, Ts...>) {
            if constexpr (all_same_abi<result_t<Ts...>, Ts...> &&
                !inherits_from<D, basic_operation_base<D>>) {
                if consteval {
                    if constexpr (fallback_cpo_invocable_r<D, R, dx::zero_t, M,
                                      Ts...>) {
                        return impl::fallback<D>(zero, mask, args...);
                    } else {
                        return fwd::select(mask, D::operator()(args...), zero);
                    }
                } else {
                    return impl::canonical<D>(zero, mask, args...);
                }
            } else {
                return impl::canonical<D>(zero, mask, args...);
            }
        } else if constexpr (fallback_cpo_invocable_r<D, R, dx::zero_t, M,
                                 Ts...>) {
            return impl::fallback<D>(zero, mask, args...);
        } else {
            return fwd::select(mask, D::operator()(args...), zero);
        }
    }

    template <simd_vector S, exact_mask_for<S> M, typename... Ts>
    requires signature_compatible<D, Ts...> &&
        extended_arguments<S, M, Ts...> && cpo_invocable<D, Ts...> &&
        equivalent_vector_with<S, result_t<Ts...>> &&
        (extendable_operation_r<D, result_t<Ts...>, S, M, Ts...> ||
            requires {
                requires simd_expression<S> || simd_expression<M> ||
                    (... || simd_expression<Ts>);
                operator()(internal::declarg<result_or_identity_t<S>>(),
                    internal::declarg<result_or_identity_t<M>>(),
                    internal::declarg<result_or_identity_t<Ts>>()...);
            } || fallback_cpo_invocable_r<D, result_t<Ts...>, S, M, Ts...>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, Ts&&... args) {
        using R DPL_NODEBUG = result_t<Ts...>;
        if constexpr (extended_cpo_invocable_r<D, R, S, M, Ts...>) {
            return impl::extended<D>(__DPL forward<S>(src),
                __DPL forward<M>(mask), __DPL forward<Ts>(args)...);
        } else if constexpr (extended_nttp_invocable_r<D, R, S, M, Ts...>) {
            constexpr D Op;
            return extended<Op>(__DPL forward<S>(src), __DPL forward<M>(mask),
                __DPL forward<Ts>(args)...);
        } else if constexpr (simd_expression<S> || simd_expression<M> ||
            (... || simd_expression<Ts>)) {
            return operator()(
                internal::forward_or_eval(__DPL forward<S>(src)),
                internal::forward_or_eval(__DPL forward<M>(mask)),
                internal::forward_or_eval(__DPL forward<Ts>(args))...);
        } else {
            return impl::fallback<D>(__DPL forward<S>(src),
                __DPL forward<M>(mask), __DPL forward<Ts>(args)...);
        }
    }

    template <simd_mask M, typename... Ts>
    requires signature_compatible<D, Ts...> && extended_arguments<M, Ts...> &&
        cpo_invocable<D, Ts...> && exact_mask_for<M, result_t<Ts...>> &&
        simd_vector<result_t<Ts...>> &&
        (extendable_operation_r<D, result_t<Ts...>, dx::zero_t, M, Ts...> ||
            requires {
                requires (simd_expression<M> || ... || simd_expression<Ts>);
                operator()(dx::zero,
                    internal::declarg<result_or_identity_t<M>>(),
                    internal::declarg<result_or_identity_t<Ts>>()...);
            } ||
            fallback_cpo_invocable_r<D, result_t<Ts...>, dx::zero_t, M, Ts...>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M&& mask, Ts&&... args) {
        using R DPL_NODEBUG = result_t<Ts...>;
        if constexpr (extended_cpo_invocable_r<D, R, dx::zero_t, M, Ts...>) {
            return impl::extended<D>(
                zero, __DPL forward<M>(mask), __DPL forward<Ts>(args)...);
        } else if constexpr (extended_nttp_invocable_r<D, R, dx::zero_t, M,
                                 Ts...>) {
            constexpr D Op;
            return extended<Op>(
                zero, __DPL forward<M>(mask), __DPL forward<Ts>(args)...);
        } else if constexpr ((simd_expression<M> || ... ||
                                 simd_expression<Ts>)) {
            return operator()(zero,
                internal::forward_or_eval(__DPL forward<M>(mask)),
                internal::forward_or_eval(__DPL forward<Ts>(args))...);
        } else {
            return impl::fallback<D>(
                zero, __DPL forward<M>(mask), __DPL forward<Ts>(args)...);
        }
    }

    template <simd_vector S, const_mask_for<S> M, typename... Ts>
    requires signature_compatible<D, Ts...> && extended_arguments<S, Ts...> &&
        cpo_invocable<D, Ts...> && equivalent_vector_with<S, result_t<Ts...>> &&
        (extendable_operation_r<D, result_t<Ts...>, S, launder_cmask_t<S, M>,
             Ts...> ||
            requires {
                requires (simd_expression<S> || ... || simd_expression<Ts>);
                operator()(internal::declarg<result_or_identity_t<S>>(),
                    internal::declarg<M>(),
                    internal::declarg<result_or_identity_t<Ts>>()...);
            } || fallback_cpo_invocable_r<D, result_t<Ts...>, S, M, Ts...>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M mask, Ts&&... args) {
        using R DPL_NODEBUG = result_t<Ts...>;
        if constexpr (extended_cpo_invocable_r<D, R, S, M, Ts...>) {
            return impl::extended<D>(
                __DPL forward<S>(src), mask, __DPL forward<Ts>(args)...);
        } else if constexpr (extended_nttp_invocable_r<D, R, S,
                                 launder_cmask_t<R, M>, Ts...>) {
            constexpr D Op;
            return extended<Op>(__DPL forward<S>(src),
                dx::to_const_mask<S>(mask), __DPL forward<Ts>(args)...);
        } else if constexpr ((simd_expression<S> || ... ||
                                 simd_expression<Ts>)) {
            return operator()( __DPL forward<S>(src), mask,
                internal::forward_or_eval(__DPL forward<Ts>(args))...);
        } else {
            return impl::fallback<D>(
                __DPL forward<S>(src), mask, __DPL forward<Ts>(args)...);
        }
    }

    template <typename M, typename... Ts>
    requires signature_compatible<D, Ts...> && extended_arguments<Ts...> &&
        cpo_invocable<D, Ts...> && const_mask_for<M, result_t<Ts...>> &&
        simd_vector<result_t<Ts...>> &&
        (extendable_operation_r<D, result_t<Ts...>, dx::zero_t,
             launder_cmask_t<result_t<Ts...>, M>, Ts...> ||
            requires {
                requires (... || simd_expression<Ts>);
                operator()(dx::zero, internal::declarg<M>(),
                    internal::declarg<result_or_identity_t<Ts>>()...);
            } ||
            fallback_cpo_invocable_r<D, result_t<Ts...>, dx::zero_t, M, Ts...>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M mask, Ts&&... args) {
        using R DPL_NODEBUG = result_t<Ts...>;
        if constexpr (extended_cpo_invocable_r<D, R, dx::zero_t, M, Ts...>) {
            return impl::extended<D>(zero, mask, __DPL forward<Ts>(args)...);
        } else if constexpr (extended_nttp_invocable_r<D, R, dx::zero_t,
                                 launder_cmask_t<R, M>, Ts...>) {
            constexpr D Op;
            return extended<Op>(
                zero, dx::to_const_mask<R>(mask), __DPL forward<Ts>(args)...);
        } else if constexpr ((... || simd_expression<Ts>)) {
            return operator()(zero, mask,
                internal::forward_or_eval(__DPL forward<Ts>(args))...);
        } else {
            return impl::fallback<D>(zero, mask, __DPL forward<Ts>(args)...);
        }
    }

    template <typename M, typename... Ts>
    requires signature_compatible<D, Ts...> && cpo_invocable<D, Ts...> &&
        simd_vector<result_t<Ts...>> &&
        (exact_mask_for<M, result_t<Ts...>> ||
            const_mask_for<remove_cvref_t<M>, result_t<Ts...>>) &&
        requires {
            operator()(
                dx::zero, internal::declarg<M>(), internal::declarg<Ts>()...);
        }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M&& mask, Ts&&... args) noexcept(
        (canonical_ornot_simd<remove_cvref_t<M>> && ... &&
            canonical_ornot_simd<remove_cvref_t<Ts>>)) {
        return operator()(
            dx::zero, __DPL forward<M>(mask), __DPL forward<Ts>(args)...);
    }
};
} // namespace datapar::internal

__DPL_DEFAULT_NAMESPACE_END
