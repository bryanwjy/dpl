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
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/concepts/mask_compatibility.h"
#  include "dpl/core/concepts/simd_vector.h"
#  include "dpl/core/immediate/const_mask.h"
#  include "dpl/core/immediate/constants/zero.h"
#  include "dpl/core/type_traits/details/declarg.h"
#  include "dpl/core/type_traits/simd_abi_type.h"
#  include "dpl/core/type_traits/simd_element_type.h"
#  include "dpl/std/utility/forward.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::internal {

/**
 * @brief A maskable operation that accumulates into a source operand.
 *
 * `src` is both an operand in the computation and the merge target
 * for inactive lanes. The result type matches `src`.
 *
 * @par Signatures
 * @code
 * op(src, args...)        // unmasked
 * op(src, mask, args...)  // masked
 * op(mask, src, args...)  // zeroing masked
 * @endcode
 *
 * @see maskable_transform_base
 * @see maskable_predicate_base
 */
template <typename D>
class maskable_accumulation_base : public maskable_operation_base<D> {
    template <typename S, typename... Ts>
    using result_t DPL_NODEBUG = cpo_result_t<D, S, Ts...>;
    template <typename S, typename... Ts>
    using basic_mask_t DPL_NODEBUG =
        basic_mask<simd_element_type_t<result_t<S, Ts...>>,
            simd_abi_type_t<result_t<S, Ts...>>>;
    using this_type = maskable_accumulation_base;

protected:
    template <canonical_vector S, canonical_ornot_simd... Ts>
    requires signature_compatible<D, S, Ts...> && cpo_invocable<D, S, Ts...> &&
        canonical_vector<result_t<S, Ts...>> &&
        same_as<S, result_t<S, Ts...>> &&
        cpo_invocable<select_t, basic_mask_t<S, Ts...>, result_t<S, Ts...>, S>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<S, Ts...> operator()(
        S src, basic_mask_t<S, Ts...> mask, Ts... args) noexcept {
        using M = basic_mask_t<S, Ts...>;
        if constexpr (canonical_cpo_invocable_r<D, S, S, M, Ts...>) {
            if constexpr (all_same_abi<S, M, Ts...>) {
                if consteval {
                    if constexpr (fallback_cpo_invocable_r<D, S, S, M, Ts...>) {
                        return impl::fallback<D>(src, mask, args...);
                    } else {
                        return fwd::select(
                            mask, D::operator()(src, args...), src);
                    }
                } else {
                    return impl::canonical<D>(src, mask, args...);
                }
            } else {
                return impl::canonical<D>(src, mask, args...);
            }
        } else if constexpr (fallback_cpo_invocable_r<D, S, S, M, Ts...>) {
            return impl::fallback<D>(src, mask, args...);
        } else {
            return fwd::select(mask, D::operator()(src, args...), src);
        }
    }

    template <canonical_vector S, canonical_ornot_simd... Ts>
    requires signature_compatible<D, S, Ts...> && cpo_invocable<D, S, Ts...> &&
        canonical_vector<result_t<S, Ts...>> &&
        same_as<S, result_t<S, Ts...>> &&
        cpo_invocable<select_t, basic_mask_t<S, Ts...>, result_t<S, Ts...>,
            zero_t>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<S, Ts...> operator()(dx::zero_t zero,
        basic_mask_t<S, Ts...> mask, S src, Ts... args) noexcept {
        using M = basic_mask_t<S, Ts...>;
        if constexpr (canonical_cpo_invocable_r<D, S, dx::zero_t, M, S,
                          Ts...>) {
            if constexpr (all_same_abi<M, S, Ts...>) {
                if consteval {
                    if constexpr (fallback_cpo_invocable_r<D, S, dx::zero_t, M,
                                      S, Ts...>) {
                        return impl::fallback<D>(zero, mask, src, args...);
                    } else {
                        return fwd::select(
                            mask, D::operator()(src, args...), zero);
                    }
                } else {
                    return impl::canonical<D>(zero, mask, src, args...);
                }
            } else {
                return impl::canonical<D>(zero, mask, src, args...);
            }
        } else if constexpr (fallback_cpo_invocable_r<D, S, dx::zero_t, M, S,
                                 Ts...>) {
            return impl::fallback<D>(zero, mask, src, args...);
        } else {
            return fwd::select(mask, D::operator()(src, args...), zero);
        }
    }

    template <canonical_vector S, typename M, canonical_ornot_simd... Ts>
    requires signature_compatible<D, S, Ts...> && cpo_invocable<D, S, Ts...> &&
        canonical_vector<result_t<S, Ts...>> &&
        same_as<S, result_t<S, Ts...>> &&
        const_mask_for<M, result_t<S, Ts...>> &&
        cpo_invocable<select_t, M, result_t<S, Ts...>, S>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<S, Ts...> operator()(
        S src, M mask, Ts... args) noexcept {
        using R = result_t<S, Ts...>;
        if constexpr (canonical_cpo_invocable_r<D, S, M, Ts...>) {
            if constexpr (all_same_abi<S, Ts...>) {
                if consteval {
                    if constexpr (fallback_cpo_invocable_r<D, S, M, Ts...>) {
                        return impl::fallback<D>(src, mask, args...);
                    } else {
                        return fwd::select(
                            mask, D::operator()(src, args...), src);
                    }
                } else {
                    return impl::canonical<D>(src, mask, args...);
                }
            } else {
                return impl::canonical<D>(src, mask, args...);
            }
        } else if constexpr (fallback_cpo_invocable_r<D, S, M, Ts...>) {
            return impl::fallback<D>(src, mask, args...);
        } else {
            return fwd::select(mask, D::operator()(src, args...), src);
        }
    }

    template <canonical_vector S, typename M, canonical_ornot_simd... Ts>
    requires signature_compatible<D, S, Ts...> && cpo_invocable<D, S, Ts...> &&
        canonical_vector<result_t<S, Ts...>> &&
        equivalent_vector_with<S, result_t<S, Ts...>> &&
        const_mask_for<M, result_t<S, Ts...>> &&
        cpo_invocable<select_t, M, result_t<S, Ts...>, dx::zero_t>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<S, Ts...> operator()(
        dx::zero_t zero, M mask, S src, Ts... args) noexcept {
        using R = result_t<S, Ts...>;
        if constexpr (canonical_cpo_invocable_r<D, dx::zero_t, M, S, Ts...>) {
            if constexpr (all_same_abi<S, Ts...>) {
                if consteval {
                    if constexpr (fallback_cpo_invocable_r<D, dx::zero_t, M, S,
                                      Ts...>) {
                        return impl::fallback<D>(zero, mask, src, args...);
                    } else {
                        return fwd::select(
                            mask, D::operator()(src, args...), zero);
                    }
                } else {
                    return impl::canonical<D>(zero, mask, src, args...);
                }
            } else {
                return impl::canonical<D>(zero, mask, src, args...);
            }
        } else if constexpr (fallback_cpo_invocable_r<D, dx::zero_t, M, S,
                                 Ts...>) {
            return impl::fallback<D>(zero, mask, src, args...);
        } else {
            return fwd::select(mask, D::operator()(src, args...), zero);
        }
    }

    template <simd_vector S, exact_mask_for<S> M, typename... Ts,
        simd_vector R = remove_cvref_t<S>>
    requires signature_compatible<D, S, Ts...> &&
        extended_arguments<S, M, Ts...> && cpo_invocable<D, S, Ts...> &&
        equivalent_vector_with<R, result_t<S, Ts...>> &&
        (extendable_operation_r<D, R, S, M, Ts...> ||
            requires {
                requires simd_expression<S> ||
                    (simd_expression<M> || ... || simd_expression<Ts>);
                // protected
                operator()(internal::declarg<result_or_identity_t<S>>(),
                    internal::declarg<result_or_identity_t<M>>(),
                    internal::declarg<result_or_identity_t<Ts>>()...);
            } || fallback_cpo_invocable_r<D, R, S, M, Ts...>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, Ts&&... args) {
        if constexpr (extended_cpo_invocable_r<D, R, S, M, Ts...>) {
            return impl::extended<D>(__DPL forward<S>(src),
                __DPL forward<M>(mask), __DPL forward<Ts>(args)...);
        } else if constexpr (extended_nttp_invocable_r<D, R, S, M, Ts...>) {
            constexpr D Op;
            return extended<Op>(__DPL forward<S>(src), __DPL forward<M>(mask),
                __DPL forward<Ts>(args)...);
        } else if constexpr (simd_expression<S> ||
            (simd_expression<M> || ... || simd_expression<Ts>)) {
            return operator()(
                internal::forward_or_eval(__DPL forward<S>(src)),
                internal::forward_or_eval(__DPL forward<M>(mask)),
                internal::forward_or_eval(__DPL forward<Ts>(args))...);
        } else {
            return impl::fallback<D>(__DPL forward<S>(src),
                __DPL forward<M>(mask), __DPL forward<Ts>(args)...);
        }
    }

    template <simd_vector S, exact_mask_for<S> M, typename... Ts,
        simd_vector R = remove_cvref_t<S>>
    requires signature_compatible<D, S, Ts...> &&
        extended_arguments<M, S, Ts...> && cpo_invocable<D, S, Ts...> &&
        equivalent_vector_with<S, result_t<S, Ts...>> &&
        (extendable_operation_r<D, R, dx::zero_t, M, S, Ts...> ||
            requires {
                requires simd_expression<M> ||
                    (simd_expression<S> || ... || simd_expression<Ts>);
                // protected
                operator()(dx::zero,
                    internal::declarg<result_or_identity_t<M>>(),
                    internal::declarg<result_or_identity_t<S>>(),
                    internal::declarg<result_or_identity_t<Ts>>()...);
            } || fallback_cpo_invocable_r<D, R, dx::zero_t, M, S, Ts...>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, S&& src, Ts&&... args) {
        if constexpr (extended_cpo_invocable_r<D, R, dx::zero_t, M, S, Ts...>) {
            return impl::extended<D>(zero, __DPL forward<M>(mask),
                __DPL forward<S>(src), __DPL forward<Ts>(args)...);
        } else if constexpr (extended_nttp_invocable_r<D, R, dx::zero_t, M, S,
                                 Ts...>) {
            constexpr D Op;
            return extended<Op>(zero, __DPL forward<M>(mask),
                __DPL forward<S>(src), __DPL forward<Ts>(args)...);
        } else if constexpr (simd_expression<M> ||
            (simd_expression<S> || ... || simd_expression<Ts>)) {
            return operator()(zero,
                internal::forward_or_eval(__DPL forward<M>(mask)),
                internal::forward_or_eval(__DPL forward<S>(src)),
                internal::forward_or_eval(__DPL forward<Ts>(args))...);
        } else {
            return impl::fallback<D>(zero, __DPL forward<M>(mask),
                __DPL forward<S>(src), __DPL forward<Ts>(args)...);
        }
    }

    template <simd_vector S, const_mask_for<S> M, typename... Ts,
        simd_vector R = remove_cvref_t<S>>
    requires signature_compatible<D, S, Ts...> &&
        extended_arguments<S, Ts...> && cpo_invocable<D, S, Ts...> &&
        equivalent_vector_with<S, result_t<S, Ts...>> &&
        (extendable_operation_r<D, R, S, launder_cmask_t<R, M>, Ts...> ||
            requires {
                requires (simd_expression<S> || ... || simd_expression<Ts>);
                // protected
                operator()(internal::declarg<result_or_identity_t<S>>(),
                    internal::declarg<M>(),
                    internal::declarg<result_or_identity_t<Ts>>()...);
            } || fallback_cpo_invocable_r<D, R, S, M, Ts...>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M mask, Ts&&... args) {
        if constexpr (extended_cpo_invocable_r<D, R, S, M, Ts...>) {
            return impl::extended<D>(
                __DPL forward<S>(src), mask, __DPL forward<Ts>(args)...);
        } else if constexpr (extended_nttp_invocable_r<D, R, S,
                                 launder_cmask_t<R, M>, Ts...>) {
            constexpr D Op;
            return extended<Op>(__DPL forward<S>(src),
                dx::to_const_mask<R>(mask), __DPL forward<Ts>(args)...);
        } else if constexpr ((simd_expression<S> || ... ||
                                 simd_expression<Ts>)) {
            return operator()(
                internal::forward_or_eval(__DPL forward<S>(src)), mask,
                internal::forward_or_eval(__DPL forward<Ts>(args))...);
        } else {
            return impl::fallback<D>(
                __DPL forward<S>(src), mask, __DPL forward<Ts>(args)...);
        }
    }

    template <simd_vector S, const_mask_for<S> M, typename... Ts,
        simd_vector R = remove_cvref_t<S>>
    requires signature_compatible<D, S, Ts...> &&
        extended_arguments<S, Ts...> && cpo_invocable<D, S, Ts...> &&
        equivalent_vector_with<S, result_t<S, Ts...>> &&
        (extendable_operation_r<D, R, dx::zero_t, launder_cmask_t<R, M>, S,
             Ts...> ||
            requires {
                requires (simd_expression<S> || ... || simd_expression<Ts>);
                // protected
                operator()(dx::zero, internal::declarg<M>(),
                    internal::declarg<result_or_identity_t<S>>(),
                    internal::declarg<result_or_identity_t<Ts>>()...);
            } || fallback_cpo_invocable_r<D, R, dx::zero_t, M, S, Ts...>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M mask, S&& src, Ts&&... args) {
        if constexpr (extended_cpo_invocable_r<D, R, dx::zero_t, M, S, Ts...>) {
            return impl::extended<D>(zero, mask, __DPL forward<S>(src),
                __DPL forward<Ts>(args)...);
        } else if constexpr (extended_nttp_invocable_r<D, R, dx::zero_t,
                                 launder_cmask_t<R, M>, S, Ts...>) {
            constexpr D Op;
            return extended<Op>(zero, dx::to_const_mask<R>(mask),
                __DPL forward<S>(src), __DPL forward<Ts>(args)...);
        } else if constexpr ((simd_expression<S> || ... ||
                                 simd_expression<Ts>)) {
            return operator()(zero, mask,
                internal::forward_or_eval(__DPL forward<S>(src)),
                internal::forward_or_eval(__DPL forward<Ts>(args))...);
        } else {
            return impl::fallback<D>(zero, mask, __DPL forward<S>(src),
                __DPL forward<Ts>(args)...);
        }
    }

    template <typename M, simd_vector S, typename... Ts>
    requires signature_compatible<D, S, Ts...> && cpo_invocable<D, S, Ts...> &&
        (exact_mask_for<M, S> || const_mask_for<M, S>) && requires {
            operator()(dx::zero, internal::declarg<M>(), internal::declarg<S>(),
                internal::declarg<Ts>()...);
        }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M&& mask, S&& src, Ts&&... args) noexcept(
        (canonical_ornot_simd<M> &&
            (canonical_ornot_simd<S> && ... && canonical_ornot_simd<Ts>))) {
        return operator()(dx::zero, __DPL forward<M>(mask),
            __DPL forward<S>(src), __DPL forward<Ts>(args)...);
    }
};

} // namespace datapar::internal

DPL_DEFAULT_NAMESPACE_END
