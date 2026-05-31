// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/internal/fwd.h" // IWYU pragma: export

#include "dpl/core/math/fma.h"
#include "dpl/core/math/internal/accuracy.h" // IWYU pragma: export
#include "dpl/core/math/internal/floating_point_simd.h"
#include "dpl/core/math/internal/polynomial.h"

#if !DPL_MODULES
#  include "dpl/core/basic/basic_vector.h" // IWYU pragma: keep
#  include "dpl/core/concepts/decayable.h"
#  include "dpl/core/concepts/operation_category.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/constants/one.h"
#  include "dpl/core/operations/arithmetic.h"
#  include "dpl/core/operations/masked.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::fmath {

template <typename S, typename M, typename... Args>
concept rt_masked_overload =
    compatible_mask_with<M, S> && dx::internal::maskable_args<S, M, Args...>;

template <typename S, typename M, typename... Args>
concept imm_masked_overload =
    const_mask_for<M, S> && dx::internal::imm_maskable_args<S, Args...>;

template <typename S, typename M, typename... Args>
concept canonical_operator_args =
    canonical_vector<S> && (const_mask_for<M, S> || canonical_mask<M>) &&
    (... && (!simd_class<Args> || canonical_class<Args>));

template <typename Op, typename S, typename M, typename... Args>
concept maskable_operator = regular_invocable<Op, Args...> && simd_vector<S> &&
    (rt_masked_overload<S, M, Args...> || imm_masked_overload<S, M, Args...>);

///
template <typename Op, typename M, typename... Args>
concept rt_zmasked_overload =
    compatible_mask_with<M, invoke_result_t<Op, Args...>> &&
    dx::internal::zmaskable_args<M, Args...>;

template <typename Op, typename M, typename... Args>
concept imm_zmasked_overload =
    compatible_mask_with<M, invoke_result_t<Op, Args...>> &&
    dx::internal::imm_zmaskable_args<Args...>;

template <typename Op, typename M, typename... Args>
concept canonical_zoperator_args =
    (const_mask_for<M, invoke_result_t<Op, Args...>> || canonical_mask<M>) &&
    (... && (!simd_class<Args> || canonical_class<Args>));

template <typename Op, typename M, typename... Args>
concept maskable_zoperator = regular_invocable<Op, Args...> &&
    (rt_zmasked_overload<Op, M, Args...> ||
        imm_zmasked_overload<Op, M, Args...>);

template <typename Op, typename M, typename... Args>
concept canonical_predicate_args =
    (const_mask_for<M, invoke_result_t<Op, Args...>> || canonical_mask<M>) &&
    (... && (!simd_class<Args> || canonical_class<Args>));

template <typename Op, typename M, typename... Args>
concept maskable_predicate =
    regular_invocable<Op, Args...> && simd_mask<invoke_result_t<Op, Args...>> &&
    (rt_zmasked_overload<Op, M, Args...> ||
        imm_zmasked_overload<Op, M, Args...>);

template <typename D>
struct masked_operation;

template <typename T>
struct decay_type {
    using type DPL_NODEBUG = T;
};
template <typename T>
using decay_type_t DPL_NODEBUG = typename decay_type<T>::type;

template <typename T>
requires requires { typename canonical_type_t<T>; }
struct decay_type<T> : canonical_type<T> {};

template <typename Op, typename... Args>
concept masked_invocable =
    requires(masked_operation<Op> const op, Args... args) { op(args...); };

template <typename Op, typename S, typename M, typename... Args>
concept decayable_masked_operator =
    decayable_vector_for<S, operation_category::lane_agnostic> &&
    (!simd_mask<M> ||
        decayable_mask_for<M, operation_category::lane_agnostic>) &&
    (... &&
        (!simd_class<Args> ||
            decayable_mask_for<Args, operation_category::lane_agnostic>)) &&
    masked_invocable<Op, decay_type_t<S>, decay_type_t<M>,
        decay_type_t<Args>...>;

template <typename Op, typename M, typename... Args>
concept decayable_masked_zoperator =
    (!simd_mask<M> ||
        decayable_mask_for<M, operation_category::lane_agnostic>) &&
    (... &&
        (!simd_class<Args> ||
            decayable_mask_for<Args, operation_category::lane_agnostic>)) &&
    masked_invocable<Op, decay_type_t<M>, decay_type_t<Args>...>;

template <typename Op, typename M, typename... Args>
concept decayable_masked_predicate =
    (!simd_mask<M> ||
        decayable_mask_for<M, operation_category::lane_agnostic>) &&
    (... &&
        (!simd_class<Args> ||
            decayable_mask_for<Args, operation_category::lane_agnostic>)) &&
    masked_invocable<Op, decay_type_t<M>, decay_type_t<Args>...>;

template <typename T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr decltype(auto) decay(T&& arg) noexcept {
    return __DPL forward<T>(arg);
}

template <typename T>
requires simd_class<remove_cvref_t<T>>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr decltype(auto) decay(T&& arg) noexcept {
    return dx::to_canonical(__DPL forward<T>(arg));
}

template <typename D>
struct masked_operation {
public:
    template <simd_vector S, typename M, typename... Args>
    requires maskable_operator<D, S, M, Args...> &&
        canonical_operator_args<S, M, Args...>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, Args... args) noexcept {
        if constexpr (requires { D::masked(src, mask, args...); }) {
            if consteval {
                return internal::masked<D>(src, mask, args...);
            } else {
                return D::masked(src, mask, args...);
            }
        } else {
            return internal::masked<D>(src, mask, args...);
        }
    }

    template <simd_vector S, typename M, typename... Args>
    requires maskable_operator<D, S, M, Args...> &&
        (!canonical_operator_args<S, M, Args...>) &&
        (requires(S src, M mask, Args... args) {
            D::masked(src, mask, args...);
        } || decayable_masked_operator<D, S, M, Args...>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, Args... args) noexcept {
        if constexpr (requires { D::masked(src, mask, args...); }) {
            return D::masked(src, mask, args...);
        } else {
            return operator()(
                fmath::decay(src), fmath::decay(mask), fmath::decay(args)...);
        }
    }

    template <typename M, typename... Args>
    requires maskable_zoperator<D, M, Args...> &&
        canonical_zoperator_args<D, M, Args...>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, Args... args) noexcept {
        if constexpr (requires { D::masked(mask, args...); }) {
            if consteval {
                return internal::masked<D>(mask, args...);
            } else {
                return D::masked(mask, args...);
            }
        } else {
            return internal::masked<D>(mask, args...);
        }
    }

    template <typename M, typename... Args>
    requires maskable_zoperator<D, M, Args...> &&
        (!canonical_zoperator_args<D, M, Args...>) &&
        (requires(M mask, Args... args) { D::masked(mask, args...); } ||
            decayable_masked_zoperator<D, M, Args...>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, Args... args) noexcept {
        if constexpr (requires { D::masked(mask, args...); }) {
            return D::masked(mask, args...);
        } else {
            return operator()(fmath::decay(mask), fmath::decay(args)...);
        }
    }

    template <typename M, typename... Args>
    requires maskable_zoperator<D, M, Args...> &&
        requires(M mask, Args... args) { operator()(mask, args...); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t, M mask, Args... args) noexcept {
        return operator()(mask, args...);
    }
};

template <typename D>
struct masked_predicate {

public:
    template <typename M, typename... Args>
    requires maskable_predicate<D, M, Args...> &&
        canonical_predicate_args<D, M, Args...>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, Args... args) noexcept {
        if constexpr (requires { D::masked(mask, args...); }) {
            if consteval {
                return internal::masked<D>(mask, args...);
            } else {
                return D::masked(mask, args...);
            }
        } else {
            return internal::masked<D>(mask, args...);
        }
    }

    template <typename M, typename... Args>
    requires maskable_predicate<D, M, Args...> &&
        (!canonical_predicate_args<D, M, Args...>) &&
        (requires(M mask, Args... args) { D::masked(mask, args...); } ||
            decayable_masked_predicate<D, M, Args...>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, Args... args) noexcept {
        if constexpr (requires { D::masked(mask, args...); }) {
            return D::masked(mask, args...);
        } else {
            return operator()(fmath::decay(mask), fmath::decay(args)...);
        }
    }
};

} // namespace datapar::fmath

DPL_DEFAULT_NAMESPACE_END
