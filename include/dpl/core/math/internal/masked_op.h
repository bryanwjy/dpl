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
#  include "dpl/core/operations/evaluate.h"
#  include "dpl/core/operations/internal/masked.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::fmath {

using internal::canonical_if_zero_t;

template <typename S, typename M, typename... Args>
concept rt_masked_overload = dx::internal::maskable_args<S, M, Args...>;

template <typename S, typename M, typename... Args>
concept imm_masked_overload = dx::internal::imm_maskable_args<S, Args...>;

template <typename Op, typename S, typename M, typename... Args>
concept canonical_operator_args = canonical_vector<S> &&
    (const_mask_for<M, canonical_if_zero_t<S, invoke_result_t<Op, Args...>>> ||
        canonical_mask<M>) &&
    (... && (!simd_class<Args> || canonical_class<Args>));

template <typename Op, typename S, typename M, typename... Args>
concept maskable_operator = regular_invocable<Op, Args...> && simd_vector<S> &&
    (rt_masked_overload<S, M, Args...> || imm_masked_overload<S, M, Args...>);

template <typename Op, typename S, typename M, typename... Args>
concept extended_masked_math_operator = maskable_operator<Op, S, M, Args...> &&
    !canonical_operator_args<Op, S, M, Args...>;

template <typename Op, typename S, typename M, typename... Args>
concept canonical_masked_math_operator = maskable_operator<Op, S, M, Args...> &&
    canonical_operator_args<Op, S, M, Args...>;

///
template <typename M, typename... Args>
concept rt_zmasked_overload = dx::internal::zmaskable_args<M, Args...>;

template <typename M, typename... Args>
concept imm_zmasked_overload = dx::internal::imm_zmaskable_args<Args...>;

template <typename Op, typename M, typename... Args>
concept canonical_zoperator_args =
    (const_mask_for<M, invoke_result_t<Op, Args...>> || canonical_mask<M>) &&
    (... && (!simd_class<Args> || canonical_class<Args>));

template <typename Op, typename M, typename... Args>
concept maskable_zoperator = regular_invocable<Op, Args...> &&
    (rt_zmasked_overload<M, Args...> || imm_zmasked_overload<M, Args...>);

template <typename Op, typename M, typename... Args>
concept extended_masked_math_zoperator = maskable_zoperator<Op, M, Args...> &&
    !canonical_zoperator_args<Op, M, Args...>;

template <typename Op, typename M, typename... Args>
concept canonical_masked_math_zoperator = maskable_zoperator<Op, M, Args...> &&
    canonical_zoperator_args<Op, M, Args...>;

///
template <typename Op, typename S, typename M, typename... Args>
concept maskable_assignment =
    regular_invocable<Op, S, Args...> && simd_vector<S> &&
    (rt_masked_overload<S, M, Args...> || imm_masked_overload<S, M, Args...>);
template <typename Op, typename S, typename M, typename... Args>
concept canonical_assignment_args = canonical_vector<S> &&
    (const_mask_for<M, invoke_result_t<Op, S, Args...>> || canonical_mask<M>) &&
    (... && (!simd_class<Args> || canonical_class<Args>));

template <typename Op, typename S, typename M, typename... Args>
concept extended_masked_math_assignment =
    maskable_assignment<Op, S, M, Args...> &&
    !canonical_assignment_args<Op, S, M, Args...>;

template <typename Op, typename S, typename M, typename... Args>
concept canonical_masked_math_assignment =
    maskable_assignment<Op, S, M, Args...> &&
    canonical_assignment_args<Op, S, M, Args...>;

template <typename Op, typename M, typename S, typename... Args>
concept extended_masked_math_zassignment =
    maskable_zoperator<Op, M, S, Args...> &&
    !canonical_zoperator_args<Op, M, S, Args...>;

template <typename Op, typename M, typename S, typename... Args>
concept canonical_masked_math_zassignment =
    maskable_zoperator<Op, M, S, Args...> &&
    canonical_zoperator_args<Op, M, S, Args...>;

///
template <typename Op, typename M, typename... Args>
concept canonical_predicate_args =
    (const_mask_for<M, invoke_result_t<Op, Args...>> || canonical_mask<M>) &&
    (... && (!simd_class<Args> || canonical_class<Args>));

template <typename Op, typename M, typename... Args>
concept maskable_predicate =
    regular_invocable<Op, Args...> && simd_mask<invoke_result_t<Op, Args...>> &&
    (rt_zmasked_overload<M, Args...> || imm_zmasked_overload<M, Args...>);

template <typename Op, typename M, typename... Args>
concept extended_masked_math_predicate = maskable_predicate<Op, M, Args...> &&
    !canonical_predicate_args<Op, M, Args...>;

template <typename Op, typename M, typename... Args>
concept canonical_masked_math_predicate = maskable_predicate<Op, M, Args...> &&
    canonical_predicate_args<Op, M, Args...>;

template <typename D>
class masked_operation;
template <typename D>
class masked_assignment;
template <typename D>
class masked_predicate;

template <typename T>
struct decay_type {
    using type DPL_NODEBUG = T;
};
template <typename T>
using decay_type_t DPL_NODEBUG = typename decay_type<T>::type;

template <typename T>
requires requires { typename canonical_type_t<T>; }
struct decay_type<T> : canonical_type<T> {};

template <typename T>
struct eval_type {
    using type DPL_NODEBUG = T;
};
template <typename T>
using eval_type_t DPL_NODEBUG = typename eval_type<T>::type;

template <typename T>
requires requires { typename simd_expression_result_t<T>; }
struct eval_type<T> : simd_expression_result<T> {};

template <typename Op, typename... Args>
concept masked_op_invocable = invocable<masked_operation<Op>, Args...>;
template <typename Op, typename... Args>
concept masked_ass_invocable = invocable<masked_assignment<Op>, Args...>;
template <typename Op, typename... Args>
concept masked_pred_invocable = invocable<masked_predicate<Op>, Args...>;

template <typename Op, typename S, typename M, typename... Args>
concept evaluatable_masked_operator =
    (simd_expression<S> || simd_expression<M> ||
        (... || simd_expression<Args>)) &&
    masked_op_invocable<Op, eval_type_t<S>, eval_type_t<M>,
        eval_type_t<Args>...>;

template <typename Op, typename S, typename M, typename... Args>
concept decayable_masked_operator =
    decayable_vector_for<S, operation_category::lane_agnostic> &&
    (!simd_mask<M> ||
        decayable_mask_for<M, operation_category::lane_agnostic>) &&
    (... &&
        (!simd_class<Args> ||
            decayable_mask_for<Args, operation_category::lane_agnostic>)) &&
    masked_op_invocable<Op, decay_type_t<S>, decay_type_t<M>,
        decay_type_t<Args>...>;

template <typename Op, typename M, typename... Args>
concept evaluatable_masked_zoperator =
    (simd_expression<M> || ... || simd_expression<Args>) &&
    masked_op_invocable<Op, eval_type_t<M>, eval_type_t<Args>...>;

template <typename Op, typename M, typename... Args>
concept decayable_masked_zoperator =
    (!simd_mask<M> ||
        decayable_mask_for<M, operation_category::lane_agnostic>) &&
    (... &&
        (!simd_class<Args> ||
            decayable_mask_for<Args, operation_category::lane_agnostic>)) &&
    masked_op_invocable<Op, decay_type_t<M>, decay_type_t<Args>...>;

template <typename Op, typename M, typename... Args>
concept evaluatable_masked_predicate =
    (simd_expression<M> || ... || simd_expression<Args>) &&
    masked_pred_invocable<Op, eval_type_t<M>, eval_type_t<Args>...>;

template <typename Op, typename M, typename... Args>
concept decayable_masked_predicate =
    (!simd_mask<M> ||
        decayable_mask_for<M, operation_category::lane_agnostic>) &&
    (... &&
        (!simd_class<Args> ||
            decayable_mask_for<Args, operation_category::lane_agnostic>)) &&
    masked_pred_invocable<Op, decay_type_t<M>, decay_type_t<Args>...>;

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

template <typename T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr decltype(auto) eval(T&& arg) noexcept {
    return __DPL forward<T>(arg);
}

template <typename T>
requires simd_class<remove_cvref_t<T>>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr decltype(auto) eval(T&& arg) noexcept {
    return dx::evaluate(__DPL forward<T>(arg));
}

template <typename D>
class masked_operation {
    template <typename... Args>
    static constexpr bool has_masked_overload =
        requires(Args... args) { D::masked(args...); };

public:
    template <simd_vector S, typename M, typename... Args>
    requires canonical_masked_math_operator<D, S, M, Args...>
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
    requires extended_masked_math_operator<D, S, M, Args...> &&
        (has_masked_overload<S, M, Args...> ||
            evaluatable_masked_operator<D, S, M, Args...> ||
            decayable_masked_operator<D, S, M, Args...>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, Args... args) noexcept {
        if constexpr (has_masked_overload<S, M, Args...>) {
            return D::masked(src, mask, args...);
        } else if constexpr (evaluatable_masked_operator<D, S, M, Args...>) {
            return operator()(
                fmath::eval(src), fmath::eval(mask), fmath::eval(args)...);
        } else {
            return operator()(
                fmath::decay(src), fmath::decay(mask), fmath::decay(args)...);
        }
    }

    template <typename M, typename... Args>
    requires canonical_masked_math_zoperator<D, M, Args...>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, Args... args) noexcept {
        if constexpr (has_masked_overload<M, Args...>) {
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
    requires extended_masked_math_zoperator<D, M, Args...> &&
        (has_masked_overload<M, Args...> ||
            evaluatable_masked_zoperator<D, M, Args...> ||
            decayable_masked_zoperator<D, M, Args...>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, Args... args) noexcept {
        if constexpr (has_masked_overload<M, Args...>) {
            return D::masked(mask, args...);
        } else if constexpr (evaluatable_masked_zoperator<D, M, Args...>) {
            return operator()(fmath::eval(mask), fmath::eval(args)...);
        } else {
            return operator()(fmath::decay(mask), fmath::decay(args)...);
        }
    }

    template <typename M, typename... Args>
    requires maskable_zoperator<D, M, Args...> &&
        masked_op_invocable<D, M, Args...>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t, M mask, Args... args) noexcept {
        return operator()(mask, args...);
    }
};

template <typename D>
class masked_assignment {
    template <typename... Args>
    static constexpr bool has_masked_overload =
        requires(Args... args) { D::masked(args...); };

public:
    template <simd_vector S, typename M, typename... Args>
    requires canonical_masked_math_assignment<D, S, M, Args...>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, Args... args) noexcept {
        if constexpr (requires { D::masked(src, mask, args...); }) {
            if consteval {
                return internal::masked<D>(src, mask, src, args...);
            } else {
                return D::masked(src, mask, args...);
            }
        } else {
            return internal::masked<D>(src, mask, src, args...);
        }
    }

    template <simd_vector S, typename M, typename... Args>
    requires extended_masked_math_assignment<D, S, M, Args...> &&
        (has_masked_overload<S, M, Args...> ||
            evaluatable_masked_operator<D, S, M, Args...> ||
            decayable_masked_operator<D, S, M, Args...>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, Args... args) noexcept {
        if constexpr (has_masked_overload<S, M, Args...>) {
            return D::masked(src, mask, args...);
        } else if constexpr (evaluatable_masked_operator<D, S, M, Args...>) {
            return operator()(
                fmath::eval(src), fmath::eval(mask), fmath::eval(args)...);
        } else {
            return operator()(
                fmath::decay(src), fmath::decay(mask), fmath::decay(args)...);
        }
    }

    template <typename M, typename S, typename... Args>
    requires canonical_masked_math_zassignment<D, M, S, Args...>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, S src, Args... args) noexcept {
        if constexpr (has_masked_overload<M, S, Args...>) {
            if consteval {
                return internal::masked<D>(mask, src, args...);
            } else {
                return D::masked(mask, src, args...);
            }
        } else {
            return internal::masked<D>(mask, src, args...);
        }
    }

    template <typename M, typename S, typename... Args>
    requires extended_masked_math_zassignment<D, M, S, Args...> &&
        (has_masked_overload<M, S, Args...> ||
            evaluatable_masked_zoperator<D, M, S, Args...> ||
            decayable_masked_zoperator<D, M, S, Args...>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, S src, Args... args) noexcept {
        if constexpr (has_masked_overload<M, S, Args...>) {
            return D::masked(mask, src, args...);
        } else if constexpr (evaluatable_masked_zoperator<D, M, S, Args...>) {
            return operator()(
                fmath::eval(mask), fmath::eval(src), fmath::eval(args)...);
        } else {
            return operator()(
                fmath::decay(mask), fmath::decay(src), fmath::decay(args)...);
        }
    }

    template <typename M, typename S, typename... Args>
    requires maskable_zoperator<D, M, S, Args...> &&
        masked_ass_invocable<D, M, S, Args...>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t, M mask, S src, Args... args) noexcept {
        return operator()(mask, src, args...);
    }
};

template <typename D>
class masked_predicate {
    template <typename... Args>
    static constexpr bool has_masked_overload =
        requires(Args... args) { D::masked(args...); };

public:
    template <typename M, typename... Args>
    requires canonical_masked_math_predicate<D, M, Args...>
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
    requires extended_masked_math_predicate<D, M, Args...> &&
        (has_masked_overload<M, Args...> ||
            evaluatable_masked_predicate<D, M, Args...> ||
            decayable_masked_predicate<D, M, Args...>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, Args... args) noexcept {
        if constexpr (has_masked_overload<M, Args...>) {
            return D::masked(mask, args...);
        } else if constexpr (evaluatable_masked_predicate<D, M, Args...>) {
            return operator()(fmath::eval(mask), fmath::eval(args)...);
        } else {
            return operator()(fmath::decay(mask), fmath::decay(args)...);
        }
    }
};

} // namespace datapar::fmath

DPL_DEFAULT_NAMESPACE_END
