// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

// IWYU pragma: always_keep

#include "dpl/core/operations/arithmetic/subtract.h"
#include "dpl/core/operations/evaluate.h"
#include "dpl/core/operations/internal/extended_operations.h"
#include "dpl/core/operations/internal/masked.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/arithmetic_type.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_equivalence.h"
#  include "dpl/core/concepts/simd_expression.h"
#  include "dpl/core/constants/zero.h"
#  include "dpl/core/type_traits/simd_expression_result.h"
#  include "dpl/std/concepts/different_from.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void negate(...) noexcept = delete;

struct negate_t;

template <typename T>
concept unqualified_canonical_negate = requires(T val) {
    {
        negate(internal::abi<T>, val)
    } -> canonical_arithmetic_result<T, T, typename T::abi_type>;
};

template <typename T>
concept unqualified_extended_negate = requires(T val) {
    { negate(val) } -> extended_operation_vector<typename T::abi_type>;
};

template <typename T>
concept expression_negate =
    simd_expression<T> && invocable<negate_t, simd_expression_result_t<T>>;

template <typename T>
concept decayable_negate =
    decayable_vector_for<T, operation_category::lane_agnostic> &&
    regular_invocable<negate_t, canonical_type_t<T>>;

template <typename T>
concept extended_negate = unqualified_extended_negate<T> ||
    expression_negate<T> || decayable_negate<T>;

template <typename S, typename M, typename T, typename A = common_abi_t<M, T>>
concept unqualified_canonical_mnegate = requires(S src, M mask, T val) {
    {
        negate(internal::abi<A>, src, mask, val)
    } -> equivalent_simd_as<canonical_if_zero_t<S, T, A>>;
};

template <typename S, typename M, typename T, typename A = common_abi_t<M, T>>
concept unqualified_extended_mnegate = requires(S src, M mask, T val) {
    { negate(src, mask, val) } -> extended_operation_vector<A>;
};

template <typename S, typename M, typename T>
concept expression_mnegate =
    (simd_expression<S> || simd_expression<M> || simd_expression<T>) &&
    invocable<negate_t, expression_result_or_zero_t<S>,
        simd_expression_result_t<M>, simd_expression_result_t<T>>;

template <typename S, typename M, typename T, typename A = common_abi_t<T, M>>
concept decayable_mnegate = decayable_vector_for<canonical_if_zero_t<S, T, A>,
                                operation_category::lane_agnostic> &&
    decayable_vector_for<T, operation_category::lane_agnostic> &&
    decayable_mask_for<M, operation_category::lane_agnostic> &&
    regular_invocable<negate_t, canonical_or_zero_t<S, T, A>,
        canonical_type_t<M>, canonical_type_t<T>>;

template <typename S, typename M, typename T, typename A = common_abi_t<T, M>>
concept extended_mnegate = unqualified_extended_mnegate<S, M, T, A> ||
    expression_mnegate<S, M, T> || decayable_mnegate<S, M, T, A>;

template <typename S, typename M, typename T,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept unqualified_canonical_imnegate = requires(S src, M mask, T val) {
    {
        negate(internal::abi<A>, src,
            internal::to_const_mask<A, negate_t, S, T>(mask), val)
    } -> equivalent_simd_as<canonical_if_zero_t<S, T, A>>;
};

template <typename S, typename M, typename T,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept unqualified_extended_imnegate = requires(S src, M mask, T val) {
    {
        negate(src, internal::to_const_mask<A, negate_t, S, T>(mask), val)
    } -> extended_operation_vector<A>;
};

template <typename S, typename M, typename T>
concept expression_imnegate = (simd_expression<S> || simd_expression<T>) &&
    invocable<negate_t, expression_result_or_zero_t<S>, M,
        simd_expression_result_t<T>>;

template <typename S, typename M, typename T,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept decayable_imnegate = decayable_vector_for<canonical_if_zero_t<S, T>,
                                 operation_category::lane_agnostic> &&
    decayable_vector_for<T, operation_category::lane_agnostic> &&
    regular_invocable<negate_t, canonical_or_zero_t<S, T, A>, M,
        canonical_type_t<T>>;

template <typename S, typename M, typename T,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept extended_imnegate = unqualified_extended_imnegate<S, M, T, A> ||
    expression_imnegate<S, M, T> || decayable_imnegate<S, M, T, A>;

struct negate_t {
private:
    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_vector<E, A>
        DPL_VECTORCALL fallback(basic_vector<E, A> val) noexcept {
        return dx::subtract(dx::zero, val);
    }

public:
    template <fixed_width_abi A, simd_element_for<A> E>
    requires arithmetic_type<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_negate<basic_vector<E, A>>) {
            if consteval {
                return fallback(val);
            } else {
                return negate(internal::abi<A>, val);
            }
        } else {
            return fallback(val);
        }
    }

    template <simd_abi A, simd_element_for<A> E>
    requires (scalable_abi<A> || !arithmetic_type<E>) &&
        unqualified_canonical_negate<basic_vector<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val) noexcept {
        return negate(internal::abi<A>, val);
    }

    template <extended_vector T>
    requires extended_negate<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val) noexcept {
        if constexpr (unqualified_extended_negate<T>) {
            return negate(val);
        } else if constexpr (expression_negate<T>) {
            return operator()(dx::evaluate(val));
        } else {
            return operator()(dx::to_canonical(val));
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E, simd_element_for<A> ME>
    requires arithmetic_type<E> && common_size_with<E, ME>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_vector<E, A> src,
        basic_mask<ME, A> mask, basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_mnegate<basic_vector<E, A>,
                          basic_mask<ME, A>, basic_vector<E, A>>) {
            if consteval {
                return internal::masked<negate_t>(src, mask, val);
            } else {
                return negate(internal::abi<A>, src, mask, val);
            }
        } else {
            return internal::masked<negate_t>(src, mask, val);
        }
    }

    template <simd_abi MA, simd_element_for<MA> ME, simd_abi TA,
        simd_element_for<TA> E>
    requires common_size_with<E, ME> &&
        (different_from<MA, TA> || scalable_abi<MA> || scalable_abi<TA> ||
            !arithmetic_type<E>) &&
        maskable_args<basic_vector<E, MA>, basic_mask<ME, MA>,
            basic_vector<E, TA>> &&
        unqualified_canonical_mnegate<basic_vector<E, MA>, basic_mask<ME, MA>,
            basic_vector<E, TA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, MA> operator()(basic_vector<E, MA> src,
        basic_mask<ME, MA> mask, basic_vector<E, TA> val) noexcept {
        return negate(internal::abi<MA>, src, mask, val);
    }

    template <simd_vector S, simd_mask M, simd_vector T>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        maskable_args<S, M, T> && extended_mnegate<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, T val) noexcept {
        if constexpr (unqualified_extended_mnegate<S, M, T>) {
            return negate(src, mask, val);
        } else if constexpr (expression_mnegate<S, M, T>) {
            return operator()(
                dx::evaluate(src), dx::evaluate(mask), dx::evaluate(val));
        } else {
            return operator()(dx::to_canonical(src), dx::to_canonical(mask),
                dx::to_canonical(val));
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E, simd_element_for<A> ME>
    requires arithmetic_type<E> && common_size_with<E, ME>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_mask<ME, A> mask, basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_mnegate<zero_t, basic_mask<ME, A>,
                          basic_vector<E, A>>) {
            if consteval {
                return internal::masked<negate_t>(mask, val);
            } else {
                return negate(internal::abi<A>, dx::zero, mask, val);
            }
        } else {
            return operator()(dx::zero_v<basic_vector<E, A>>, mask, val);
        }
    }

    template <simd_abi TA, simd_element_for<TA> E, common_abi_with<TA> MA,
        simd_element_for<MA> ME>
    requires common_size_with<E, ME> &&
        (different_from<MA, TA> || scalable_abi<MA> || scalable_abi<TA> ||
            !arithmetic_type<E>) &&
        zmaskable_args<basic_mask<ME, MA>, basic_vector<E, TA>> &&
        unqualified_canonical_mnegate<dx::zero_t, basic_mask<ME, MA>,
            basic_vector<E, TA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        basic_mask<ME, MA> mask, basic_vector<E, TA> val) noexcept {
        return negate(internal::abi<MA>, dx::zero, mask, val);
    }

    template <simd_mask M, simd_vector T>
    requires (extended_mask<M> || extended_vector<T>) && zmaskable_args<M, T> &&
        extended_mnegate<zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val) noexcept {
        if constexpr (unqualified_extended_mnegate<dx::zero_t, M, T>) {
            return negate(dx::zero, mask, val);
        } else if constexpr (expression_mnegate<dx::zero_t, M, T>) {
            return operator()(dx::evaluate(mask), dx::evaluate(val));
        } else {
            return operator()(dx::to_canonical(mask), dx::to_canonical(val));
        }
    }

    template <simd_mask M, simd_vector T>
    requires invocable<negate_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t, M mask, T val) noexcept {
        return operator()(mask, val);
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        const_mask_for<basic_vector<E, A>> M>
    requires arithmetic_type<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> src, M mask, basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_imnegate<basic_vector<E, A>, M,
                          basic_vector<E, A>>) {
            if consteval {
                return internal::masked<negate_t>(src, mask, val);
            } else {
                return negate(internal::abi<A>, src,
                    dx::to_compatible_const_mask<basic_vector<E, A>>(mask),
                    val);
            }
        } else {
            return internal::masked<negate_t>(src, mask, val);
        }
    }

    template <simd_abi SA, simd_element_for<SA> E,
        const_mask_for<basic_vector<E, SA>> M, simd_abi TA>
    requires (different_from<SA, TA> || scalable_abi<SA> || scalable_abi<TA> ||
                 !arithmetic_type<E>) &&
        simd_element_for<E, TA> &&
        imm_maskable_args<basic_vector<E, SA>, basic_vector<E, TA>> &&
        unqualified_canonical_imnegate<basic_vector<E, SA>, M,
            basic_vector<E, TA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, SA> operator()(
        basic_vector<E, SA> src, M mask, basic_vector<E, TA> val) noexcept {
        return negate(internal::abi<SA>, src,
            dx::to_compatible_const_mask<basic_vector<E, SA>>(mask), val);
    }

    template <simd_vector S, const_mask_for<S> M, simd_vector T>
    requires (extended_vector<S> || extended_vector<T>) &&
        imm_maskable_args<S, T> && extended_imnegate<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, T val) noexcept {
        if constexpr (unqualified_extended_imnegate<S, M, T>) {
            return negate(src, dx::to_compatible_const_mask<S>(mask), val);
        } else if constexpr (expression_imnegate<S, M, T>) {
            return operator()(
                dx::evaluate(src), dx::evaluate(mask), dx::evaluate(val));
        } else {
            return operator()(
                dx::to_canonical(src), mask, dx::to_canonical(val));
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        const_mask_for<basic_vector<E, A>> M>
    requires arithmetic_type<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        M mask, basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_imnegate<zero_t, M,
                          basic_vector<E, A>>) {
            if consteval {
                return internal::masked<negate_t>(mask, val);
            } else {
                return negate(internal::abi<A>, dx::zero,
                    dx::to_compatible_const_mask<basic_vector<E, A>>(mask),
                    val);
            }
        } else {
            return operator()(dx::broadcast<E, A>(dx::zero), mask, val);
        }
    }

    template <simd_abi TA, simd_element_for<TA> E,
        const_mask_for<basic_vector<E, TA>> M>
    requires (scalable_abi<TA> || !arithmetic_type<E>) &&
        imm_zmaskable_args<basic_vector<E, TA>> &&
        unqualified_canonical_imnegate<zero_t, M, basic_vector<E, TA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, TA> operator()(
        M mask, basic_vector<E, TA> val) noexcept {
        return negate(internal::abi<TA>, dx::zero,
            dx::to_compatible_const_mask<basic_vector<E, TA>>(mask), val);
    }

    template <extended_vector T, const_mask_for<T> M>
    requires imm_zmaskable_args<T> && extended_imnegate<zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val) noexcept {
        if constexpr (unqualified_extended_imnegate<zero_t, M, T>) {
            return negate(dx::zero, dx::to_compatible_const_mask<T>(mask), val);
        } else if constexpr (expression_imnegate<dx::zero_t, M, T>) {
            return operator()(mask, dx::evaluate(val));
        } else {
            return operator()(mask, dx::to_canonical(val));
        }
    }

    template <simd_vector T, const_mask_for<T> M>
    requires invocable<negate_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t, M mask, T val) noexcept {
        return operator()(mask, val);
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::negate_t negate{};
} // namespace cpo
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
