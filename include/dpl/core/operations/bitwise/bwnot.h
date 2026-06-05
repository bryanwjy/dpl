// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

// IWYU pragma: always_keep

#include "dpl/core/operations/bitwise/bwandnot.h"
#include "dpl/core/operations/bitwise/result.h"
#include "dpl/core/operations/internal/masked.h"
#include "dpl/core/operations/internal/transform.h"
#if !DPL_MODULES
#  include "dpl/core/basic/internal/abi.h"
#  include "dpl/core/concepts/decayable.h"
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/constants/all_bits.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void bwnot(...) noexcept = delete;

struct bwnot_t;

template <typename T>
concept unqualified_canonical_bwnot = requires(T val) {
    {
        bwnot(internal::abi<T>, val)
    } -> canonical_bitwise_result<T, T, typename T::abi_type>;
};

template <typename T>
concept unqualified_extended_bwnot = requires(T val) {
    { bwnot(val) } -> vector_with_common_abi<typename T::abi_type>;
};

template <typename T>
concept expression_bwnot =
    simd_expression<T> && invocable<bwnot_t, simd_expression_result_t<T>>;

template <typename T>
concept decayable_bwnot =
    decayable_vector_for<T, operation_category::lane_agnostic> &&
    regular_invocable<bwnot_t, canonical_type_t<T>>;

template <typename T>
concept extended_bwnot =
    unqualified_extended_bwnot<T> || expression_bwnot<T> || decayable_bwnot<T>;

template <typename S, typename M, typename T, typename A = common_abi_t<M, T>>
concept unqualified_canonical_mbwnot = requires(S src, M mask, T val) {
    {
        bwnot(internal::abi<A>, src, mask, val)
    } -> equivalent_simd_type_with<canonical_if_zero_t<S, T, A>>;
};

template <typename S, typename M, typename T, typename A = common_abi_t<M, T>>
concept unqualified_extended_mbwnot = requires(S src, M mask, T val) {
    { bwnot(src, mask, val) } -> vector_with_common_abi<A>;
};

template <typename S, typename M, typename T>
concept expression_mbwnot =
    (simd_expression<S> || simd_expression<M> || simd_expression<T>) &&
    invocable<bwnot_t, expression_result_or_zero_t<S>,
        simd_expression_result_t<M>, simd_expression_result_t<T>>;

template <typename S, typename M, typename T, typename A = common_abi_t<T, M>>
concept decayable_mbwnot = decayable_vector_for<canonical_if_zero_t<S, T, A>,
                               operation_category::lane_agnostic> &&
    decayable_vector_for<T, operation_category::lane_agnostic> &&
    decayable_mask_for<M, operation_category::lane_agnostic> &&
    regular_invocable<bwnot_t, canonical_or_zero_t<S, T, A>,
        canonical_type_t<M>, canonical_type_t<T>>;

template <typename S, typename M, typename T, typename A = common_abi_t<T, M>>
concept extended_mbwnot = unqualified_extended_mbwnot<S, M, T, A> ||
    expression_mbwnot<S, M, T> || decayable_mbwnot<S, M, T, A>;

template <typename S, typename M, typename T,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept unqualified_canonical_imbwnot = requires(S src, M mask, T val) {
    {
        bwnot(internal::abi<A>, src,
            internal::to_const_mask<A, bwnot_t, S, T>(mask), val)
    } -> equivalent_simd_type_with<canonical_if_zero_t<S, T, A>>;
};

template <typename S, typename M, typename T,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept unqualified_extended_imbwnot = requires(S src, M mask, T val) {
    {
        bwnot(src, internal::to_const_mask<A, bwnot_t, S, T>(mask), val)
    } -> vector_with_common_abi<A>;
};

template <typename S, typename M, typename T>
concept expression_imbwnot = (simd_expression<S> || simd_expression<T>) &&
    invocable<bwnot_t, expression_result_or_zero_t<S>, M,
        simd_expression_result_t<T>>;

template <typename S, typename M, typename T,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept decayable_imbwnot = decayable_vector_for<canonical_if_zero_t<S, T>,
                                operation_category::lane_agnostic> &&
    decayable_vector_for<T, operation_category::lane_agnostic> &&
    regular_invocable<bwnot_t, canonical_or_zero_t<S, T, A>, M,
        canonical_type_t<T>>;

template <typename S, typename M, typename T,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept extended_imbwnot = unqualified_extended_imbwnot<S, M, T, A> ||
    expression_imbwnot<S, M, T> || decayable_imbwnot<S, M, T, A>;

///

template <typename T, typename A = typename T::abi_type>
concept unqualified_canonical_mask_bwnot = requires(T val) {
    { bwnot(internal::abi<A>, val) } -> canonical_bitwise_mask<T, T, A>;
};

template <typename T, typename A = typename T::abi_type>
concept unqualified_extended_mask_bwnot = requires(T val) {
    { bwnot(val) } -> mask_with_common_abi<A>;
};

template <typename T>
concept expression_mask_bwnot = simd_expression<T> &&
    regular_invocable<bwnot_t, simd_expression_result_t<T>>;

template <typename T, typename A = typename T::abi_type>
concept decayable_mask_bwnot =
    decayable_mask_for<T, operation_category::lane_agnostic> &&
    regular_invocable<bwnot_t, canonical_type_t<T>>;

template <typename T, typename A = typename T::abi_type>
concept extended_mask_bwnot = unqualified_extended_mask_bwnot<T, A> ||
    expression_mask_bwnot<T> || decayable_mask_bwnot<T, A>;

struct bwnot_t {
private:
    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_vector<E, A> val) noexcept {
        if constexpr (is_invocable_v<bwandnot_t, basic_vector<E, A>,
                          dx::all_bits_t>) {
            return bwandnot_t::operator()(val, dx::all_bits);
        } else {
            using bit_type = bit_type_t<sizeof(E) * char_bit_v>;
            return internal::transform<basic_vector<E, A>>(
                [](auto val) {
                    auto const promoted = ~__DPL bit_cast<bit_type>(val);
                    return __DPL bit_cast<E>(static_cast<bit_type>(promoted));
                },
                val);
        }
    }

    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_mask<E, A> val) noexcept {
        if constexpr (is_invocable_v<bwandnot_t, basic_mask<E, A>>) {
            constexpr auto all = dx::broadcast<basic_mask<E, A>>(true_type{});
            return bwandnot_t::operator()(val, all);
        } else {
            return internal::transform<basic_vector<E, A>>(
                [](auto val) { return !val; }, val);
        }
    }

public:
    template <fixed_width_abi A, simd_element_for<A> E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_bwnot<basic_vector<E, A>>) {
            if consteval {
                return fallback(val);
            } else {
                return bwnot(internal::abi<A>, val);
            }
        } else {
            return fallback(val);
        }
    }

    template <scalable_abi A, simd_element_for<A> E>
    requires unqualified_canonical_bwnot<basic_vector<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val) noexcept {
        return bwnot(internal::abi<A>, val);
    }

    template <extended_vector T>
    requires extended_bwnot<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val) noexcept {
        if constexpr (unqualified_extended_bwnot<T>) {
            return bwnot(val);
        } else if constexpr (expression_bwnot<T>) {
            return operator()(dx::evaluate(val));
        } else {
            return operator()(dx::to_canonical(val));
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E, simd_element_for<A> ME>
    requires common_size_with<E, ME>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_vector<E, A> src,
        basic_mask<ME, A> mask, basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_mbwnot<basic_vector<E, A>,
                          basic_mask<ME, A>, basic_vector<E, A>>) {
            if consteval {
                return internal::masked<bwnot_t>(src, mask, val);
            } else {
                return bwnot(internal::abi<A>, src, mask, val);
            }
        } else {
            return internal::masked<bwnot_t>(src, mask, val);
        }
    }

    template <simd_abi MA, simd_element_for<MA> ME, simd_abi TA,
        simd_element_for<TA> E>
    requires common_size_with<E, ME> &&
        (different_from<MA, TA> || scalable_abi<MA> || scalable_abi<TA>) &&
        maskable_args<basic_vector<E, MA>, basic_mask<ME, MA>,
            basic_vector<E, TA>> &&
        unqualified_canonical_mbwnot<basic_vector<E, MA>, basic_mask<ME, MA>,
            basic_vector<E, TA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, MA> operator()(basic_vector<E, MA> src,
        basic_mask<ME, MA> mask, basic_vector<E, TA> val) noexcept {
        return bwnot(internal::abi<MA>, src, mask, val);
    }

    template <simd_vector S, simd_mask M, simd_vector T>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        maskable_args<S, M, T> && extended_mbwnot<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, T val) noexcept {
        if constexpr (unqualified_extended_mbwnot<S, M, T>) {
            return bwnot(src, mask, val);
        } else if constexpr (expression_mbwnot<S, M, T>) {
            return operator()(
                dx::evaluate(src), dx::evaluate(mask), dx::evaluate(val));
        } else {
            return operator()(dx::to_canonical(src), dx::to_canonical(mask),
                dx::to_canonical(val));
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E, simd_element_for<A> ME>
    requires common_size_with<E, ME>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_mask<ME, A> mask, basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_mbwnot<zero_t, basic_mask<ME, A>,
                          basic_vector<E, A>>) {
            if consteval {
                return internal::masked<bwnot_t>(mask, val);
            } else {
                return bwnot(internal::abi<A>, dx::zero, mask, val);
            }
        } else {
            return operator()(dx::zero_v<basic_vector<E, A>>, mask, val);
        }
    }

    template <simd_abi TA, simd_element_for<TA> E, common_abi_with<TA> MA,
        simd_element_for<MA> ME>
    requires common_size_with<E, ME> &&
        (different_from<MA, TA> || scalable_abi<MA> || scalable_abi<TA>) &&
        zmaskable_args<basic_mask<ME, MA>, basic_vector<E, TA>> &&
        unqualified_canonical_mbwnot<dx::zero_t, basic_mask<ME, MA>,
            basic_vector<E, TA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        basic_mask<ME, MA> mask, basic_vector<E, TA> val) noexcept {
        return bwnot(internal::abi<MA>, dx::zero, mask, val);
    }

    template <simd_mask M, simd_vector T>
    requires (extended_mask<M> || extended_vector<T>) && zmaskable_args<M, T> &&
        extended_mbwnot<zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val) noexcept {
        if constexpr (unqualified_extended_mbwnot<dx::zero_t, M, T>) {
            return bwnot(dx::zero, mask, val);
        } else if constexpr (expression_mbwnot<dx::zero_t, M, T>) {
            return operator()(dx::evaluate(mask), dx::evaluate(val));
        } else {
            return operator()(dx::to_canonical(mask), dx::to_canonical(val));
        }
    }

    template <simd_mask M, simd_vector T>
    requires invocable<bwnot_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t, M mask, T val) noexcept {
        return operator()(mask, val);
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        const_mask_for<basic_vector<E, A>> M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> src, M mask, basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_imbwnot<basic_vector<E, A>, M,
                          basic_vector<E, A>>) {
            if consteval {
                return internal::masked<bwnot_t>(src, mask, val);
            } else {
                return bwnot(internal::abi<A>, src,
                    dx::to_compatible_const_mask<basic_vector<E, A>>(mask),
                    val);
            }
        } else {
            return internal::masked<bwnot_t>(src, mask, val);
        }
    }

    template <simd_abi SA, simd_element_for<SA> E,
        const_mask_for<basic_vector<E, SA>> M, simd_abi TA>
    requires (different_from<SA, TA> || scalable_abi<SA> || scalable_abi<TA>) &&
        simd_element_for<E, TA> &&
        imm_maskable_args<basic_vector<E, SA>, basic_vector<E, TA>> &&
        unqualified_canonical_imbwnot<basic_vector<E, SA>, M,
            basic_vector<E, TA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, SA> operator()(
        basic_vector<E, SA> src, M mask, basic_vector<E, TA> val) noexcept {
        return bwnot(internal::abi<SA>, src,
            dx::to_compatible_const_mask<basic_vector<E, SA>>(mask), val);
    }

    template <simd_vector S, const_mask_for<S> M, simd_vector T>
    requires (extended_vector<S> || extended_vector<T>) &&
        imm_maskable_args<S, T> && extended_imbwnot<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, T val) noexcept {
        if constexpr (unqualified_extended_imbwnot<S, M, T>) {
            return bwnot(src, dx::to_compatible_const_mask<S>(mask), val);
        } else if constexpr (expression_imbwnot<S, M, T>) {
            return operator()(
                dx::evaluate(src), dx::evaluate(mask), dx::evaluate(val));
        } else {
            return operator()(
                dx::to_canonical(src), mask, dx::to_canonical(val));
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        const_mask_for<basic_vector<E, A>> M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        M mask, basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_imbwnot<zero_t, M,
                          basic_vector<E, A>>) {
            if consteval {
                return internal::masked<bwnot_t>(mask, val);
            } else {
                return bwnot(internal::abi<A>, dx::zero,
                    dx::to_compatible_const_mask<basic_vector<E, A>>(mask),
                    val);
            }
        } else {
            return operator()(dx::broadcast<E, A>(dx::zero), mask, val);
        }
    }

    template <scalable_abi TA, simd_element_for<TA> E,
        const_mask_for<basic_vector<E, TA>> M>
    requires imm_zmaskable_args<basic_vector<E, TA>> &&
        unqualified_canonical_imbwnot<zero_t, M, basic_vector<E, TA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, TA> operator()(
        M mask, basic_vector<E, TA> val) noexcept {
        return bwnot(internal::abi<TA>, dx::zero,
            dx::to_compatible_const_mask<basic_vector<E, TA>>(mask), val);
    }

    template <extended_vector T, const_mask_for<T> M>
    requires imm_zmaskable_args<T> && extended_imbwnot<zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val) noexcept {
        if constexpr (unqualified_extended_imbwnot<zero_t, M, T>) {
            return bwnot(dx::zero, dx::to_compatible_const_mask<T>(mask), val);
        } else if constexpr (expression_imbwnot<dx::zero_t, M, T>) {
            return operator()(mask, dx::evaluate(val));
        } else {
            return operator()(mask, dx::to_canonical(val));
        }
    }

    template <simd_vector T, const_mask_for<T> M>
    requires invocable<bwnot_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t, M mask, T val) noexcept {
        return operator()(mask, val);
    }
    ///

    template <fixed_width_abi A, simd_element_for<A> E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<E, A> operator()(
        basic_mask<E, A> val) noexcept {
        if constexpr (unqualified_canonical_mask_bwnot<basic_mask<E, A>>) {
            if consteval {
                return fallback(val);
            } else {
                return bwnot(internal::abi<A>, val);
            }
        } else {
            return fallback(val);
        }
    }

    template <simd_abi A, simd_element_for<A> E>
    requires scalable_abi<A> &&
        unqualified_canonical_mask_bwnot<basic_mask<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<E, A> operator()(
        basic_mask<E, A> val) noexcept {
        return bwnot(internal::abi<A>, val);
    }

    template <extended_mask T>
    requires extended_mask_bwnot<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val) noexcept {
        if constexpr (unqualified_extended_mask_bwnot<T>) {
            return bwnot(val);
        } else {
            return operator()(dx::to_canonical(val));
        }
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::bwnot_t bwnot{};
} // namespace cpo
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
