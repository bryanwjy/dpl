// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/operations/bit/result.h"
#include "dpl/core/operations/masked.h"
#include "dpl/core/operations/transform.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/decayable.h"
#  include "dpl/core/concepts/operation_category.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/std/bit/bit_cast.h"
#  include "dpl/std/bit/countr.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void popcount(...) noexcept = delete;

struct popcount_t;

template <typename T>
concept unqualified_canonical_popcount = requires(T val) {
    {
        popcount(internal::abi<T>, val)
    } -> canonical_vector_bit<T, typename T::abi_type>;
};

template <typename T>
concept unqualified_extended_popcount = requires(T val) {
    { popcount(val) } -> extended_operation_vector<typename T::abi_type>;
};

template <typename T>
concept expression_popcount =
    simd_expression<T> && invocable<popcount_t, simd_expression_result_t<T>>;

template <typename T>
concept decayable_popcount =
    decayable_vector_for<T, operation_category::lane_agnostic> &&
    regular_invocable<popcount_t, canonical_type_t<T>>;

template <typename T>
concept extended_popcount = unqualified_extended_popcount<T> ||
    expression_popcount<T> || decayable_popcount<T>;

template <typename S, typename M, typename T, typename A = common_abi_t<M, T>>
concept unqualified_canonical_mpopcount = requires(S src, M mask, T val) {
    {
        popcount(internal::abi<A>, src, mask, val)
    } -> equivalent_simd_as<canonical_if_zero_t<S, T, A>>;
};

template <typename S, typename M, typename T, typename A = common_abi_t<M, T>>
concept unqualified_extended_mpopcount = requires(S src, M mask, T val) {
    { popcount(src, mask, val) } -> extended_operation_vector<A>;
};

template <typename S, typename M, typename T>
concept expression_mpopcount =
    (simd_expression<S> || simd_expression<M> || simd_expression<T>) &&
    invocable<popcount_t, expression_result_or_zero_t<S>,
        simd_expression_result_t<M>, simd_expression_result_t<T>>;

template <typename S, typename M, typename T, typename A = common_abi_t<T, M>>
concept decayable_mpopcount = decayable_vector_for<canonical_if_zero_t<S, T, A>,
                                  operation_category::lane_agnostic> &&
    decayable_vector_for<T, operation_category::lane_agnostic> &&
    decayable_mask_for<M, operation_category::lane_agnostic> &&
    regular_invocable<popcount_t, canonical_or_zero_t<S, T, A>,
        canonical_type_t<M>, canonical_type_t<T>>;

template <typename S, typename M, typename T, typename A = common_abi_t<T, M>>
concept extended_mpopcount = unqualified_extended_mpopcount<S, M, T, A> ||
    expression_mpopcount<S, M, T> || decayable_mpopcount<S, M, T, A>;

template <typename S, typename M, typename T,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept unqualified_canonical_impopcount = requires(S src, M mask, T val) {
    {
        popcount(internal::abi<A>, src,
            internal::to_const_mask<A, popcount_t, S, T>(mask), val)
    } -> equivalent_simd_as<canonical_if_zero_t<S, T, A>>;
};

template <typename S, typename M, typename T,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept unqualified_extended_impopcount = requires(S src, M mask, T val) {
    {
        popcount(src, internal::to_const_mask<A, popcount_t, S, T>(mask), val)
    } -> extended_operation_vector<A>;
};

template <typename S, typename M, typename T>
concept expression_impopcount = (simd_expression<S> || simd_expression<T>) &&
    invocable<popcount_t, expression_result_or_zero_t<S>, M,
        simd_expression_result_t<T>>;

template <typename S, typename M, typename T,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept decayable_impopcount = decayable_vector_for<canonical_if_zero_t<S, T>,
                                   operation_category::lane_agnostic> &&
    decayable_vector_for<T, operation_category::lane_agnostic> &&
    regular_invocable<popcount_t, canonical_or_zero_t<S, T, A>, M,
        canonical_type_t<T>>;

template <typename S, typename M, typename T,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept extended_impopcount = unqualified_extended_impopcount<S, M, T, A> ||
    expression_impopcount<S, M, T> || decayable_impopcount<S, M, T, A>;

///
template <typename T>
concept unqualified_canonical_mask_popcount = requires(T val) {
    { popcount(internal::abi<T>, val) } -> core_convertible_to<size_t>;
};

template <typename T>
concept unqualified_extended_mask_popcount = requires(T val) {
    { popcount(val) } -> core_convertible_to<size_t>;
};

template <typename T>
concept expression_mask_popcount =
    simd_expression<T> && invocable<popcount_t, simd_expression_result_t<T>>;

template <typename T>
concept decayable_mask_popcount =
    decayable_mask_for<T, operation_category::lane_reduction> &&
    regular_invocable<popcount_t, canonical_type_t<T>>;

template <typename T>
concept extended_mask_popcount = unqualified_extended_mask_popcount<T> ||
    expression_mask_popcount<T> || decayable_mask_popcount<T>;
///

struct popcount_t {
private:
    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_vector<E, A> val) noexcept {
        using ubit = unsigned_representation_t<E>;
        return internal::transform<basic_vector<ubit, A>>(
            [](auto val) {
                auto const count = __DPL popcount(__DPL bit_cast<ubit>(val));
                return static_cast<ubit>(count);
            },
            val);
    }

    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr size_t DPL_VECTORCALL fallback(
        basic_mask<E, A> val) noexcept {
        return []<size_t I>(this auto self, auto val, immediate<I>) {
            if constexpr (I > 0) {
                return static_cast<size_t>(val[I]) + self(val, imm<I - 1>);
            } else {
                return static_cast<size_t>(val[I]);
            }
        }(val, imm<simd_abi_traits<E, A>::size - 1>);
    }

public:
    template <fixed_width_abi A, simd_element_for<A> E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<unsigned_representation_t<E>, A> operator()(
        basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_popcount<basic_vector<E, A>>) {
            if consteval {
                return fallback(val);
            } else {
                return popcount(internal::abi<A>, val);
            }
        } else {
            return fallback(val);
        }
    }

    template <scalable_abi A, simd_element_for<A> E>
    requires unqualified_canonical_popcount<basic_vector<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<unsigned_representation_t<E>, A> operator()(
        basic_vector<E, A> val) noexcept {
        return popcount(internal::abi<A>, val);
    }

    template <extended_vector T>
    requires extended_popcount<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val) noexcept {
        if constexpr (unqualified_extended_popcount<T>) {
            return popcount(val);
        } else if constexpr (expression_popcount<T>) {
            return operator()(dx::evaluate(val));
        } else {
            return operator()(dx::to_canonical(val));
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E, simd_element_for<A> ME>
    requires common_size_with<E, ME>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<unsigned_representation_t<E>, A> operator()(
        basic_vector<unsigned_representation_t<E>, A> src,
        basic_mask<ME, A> mask, basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_mpopcount<
                          basic_vector<unsigned_representation_t<E>, A>,
                          basic_mask<ME, A>, basic_vector<E, A>>) {
            if consteval {
                return internal::masked<popcount_t>(src, mask, val);
            } else {
                return popcount(internal::abi<A>, src, mask, val);
            }
        } else {
            return internal::masked<popcount_t>(src, mask, val);
        }
    }

    template <simd_abi MA, simd_element_for<MA> ME, simd_abi TA,
        simd_element_for<TA> E>
    requires (different_from<MA, TA> || scalable_abi<MA> || scalable_abi<TA>) &&
        maskable_args<basic_vector<unsigned_representation_t<E>, MA>,
            basic_mask<ME, MA>, basic_vector<E, TA>> &&
        unqualified_canonical_mpopcount<
            basic_vector<unsigned_representation_t<E>, MA>, basic_mask<ME, MA>,
            basic_vector<E, TA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, MA> operator()(basic_vector<E, MA> src,
        basic_mask<ME, MA> mask, basic_vector<E, TA> val) noexcept {
        return popcount(internal::abi<MA>, src, mask, val);
    }

    template <simd_vector S, simd_mask M, simd_vector T>
    requires extended_vector_bit<S, T> &&
        (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        maskable_args<S, M, T> && extended_mpopcount<popcount_t, S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, T val) noexcept {
        if constexpr (unqualified_extended_mpopcount<S, M, T>) {
            return popcount(src, mask, val);
        } else if constexpr (expression_mpopcount<S, M, T>) {
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
    static constexpr basic_vector<unsigned_representation_t<E>, A> operator()(
        basic_mask<ME, A> mask, basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_mpopcount<zero_t, basic_mask<ME, A>,
                          basic_vector<E, A>>) {
            if consteval {
                return internal::masked<popcount_t>(mask, val);
            } else {
                return popcount(internal::abi<A>, dx::zero, mask, val);
            }
        } else {
            return operator()(dx::zero_v<basic_vector<E, A>>, mask, val);
        }
    }

    template <simd_abi MA, simd_element_for<MA> ME, simd_abi TA,
        simd_element_for<TA> E>
    requires (different_from<MA, TA> || scalable_abi<MA> || scalable_abi<TA>) &&
        zmaskable_args<basic_mask<ME, MA>, basic_vector<E, TA>> &&
        unqualified_canonical_mpopcount<dx::zero_t, basic_mask<ME, MA>,
            basic_vector<E, TA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        basic_mask<ME, MA> mask, basic_vector<E, TA> val) noexcept {
        return popcount(internal::abi<MA>, dx::zero, mask, val);
    }

    template <simd_mask M, simd_vector T>
    requires (extended_mask<M> || extended_vector<T>) && zmaskable_args<M, T> &&
        extended_mpopcount<popcount_t, zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val) noexcept {
        if constexpr (unqualified_extended_mpopcount<dx::zero_t, M, T>) {
            return popcount(dx::zero, mask, val);
        } else if constexpr (expression_mpopcount<dx::zero_t, M, T>) {
            return operator()(dx::evaluate(mask), dx::evaluate(val));
        } else {
            return operator()(dx::to_canonical(mask), dx::to_canonical(val));
        }
    }

    template <simd_mask M, simd_vector T>
    requires requires(M mask, T val) { popcount_t::operator()(mask, val); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t, M mask, T val) noexcept {
        return operator()(mask, val);
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        const_mask_for<basic_vector<E, A>> M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<unsigned_representation_t<E>, A> operator()(
        basic_vector<unsigned_representation_t<E>, A> src, M mask,
        basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_impopcount<
                          basic_vector<unsigned_representation_t<E>, A>, M,
                          basic_vector<E, A>>) {
            if consteval {
                return internal::masked<popcount_t>(src, mask, val);
            } else {
                return popcount(internal::abi<A>, src,
                    dx::to_compatible_const_mask<basic_vector<E, A>>(mask),
                    val);
            }
        } else {
            return internal::masked<popcount_t>(src, mask, val);
        }
    }

    template <simd_abi SA, simd_abi TA, simd_element_for<TA> E,
        const_mask_for<basic_vector<unsigned_representation_t<E>, SA>> M>
    requires simd_element_for<unsigned_representation_t<E>, SA> &&
        (different_from<SA, TA> || scalable_abi<SA> || scalable_abi<TA>) &&
        imm_maskable_args<basic_vector<unsigned_representation_t<E>, SA>,
            basic_vector<E, TA>> &&
        unqualified_canonical_impopcount<
            basic_vector<unsigned_representation_t<E>, SA>, M,
            basic_vector<E, TA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<unsigned_representation_t<E>, SA> operator()(
        basic_vector<unsigned_representation_t<E>, SA> src, M mask,
        basic_vector<E, TA> val) noexcept {
        return popcount(internal::abi<SA>, src,
            dx::to_compatible_const_mask<basic_vector<E, SA>>(mask), val);
    }

    template <simd_vector S, const_mask_for<S> M, simd_vector T>
    requires extended_vector_bit<S, T> &&
        (extended_vector<S> || extended_vector<T>) && imm_maskable_args<S, T> &&
        extended_impopcount<popcount_t, S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, T val) noexcept {
        constexpr auto cmask = dx::to_compatible_const_mask<S>(mask);
        if constexpr (unqualified_extended_impopcount<S, M, T>) {
            return popcount(src, cmask, val);
        } else if constexpr (expression_impopcount<S, M, T>) {
            return operator()(dx::evaluate(src), cmask, dx::evaluate(val));
        } else {
            return operator()(
                dx::to_canonical(src), cmask, dx::to_canonical(val));
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        const_mask_for<basic_vector<E, A>> M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<unsigned_representation_t<E>, A> operator()(
        M mask, basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_impopcount<zero_t, M,
                          basic_vector<E, A>>) {
            if consteval {
                return internal::masked<popcount_t>(mask, val);
            } else {
                return popcount(internal::abi<A>, dx::zero,
                    dx::to_compatible_const_mask<basic_vector<E, A>>(mask),
                    val);
            }
        } else {
            return operator()(dx::zero_v<basic_vector<E, A>>, mask, val);
        }
    }

    template <scalable_abi TA, simd_element_for<TA> E,
        const_mask_for<basic_vector<E, TA>> M>
    requires imm_zmaskable_args<basic_vector<E, TA>> &&
        unqualified_canonical_impopcount<zero_t, M, basic_vector<E, TA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<unsigned_representation_t<E>, TA> operator()(
        M mask, basic_vector<E, TA> val) noexcept {
        return popcount(internal::abi<TA>, dx::zero,
            dx::to_compatible_const_mask<basic_vector<E, TA>>(mask), val);
    }

    template <extended_vector T, const_mask_for<T> M>
    requires imm_zmaskable_args<T> &&
        extended_impopcount<popcount_t, zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val) noexcept {
        constexpr auto cmask = dx::to_compatible_const_mask<T>(mask);
        if constexpr (unqualified_extended_impopcount<zero_t, M, T>) {
            return popcount(dx::zero, cmask, val);
        } else if constexpr (expression_impopcount<zero_t, M, T>) {
            return operator()(cmask, dx::evaluate(val));
        } else {
            return operator()(cmask, dx::to_canonical(val));
        }
    }

    template <simd_vector T, const_mask_for<T> M>
    requires requires(M mask, T val) { popcount_t::operator()(mask, val); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t, M mask, T val) noexcept {
        return operator()(mask, val);
    }
    ///

    template <fixed_width_abi A, simd_element_for<A> E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr size_t operator()(basic_mask<E, A> val) noexcept {
        if constexpr (unqualified_canonical_mask_popcount<basic_mask<E, A>>) {
            if consteval {
                return fallback(val);
            } else {
                return popcount(internal::abi<A>, val);
            }
        } else {
            return fallback(val);
        }
    }

    template <scalable_abi A, simd_element_for<A> E>
    requires unqualified_canonical_mask_popcount<basic_mask<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr size_t operator()(basic_mask<E, A> val) noexcept {
        return popcount(internal::abi<A>, val);
    }

    template <extended_mask T>
    requires extended_mask_popcount<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr size_t operator()(T val) noexcept {
        if constexpr (unqualified_extended_mask_popcount<T>) {
            return popcount(val);
        } else if constexpr (expression_mask_popcount<T>) {
            return operator()(dx::evaluate(val));
        } else {
            return operator()(dx::to_canonical(val));
        }
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::popcount_t popcount{};
}
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
