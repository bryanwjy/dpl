// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/operations/internal/masked.h"
#include "dpl/core/operations/internal/transform.h"

#if !DPL_MODULES
#  include "dpl/core/basic/initialize.h"
#  include "dpl/core/basic/internal/abi.h"
#  include "dpl/core/concepts/decayable.h"
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/type_traits/representation.h"
#  include "dpl/core/type_traits/simd_abi_type.h"
#  include "dpl/core/type_traits/simd_element_type.h"
#  include "dpl/std/bit/bit_cast.h"
#  include "dpl/std/bit/byteswap.h"
#  include "dpl/std/utility/bitset.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void byteswap(...) noexcept = delete;

template <typename T, typename U>
concept canonical_byteswap_vector =
    simd_vector<T> && same_as<typename T::value_type, typename U::value_type> &&
    same_as<typename T::abi_type, typename U::abi_type>;

template <typename T, typename U>
concept canonical_byteswap_mask =
    simd_mask<T> && same_as<simd_element_type_t<T>, simd_element_type_t<U>> &&
    same_as<simd_abi_type_t<T>, simd_abi_type_t<U>>;

struct byteswap_t;

template <typename T>
concept unqualified_canonical_byteswap = requires(T val) {
    { byteswap(internal::abi<T>, val) } -> equivalent_vector_with<T>;
};

template <typename T>
concept unqualified_extended_byteswap = requires(T val) {
    { byteswap(val) } -> vector_with_common_abi<typename T::abi_type>;
};

template <typename T>
concept expression_byteswap =
    simd_expression<T> && invocable<byteswap_t, simd_expression_result_t<T>>;

template <typename T>
concept decayable_byteswap =
    decayable_vector_for<T, operation_category::lane_agnostic> &&
    regular_invocable<byteswap_t, canonical_type_t<T>>;

template <typename T>
concept extended_byteswap = unqualified_extended_byteswap<T> ||
    expression_byteswap<T> || decayable_byteswap<T>;

template <typename S, typename M, typename T, typename A = common_abi_t<M, T>>
concept unqualified_canonical_mbyteswap = requires(S src, M mask, T val) {
    {
        byteswap(internal::abi<A>, src, mask, val)
    } -> equivalent_simd_type_with<canonical_if_zero_t<S, T, A>>;
};

template <typename S, typename M, typename T, typename A = common_abi_t<M, T>>
concept unqualified_extended_mbyteswap = requires(S src, M mask, T val) {
    { byteswap(src, mask, val) } -> vector_with_common_abi<A>;
};

template <typename S, typename M, typename T>
concept expression_mbyteswap =
    (simd_expression<S> || simd_expression<M> || simd_expression<T>) &&
    invocable<byteswap_t, expression_result_or_zero_t<S>,
        simd_expression_result_t<M>, simd_expression_result_t<T>>;

template <typename S, typename M, typename T, typename A = common_abi_t<T, M>>
concept decayable_mbyteswap = decayable_vector_for<canonical_if_zero_t<S, T, A>,
                                  operation_category::lane_agnostic> &&
    decayable_vector_for<T, operation_category::lane_agnostic> &&
    decayable_mask_for<M, operation_category::lane_agnostic> &&
    regular_invocable<byteswap_t, canonical_or_zero_t<S, T, A>,
        canonical_type_t<M>, canonical_type_t<T>>;

template <typename S, typename M, typename T, typename A = common_abi_t<T, M>>
concept extended_mbyteswap = unqualified_extended_mbyteswap<S, M, T, A> ||
    expression_mbyteswap<S, M, T> || decayable_mbyteswap<S, M, T, A>;

template <typename S, typename M, typename T,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept unqualified_canonical_imbyteswap = requires(S src, M mask, T val) {
    {
        byteswap(internal::abi<A>, src,
            internal::to_const_mask<A, byteswap_t, S, T>(mask), val)
    } -> equivalent_simd_type_with<canonical_if_zero_t<S, T, A>>;
};

template <typename S, typename M, typename T,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept unqualified_extended_imbyteswap = requires(S src, M mask, T val) {
    {
        byteswap(src, internal::to_const_mask<A, byteswap_t, S, T>(mask), val)
    } -> vector_with_common_abi<A>;
};

template <typename S, typename M, typename T>
concept expression_imbyteswap = (simd_expression<S> || simd_expression<T>) &&
    invocable<byteswap_t, expression_result_or_zero_t<S>, M,
        simd_expression_result_t<T>>;

template <typename S, typename M, typename T,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept decayable_imbyteswap = decayable_vector_for<canonical_if_zero_t<S, T>,
                                   operation_category::lane_agnostic> &&
    decayable_vector_for<T, operation_category::lane_agnostic> &&
    regular_invocable<byteswap_t, canonical_or_zero_t<S, T, A>, M,
        canonical_type_t<T>>;

template <typename S, typename M, typename T,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept extended_imbyteswap = unqualified_extended_imbyteswap<S, M, T, A> ||
    expression_imbyteswap<S, M, T> || decayable_imbyteswap<S, M, T, A>;

///
template <typename T>
concept unqualified_canonical_mask_byteswap = requires(T val) {
    { byteswap(internal::abi<T>, val) } -> canonical_byteswap_mask<T>;
};

template <typename T>
concept unqualified_extended_mask_byteswap = requires(T val) {
    { byteswap(val) } -> vector_with_common_abi<typename T::abi_type>;
};

template <typename T>
concept expression_mask_byteswap =
    mask_expression<T> && invocable<byteswap_t, simd_expression_result_t<T>>;

template <typename T>
concept decayable_mask_byteswap =
    decayable_mask_for<T, operation_category::lane_agnostic> &&
    regular_invocable<byteswap_t, canonical_type_t<T>>;

template <typename T>
concept extended_mask_byteswap = unqualified_extended_mask_byteswap<T> ||
    expression_mask_byteswap<T> || decayable_mask_byteswap<T>;

///

struct byteswap_t {
private:
    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_vector<E, A> arg) noexcept {
        using ubit = unsigned_representation_t<E>;
        return internal::transform<basic_vector<ubit, A>>(
            [](auto val) {
                auto const count = __DPL byteswap(__DPL bit_cast<ubit>(val));
                return static_cast<ubit>(count);
            },
            arg);
    }

    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_mask<E, A> arg) noexcept {
        using ubit = unsigned_representation_t<E>;

        return [&]<size_t... Is>(index_sequence<Is...>) {
            constexpr auto simd_size = simd_abi_traits<E, A>::size();
            return dx::initialize<E, A>(
                bitset<simd_size>(arg[imm<simd_size - 1 - Is>]...));
        }(iota_sequence<E, A>);
    }

public:
    template <fixed_width_abi A, simd_element_for<A> E>
    requires integral<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_byteswap<basic_vector<E, A>>) {
            if consteval {
                return fallback(val);
            } else {
                return byteswap(internal::abi<A>, val);
            }
        } else {
            return fallback(val);
        }
    }

    template <simd_abi A, simd_element_for<A> E>
    requires (scalable_abi<A> || !integral<E>) &&
        unqualified_canonical_byteswap<basic_vector<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val) noexcept {
        return byteswap(internal::abi<A>, val);
    }

    template <extended_vector T>
    requires extended_byteswap<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val) noexcept {
        if constexpr (unqualified_extended_byteswap<T>) {
            return byteswap(val);
        } else if constexpr (expression_byteswap<T>) {
            return operator()(dx::evaluate(val));
        } else {
            return operator()(dx::to_canonical(val));
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E, simd_element_for<A> ME>
    requires integral<E> && common_size_with<E, ME>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_vector<E, A> src,
        basic_mask<ME, A> mask, basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_mbyteswap<basic_vector<E, A>,
                          basic_mask<ME, A>, basic_vector<E, A>>) {
            if consteval {
                return internal::masked<byteswap_t>(src, mask, val);
            } else {
                return byteswap(internal::abi<A>, src, mask, val);
            }
        } else {
            return internal::masked<byteswap_t>(src, mask, val);
        }
    }

    template <simd_abi MA, simd_element_for<MA> ME, simd_abi TA,
        simd_element_for<TA> E>
    requires (different_from<MA, TA> || scalable_abi<MA> || scalable_abi<TA> ||
                 !integral<E>) &&
        maskable_args<basic_vector<E, MA>, basic_mask<ME, MA>,
            basic_vector<E, TA>> &&
        unqualified_canonical_mbyteswap<basic_vector<E, MA>, basic_mask<ME, MA>,
            basic_vector<E, TA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, MA> operator()(basic_vector<E, MA> src,
        basic_mask<ME, MA> mask, basic_vector<E, TA> val) noexcept {
        return byteswap(internal::abi<MA>, src, mask, val);
    }

    template <simd_vector S, simd_mask M, simd_vector T>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        maskable_args<S, M, T> && extended_mbyteswap<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, T val) noexcept {
        if constexpr (unqualified_extended_mbyteswap<S, M, T>) {
            return byteswap(src, mask, val);
        } else if constexpr (expression_mbyteswap<S, M, T>) {
            return operator()(
                dx::evaluate(src), dx::evaluate(mask), dx::evaluate(val));
        } else {
            return operator()(dx::to_canonical(src), dx::to_canonical(mask),
                dx::to_canonical(val));
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E, simd_element_for<A> ME>
    requires integral<E> && common_size_with<E, ME>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_mask<ME, A> mask, basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_mbyteswap<zero_t, basic_mask<ME, A>,
                          basic_vector<E, A>>) {
            if consteval {
                return internal::masked<byteswap_t>(mask, val);
            } else {
                return byteswap(internal::abi<A>, dx::zero, mask, val);
            }
        } else {
            return operator()(dx::zero_v<basic_vector<E, A>>, mask, val);
        }
    }

    template <simd_abi TA, simd_element_for<TA> E, common_abi_with<TA> MA,
        simd_element_for<MA> ME>
    requires (different_from<MA, TA> || scalable_abi<MA> || scalable_abi<TA> ||
                 !integral<E>) &&
        zmaskable_args<basic_mask<ME, MA>, basic_vector<E, TA>> &&
        unqualified_canonical_mbyteswap<dx::zero_t, basic_mask<ME, MA>,
            basic_vector<E, TA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        basic_mask<ME, MA> mask, basic_vector<E, TA> val) noexcept {
        return byteswap(internal::abi<MA>, dx::zero, mask, val);
    }

    template <simd_mask M, simd_vector T>
    requires (extended_mask<M> || extended_vector<T>) && zmaskable_args<M, T> &&
        extended_mbyteswap<zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val) noexcept {
        if constexpr (unqualified_extended_mbyteswap<dx::zero_t, M, T>) {
            return byteswap(dx::zero, mask, val);
        } else if constexpr (expression_mbyteswap<dx::zero_t, M, T>) {
            return operator()(dx::evaluate(mask), dx::evaluate(val));
        } else {
            return operator()(dx::to_canonical(mask), dx::to_canonical(val));
        }
    }

    template <simd_mask M, simd_vector T>
    requires invocable<byteswap_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t, M mask, T val) noexcept {
        return operator()(mask, val);
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        const_mask_for<basic_vector<E, A>> M>
    requires integral<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> src, M mask, basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_imbyteswap<basic_vector<E, A>, M,
                          basic_vector<E, A>>) {
            if consteval {
                return internal::masked<byteswap_t>(src, mask, val);
            } else {
                return byteswap(internal::abi<A>, src,
                    dx::to_compatible_const_mask<basic_vector<E, A>>(mask),
                    val);
            }
        } else {
            return internal::masked<byteswap_t>(src, mask, val);
        }
    }

    template <simd_abi SA, simd_element_for<SA> E,
        const_mask_for<basic_vector<E, SA>> M, simd_abi TA>
    requires (different_from<SA, TA> || scalable_abi<SA> || scalable_abi<TA> ||
                 !integral<E>) &&
        simd_element_for<E, TA> &&
        imm_maskable_args<basic_vector<E, SA>, basic_vector<E, TA>> &&
        unqualified_canonical_imbyteswap<basic_vector<E, SA>, M,
            basic_vector<E, TA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, SA> operator()(
        basic_vector<E, SA> src, M mask, basic_vector<E, TA> val) noexcept {
        return byteswap(internal::abi<SA>, src,
            dx::to_compatible_const_mask<basic_vector<E, SA>>(mask), val);
    }

    template <simd_vector S, const_mask_for<S> M, simd_vector T>
    requires (extended_vector<S> || extended_vector<T>) &&
        imm_maskable_args<S, T> && extended_imbyteswap<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, T val) noexcept {
        if constexpr (unqualified_extended_imbyteswap<S, M, T>) {
            return byteswap(src, dx::to_compatible_const_mask<S>(mask), val);
        } else if constexpr (expression_imbyteswap<S, M, T>) {
            return operator()(
                dx::evaluate(src), dx::evaluate(mask), dx::evaluate(val));
        } else {
            return operator()(
                dx::to_canonical(src), mask, dx::to_canonical(val));
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        const_mask_for<basic_vector<E, A>> M>
    requires integral<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        M mask, basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_imbyteswap<zero_t, M,
                          basic_vector<E, A>>) {
            if consteval {
                return internal::masked<byteswap_t>(mask, val);
            } else {
                return byteswap(internal::abi<A>, dx::zero,
                    dx::to_compatible_const_mask<basic_vector<E, A>>(mask),
                    val);
            }
        } else {
            return operator()(dx::broadcast<E, A>(dx::zero), mask, val);
        }
    }

    template <simd_abi TA, simd_element_for<TA> E,
        const_mask_for<basic_vector<E, TA>> M>
    requires (scalable_abi<TA> || !integral<E>) &&
        imm_zmaskable_args<basic_vector<E, TA>> &&
        unqualified_canonical_imbyteswap<zero_t, M, basic_vector<E, TA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, TA> operator()(
        M mask, basic_vector<E, TA> val) noexcept {
        return byteswap(internal::abi<TA>, dx::zero,
            dx::to_compatible_const_mask<basic_vector<E, TA>>(mask), val);
    }

    template <extended_vector T, const_mask_for<T> M>
    requires imm_zmaskable_args<T> && extended_imbyteswap<zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val) noexcept {
        if constexpr (unqualified_extended_imbyteswap<zero_t, M, T>) {
            return byteswap(
                dx::zero, dx::to_compatible_const_mask<T>(mask), val);
        } else if constexpr (expression_imbyteswap<dx::zero_t, M, T>) {
            return operator()(mask, dx::evaluate(val));
        } else {
            return operator()(mask, dx::to_canonical(val));
        }
    }

    template <simd_vector T, const_mask_for<T> M>
    requires invocable<byteswap_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t, M mask, T val) noexcept {
        return operator()(mask, val);
    }

    ///
    template <fixed_width_abi A, simd_element_for<A> E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<E, A> operator()(
        basic_mask<E, A> val) noexcept {
        if constexpr (unqualified_canonical_mask_byteswap<basic_mask<E, A>>) {
            if consteval {
                return fallback(val);
            } else {
                return byteswap(internal::abi<A>, val);
            }
        } else {
            return fallback(val);
        }
    }

    template <scalable_abi A, simd_element_for<A> E>
    requires unqualified_canonical_mask_byteswap<basic_mask<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<E, A> operator()(
        basic_mask<E, A> val) noexcept {
        return byteswap(internal::abi<A>, val);
    }

    template <extended_mask T>
    requires extended_mask_byteswap<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val) noexcept {
        if constexpr (unqualified_extended_mask_byteswap<T>) {
            return byteswap(val);
        } else if constexpr (expression_mask_byteswap<T>) {
            return operator()(dx::evaluate(val));
        } else {
            return operator()(dx::to_canonical(val));
        }
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::byteswap_t byteswap{};
}
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
