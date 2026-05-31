// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

// IWYU pragma: always_keep

#include "dpl/core/operations/bitwise/bwandnot.h"
#include "dpl/core/operations/bitwise/result.h"
#include "dpl/core/operations/masked.h"
#include "dpl/core/operations/transform.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/decayable.h"
#  include "dpl/core/concepts/operation_category.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_equivalence.h"
#  include "dpl/core/constants/all_bits.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void bwnot(...) noexcept = delete;
template <auto>
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
    { bwnot(val) } -> extended_bitwise_result<T, T, typename T::abi_type>;
};

template <typename T>
concept unqualified_bwnot = unqualified_extended_bwnot<T> ||
    (decayable_vector_for<T, operation_category::lane_agnostic> &&
        regular_invocable<bwnot_t, canonical_type_t<T>>);

template <typename T, typename A = typename T::abi_type>
concept unqualified_canonical_mask_bwnot = requires(T val) {
    { bwnot(internal::abi<A>, val) } -> canonical_bitwise_mask<T, T, A>;
};

template <typename T, typename A = typename T::abi_type>
concept unqualified_extended_mask_bwnot = requires(T val) {
    { bwnot(val) } -> extended_bitwise_mask<T, T, A>;
};

template <typename T, typename A = typename T::abi_type>
concept unqualified_mask_bwnot = unqualified_canonical_mask_bwnot<T, A> ||
    unqualified_extended_mask_bwnot<T, A> ||
    (decayable_mask_for<T, operation_category::lane_agnostic> &&
        regular_invocable<bwnot_t, canonical_type_t<T>, A>);

template <typename S, typename C, typename T, typename A = common_abi_t<C, T>>
concept unqualified_canonical_mbwnot = requires(S src, C mask, T val) {
    {
        bwnot(internal::abi<A>, src, mask, val)
    } -> equivalent_simd_as<canonical_if_zero_t<S, T, A>>;
};

template <typename S, typename C, typename T, typename A = common_abi_t<C, T>>
concept unqualified_extended_mbwnot = requires(S src, C mask, T val) {
    {
        bwnot(src, mask, val)
    } -> equivalent_simd_as<canonical_if_zero_t<S, T, A>>;
};

template <typename S, typename C, typename T, typename A = common_abi_t<T, C>>
concept decayable_mbwnot = decayable_vector_for<canonical_if_zero_t<S, T, A>,
                               operation_category::lane_agnostic> &&
    decayable_vector_for<T, operation_category::lane_agnostic> &&
    decayable_mask_for<C, operation_category::lane_agnostic> &&
    requires(bwnot_t op, canonical_or_zero_t<S, T, A> s, canonical_type_t<C> c,
        canonical_type_t<T> t) { op(s, c, t); };

template <typename S, typename C, typename T, typename A = common_abi_t<T, C>>
concept extended_mbwnot =
    unqualified_extended_mbwnot<S, C, T, A> || decayable_mbwnot<S, C, T, A>;

template <typename S, typename M, typename T,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept unqualified_canonical_imbwnot = requires(S src, M mask, T val) {
    {
        bwnot(internal::abi<A>, src,
            internal::to_const_mask<A, bwnot_t, S, T>(mask), val)
    } -> equivalent_simd_as<canonical_if_zero_t<S, T, A>>;
};

template <typename S, typename M, typename T,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept unqualified_extended_imbwnot = requires(S src, M mask, T val) {
    {
        bwnot(src, internal::to_const_mask<A, bwnot_t, S, T>(mask), val)
    } -> equivalent_simd_as<canonical_if_zero_t<S, T, A>>;
};

template <typename S, typename M, typename T,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept decayable_imbwnot = decayable_vector_for<canonical_if_zero_t<S, T>,
                                operation_category::lane_agnostic> &&
    decayable_vector_for<T, operation_category::lane_agnostic> &&
    requires(bwnot_t op, canonical_or_zero_t<S, T, A> s, M mask,
        canonical_type_t<T> t) { op(s, mask, t); };

template <typename S, typename M, typename T,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept extended_imbwnot =
    unqualified_extended_imbwnot<S, M, T, A> || decayable_imbwnot<S, M, T, A>;

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

    template <simd_abi A, simd_element_for<A> E>
    requires scalable_abi<A> && unqualified_canonical_bwnot<basic_vector<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val) noexcept {
        return bwnot(internal::abi<A>, val);
    }

    template <extended_vector T>
    requires unqualified_bwnot<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val) noexcept {
        if constexpr (unqualified_extended_bwnot<T>) {
            return bwnot(val);
        } else {
            return operator()(dx::to_canonical(val));
        }
    }

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
    requires unqualified_mask_bwnot<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val) noexcept {
        if constexpr (unqualified_extended_mask_bwnot<T>) {
            return bwnot(val);
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

    template <simd_vector S, simd_mask Mask, simd_vector T>
    requires (extended_vector<S> || extended_mask<Mask> ||
                 extended_vector<T>) &&
        maskable_args<S, Mask, T> && extended_mbwnot<S, Mask, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, Mask mask, T val) noexcept {
        if constexpr (unqualified_extended_mbwnot<S, Mask, T>) {
            return bwnot(src, mask, val);
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
        if constexpr (unqualified_canonical_mbwnot<dx::zero_t,
                          basic_mask<ME, A>, basic_vector<E, A>>) {
            if consteval {
                return internal::masked<bwnot_t>(mask, val);
            } else {
                return bwnot(internal::abi<A>, dx::zero, mask, val);
            }
        } else {
            return operator()(dx::zero_v<basic_vector<E, A>>, mask, val);
        }
    }

    template <simd_abi MA, simd_element_for<MA> E, simd_abi TA,
        simd_element_for<TA> E>
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
        extended_mbwnot<dx::zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val) noexcept {
        if constexpr (unqualified_extended_mbwnot<dx::zero_t, M, T>) {
            return bwnot(mask, val);
        } else {
            return operator()(dx::to_canonical(mask), dx::to_canonical(val));
        }
    }

    template <simd_mask M, simd_vector T>
    requires requires(M mask, T val) { bwnot_t::operator()(mask, val); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t, M mask, T val) noexcept {
        return operator()(mask, val);
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        const_mask_for<basic_vector<E, A>> Mask>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> src, Mask mask, basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_imbwnot<basic_vector<E, A>, Mask,
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
        const_mask_for<basic_vector<E, SA>> Mask, simd_abi InA>
    requires (different_from<SA, InA> || scalable_abi<SA> ||
                 scalable_abi<InA>) &&
        simd_element_for<E, InA> &&
        imm_maskable_args<basic_vector<E, SA>, basic_vector<E, InA>> &&
        unqualified_canonical_imbwnot<basic_vector<E, SA>, Mask,
            basic_vector<E, InA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, SA> operator()(
        basic_vector<E, SA> src, Mask mask, basic_vector<E, InA> val) noexcept {
        return bwnot(internal::abi<SA>, src,
            dx::to_compatible_const_mask<basic_vector<E, SA>>(mask), val);
    }

    template <simd_vector S, const_mask_for<S> Mask, simd_vector T>
    requires (extended_vector<S> || extended_vector<T>) &&
        imm_maskable_args<S, T> && extended_imbwnot<S, Mask, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, Mask mask, T val) noexcept {
        if constexpr (unqualified_extended_mbwnot<S, Mask, T>) {
            return bwnot(src, dx::to_compatible_const_mask<S>(mask), val);
        } else {
            return operator()(dx::to_canonical(src),
                dx::to_compatible_const_mask<S>(mask), dx::to_canonical(val));
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        const_mask_for<basic_vector<E, A>> Mask>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        Mask mask, basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_imbwnot<dx::zero_t, Mask,
                          basic_vector<E, A>>) {
            if consteval {
                return internal::masked<bwnot_t>(mask, val);
            } else {
                return bwnot(internal::abi<A>, dx::zero,
                    dx::to_compatible_const_mask<basic_vector<E, A>>(mask),
                    val);
            }
        } else {
            return operator()(dx::zero_v<basic_vector<E, A>>, mask, val);
        }
    }

    template <simd_abi TA, simd_element_for<TA> E,
        const_mask_for<basic_vector<E, TA>> M>
    requires scalable_abi<TA> && simd_element_for<E, TA> &&
        imm_zmaskable_args<basic_vector<E, TA>> &&
        unqualified_canonical_imbwnot<dx::zero_t, M, basic_vector<E, TA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, TA> operator()(
        M mask, basic_vector<E, TA> val) noexcept {
        return bwnot(internal::abi<TA>, dx::zero,
            dx::to_compatible_const_mask<basic_vector<E, TA>>(mask), val);
    }

    template <extended_vector T, const_mask_for<T> M>
    requires imm_zmaskable_args<T> && extended_imbwnot<dx::zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val) noexcept {
        if constexpr (unqualified_extended_imbwnot<dx::zero_t, M, T>) {
            return bwnot(dx::zero, dx::to_compatible_const_mask<T>(mask), val);
        } else {
            return operator()(
                dx::to_compatible_const_mask<T>(mask), dx::to_canonical(val));
        }
    }

    template <simd_vector T, const_mask_for<T> M>
    requires requires(M mask, T val) { bwnot_t::operator()(mask, val); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t, M mask, T val) noexcept {
        return operator()(mask, val);
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::bwnot_t bwnot{};
} // namespace cpo
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
