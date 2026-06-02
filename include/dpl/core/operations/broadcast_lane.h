// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/operations/masked.h"
#include "dpl/core/operations/permute.h"

#if !DPL_MODULES
#  include "dpl/core/basic/broadcast.h"
#  include "dpl/core/basic/immediate.h"
#  include "dpl/core/basic/to_canonical.h"
#  include "dpl/core/concepts/common_size_with.h"
#  include "dpl/core/concepts/decayable.h"
#  include "dpl/core/concepts/operation_category.h"
#  include "dpl/core/concepts/simd_equivalence.h"
#  include "dpl/core/type_traits/iota_sequence.h"
#  include "dpl/std/utility/sequence.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void broadcast_lane(...) noexcept = delete;

template <size_t I>
struct broadcast_lanei_t;
struct broadcast_lane_t;

template <typename T, typename N>
concept unqualified_canonical_broadcast_lanei = requires(T val, N idx) {
    {
        broadcast_lane(internal::abi<T>, val, idx)
    } -> canonical_permute_result<T>;
};

template <typename T, typename N>
concept unqualified_extended_broadcast_lanei = requires(T val, N idx) {
    { broadcast_lane(val, idx) } -> permute_result<T>;
};

template <typename T, typename N>
concept unqualified_broadcast_lanei =
    unqualified_extended_broadcast_lanei<T, N> ||
    (decayable_vector_for<T, operation_category::lane_permutation> &&
        regular_invocable<broadcast_lanei_t<N::value>, T>);

template <typename S, typename C, typename L, typename R,
    typename A = common_abi_t<C, L>>
concept unqualified_canonical_mbroadcast_lanei =
    requires(S src, C mask, L val, R idx) {
        {
            broadcast_lane(internal::abi<A>, src, mask, val, idx)
        } -> equivalent_simd_as<canonical_if_zero_t<S, L, A>>;
    };

template <typename S, typename C, typename L, typename R,
    typename A = common_abi_t<C, L>>
concept unqualified_extended_mbroadcast_lanei =
    requires(S src, C mask, L val, R idx) {
        {
            broadcast_lane(src, mask, val, idx)
        } -> equivalent_simd_as<canonical_if_zero_t<S, L, A>>;
    };

template <typename S, typename C, typename L, typename R,
    typename A = common_abi_t<L, C>>
concept decayable_mbroadcast_lanei =
    decayable_vector_for<canonical_if_zero_t<S, L, A>,
        operation_category::lane_agnostic> &&
    decayable_vector_for<L, operation_category::lane_agnostic> &&
    decayable_mask_for<C, operation_category::lane_agnostic> &&
    requires(broadcast_lanei_t<R::value> op, canonical_or_zero_t<S, L, A> src,
        canonical_type_t<C> mask, canonical_type_t<L> val,
        R idx) { op(src, mask, val, idx); };

template <typename S, typename C, typename L, typename R,
    typename A = common_abi_t<L, C>>
concept extended_mbroadcast_lanei =
    unqualified_extended_mbroadcast_lanei<S, C, L, R, A> ||
    decayable_mbroadcast_lanei<S, C, L, R, A>;

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<canonical_if_zero_t<S, L>, L>>
concept unqualified_canonical_imbroadcast_lanei = requires(
    S src, M mask, L val, R idx) {
    {
        broadcast_lane(internal::abi<A>, src,
            internal::to_const_mask<A, broadcast_lanei_t<R::value>, S, L, R>(
                mask),
            val, idx)
    } -> equivalent_simd_as<canonical_if_zero_t<S, L, A>>;
};

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<canonical_if_zero_t<S, L>, L>>
concept unqualified_extended_imbroadcast_lanei = requires(
    S src, M mask, L val, R idx) {
    {
        broadcast_lane(src,
            internal::to_const_mask<A, broadcast_lanei_t<R::value>, S, L, R>(
                mask),
            val, idx)
    } -> equivalent_simd_as<canonical_if_zero_t<S, L, A>>;
};

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<canonical_if_zero_t<S, L>, L>>
concept decayable_imbroadcast_lanei =
    decayable_vector_for<canonical_if_zero_t<S, L>,
        operation_category::lane_agnostic> &&
    decayable_vector_for<L, operation_category::lane_agnostic> &&
    requires(broadcast_lanei_t<R::value> broadcast_lane,
        canonical_or_zero_t<S, L, A> src, M mask, canonical_type_t<L> val,
        R idx) { broadcast_lane(src, mask, val, idx); };

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<canonical_if_zero_t<S, L>, L>>
concept extended_imbroadcast_lanei =
    unqualified_extended_imbroadcast_lanei<S, M, L, R, A> ||
    decayable_imbroadcast_lanei<S, M, L, R, A>;

template <typename T>
concept unqualified_canonical_broadcast_lane = requires(T val, size_t idx) {
    {
        broadcast_lane(internal::abi<T>, val, idx)
    } -> canonical_permute_result<T>;
};

template <typename T>
concept unqualified_extended_broadcast_lane = requires(T val, size_t idx) {
    { broadcast_lane(val, idx) } -> permute_result<T>;
};

template <typename T>
concept unqualified_broadcast_lane = unqualified_extended_broadcast_lane<T> ||
    (decayable_vector_for<T, operation_category::lane_permutation> &&
        regular_invocable<broadcast_lane_t, T>);

template <typename S, typename C, typename T, typename A = common_abi_t<C, T>>
concept unqualified_canonical_mbroadcast_lane =
    requires(S src, C mask, T val, size_t idx) {
        {
            broadcast_lane(internal::abi<A>, src, mask, val, idx)
        } -> equivalent_simd_as<canonical_if_zero_t<S, T, A>>;
    };

template <typename S, typename C, typename T, typename A = common_abi_t<C, T>>
concept unqualified_extended_mbroadcast_lane =
    requires(S src, C mask, T val, size_t idx) {
        {
            broadcast_lane(src, mask, val, idx)
        } -> equivalent_simd_as<canonical_if_zero_t<S, T, A>>;
    };

template <typename S, typename C, typename T, typename A = common_abi_t<T, C>>
concept decayable_mbroadcast_lane =
    decayable_vector_for<canonical_if_zero_t<S, T, A>,
        operation_category::lane_agnostic> &&
    decayable_vector_for<T, operation_category::lane_agnostic> &&
    decayable_mask_for<C, operation_category::lane_agnostic> &&
    requires(broadcast_lane_t broadcast_lane, canonical_or_zero_t<S, T, A> s,
        canonical_type_t<C> c,
        canonical_type_t<T> t) { broadcast_lane(s, c, t); };

template <typename S, typename C, typename T, typename A = common_abi_t<T, C>>
concept extended_mbroadcast_lane =
    unqualified_extended_mbroadcast_lane<S, C, T, A> ||
    decayable_mbroadcast_lane<S, C, T, A>;

template <typename S, typename M, typename T,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept unqualified_canonical_imbroadcast_lane = requires(
    S src, M mask, T val, size_t idx) {
    {
        broadcast_lane(internal::abi<A>, src,
            internal::to_const_mask<A, broadcast_lane_t, S, T, size_t>(mask),
            val, idx)
    } -> equivalent_simd_as<canonical_if_zero_t<S, T, A>>;
};

template <typename S, typename M, typename T,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept unqualified_extended_imbroadcast_lane = requires(
    S src, M mask, T val, size_t idx) {
    {
        broadcast_lane(src,
            internal::to_const_mask<A, broadcast_lane_t, S, T, size_t>(mask),
            val, idx)
    } -> equivalent_simd_as<canonical_if_zero_t<S, T, A>>;
};

template <typename S, typename M, typename T,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept decayable_imbroadcast_lane =
    decayable_vector_for<canonical_if_zero_t<S, T>,
        operation_category::lane_agnostic> &&
    decayable_vector_for<T, operation_category::lane_agnostic> &&
    requires(broadcast_lane_t broadcast_lane, canonical_or_zero_t<S, T, A> src,
        M mask, canonical_type_t<T> val,
        size_t idx) { broadcast_lane(src, mask, val, idx); };

template <typename S, typename M, typename T,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept extended_imbroadcast_lane =
    unqualified_extended_imbroadcast_lane<S, M, T, A> ||
    decayable_imbroadcast_lane<S, M, T, A>;

struct broadcast_lane_t {
private:
    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_vector<E, A>
        DPL_VECTORCALL fallback(basic_vector<E, A> arg, size_t idx) noexcept {
        using sint = signed_representation_t<E>;
        return dx::permute(arg, dx::broadcast<sint, A>(static_cast<sint>(idx)));
    }

    template <typename E, typename A, size_t I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_vector<E, A>
        DPL_VECTORCALL fallbacki(
            basic_vector<E, A> arg, immediate<I> idx) noexcept {
        if constexpr (I >= simd_abi_traits<E, A>::size) {
            return dx::broadcast<E, A>(dx::zero);
        } else {
            return [&]<size_t... Is>(index_sequence<Is...>) {
                return permutei_t<((Is / Is) * I)...>::operator()(arg);
            }(iota_sequence<E, A>);
        }
    }

public:
    template <fixed_width_abi A, simd_element_for<A> E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val, size_t idx) noexcept {
        if constexpr (unqualified_canonical_broadcast_lane<
                          basic_vector<E, A>>) {
            if consteval {
                return fallback(val, idx);
            } else {
                return broadcast_lane(internal::abi<A>, val, idx);
            }
        } else {
            return fallback(val, idx);
        }
    }

    template <scalable_abi A, simd_element_for<A> E>
    requires unqualified_canonical_broadcast_lane<basic_vector<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val, size_t idx) noexcept {
        return broadcast_lane(internal::abi<A>, val, idx);
    }

    template <extended_vector T>
    requires unqualified_broadcast_lane<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val, size_t idx) noexcept {
        if constexpr (unqualified_extended_broadcast_lane<T>) {
            return broadcast_lane(val, idx);
        } else {
            return operator()(dx::to_canonical(val), idx);
        }
    }
    ///

    template <fixed_width_abi A, simd_element_for<A> E,
        integral_constant_like I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val, I idx) noexcept {
        if constexpr (unqualified_canonical_broadcast_lanei<basic_vector<E, A>,
                          I>) {
            if consteval {
                return broadcast_lane_t::fallbacki(val, idx);
            } else {
                return broadcast_lane(internal::abi<A>, val, idx);
            }
        } else {
            return broadcast_lane_t::fallbacki(val, idx);
        }
    }

    template <extended_vector T, integral_constant_like I>
    requires unqualified_broadcast_lanei<T, I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val, I idx) noexcept {
        if constexpr (unqualified_extended_broadcast_lanei<T, I>) {
            return broadcast_lane(val, idx);
        } else {
            return operator()(dx::to_canonical(val), idx);
        }
    }
    ///

    template <fixed_width_abi A, simd_element_for<A> E, simd_element_for<A> ME>
    requires common_size_with<E, ME>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_vector<E, A> pass,
        basic_mask<ME, A> mask, basic_vector<E, A> val, size_t idx) noexcept {
        if constexpr (unqualified_canonical_mbroadcast_lane<basic_vector<E, A>,
                          basic_mask<ME, A>, basic_vector<E, A>>) {
            if consteval {
                return internal::masked<broadcast_lane_t>(pass, mask, val, idx);
            } else {
                return broadcast_lane(internal::abi<A>, pass, mask, val, idx);
            }
        } else {
            return internal::masked<broadcast_lane_t>(pass, mask, val, idx);
        }
    }

    template <simd_abi MA, simd_element_for<MA> ME, simd_abi TA,
        simd_element_for<TA> E>
    requires common_size_with<E, ME> &&
        (different_from<MA, TA> || scalable_abi<MA> || scalable_abi<TA>) &&
        maskable_args<basic_vector<E, MA>, basic_mask<ME, MA>,
            basic_vector<E, TA>> &&
        unqualified_canonical_mbroadcast_lane<basic_vector<E, MA>,
            basic_mask<ME, MA>, basic_vector<E, TA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, MA> operator()(basic_vector<E, MA> pass,
        basic_mask<ME, MA> mask, basic_vector<E, TA> val, size_t idx) noexcept {
        return broadcast_lane(internal::abi<MA>, pass, mask, val, idx);
    }

    template <simd_vector S, simd_mask M, simd_vector T>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        maskable_args<S, M, T> && extended_mbroadcast_lane<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        S pass, M mask, T arg, size_t idx) noexcept {
        if constexpr (unqualified_extended_mbroadcast_lane<S, M, T>) {
            return broadcast_lane(pass, mask, arg, idx);
        } else {
            return operator()(dx::to_canonical(pass), dx::to_canonical(mask),
                dx::to_canonical(arg), idx);
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E, simd_element_for<A> ME>
    requires common_size_with<E, ME>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_mask<ME, A> mask, basic_vector<E, A> val, size_t idx) noexcept {
        if constexpr (unqualified_canonical_mbroadcast_lane<zero_t,
                          basic_mask<ME, A>, basic_vector<E, A>>) {
            if consteval {
                return internal::masked<broadcast_lane_t>(mask, val, idx);
            } else {
                return broadcast_lane(
                    internal::abi<A>, dx::zero, mask, val, idx);
            }
        } else {
            return operator()(dx::zero_v<basic_vector<E, A>>, mask, val, idx);
        }
    }

    template <simd_abi MA, simd_element_for<MA> ME, simd_abi TA,
        simd_element_for<TA> E>
    requires common_size_with<E, ME> &&
        (different_from<MA, TA> || scalable_abi<MA> || scalable_abi<TA>) &&
        zmaskable_args<basic_mask<ME, MA>, basic_vector<E, TA>> &&
        unqualified_canonical_mbroadcast_lane<dx::zero_t, basic_mask<ME, MA>,
            basic_vector<E, TA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        basic_mask<ME, MA> mask, basic_vector<E, TA> val, size_t idx) noexcept {
        return broadcast_lane(internal::abi<MA>, dx::zero, mask, val, idx);
    }

    template <simd_mask M, simd_vector T>
    requires (extended_mask<M> || extended_vector<T>) && zmaskable_args<M, T> &&
        extended_mbroadcast_lane<zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T arg, size_t idx) noexcept {
        if constexpr (unqualified_extended_mbroadcast_lane<dx::zero_t, M, T>) {
            return broadcast_lane(mask, arg, idx);
        } else {
            return operator()(
                dx::to_canonical(mask), dx::to_canonical(arg), idx);
        }
    }

    template <simd_mask M, simd_vector T>
    requires requires(M mask, T arg, size_t idx) {
        broadcast_lane_t::operator()(mask, arg, idx);
    }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t, M mask, T arg, size_t idx) noexcept {
        return operator()(mask, arg, idx);
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        const_mask_for<basic_vector<E, A>> M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_vector<E, A> pass,
        M mask, basic_vector<E, A> val, size_t idx) noexcept {
        if constexpr (unqualified_canonical_imbroadcast_lane<basic_vector<E, A>,
                          M, basic_vector<E, A>>) {
            if consteval {
                return internal::masked<broadcast_lane_t>(pass, mask, val, idx);
            } else {
                return broadcast_lane(internal::abi<A>, pass,
                    dx::to_compatible_const_mask<basic_vector<E, A>>(mask), val,
                    idx);
            }
        } else {
            return internal::masked<broadcast_lane_t>(pass, mask, val, idx);
        }
    }

    template <simd_abi SA, simd_element_for<SA> E,
        const_mask_for<basic_vector<E, SA>> M, simd_abi TA>
    requires (different_from<SA, TA> || scalable_abi<SA> || scalable_abi<TA>) &&
        simd_element_for<E, TA> &&
        imm_maskable_args<basic_vector<E, SA>, basic_vector<E, TA>> &&
        unqualified_canonical_imbroadcast_lane<basic_vector<E, SA>, M,
            basic_vector<E, TA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, SA> operator()(basic_vector<E, SA> pass,
        M mask, basic_vector<E, TA> val, size_t idx) noexcept {
        return broadcast_lane(internal::abi<SA>, pass,
            dx::to_compatible_const_mask<basic_vector<E, SA>>(mask), val, idx);
    }

    template <simd_vector S, const_mask_for<S> M, simd_vector T>
    requires (extended_vector<S> || extended_vector<T>) &&
        imm_maskable_args<S, T> && extended_imbroadcast_lane<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        S pass, M mask, T arg, size_t idx) noexcept {
        if constexpr (unqualified_extended_mbroadcast_lane<S, M, T>) {
            return broadcast_lane(
                pass, dx::to_compatible_const_mask<S>(mask), arg, idx);
        } else {
            return operator()(dx::to_canonical(pass),
                dx::to_compatible_const_mask<S>(mask), dx::to_canonical(arg),
                idx);
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        const_mask_for<basic_vector<E, A>> M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        M mask, basic_vector<E, A> val, size_t idx) noexcept {
        if constexpr (unqualified_canonical_imbroadcast_lane<zero_t, M,
                          basic_vector<E, A>>) {
            if consteval {
                return internal::masked<broadcast_lane_t>(mask, val, idx);
            } else {
                return broadcast_lane(internal::abi<A>, dx::zero,
                    dx::to_compatible_const_mask<basic_vector<E, A>>(mask), val,
                    idx);
            }
        } else {
            return operator()(dx::zero_v<basic_vector<E, A>>, mask, val, idx);
        }
    }

    template <simd_abi TA, simd_element_for<TA> E,
        const_mask_for<basic_vector<E, TA>> M>
    requires scalable_abi<TA> && simd_element_for<E, TA> &&
        imm_zmaskable_args<basic_vector<E, TA>> &&
        unqualified_canonical_imbroadcast_lane<zero_t, M, basic_vector<E, TA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, TA> operator()(
        M mask, basic_vector<E, TA> val, size_t idx) noexcept {
        return broadcast_lane(internal::abi<TA>, dx::zero,
            dx::to_compatible_const_mask<basic_vector<E, TA>>(mask), val, idx);
    }

    template <extended_vector T, const_mask_for<T> M>
    requires imm_zmaskable_args<T> && extended_imbroadcast_lane<zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T arg, size_t idx) noexcept {
        if constexpr (unqualified_extended_mbroadcast_lane<zero_t, M, T>) {
            return broadcast_lane(
                dx::zero, dx::to_compatible_const_mask<T>(mask), arg, idx);
        } else {
            return operator()(dx::to_compatible_const_mask<T>(mask),
                dx::to_canonical(arg), idx);
        }
    }

    template <simd_vector T, const_mask_for<T> M>
    requires requires(M mask, T arg, size_t idx) {
        broadcast_lane_t::operator()(mask, arg, idx);
    }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t, M mask, T arg, size_t idx) noexcept {
        return operator()(mask, arg, idx);
    }

    ///
    template <fixed_width_abi A, simd_element_for<A> E, common_size_with<E> ME,
        integral_constant_like I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_vector<E, A> pass,
        basic_mask<ME, A> mask, basic_vector<E, A> val, I idx) noexcept {
        if constexpr (unqualified_canonical_mbroadcast_lanei<basic_vector<E, A>,
                          basic_mask<ME, A>, basic_vector<E, A>, I>) {
            if consteval {
                return internal::masked<broadcast_lane_t>(pass, mask, val, idx);
            } else {
                return broadcast_lane(internal::abi<A>, pass, mask, val, idx);
            }
        } else {
            return internal::masked<broadcast_lane_t>(pass, mask, val, idx);
        }
    }

    template <simd_abi MA, simd_element_for<MA> ME, simd_abi TA,
        simd_element_for<TA> E, integral_constant_like I>
    requires common_size_with<E, ME> &&
        (different_from<MA, TA> || scalable_abi<MA> || scalable_abi<TA>) &&
        maskable_args<basic_vector<E, MA>, basic_mask<ME, MA>,
            basic_vector<E, TA>> &&
        unqualified_canonical_mbroadcast_lanei<basic_vector<E, MA>,
            basic_mask<ME, MA>, basic_vector<E, TA>, I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, MA> operator()(basic_vector<E, MA> pass,
        basic_mask<ME, MA> mask, basic_vector<E, TA> val, I idx) noexcept {
        return broadcast_lane(internal::abi<MA>, pass, mask, val, idx);
    }

    template <simd_vector S, simd_mask M, simd_vector T,
        integral_constant_like I>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        maskable_args<S, M, T> && extended_mbroadcast_lanei<S, M, T, I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S pass, M mask, T arg, I idx) noexcept {
        if constexpr (unqualified_extended_mbroadcast_lanei<S, M, T, I>) {
            return broadcast_lane(pass, mask, arg, idx);
        } else {
            return operator()(dx::to_canonical(pass), dx::to_canonical(mask),
                dx::to_canonical(arg), idx);
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E, common_size_with<E> ME,
        integral_constant_like I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_mask<ME, A> mask, basic_vector<E, A> val, I idx) noexcept {
        if constexpr (unqualified_canonical_mbroadcast_lanei<zero_t,
                          basic_mask<ME, A>, basic_vector<E, A>, I>) {
            if consteval {
                return internal::masked<broadcast_lane_t>(mask, val, idx);
            } else {
                return broadcast_lane(
                    internal::abi<A>, dx::zero, mask, val, idx);
            }
        } else {
            return operator()(dx::zero_v<basic_vector<E, A>>, mask, val, idx);
        }
    }

    template <simd_abi MA, simd_element_for<MA> ME, simd_abi TA,
        simd_element_for<TA> E, integral_constant_like I>
    requires common_size_with<E, ME> &&
        (different_from<MA, TA> || scalable_abi<MA> || scalable_abi<TA>) &&
        zmaskable_args<basic_mask<ME, MA>, basic_vector<E, TA>> &&
        unqualified_canonical_mbroadcast_lanei<dx::zero_t, basic_mask<ME, MA>,
            basic_vector<E, TA>, I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        basic_mask<ME, MA> mask, basic_vector<E, TA> val, I idx) noexcept {
        return broadcast_lane(internal::abi<MA>, dx::zero, mask, val, idx);
    }

    template <simd_mask M, simd_vector T, integral_constant_like I>
    requires (extended_mask<M> || extended_vector<T>) && zmaskable_args<M, T> &&
        extended_mbroadcast_lanei<zero_t, M, T, I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T arg, I idx) noexcept {
        if constexpr (unqualified_extended_mbroadcast_lanei<dx::zero_t, M, T,
                          I>) {
            return broadcast_lane(mask, arg, idx);
        } else {
            return operator()(
                dx::to_canonical(mask), dx::to_canonical(arg), idx);
        }
    }

    template <simd_mask M, simd_vector T, integral_constant_like I>
    requires requires(
        M mask, T arg, I idx) { broadcast_lane_t::operator()(mask, arg, idx); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t, M mask, T arg, I idx) noexcept {
        return operator()(mask, arg, idx);
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        const_mask_for<basic_vector<E, A>> M, integral_constant_like I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_vector<E, A> pass,
        M mask, basic_vector<E, A> val, I idx) noexcept {
        if constexpr (unqualified_canonical_imbroadcast_lanei<
                          basic_vector<E, A>, M, basic_vector<E, A>, I>) {
            if consteval {
                return internal::masked<broadcast_lane_t>(pass, mask, val, idx);
            } else {
                return broadcast_lane(internal::abi<A>, pass,
                    dx::to_compatible_const_mask<basic_vector<E, A>>(mask), val,
                    idx);
            }
        } else {
            return internal::masked<broadcast_lane_t>(pass, mask, val, idx);
        }
    }

    template <simd_abi SA, simd_element_for<SA> E,
        const_mask_for<basic_vector<E, SA>> M, simd_abi TA,
        integral_constant_like I>
    requires (different_from<SA, TA> || scalable_abi<SA> || scalable_abi<TA>) &&
        simd_element_for<E, TA> &&
        imm_maskable_args<basic_vector<E, SA>, basic_vector<E, TA>> &&
        unqualified_canonical_imbroadcast_lanei<basic_vector<E, SA>, M,
            basic_vector<E, TA>, I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, SA> operator()(basic_vector<E, SA> pass,
        M mask, basic_vector<E, TA> val, I idx) noexcept {
        return broadcast_lane(internal::abi<SA>, pass,
            dx::to_compatible_const_mask<basic_vector<E, SA>>(mask), val, idx);
    }

    template <simd_vector S, const_mask_for<S> M, simd_vector T,
        integral_constant_like I>
    requires (extended_vector<S> || extended_vector<T>) &&
        imm_maskable_args<S, T> && extended_imbroadcast_lanei<S, M, T, I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S pass, M mask, T arg, I idx) noexcept {
        if constexpr (unqualified_extended_mbroadcast_lanei<S, M, T, I>) {
            return broadcast_lane(
                pass, dx::to_compatible_const_mask<S>(mask), arg, idx);
        } else {
            return operator()(dx::to_canonical(pass),
                dx::to_compatible_const_mask<S>(mask), dx::to_canonical(arg),
                idx);
        }
    }

    template <fixed_width_abi A, simd_element_for<A> E,
        const_mask_for<basic_vector<E, A>> M, integral_constant_like I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        M mask, basic_vector<E, A> val, I idx) noexcept {
        if constexpr (unqualified_canonical_imbroadcast_lanei<zero_t, M,
                          basic_vector<E, A>, I>) {
            if consteval {
                return internal::masked<broadcast_lane_t>(mask, val, idx);
            } else {
                return broadcast_lane(internal::abi<A>, dx::zero,
                    dx::to_compatible_const_mask<basic_vector<E, A>>(mask), val,
                    idx);
            }
        } else {
            return operator()(dx::zero_v<basic_vector<E, A>>, mask, val, idx);
        }
    }

    template <simd_abi TA, simd_element_for<TA> E,
        const_mask_for<basic_vector<E, TA>> M, integral_constant_like I>
    requires scalable_abi<TA> && simd_element_for<E, TA> &&
        imm_zmaskable_args<basic_vector<E, TA>> &&
        unqualified_canonical_imbroadcast_lanei<zero_t, M, basic_vector<E, TA>,
            I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, TA> operator()(
        M mask, basic_vector<E, TA> val, I idx) noexcept {
        return broadcast_lane(internal::abi<TA>, dx::zero,
            dx::to_compatible_const_mask<basic_vector<E, TA>>(mask), val, idx);
    }

    template <extended_vector T, const_mask_for<T> M, integral_constant_like I>
    requires imm_zmaskable_args<T> &&
        extended_imbroadcast_lanei<zero_t, M, T, I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T arg, I idx) noexcept {
        if constexpr (unqualified_extended_mbroadcast_lanei<zero_t, M, T, I>) {
            return broadcast_lane(
                dx::zero, dx::to_compatible_const_mask<T>(mask), arg, idx);
        } else {
            return operator()(dx::to_compatible_const_mask<T>(mask),
                dx::to_canonical(arg), idx);
        }
    }

    template <simd_vector T, const_mask_for<T> M, integral_constant_like I>
    requires requires(
        M mask, T arg, I idx) { broadcast_lane_t::operator()(mask, arg, idx); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t, M mask, T arg, I idx) noexcept {
        return operator()(mask, arg, idx);
    }
};

template <size_t I>
struct broadcast_lanei_t {
    template <simd_vector T>
    requires regular_invocable<broadcast_lane_t, T, size_t>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val) noexcept {
        return broadcast_lane_t::operator()(val, imm<I>);
    }

    template <typename S, typename M, simd_vector T>
    requires regular_invocable<broadcast_lane_t, S, M, T, size_t>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, T val) noexcept {
        return broadcast_lane_t::operator()(src, mask, val, imm<I>);
    }

    template <typename M, simd_vector T>
    requires regular_invocable<broadcast_lane_t, M, T, size_t>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val) noexcept {
        return broadcast_lane_t::operator()(mask, val, imm<I>);
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT template <size_t I>
inline constexpr internal::broadcast_lanei_t<I> broadcast_lanei{};
DPL_EXPORT inline constexpr internal::broadcast_lane_t broadcast_lane{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
