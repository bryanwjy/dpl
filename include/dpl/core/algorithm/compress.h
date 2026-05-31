// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/algorithm/shift.h"
#include "dpl/core/algorithm/slide.h"

#if !DPL_MODULES
#  include "dpl/core/basic/immediate.h"
#  include "dpl/core/concepts/common_abi_with.h"
#  include "dpl/core/concepts/decayable.h"
#  include "dpl/core/concepts/operation_category.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/operations/bit.h"
#  include "dpl/core/operations/select.h"
#  include "dpl/core/type_traits/array_for.h"
#  include "dpl/core/type_traits/basic_type.h"
#  include "dpl/core/type_traits/simd_abi_traits.h"
#  include "dpl/std/concepts/invocable.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void compress(...) noexcept = delete;

struct compress_t;

template <typename S, typename M, typename T, typename A = common_abi_t<M, T>>
concept unqualified_canonical_compress = requires(S src, M mask, T val) {
    {
        compress(internal::abi<A>, src, mask, val)
    } -> equivalent_simd_as<canonical_if_zero_t<S, T, A>>;
};

template <typename S, typename M, typename T, typename A = common_abi_t<M, T>>
concept unqualified_extended_compress = requires(S src, M mask, T val) {
    {
        compress(src, mask, val)
    } -> equivalent_simd_as<canonical_if_zero_t<S, T, A>>;
};

template <typename Op, typename S, typename M, typename T,
    typename A = common_abi_t<T, M>>
concept decayable_compress = decayable_vector_for<canonical_if_zero_t<S, T, A>,
                                 operation_category::lane_agnostic> &&
    decayable_vector_for<T, operation_category::lane_agnostic> &&
    decayable_mask_for<M, operation_category::lane_agnostic> &&
    requires(canonical_or_zero_t<S, T, A> s, canonical_type_t<M> c,
        canonical_type_t<T> t) { Op::operator()(s, c, t); };

template <typename Op, typename S, typename M, typename T,
    typename A = common_abi_t<T, M>>
concept extended_compress = unqualified_extended_compress<S, M, T, A> ||
    decayable_compress<Op, S, M, T, A>;

template <typename S, typename M, typename R, typename A = common_abi_t<S, R>>
concept unqualified_canonical_compressi = requires(S src, R val) {
    {
        compress(internal::abi<A>, src, internal::select_mask<M, S, R>(), val)
    } -> equivalent_simd_as<S>;
};

template <typename S, typename M, typename R, typename A = common_abi_t<S, R>>
concept unqualified_extended_compressi = requires(S src, R val) {
    {
        compress(src, internal::select_mask<M, S, R>(), val)
    } -> equivalent_simd_as<S>;
};

template <typename S, typename M, typename R, typename A = common_abi_t<M, R>>
concept unqualified_compressi = unqualified_extended_compress<S, M, R, A> ||
    (decayable_vector_for<S, operation_category::lane_permutation> &&
        decayable_vector_for<R, operation_category::lane_permutation> &&
        regular_invocable<compress_t, canonical_type_t<S>, M,
            canonical_type_t<R>>);

struct compress_t {
private:
    template <typename EM, typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static consteval auto DPL_VECTORCALL fallback(basic_vector<E, A> src,
        basic_mask<EM, A> mask, basic_vector<E, A> val) noexcept {
        using mask_type = basic_mask<EM, A>;
        [&]<size_t I, size_t J>(
            this auto self, immediate<I> idx, immediate<J> jdx) consteval {
            if constexpr (I == simd_abi_traits<E, A>::size) {
                return;
            } else {
                if (mask[idx]) {
                    src[jdx] = val[idx];
                    self(imm<I + 1>, imm<J + 1>);
                } else {
                    self(imm<I + 1>, imm<J>);
                }
            }
        }(imm<0zu>, imm<0zu>);
        return src;
    }

    template <auto V, typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallbacki(
        basic_vector<E, A> src, basic_vector<E, A> val) noexcept {
        static constexpr auto mask =
            dx::to_compatible_const_mask<basic_vector<E, A>>(imm<V>);
        using I = signed_representation_t<E>;
        constexpr auto idx = []<size_t I>(this auto self, array_for<I, A> dst,
                                 immediate<I> idx) {
            if constexpr (I == simd_abi_traits<E, A>::size) {
                return dx::load<I, 0>(dst.data);
            } else if constexpr (mask[idx]) {
                using type = remove_cvref_t<decltype(mask)>;
                auto const j =
                    __DPL popcount(type::value << (type::width - idx));
                dst.data[j] = static_cast<I>(type::value);
                return self(self, dst, imm<I + 1>);
            } else {
                return self(self, dst, imm<I + 1>);
            }
        }(array_for<I, A>{}, imm<0zu>);

        constexpr auto selection = (1zu << dx::popcount(mask)) - 1;
        constexpr auto smask =
            dx::to_compatible_const_mask<basic_vector<E, A>>(imm<selection>);
        return dx::select(smask, dx::permute(val, idx), src);
    }

public:
    template <fixed_width_abi A, simd_element_for<A> E, simd_element_for<A> ME>
    requires common_size_with<E, ME> &&
        unqualified_canonical_compress<basic_vector<E, A>, basic_mask<ME, A>,
            basic_vector<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_vector<E, A> pass,
        basic_mask<ME, A> mask, basic_vector<E, A> val) noexcept {
        // Only callable if backend supports it for now
        // There is no way to do this efficiently without better backend support
        // or more complex primitives, e.g. in/exclusive scan,
        // masked-scatter.
        if consteval {
            return compress_t::fallback(pass, mask, val);
        } else {
            return compress(internal::abi<A>, pass, mask, val);
        }
    }

    template <simd_abi SA, simd_element_for<SA> ME, simd_abi TA,
        simd_element_for<TA> E>
    requires common_size_with<E, ME> && different_from<SA, TA> &&
        maskable_args<basic_vector<E, SA>, basic_mask<ME, SA>,
            basic_vector<E, TA>> &&
        unqualified_canonical_compress<basic_vector<E, SA>, basic_mask<ME, SA>,
            basic_vector<E, TA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, SA> operator()(basic_vector<E, SA> pass,
        basic_mask<ME, SA> mask, basic_vector<E, TA> val) noexcept {
        return compress(internal::abi<SA>, pass, mask, val);
    }

    template <simd_vector S, simd_mask M, simd_vector Arg>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<Arg>) &&
        maskable_args<S, M, Arg> && extended_compress<compress_t, S, M, Arg>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S pass, M mask, Arg arg) noexcept {
        if constexpr (unqualified_extended_compress<S, M, Arg>) {
            return compress(pass, mask, arg);
        } else {
            return operator()(dx::to_canonical(pass), dx::to_canonical(mask),
                dx::to_canonical(arg));
        }
    }

    template <simd_abi A, simd_element_for<A> E, simd_element_for<A> ME>
    requires common_size_with<E, ME>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_mask<ME, A> mask, basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_compress<zero_t, basic_mask<ME, A>,
                          basic_vector<E, A>>) {
            if consteval {
                return operator()(dx::broadcast<E, A>(dx::zero), mask, val);
            } else {
                return compress(internal::abi<A>, dx::zero, mask, val);
            }
        } else {
            return operator()(dx::broadcast<E, A>(dx::zero), mask, val);
        }
    }

    template <simd_abi TA, simd_element_for<TA> E, common_abi_with<TA> MA,
        simd_element_for<MA> ME>
    requires common_size_with<E, ME> && different_from<MA, TA> &&
        zmaskable_args<basic_mask<ME, MA>, basic_vector<E, TA>> &&
        unqualified_canonical_compress<dx::zero_t, basic_mask<ME, MA>,
            basic_vector<E, TA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        basic_mask<ME, MA> mask, basic_vector<E, TA> val) noexcept {
        return compress(internal::abi<MA>, dx::zero, mask, val);
    }

    template <simd_mask M, simd_vector T>
    requires (extended_mask<M> || extended_vector<T>) && zmaskable_args<M, T> &&
        extended_compress<compress_t, zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T arg) noexcept {
        if constexpr (unqualified_extended_compress<dx::zero_t, M, T>) {
            return compress(mask, arg);
        } else {
            return operator()(dx::to_canonical(mask), dx::to_canonical(arg));
        }
    }

    template <simd_mask M, simd_vector T>
    requires requires(M mask, T arg) { compress_t::operator()(mask, arg); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t, M mask, T arg) noexcept {
        return operator()(mask, arg);
    }

    template <typename M, simd_abi A, simd_element_for<A> E>
    requires const_mask_for<M, basic_vector<E, A>> &&
        imm_maskable_args<basic_vector<E, A>, basic_vector<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> src, M mask, basic_vector<E, A> val) noexcept {
        constexpr auto cmask =
            dx::to_compatible_const_mask<basic_vector<E, A>>(mask);
        constexpr auto V = cmask();
        if constexpr (unqualified_canonical_compressi<basic_vector<E, A>, M,
                          basic_vector<E, A>>) {
            if consteval {
                return fallbacki<V>(src, val);
            } else {
                return compress(internal::abi<A>, src, cmask, val);
            }
        } else {
            return fallbacki<V>(src, val);
        }
    }

    template <simd_abi SA, simd_element_for<SA> E, typename M, simd_abi RA>
    requires simd_element_for<E, RA> && different_from<SA, RA> &&
        const_mask_for<M, basic_vector<E, SA>> &&
        imm_maskable_args<basic_vector<E, SA>, basic_vector<E, RA>> &&
        unqualified_canonical_compressi<basic_vector<E, SA>, M,
            basic_vector<E, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        basic_vector<E, SA> src, M mask, basic_vector<E, RA> val) noexcept {
        using A = common_abi_t<SA, RA>;
        constexpr auto cmask =
            dx::to_compatible_const_mask<basic_vector<E, A>>(mask);
        return compress(internal::abi<A>, src, cmask, val);
    }

    template <fixed_width_vector S, typename M, fixed_width_vector R>
    requires common_size_with<typename S::value_type, typename R::value_type> &&
        (extended_vector<S> || extended_vector<R>) && const_mask_for<M, S> &&
        imm_maskable_args<S, R> && unqualified_compressi<S, M, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, R val) noexcept {
        if constexpr (unqualified_extended_compressi<S, M, R>) {
            constexpr auto cmask =
                dx::to_compatible_const_mask<basic_vector<E, A>>(mask);
            return compress(src, cmask, val);
        } else {
            return operator()(
                dx::to_canonical(src), mask, dx::to_canonical(val));
        }
    }

    template <simd_abi A, simd_element_for<A> E,
        const_mask_for<basic_vector<E, A>> M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> pass, M mask, basic_vector<E, A> val) noexcept {
        constexpr auto cmask =
            dx::to_compatible_const_mask<basic_vector<E, A>>(mask);
        constexpr auto V = cmask();
        if constexpr (unqualified_canonical_compressi<basic_vector<E, A>, M,
                          basic_vector<E, A>>) {
            if consteval {
                return compress_t::fallbacki<V>(pass, val);
            } else {
                return compress(internal::abi<A>, pass, cmask, val);
            }
        } else {
            return compress_t::fallbacki<V>(pass, val);
        }
    }

    template <simd_abi SA, simd_element_for<SA> E,
        const_mask_for<basic_vector<E, SA>> M, simd_abi TA>
    requires different_from<SA, TA> && simd_element_for<E, TA> &&
        imm_maskable_args<basic_vector<E, SA>, basic_vector<E, TA>> &&
        unqualified_canonical_compressi<basic_vector<E, SA>, M,
            basic_vector<E, TA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, SA> operator()(
        basic_vector<E, SA> pass, M mask, basic_vector<E, TA> val) noexcept {
        return compress(internal::abi<SA>, pass,
            dx::to_compatible_const_mask<basic_vector<E, SA>>(mask), val);
    }

    template <simd_vector S, const_mask_for<S> M, simd_vector T>
    requires (extended_vector<S> || extended_vector<T>) &&
        imm_maskable_args<S, T> && extended_compressi<compress_t, S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S pass, M mask, T val) noexcept {
        if constexpr (unqualified_extended_compress<S, M, T>) {
            return compress(pass, dx::to_compatible_const_mask<S>(mask), val);
        } else {
            return operator()(
                dx::to_canonical(pass), mask, dx::to_canonical(val));
        }
    }

    template <simd_abi A, simd_element_for<A> E,
        const_mask_for<basic_vector<E, A>> M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        M mask, basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_compressi<zero_t, M,
                          basic_vector<E, A>>) {
            if consteval {
                return operator()(dx::broadcast<E, A>(dx::zero), mask, val);
            } else {
                return compress(internal::abi<A>, dx::zero,
                    dx::to_compatible_const_mask<basic_vector<E, A>>(mask),
                    val);
            }
        } else {
            return operator()(dx::broadcast<E, A>(dx::zero), mask, val);
        }
    }

    template <extended_vector T, const_mask_for<T> M>
    requires imm_zmaskable_args<T> &&
        extended_compressi<compress_t, zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val) noexcept {
        if constexpr (unqualified_extended_compressi<zero_t, M, T>) {
            return compress(
                dx::zero, dx::to_compatible_const_mask<T>(mask), val);
        } else {
            return operator()(mask, dx::to_canonical(val));
        }
    }

    template <simd_vector T, const_mask_for<T> M>
    requires requires(M mask, T val) { compress_t::operator()(mask, val); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t, M mask, T val) noexcept {
        return operator()(mask, val);
    }
};

template <auto V>
struct compressi_t {};

template <integral auto V>
struct compressi_t<V> {
private:
    template <typename T>
    using mask_type DPL_NODEBUG = const_mask<simd_abi_traits<T>::size, V>;

public:
    template <simd_vector S, simd_vector R>
    requires requires {
        typename mask_type<S>;
        requires regular_invocable<compress_t, S, mask_type<S>, R>;
    }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, R val) noexcept {
        constexpr mask_type<S> mask{};
        return compress_t::operator()(src, mask, val);
    }

    template <simd_class S, typename R>
    requires (!simd_class<R>) && requires { typename mask_type<S>; } &&
        regular_invocable<compress_t, S, mask_type<S>, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, R val) noexcept {
        constexpr mask_type<S> mask{};
        return compress_t::operator()(src, mask, val);
    }

    template <typename S, simd_class R>
    requires (!simd_class<S>) && requires { typename mask_type<R>; } &&
        regular_invocable<compress_t, S, mask_type<R>, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, R val) noexcept {
        constexpr mask_type<R> mask{};
        return compress_t::operator()(src, mask, val);
    }
};

} // namespace datapar::internal

namespace datapar {
DPL_EXPORT template <auto V>
inline constexpr internal::compressi_t<V> compressi{};
DPL_EXPORT inline constexpr internal::compress_t compress{};
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END