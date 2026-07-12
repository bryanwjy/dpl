// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/basic/initialize.h"
#  include "dpl/core/basic/internal/abi.h"
#  include "dpl/core/basic/internal/iota_sequence.h"
#  include "dpl/core/concepts/common_size_with.h"
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/concepts/mask_compatibility.h"
#  include "dpl/core/dispatch/maskable/transform.h"
#  include "dpl/core/dispatch/operation/primitive.h"
#  include "dpl/core/immediate/immediate.h"
#  include "dpl/core/type_traits/simd_abi_type.h"
#  include "dpl/core/type_traits/simd_element_type.h"
#  include "dpl/std/concepts/integer_sequence_like.h"
#  include "dpl/std/type_traits/sequence.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void permute(...) noexcept = delete;

struct DPL_EMPTY_BASES permute_t :
    private primitive_operation_base<permute_t>,
    private maskable_transform_base<permute_t> {
    using operation_base<permute_t>::operator();
    using maskable_transform_base<permute_t>::operator();
};

template <>
struct operation_signature<permute_t> {
    template <simd_vector L, typename R>
    requires index_sequence_like<R> ||
        (canonical_vector<R> && integral<simd_element_type_t<R>>)
    static consteval void operator()(L&&, R) noexcept {}
};

template <>
struct fallback_impl<permute_t> {
    template <canonical_simd_type T, common_size_with<simd_element_type_t<T>> E,
        same_as<simd_abi_type_t<T>> A>
    requires integral<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(
        T arg, basic_vector<E, A> idx) noexcept {
        return []<size_t... Is>(
                   T arg, basic_vector<E, A> idx, index_sequence<Is...>) {
            constexpr auto simd_size = simd_abi_traits<T>::size();
            using TE = simd_element_type_t<T>;
            auto const zero = TE();
            return dx::initialize<T>(
                (idx[imm<Is>] < simd_size ? arg[idx[imm<Is>]] : zero)...);
        }(arg, idx, iota_sequence<T>);
    }

    template <canonical_simd_type T, index_sequence_like I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(T arg, I idx) noexcept {
        static_assert(I::size() <= simd_abi_traits<T>::size());
        if constexpr (I::size() < simd_abi_traits<T>::size()) {
            constexpr auto seq = __DPL to_index_sequence(idx);
            constexpr auto D = simd_abi_traits<T>::size() - I::size();
            constexpr auto all = []<size_t... Is, size_t... Js>(
                                     index_sequence<Is...>,
                                     index_sequence<Js...>) {
                return index_sequence<Is..., (sizeof...(Is) + Js)...>{};
            }(seq, make_index_sequence<D>{});
            return operator()(arg, all);
        } else {
            return __DPL apply(
                [&](auto... i) {
                    constexpr auto simd_size = simd_abi_traits<T>::size();
                    using TE = simd_element_type_t<T>;
                    auto const zero = TE();
                    return dx::initialize<T>(
                        (i < simd_size ? arg[i] : zero)...);
                },
                idx);
        }
    }
};

template <typename S, typename M, typename T, typename I>
concept unqualified_canonical_mpermute = cpo_invocable<permute_t, T, I> &&
    (!simd_type<S> || same_as<S, cpo_result_t<permute_t, T, I>>) && requires {
        {
            permute(internal::abi<cpo_result_t<permute_t, T, I>>,
                internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>(), internal::declarg<I>())
        } -> same_as<cpo_result_t<permute_t, T, I>>;
    };

template <typename T>
using launder_sequence_t DPL_NODEBUG =
    decltype(__DPL to_index_sequence(internal::declarg<T>()));

template <typename T, typename L>
concept index_vector_for = simd_vector<T> && simd_vector<L> &&
    common_abi_with<simd_abi_type_t<T>, simd_abi_type_t<L>> &&
    integral<simd_element_type_t<T>>;

template <>
struct canonical_impl<permute_t> {
private:
    template <typename T, typename A = simd_abi_type_t<T>>
    using result_t DPL_NODEBUG = basic_vector<simd_element_type_t<T>, A>;

    template <typename T, typename A = simd_abi_type_t<T>>
    using mask_t DPL_NODEBUG = basic_mask<simd_element_type_t<T>, A>;

public:
    template <simd_abi A, simd_element_for<A> E, index_sequence_like R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val, R idx) noexcept
    requires requires(
        launder_sequence_t<R> seq) { permute(internal::abi<A>, val, seq); }
    {
        constexpr auto seq = __DPL to_index_sequence(idx);
        return permute(internal::abi<A>, val, seq);
    }

    template <canonical_vector T, index_sequence_like R>
    requires unqualified_canonical_mpermute<result_t<T>, mask_t<T>, T,
        launder_sequence_t<R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<T> operator()(
        result_t<T> src, mask_t<T> mask, T val, R idx) noexcept {
        constexpr auto seq = __DPL to_index_sequence(idx);
        return permute(internal::abi<result_t<T>>, src, mask, val, seq);
    }

    template <canonical_vector T, const_mask_for<T> M, index_sequence_like R>
    requires canonical_vector<T> &&
        unqualified_canonical_mpermute<result_t<T>, launder_cmask_t<T, M>, T,
            launder_sequence_t<R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<T> operator()(
        result_t<T> src, M cmask, T val, R idx) noexcept {
        constexpr auto seq = __DPL to_index_sequence(idx);
        return permute(internal::abi<result_t<T>>, src,
            dx::to_const_mask<T>(cmask), val, seq);
    }

    template <canonical_vector T, index_sequence_like R>
    requires unqualified_canonical_mpermute<dx::zero_t, mask_t<T>, T,
        launder_sequence_t<R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<T> operator()(
        dx::zero_t zero, mask_t<T> mask, T val, R idx) noexcept {
        constexpr auto seq = __DPL to_index_sequence(idx);
        return permute(internal::abi<result_t<T>>, zero, mask, val, seq);
    }

    template <canonical_vector T, const_mask_for<T> M, index_sequence_like R>
    requires canonical_vector<T> &&
        unqualified_canonical_mpermute<dx::zero_t, launder_cmask_t<T, M>, T,
            launder_sequence_t<R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<T> operator()(
        dx::zero_t zero, M cmask, T val, R idx) noexcept {
        constexpr auto seq = __DPL to_index_sequence(idx);
        return permute(
            internal::abi<T>, zero, dx::to_const_mask<T>(cmask), val, seq);
    }

    ///

    template <canonical_vector T, index_vector_for<T> R,
        typename A = common_abi_t<T, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<T, A> operator()(T val, R idx) noexcept
    requires requires { permute(internal::abi<A>, val, idx); }
    {
        return permute(internal::abi<A>, val, idx);
    }

    template <canonical_vector T, index_vector_for<T> R,
        typename A = common_abi_t<T, R>>
    requires unqualified_canonical_mpermute<result_t<T, A>, mask_t<T, A>, T, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<T, A> operator()(
        result_t<T, A> src, mask_t<T, A> mask, T val, R idx) noexcept {
        return permute(internal::abi<A>, src, mask, val, idx);
    }

    template <canonical_vector T, index_vector_for<T> R,
        result_cmask_for<permute_t, T, R> M, typename A = common_abi_t<T, R>>
    requires unqualified_canonical_mpermute<cpo_result_t<permute_t, T, R>,
        launder_cmask_t<result_t<T, A>, M>, T, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<T, A> operator()(
        result_t<T, A> src, M cmask, T val, R idx) noexcept {
        return permute(internal::abi<A>, src,
            dx::to_const_mask<result_t<T, A>>(cmask), val, idx);
    }

    template <canonical_vector T, index_vector_for<T> R,
        typename A = common_abi_t<T, R>>
    requires unqualified_canonical_mpermute<dx::zero_t, mask_t<T, A>, T, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<T, A> operator()(
        dx::zero_t zero, mask_t<T, A> mask, T val, R idx) noexcept {
        return permute(internal::abi<A>, zero, mask, val, idx);
    }

    template <canonical_vector T, index_vector_for<T> R,
        result_cmask_for<permute_t, T, R> M, typename A = common_abi_t<T, R>>
    requires unqualified_canonical_mpermute<dx::zero_t,
        launder_cmask_t<cpo_result_t<permute_t, T, R>, M>, T, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<T, A> operator()(
        dx::zero_t zero, M cmask, T val, R idx) noexcept {
        return permute(internal::abi<A>, zero,
            dx::to_const_mask<result_t<T, A>>(cmask), val, idx);
    }
};

template <typename T, typename I, typename A = common_abi_t<T, I>>
concept unqualified_extended_permute = requires {
    {
        permute(internal::declarg<T>(), internal::declarg<I>())
    } -> vector_with_common_abi<A>;
};

template <typename S, typename M, typename T, typename I>
concept unqualified_extended_mpermute = cpo_invocable<permute_t, T, I> &&
    (!simd_type<S> ||
        equivalent_vector_with<S, cpo_result_t<permute_t, T, I>>) &&
    requires {
        {
            permute(internal::abi<cpo_result_t<permute_t, T, I>>,
                internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>(), internal::declarg<I>())
        } -> equivalent_vector_with<cpo_result_t<permute_t, T, I>>;
    };

template <>
struct extended_impl<permute_t> {
public:
    template <extended_vector L, index_sequence_like R>
    requires unqualified_extended_permute<L, launder_sequence_t<R>,
        simd_abi_type_t<L>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& val, R idx) {
        return permute(__DPL forward<L>(val), __DPL to_index_sequence(idx));
    }

    template <simd_vector S, exact_mask_for<S> M, common_vector_with<S> T,
        index_sequence_like R>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mpermute<S, M, T, launder_sequence_t<R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M mask, T&& val, R idx) {
        return permute( __DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val), __DPL to_index_sequence(idx));
    }

    template <simd_vector S, const_mask_for<S> M, common_vector_with<S> T,
        index_sequence_like R>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_mpermute<S, launder_cmask_t<S, M>, T,
            launder_sequence_t<R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M cmask, T&& val, R idx) {
        return permute( __DPL forward<S>(src), dx::to_const_mask<S>(cmask),
            __DPL forward<T>(val), __DPL to_index_sequence(idx));
    }

    template <simd_vector T, simd_mask M, index_sequence_like R>
    requires (extended_mask<M> || extended_vector<T>) &&
        cpo_invocable<permute_t, T, R> &&
        exact_mask_for<M, cpo_result_t<permute_t, T, R>> &&
        unqualified_extended_mpermute<dx::zero_t, M, T, launder_sequence_t<R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, T&& val, R idx) {
        return permute(zero, __DPL forward<M>(mask), __DPL forward<T>(val),
            __DPL to_index_sequence(idx));
    }

    template <extended_vector T, index_sequence_like R,
        result_cmask_for<permute_t, T, R> M>
    requires unqualified_extended_mpermute<dx::zero_t,
        launder_cmask_t<cpo_result_t<permute_t, T, R>, M>, T,
        launder_sequence_t<R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M cmask, T&& val, R idx) {
        return permute(zero,
            dx::to_const_mask<cpo_result_t<permute_t, T, R>>(cmask),
            __DPL forward<T>(val), __DPL to_index_sequence(idx));
    }

    ///
    template <extended_vector L, index_vector_for<L> R>
    requires unqualified_extended_permute<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& val, R idx) {
        return permute(__DPL forward<L>(val), idx);
    }

    template <simd_vector S, exact_mask_for<S> M, common_vector_with<S> T,
        index_vector_for<T> R>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mpermute<S, M, T, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M mask, T&& val, R idx) {
        return permute( __DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val), idx);
    }

    template <simd_vector S, const_mask_for<S> M, common_vector_with<S> T,
        index_vector_for<T> R>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_mpermute<S, launder_cmask_t<S, M>, T, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M cmask, T&& val, R idx) {
        return permute( __DPL forward<S>(src), dx::to_const_mask<S>(cmask),
            __DPL forward<T>(val), idx);
    }

    template <simd_vector T, simd_mask M, index_vector_for<T> R>
    requires (extended_mask<M> || extended_vector<T>) &&
        cpo_invocable<permute_t, T, R> &&
        exact_mask_for<M, cpo_result_t<permute_t, T, R>> &&
        unqualified_extended_mpermute<dx::zero_t, M, T, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, T&& val, R idx) {
        return permute(
            zero, __DPL forward<M>(mask), __DPL forward<T>(val), idx);
    }

    template <extended_vector T, index_vector_for<T> R,
        result_cmask_for<permute_t, T, R> M>
    requires unqualified_extended_mpermute<dx::zero_t,
        launder_cmask_t<cpo_result_t<permute_t, T, R>, M>, T, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M cmask, T&& val, R idx) {
        return permute(zero,
            dx::to_const_mask<cpo_result_t<permute_t, T, R>>(cmask),
            __DPL forward<T>(val), idx);
    }
};

template <size_t... Is>
struct permutei_t {
private:
    using seq_t DPL_NODEBUG = index_sequence<Is...>;
    static constexpr seq_t seq{};

public:
    template <typename... Ts>
    requires cpo_invocable<permute_t, Ts..., seq_t>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(Ts&&... args) noexcept(
        (... && (!simd_type<Ts> || canonical_simd_type<Ts>))) {
        return permute_t::operator()(__DPL forward<Ts>(args)..., seq);
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::permute_t permute{};
template <size_t... Is>
inline constexpr internal::permutei_t<Is...> permutei{};
} // namespace cpo
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
