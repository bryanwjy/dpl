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
#  include "dpl/std/type_traits/remove_cvref.h"
#  include "dpl/std/type_traits/sequence.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void permute(...) noexcept = delete;

/**
 * @brief Rearranges lanes of a vector according to an index.
 *
 * Lane @c i of the result holds <tt>val[idx[i]]</tt>.
 *
 * The index may be a vector with an signed-integral element type, or an
 * @c index_sequence_like carrying the pattern as a non-type template parameter.
 * Prefer the latter where the pattern is known at compile time: backends can
 * select an immediate-operand instruction. @ref permutei is a convenience alias
 * for that form.
 *
 * Indices are element-granular: an index of @c n selects lane @c n regardless
 * of element width. Backends translate to the granularity their instruction
 * requires.
 *
 * @param val Source vector supplying lane values.
 * @param idx Index selecting a source lane per output lane.
 * @return A vector whose lane @c i is <tt>val[idx[i]]</tt>.
 *
 * @pre Every index lies in <tt>[0, size)</tt>.
 * @pre @c size is a power of two.
 *
 * @par Out-of-range indices
 * ABI implementation-defined, possibly even undefined. It may differ between
 * ABIs, between element types on the same ABI, and between DPL versions -- some
 * hardware wraps the index, some yields zero, some ignores the high bits it
 * does not need. Requiring one policy would cost several instructions per
 * permute on backends whose native behavior differs, and on scalable ABIs a
 * wrapping policy would make results depend on the runtime vector length.
 *
 * @warning Do not rely on observed out-of-range behavior. Use @ref lookup for a
 *          defined policy, at explicit cost.
 *
 * @par Constant evaluation
 * Out-of-range indices are ill-formed during constant evaluation. @c if
 * @c consteval bypasses ADL for the reference implementation, which diagnoses
 * rather than producing an unspecified value -- there is no implementation to
 * be defined by at compile time.
 *
 * @sa lookup   Defined out-of-range policy.
 * @sa permutei Compile-time index pattern.
 */
struct DPL_EMPTY_BASES permute_t :
    public primitive_operation_base<permute_t>,
    public maskable_transform_base<permute_t> {
    using operation_base<permute_t>::operator();
    using maskable_transform_base<permute_t>::operator();
};

template <>
struct operation_signature<permute_t> {
    template <simd_vector L, typename R>
    requires (canonical_vector<R> && integral<simd_element_type_t<R>>) ||
        index_sequence_like<R>
    static consteval void operator()(L&&, R) noexcept {}
};

template <typename T, typename L>
concept vindex_for = simd_vector<T> && simd_vector<L> &&
    same_abi_as<simd_abi_type_t<T>, simd_abi_type_t<L>> &&
    integral<simd_element_type_t<T>>;

template <typename T, typename L>
concept canonical_vindex_for =
    vindex_for<T, L> && canonical_vector<T> && canonical_vector<L>;

template <>
struct fallback_impl<permute_t> {

    template <canonical_vector T, canonical_vindex_for<T> R>
    requires fixed_width_abi<simd_abi_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(T arg, R idx) noexcept {
        return []<size_t... Is>(T arg, R idx, index_sequence<Is...>) {
            return dx::initialize<T>(arg[idx[imm<Is>]]...);
        }(arg, idx, iota_sequence<T>);
    }

    template <canonical_vector T, index_sequence_like I>
    requires fixed_width_abi<simd_abi_type_t<T>>
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
            static_assert([]<size_t... Is>(index_sequence<Is...>) {
                return (... && (Is < simd_abi_traits<T>::size()));
            }(__DPL to_index_sequence(idx)));
            return __DPL apply(
                [&](auto... i) {
                    constexpr auto simd_size = simd_abi_traits<T>::size();
                    using TE = simd_element_type_t<T>;
                    auto const zero = TE();
                    return dx::initialize<T>(arg[i]...);
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

template <>
struct canonical_impl<permute_t> {
public:
    template <canonical_vector T, index_sequence_like R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val, R idx) noexcept
    requires requires(
        launder_sequence_t<R> seq) { permute(internal::abi<T>, val, seq); }
    {
        constexpr auto seq = __DPL to_index_sequence(idx);
        return permute(internal::abi<T>, val, seq);
    }

    template <canonical_vector T, index_sequence_like R>
    requires unqualified_canonical_mpermute<T, simd_mask_type_t<T>, T,
        launder_sequence_t<R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(type_identity_t<T> src,
        simd_mask_type_t<T> mask, T val, R idx) noexcept {
        constexpr auto seq = __DPL to_index_sequence(idx);
        return permute(internal::abi<T>, src, mask, val, seq);
    }

    template <canonical_vector T, const_mask_for<T> M, index_sequence_like R>
    requires canonical_vector<T> &&
        unqualified_canonical_mpermute<T, launder_cmask_t<T, M>, T,
            launder_sequence_t<R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, M cmask, T val, R idx) noexcept {
        constexpr auto seq = __DPL to_index_sequence(idx);
        return permute(
            internal::abi<T>, src, dx::to_const_mask<T>(cmask), val, seq);
    }

    template <canonical_vector T, index_sequence_like R>
    requires unqualified_canonical_mpermute<dx::zero_t, simd_mask_type_t<T>, T,
        launder_sequence_t<R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, simd_mask_type_t<T> mask, T val, R idx) noexcept {
        constexpr auto seq = __DPL to_index_sequence(idx);
        return permute(internal::abi<T>, zero, mask, val, seq);
    }

    template <canonical_vector T, const_mask_for<T> M, index_sequence_like R>
    requires canonical_vector<T> &&
        unqualified_canonical_mpermute<dx::zero_t, launder_cmask_t<T, M>, T,
            launder_sequence_t<R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, M cmask, T val, R idx) noexcept {
        constexpr auto seq = __DPL to_index_sequence(idx);
        return permute(
            internal::abi<T>, zero, dx::to_const_mask<T>(cmask), val, seq);
    }

    ///

    template <canonical_vector T, canonical_vindex_for<T> R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val, R idx) noexcept
    requires requires { permute(internal::abi<T>, val, idx); }
    {
        return permute(internal::abi<T>, val, idx);
    }

    template <canonical_vector T, canonical_vindex_for<T> R>
    requires unqualified_canonical_mpermute<T, simd_mask_type_t<T>, T, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(type_identity_t<T> src,
        simd_mask_type_t<T> mask, T val, R idx) noexcept {
        return permute(internal::abi<T>, src, mask, val, idx);
    }

    template <canonical_vector T, canonical_vindex_for<T> R,
        result_cmask_for<permute_t, T, R> M>
    requires unqualified_canonical_mpermute<T, launder_cmask_t<T, M>, T, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, M cmask, T val, R idx) noexcept {
        return permute(
            internal::abi<T>, src, dx::to_const_mask<T>(cmask), val, idx);
    }

    template <canonical_vector T, canonical_vindex_for<T> R>
    requires unqualified_canonical_mpermute<dx::zero_t, simd_mask_type_t<T>, T,
        R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, simd_mask_type_t<T> mask, T val, R idx) noexcept {
        return permute(internal::abi<T>, zero, mask, val, idx);
    }

    template <canonical_vector T, canonical_vindex_for<T> R,
        result_cmask_for<permute_t, T, R> M>
    requires unqualified_canonical_mpermute<dx::zero_t,
        launder_cmask_t<cpo_result_t<permute_t, T, R>, M>, T, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, M cmask, T val, R idx) noexcept {
        return permute(
            internal::abi<T>, zero, dx::to_const_mask<T>(cmask), val, idx);
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

    template <simd_vector S, exact_mask_for<S> M, vector_subsumed_by<S> T,
        index_sequence_like R>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mpermute<S, M, T, launder_sequence_t<R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M mask, T&& val, R idx) {
        return permute( __DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val), __DPL to_index_sequence(idx));
    }

    template <simd_vector S, const_mask_for<S> M, vector_subsumed_by<S> T,
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
    template <extended_vector L, vindex_for<L> R>
    requires unqualified_extended_permute<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& val, R idx) {
        return permute(__DPL forward<L>(val), idx);
    }

    template <simd_vector S, exact_mask_for<S> M, vector_subsumed_by<S> T,
        vindex_for<T> R>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mpermute<S, M, T, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M mask, T&& val, R idx) {
        return permute( __DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val), idx);
    }

    template <simd_vector S, const_mask_for<S> M, vector_subsumed_by<S> T,
        vindex_for<T> R>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_mpermute<S, launder_cmask_t<S, M>, T, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M cmask, T&& val, R idx) {
        return permute( __DPL forward<S>(src), dx::to_const_mask<S>(cmask),
            __DPL forward<T>(val), idx);
    }

    template <simd_vector T, simd_mask M, vindex_for<T> R>
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

    template <extended_vector T, vindex_for<T> R,
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

public:
    template <typename... Ts>
    requires cpo_invocable<permute_t, Ts..., seq_t>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(Ts&&... args) noexcept(
        (... && (!simd_type<Ts> || canonical_simd_type<Ts>))) {
        constexpr seq_t seq{};
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
