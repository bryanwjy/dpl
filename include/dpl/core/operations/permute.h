// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/basic/initialize.h"
#  include "dpl/core/basic/internal/abi.h"
#  include "dpl/core/basic/internal/iota_sequence.h"
#  include "dpl/core/concepts/common_size_with.h"
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/dispatch/maskable/transform.h"
#  include "dpl/core/dispatch/operation/primitive.h"
#  include "dpl/core/immediate/immediate.h"
#  include "dpl/core/type_traits/simd_abi_type.h"
#  include "dpl/core/type_traits/simd_element_type.h"
#  include "dpl/std/utility/integer_sequence_like.h"
#  include "dpl/std/utility/sequence.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
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

template <>
struct canonical_impl<permute_t> {
    template <simd_abi A, simd_element_for<A> E, index_sequence_like R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val, R idx) noexcept
    requires requires {
        permute(internal::abi<A>, val, __DPL to_index_sequence(idx));
    }
    {
        return permute(internal::abi<A>, val, __DPL to_index_sequence(idx));
    }

    template <canonical_vector L, common_abi_with<simd_abi_type_t<L>> RA,
        simd_element_for<RA> RE, typename A = common_abi_t<L, RA>>
    requires integral<RE>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<simd_element_type_t<L>, A> operator()(
        L val, basic_vector<RE, RA> idx) noexcept
    requires requires { permute(internal::abi<A>, val, idx); }
    {
        return permute(internal::abi<A>, val, idx);
    }
};

template <typename T, typename I, typename A = simd_abi_type_t<T>>
concept unqualified_extended_permutei = requires(I idx) {
    {
        permute(internal::declarg<T>(), __DPL to_index_sequence(idx))
    } -> vector_with_common_abi<A>;
};

template <typename T, typename I,
    typename A = common_abi_t<remove_cvref_t<T>, I>>
concept unqualified_extended_permute = requires(I idx) {
    { permute(internal::declarg<T>(), idx) } -> vector_with_common_abi<A>;
};

template <>
struct extended_impl<permute_t> {

    template <simd_type L, index_sequence_like R>
    requires unqualified_extended_permutei<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& val, R idx) {
        return permute(__DPL forward<L>(val), __DPL to_index_sequence(idx));
    }

    template <simd_type L, common_abi_with<simd_abi_type_t<L>> RA,
        simd_element_for<RA> RE, typename A = common_abi_t<L, RA>>
    requires unqualified_extended_permute<L, basic_vector<RE, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& val, basic_vector<RE, RA> idx) {
        return permute(__DPL forward<L>(val), idx);
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
DPL_EXPORT inline constexpr internal::permute_t permute{};
DPL_EXPORT template <size_t... Is>
inline constexpr internal::permutei_t<Is...> permutei{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
