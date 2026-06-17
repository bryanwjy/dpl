// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/operations/abi_promotion.h"
#include "dpl/core/operations/pack_mask.h"
#include "dpl/core/operations/split_result.h"

#if !DPL_MODULES
#  include "dpl/core/basic/initialize.h"
#  include "dpl/core/basic/internal/abi.h"
#  include "dpl/core/basic/load.h"
#  include "dpl/core/basic/store.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_type.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/operation/primitive.h"
#  include "dpl/core/operations/internal/array_for.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
/**
 * @brief Splits a SIMD value into @p N equally sized SIMD values.
 *
 * Decomposes the SIMD value @p x into @p N values of a smaller ABI and returns
 * them as a @c split_result aggregate.
 *
 * The destination ABI is determined by recursively applying
 * @c demote_abi to the source ABI until the resulting ABI width,
 * multiplied by @p N, is equal to the width of the source ABI.
 *
 * Formally, if @c A is the ABI of @p x and @c B is the resulting ABI,
 * then @c B is the unique ABI reached through repeated application of
 * @c demote_abi such that:
 *
 * @code
 * B::size * N == A::size
 * @endcode
 *
 * If no such ABI exists, the expression is ill-formed.
 *
 * Resolution is performed through ADL. Implementations may provide either:
 *
 * @code
 * split<TargetAbi>(internal::abi<SourceAbi>, x)
 * @endcode
 *
 * or
 *
 * @code
 * split(internal::abi<TargetAbi>, x)
 * @endcode
 *
 * When both are present, the source-owned overload is preferred.
 * This reflects the principle that decomposition is primarily defined
 * by the source ABI, since the source determines how its representation
 * is partitioned into smaller values.
 *
 * The target-owned form exists to support extension libraries where
 * source and destination ABI definitions are distributed independently.
 *
 * @tparam N The number of SIMD values to split into.
 * @tparam T The SIMD type of the source value.
 *
 * @param x The SIMD value to split.
 *
 * @return A @c split_result<U, N>, where @c U is the SIMD type formed
 *         using the resolved destination ABI.
 *
 * @note This operation is only defined for fixed-width ABI families.
 *       The demotion chain must resolve to an exact partition of the
 *       source ABI width.
 *
 * @see demote_abi
 * @see split_result
 * @see concat
 */
template <size_t N>
struct split_t;

template <typename>
void split(...) noexcept = delete;
void split(...) noexcept = delete;

template <size_t N>
struct split_t : private primitive_operation_base<split_t<N>> {
    using primitive_operation_base<split_t<N>>::operator();
};

template <typename T, size_t N>
concept splittable = fixed_width_simd_type<T> &&
    ((simd_abi_type_t<T>::size % N) == 0) && requires {
        typename demote_abi_t<simd_abi_type_t<T>>;
        typename split_target_t<N, simd_abi_type_t<T>>;
    };

template <size_t N>
struct fallback_impl<split_t<N>> {
    template <fixed_width_abi A, simd_element_for<A> E>
    requires splittable<basic_vector<E, A>, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static consteval auto operator()(basic_vector<E, A> src) noexcept {
        using ToA = split_target_t<N, A>;
        array_for<E, A> buffer{};
        dx::store(src, buffer.data);
#if DPL_CXX26
        constexpr auto [... is] = make_index_sequence<N>{};
        constexpr auto S = simd_abi_traits<ToA, E>::size;
        return dx::make_split_result(dx::load<ToA>(buffer.data + is * S)...);
#else
        constexpr make_index_sequence<N> iseq{};
        return [&]<size_t... Is>(index_sequence<Is...>) {
            constexpr auto S = simd_abi_traits<ToA, E>::size;
            return dx::make_split_result(
                dx::load<E, ToA>(buffer.data + Is * S)...);
        }(iseq);
#endif
    }

    template <fixed_width_abi A, simd_element_for<A> E>
    requires splittable<basic_mask<E, A>, N> &&
        cpo_invocable<pack_mask_t, basic_mask<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static consteval auto operator()(basic_mask<E, A> src) noexcept {
        using ToA = split_target_t<N, A>;
        auto const set = dx::pack_mask(src);
        constexpr auto S = simd_abi_traits<ToA, E>::size;
        constexpr auto chunk = A::size / N;
        using bitset_t = bitset<chunk>;
        bitset<chunk> data[N];
        for (auto& set : data) {
            set = __DPL truncate<chunk>(set >>= chunk);
        }

        return __DPL apply(
            [](auto const&... set) {
                return dx::make_split_result(dx::initialize<E, ToA>(set)...);
            }(),
            data);
    }
};

template <typename T, size_t N>
concept unqualified_split_into = requires(T arg) {
    split<split_target_t<N, simd_abi_type_t<T>>>(internal::abi<T>, arg);
};

template <typename T, size_t N>
concept unqualified_split_outof = requires(T arg) {
    split(internal::abi<split_target_t<N, simd_abi_type_t<T>>>, arg);
};

template <typename T, size_t N>
concept unqualified_canonical_split =
    unqualified_split_into<T, N> || unqualified_split_outof<T, N>;

template <size_t N>
struct canonical_impl<split_t<N>> {
private:
    template <typename T>
    using result_t DPL_NODEBUG = conditional_t<simd_vector<T>,
        basic_vector<simd_element_type_t<T>,
            split_target_t<N, simd_abi_type_t<T>>>,
        basic_mask<simd_element_type_t<T>,
            split_target_t<N, simd_abi_type_t<T>>>>;

public:
    template <canonical_simd_type T>
    requires splittable<T, N> && unqualified_canonical_split<T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr result_t<T>
        DPL_VECTORCALL operator()(T src) noexcept {
        using ToA = split_target_t<N, simd_abi_type_t<T>>;
        if constexpr (unqualified_split_into<T, N>) {
            return split<ToA>(internal::abi<T>, src);
        } else {
            return split(internal::abi<ToA>, src);
        }
    }
};

template <typename T, typename U, size_t N>
inline constexpr bool equivalent_split_result = false;

template <size_t N, fixed_width_simd_type T, fixed_width_simd_type U>
inline constexpr bool equivalent_split_result<dx::split_result<T, N>, U, N> =
    common_simd_type_with<T, U> &&
    (simd_abi_type_t<U>::size == simd_abi_type_t<T>::size * N);

template <typename T, typename From, size_t N>
concept equivalent_split_result_as = equivalent_split_result<T, From, N>;

template <typename T, size_t N>
concept unqualified_extended_split = requires {
    { split<N>(internal::declarg<T>()) } -> equivalent_split_result_as<T, N>;
};

template <size_t N>
struct extended_impl<split_t<N>> {
public:
    template <canonical_simd_type T>
    requires unqualified_extended_split<T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(T&& src) noexcept {
        return split<N>(__DPL forward<T>(src));
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT template <size_t N>
inline constexpr internal::split_t<N> split{};
}
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
