// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_class.h"
#  include "dpl/core/type_traits/promote_abi.h"
#  include "dpl/core/utility/split_result.h"
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
template <typename>
struct split_t {};

template <typename>
void split(...) noexcept = delete;
void split(...) noexcept = delete;

template <typename T, typename U>
inline constexpr bool equivalent_split_result = false;

template <size_t N, fixed_width_class T, fixed_width_class U>
inline constexpr bool equivalent_split_result<split_result<T, N>, U> =
    equivalent_class_as<T, U> && (N * T::abi_type::size == U::abi_type::size);

template <typename T, typename U>
concept equivalent_split_result_as = equivalent_split_result<T, U>;

template <size_t Target, typename C>
struct split_target {};

template <size_t Target, typename C>
requires (Target == C::size)
struct split_target<Target, C> {
    using type DPL_NODEBUG = C;
};

template <size_t Target, typename C>
requires (Target < C::size) && requires { typename demote_abi_t<C>; }
struct split_target<Target, C> : split_target<Target, demote_abi_t<C>> {};

template <typename Source>
using split_target_t DPL_NODEBUG =
    typename demote<Source::size / N, demote_abi_t<Source>>::type;

template <typename T>
concept unqualified_split_into = requires(T arg) {
    {
        split<split_target_t<typename T::abi_type>>(internal::abi<T>, arg)
    } -> equivalent_split_result_as<rebind_simd_t<T, simd_lane_type_t<T>,
        split_target_t<typename T::abi_type>>>;
};

template <typename T>
concept unqualified_split_outof = requires(T arg) {
    {
        split(internal::abi<split_target_t<typename T::abi_type>>, arg)
    } -> equivalent_split_result_as<rebind_simd_t<T, simd_lane_type_t<T>,
        split_target_t<typename T::abi_type>>>;
};

template <typename T, size_t N>
concept splittable = fixed_width_class<T> && requires {
    typename demote_abi_t<typename T::abi_type>;
    typename split_target_t<typename T::abi_type>;
} && (split_target_t<typename T::abi_type>::size * N) == T::abi_type::size;

template <size_t N>
struct split_t {
private:
    template <typename E, typename A0>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static consteval auto fallback(basic_simd<E, A0> src) noexcept {
        constexpr make_index_sequence<A0::size / A::size> seq{};
        array_for<E, A0> buffer{};
        dx::store(src, buffer.data);
        return [&]<size_t... Is>(index_sequence<Is...>) {
            constexpr auto stride = simd_abi_traits<A, E>::size;
            return dx::make_split_result(
                dx::load<A>(buffer.data + Is * stride)...);
        }(seq);
    }

    template <typename E, typename A0>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static consteval auto fallback(basic_mask<E, A0> src) noexcept {
        constexpr make_index_sequence<N> iseq{};
        return [&]<size_t... Is>(index_sequence<Is...>) {
            constexpr auto S = simd_abi_traits<A, E>::size;
            constexpr make_index_sequence<S> jseq{};
            return dx::make_split_result(
                [&]<size_t O, size_t... Js>(
                    immediate<O>, index_sequence<Js...>) {
                    return dx::initialize<A>(src[O + Js]...);
                }(imm<(Is * S)>, jseq)...);
        }(iseq);
    }

public:
    template <splittable<N> T>
    requires (
        unqualified_split_into<T> || unqualified_split_into<basic_type_t<T>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto operator()(T src) noexcept {
        if constexpr (unqualified_split_into<T>) {
            using A = split_target_t<typename T::abi_type>;
            if constexpr (basic_simd_class<T>) {
                if consteval {
                    return fallback(src);
                } else {
                    return split<A>(internal::abi<T>, src);
                }
            } else {
                return split<A>(internal::abi<T>, src);
            }
        } else {
            return operator()(dx::to_basic_type(src));
        }
    }

    template <splittable<N> T>
    requires (!unqualified_split_into<T> &&
                 !unqualified_split_into<basic_type_t<T>>) &&
        (unqualified_split_outof<T> || unqualified_split_outof<basic_type_t<T>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto operator()(T src) noexcept {
        if constexpr (unqualified_split_outof<T>) {
            using A = split_target_t<typename T::abi_type>;
            if constexpr (basic_simd_class<T>) {
                if consteval {
                    return fallback(src);
                } else {
                    return split(internal::abi<A>, src);
                }
            } else {
                return split(internal::abi<A>, src);
            }
        } else {
            return operator()(dx::to_basic_type(src));
        }
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT template <size_t N>
inline constexpr split_t<N> split{};
}
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
