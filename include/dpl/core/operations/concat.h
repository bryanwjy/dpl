// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/basic/reinterpret.h"
#  include "dpl/core/concepts/basic_element.h"
#  include "dpl/core/concepts/simd_mask_type.h"
#  include "dpl/core/concepts/simd_type.h"
#  include "dpl/core/type_traits/common_abi.h"
#  include "dpl/core/type_traits/promote_abi.h"
#endif

#if DPL_HAS_CXX26_EXTENSIONS
DPL_DISABLE_WARNING_PUSH()
DPL_DISABLE_WARNING("-Wc++26-extensions")
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

/**
 * @brief Concatenates two or more SIMD values into a single wider SIMD value.
 *
 * Concatenation combines the lane contents of the input operands into a single
 * SIMD object whose ABI is determined through ABI compatibility normalization
 * followed by iterative promotion.
 *
 * Target resolution proceeds as follows:
 *
 * 1. The operand ABIs are normalized using @c common_abi.
 * 2. The resulting ABI is iteratively widened via @c promote_abi.
 * 3. Resolution succeeds when the candidate ABI width exactly matches the
 *    aggregate lane width of all input operands.
 *
 * Formally, if @c A is the normalized ABI of the operand pack and @c B is the
 * resolved target ABI, then @c B is the unique ABI reached through repeated
 * application of @c promote_abi such that:
 *
 * @code
 * B::size == (... + Ts::size)
 * @endcode
 *
 * If no such ABI exists, the expression is ill-formed.
 *
 * Resolution is performed through ADL. Implementations may provide either:
 *
 * @code
 * concat<TargetAbi>(SourceAbi{}, xs...)
 * @endcode
 *
 * or
 *
 * @code
 * concat<SourceAbi>(TargetAbi{}, xs...)
 * @endcode
 *
 * where @c SourceAbi is the normalized operand ABI determined by
 * @c common_abi.
 *
 * When both forms are present, the source-owned overload is preferred.
 *
 * The target-owned form exists to support extension scenarios where source and
 * target ABI libraries are defined independently.
 *
 * There is no implicit truncation, padding, or partial fill.
 *
 * @tparam Ts SIMD operand types.
 * @param xs SIMD operands to concatenate.
 *
 * @return A SIMD object whose ABI is the resolved concatenation target.
 *
 * @note This operation is only defined for fixed-width ABI families. Every ABI
 *       reachable through promotion must be fixed-width.
 *
 * @note The public API is ABI-agnostic. Target resolution and backend
 *       realization are delegated to ABI libraries through @c concat_target_t,
 *       @c promote_abi, and ADL.
 *
 * @pre @c concat_target_t<Ts...> is well-formed and all ABI libraries required
 * for resolution (including intermediate promotion targets) are visible at the
 *      call site.
 *
 * @see common_abi
 * @see promote_abi
 * @see split
 */
struct concat_t;
template <typename>
void concat(...) noexcept = delete;

template <size_t N, typename C>
struct concat_target {};

template <size_t N, typename C>
requires (N == C::size)
struct concat_target<N, C> {
    using type DPL_NODEBUG = C;
};

template <size_t N, typename C>
requires (N > C::size) && requires { typename promote_abi_t<C>; }
struct concat_target<N, C> : concat_target<N, promote_abi_t<C>> {};

template <typename... As>
using concat_target_t DPL_NODEBUG =
    typename concat_target<(0zu + ... + As::size),
        promote_abi_t<common_abi_t<As...>>>::type;

template <typename T, typename... Ts>
concept concatable = (fixed_width_class<T> && ... && fixed_width_class<Ts>) &&
    (... && same_as<simd_lane_type_t<T>, simd_lane_type_t<Ts>>) &&
    requires {
        typename common_abi_t<typename T::abi_type, typename Ts::abi_type...>;
        typename promote_abi_t<
            common_abi_t<typename T::abi_type, typename Ts::abi_type...>>;
        typename concat_target_t<typename T::abi_type,
            typename Ts::abi_type...>;
    } &&
    concat_target_t<typename T::abi_type>::size ==
        (T::abi_type::size + ... + Ts::abi_type::size);

template <typename T, typename... Ts>
concept unqualified_concat_to = requires(T arg, Ts... args) {
    {
        concat<concat_target_t<typename T::abi_type, typename Ts::abi_type...>>(
            internal::abi<
                common_abi_t<typename T::abi_type, typename Ts::abi_type...>>,
            arg, args...)
    } -> equivalent_class_as<rebind_simd_t<T, simd_lane_type_t<T>,
        concat_target_t<typename T::abi_type, typename Ts::abi_type...>>>;
};

template <typename T, typename... Ts>
concept unqualified_concat_from = requires(T arg, Ts... args) {
    {
        concat<common_abi_t<typename T::abi_type, typename Ts::abi_type...>>(
            internal::abi<concat_target_t<typename T::abi_type,
                typename Ts::abi_type...>>,
            arg, args...)
    } -> equivalent_class_as<rebind_simd_t<T, simd_lane_type_t<T>,
        concat_target_t<typename T::abi_type, typename Ts::abi_type...>>>;
};

struct concat_t {
private:
    template <typename AT, typename E, typename... As>
    static consteval auto fallback(basic_simd<E, As>... args) noexcept {
        array_for<E, AT> buffer{};
        auto* ptr = buffer.data;
#if __cpp_expansion_statements >= 202506L && DPL_HAS_CXX26_EXTENSIONS
        template for (auto const& arg : {args...}) {
            dx::store(arg, ptr);
            ptr += arg.size();
        }
#else
        (..., [&ptr]<typename A>(basic_simd<E, A> arg) {
            dx::store(arg, ptr);
            ptr += arg.size();
        }(args));
#endif
        return dx::load<AT>(buffer.data);
    }

    template <typename AT, typename E, typename... As>
    static consteval auto fallback(basic_simd_mask<E, As>... args) noexcept {
        bool buffer[AT::size]{};
#if __cpp_expansion_statements >= 202506L && DPL_HAS_CXX26_EXTENSIONS
        template for (auto* ptr = buffer; auto const& arg : {args...}) {
            template for (auto const idx : iota_sequence<E, A>) {
                ptr[idx] = arg[idx];
            }
            ptr += arg.size();
        }
#else
        (..., [ptr = buffer]<typename A>(basic_simd_mask<E, A> arg) {
            [&]<size_t I = 0>(this auto self, immediate<I> idx = {}) {
                if constexpr (I < simd_abi_traits<E, A>::size) {
                    ptr[idx] = arg[idx];
                    self(imm<I + 1>);
                }
            }();
            ptr += arg.size();
        }(args));
#endif

        return [&]<size_t... Is>(index_sequence<Is...>) {
            return dx::initialize<AT>(buffer[Is]...);
        }(iota_sequence<E, AT>);
    }

public:
    template <fixed_width_class T, fixed_width_class... Ts>
    requires concatable<T, Ts...> &&
        (unqualified_concat_to<T, Ts...> ||
            unqualified_concat_to<basic_type_t<T>, basic_type_t<Ts>...>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        T arg, Ts... args) noexcept {
        if constexpr (unqualified_concat_to<T, Ts...>) {
            using To =
                concat_target_t<typename T::abi_type, typename Ts::abi_type...>;
            using From =
                common_abi_t<typename T::abi_type, typename Ts::abi_type...>;
            if constexpr ((basic_simd_type<T> && ... && basic_simd_type<Ts>)) {
                if consteval {
                    return fallback<To>(arg, args...);
                } else {
                    return concat<To>(internal::abi<From>, arg, args...);
                }
            } else {
                return concat<To>(internal::abi<From>, arg, args...);
            }
        } else {
            return operator()(
                dx::to_basic_type(arg), dx::to_basic_type(args)...);
        }
    }

    template <fixed_width_class T, fixed_width_class... Ts>
    requires concatable<T, Ts...> &&
        (!unqualified_concat_to<T, Ts...> &&
            !unqualified_concat_to<basic_type_t<T>, basic_type_t<Ts>...>) &&
        (unqualified_concat_from<T, Ts...> ||
            unqualified_concat_from<basic_type_t<T>, basic_type_t<Ts>...>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        T arg, Ts... args) noexcept {
        if constexpr (unqualified_concat_from<T, Ts...>) {
            using To =
                concat_target_t<typename T::abi_type, typename Ts::abi_type...>;
            using From =
                common_abi_t<typename T::abi_type, typename Ts::abi_type...>;
            if constexpr ((... && basic_simd_type<Ts>)) {
                if consteval {
                    return fallback<To>(arg, args...);
                } else {
                    return concat<From>(internal::abi<To>, arg, args...);
                }
            } else {
                return concat<From>(internal::abi<To>, arg, args...);
            }
        } else {
            return operator()(
                dx::to_basic_type(arg), dx::to_basic_type(args)...);
        }
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::concat_t concat{};
}
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END

#if DPL_HAS_CXX26_EXTENSIONS
DPL_DISABLE_WARNING_POP()
#endif
