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

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

void concat(...) noexcept = delete;

template <typename A, typename T, typename... Ts>
concept unqualified_concat = requires(T arg, Ts... args) {
    {
        concat(internal::abi<A>, arg, args...)
    } -> simd_with<simd_lane_type_t<T>, A>;
};

struct concat_t {
private:
    template <size_t N, typename C>
    struct promote {};

    template <size_t N, typename C>
    requires (N == C::size)
    struct promote<N, C> {
        using type DPL_NODEBUG = C;
    };

    template <size_t N, typename C>
    requires (N > C::size) && requires { typename promote_abi_t<C>; }
    struct promote<N, C> : promote<N, promote_abi_t<C>> {};

    template <typename... Ts>
    using target DPL_NODEBUG =
        typename promote<(0zu + ... + Ts::abi_type::size),
            promote_abi_t<common_abi_t<typename Ts::abi_type...>>>::type;

    template <typename AT, typename E, typename... As>
    static consteval auto fallback(basic_simd<E, As>... args) noexcept {
        array_for<E, AT> buffer{};
        auto* ptr = buffer.data;
        (..., [&ptr]<typename A>(basic_simd<E, A> arg) {
            dx::store(arg, ptr);
            ptr += arg.size();
        }(args));
        return dx::load<AT>(buffer.data);
    }

    template <typename AT, typename E, typename... As>
    static consteval auto fallback(basic_simd_mask<E, As>... args) noexcept {
        array_for<bool, AT> buffer{};
        auto* ptr = buffer.data;
        (..., [&ptr]<typename A>(basic_simd_mask<E, A> arg) {
            [&]<size_t I = 0>(this auto self, immediate<I> idx = {}) {
                if constexpr (I < simd_abi_traits<E, A>::size) {
                    ptr[idx] = arg[idx];
                    self(imm<I + 1>);
                }
            }();
            ptr += arg.size();
        }(args));
        return [&]<size_t... Is>(index_sequence<Is...>) {
            return dx::initialize<AT>(buffer.data[Is]...);
        }(iota_sequence<E, AT>);
    }

public:
    template <fixed_width_simd T, fixed_width_simd... Ts>
    requires (... &&
                 same_as<typename T::value_type, typename Ts::value_type>) &&
        requires {
            typename common_abi_t<T, Ts...>;
            typename promote_abi_t<common_abi_t<T, Ts...>>;
            typename target<T, Ts...>;
            requires (target<T, Ts...>::size ==
                (T::abi_type::size + ... + Ts::abi_type::size));
        } &&
        (unqualified_concat<target<T, Ts...>, T, Ts...> ||
            unqualified_concat<target<T, Ts...>, basic_type_t<T>,
                basic_type_t<Ts>...>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        T arg, Ts... args) noexcept {
        if constexpr (unqualified_concat<target<T, Ts...>, T, Ts...>) {
            if constexpr ((basic_simd_type<T> && ... && basic_simd_type<Ts>)) {
                if consteval {
                    return fallback(arg, args...);
                } else {
                    return concat(
                        internal::abi<target<T, Ts...>>, arg, args...);
                }
            }

            return concat(internal::abi<target<T, Ts...>>, arg, args...);
        } else {
            return operator()(internal::abi<target<T, Ts...>>,
                dx::to_basic_type(arg), dx::to_basic_type(args)...);
        }
    }

    template <fixed_width_mask T, fixed_width_mask... Ts>
    requires (... && common_size_simd_with<T, Ts>) &&
        requires {
            typename common_abi_t<T, Ts...>;
            typename promote_abi_t<common_abi_t<T, Ts...>>;
            typename target<T, Ts...>;
            requires (target<T, Ts...>::size ==
                (T::abi_type::size + ... + Ts::abi_type::size));
        } &&
        (unqualified_concat<target<T, Ts...>, T, Ts...> ||
            unqualified_concat<target<T, Ts...>, basic_type_t<T>,
                basic_type_t<Ts>...>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        T arg, Ts... args) noexcept {
        using E = common_size_type_t<T, Ts...>;
        if constexpr ((same_as<E, simd_lane_type_t<T>> && ... &&
                          same_as<E, simd_lane_type_t<Ts>>)) {
            if constexpr (unqualified_concat<target<T, Ts...>, T, Ts...>) {
                if constexpr ((basic_simd_mask_type<T> && ... &&
                                  basic_simd_mask_type<Ts>)) {
                    if consteval {
                        using result = simd_lane_type_t<decltype(concat(
                            internal::abi<target<T, Ts...>>, arg, args...))>;
                        return dx::reinterpret<result>(fallback(arg, args...));
                    } else {
                        return concat(
                            internal::abi<target<T, Ts...>>, arg, args...);
                    }
                } else {
                    return concat(
                        internal::abi<target<T, Ts...>>, arg, args...);
                }
            } else {
                return operator()(
                    internal::abi<target<T, Ts...>>, arg, args...);
            }
        } else {
            return operator()(internal::abi<target<T, Ts...>>,
                dx::reinterpret<E>(arg), dx::reinterpret<E>(args)...);
        }
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
/**
 * Concatenates one or more SIMD objects into a single SIMD value.
 *
 * Concatenation is performed by selecting a target ABI whose capacity exactly
 * matches the aggregate lane width of the input operands. Target selection is
 * resolved through ABI compatibility normalization (`common_abi`) followed by
 * iterative widening via `promote_abi`.
 *
 * The operation is only well-formed if a valid target ABI can be resolved and
 * the corresponding ABI library/module is visible at the call site. In
 * particular, any ABI required during target resolution (including intermediate
 * promotion targets) must be available in the current translation unit.
 *
 * There is no implicit truncation, padding, or partial fill.
 *
 * @tparam Ts SIMD operand types.
 * @param xs SIMD operands to concatenate.
 *
 * @return A SIMD object whose ABI is the resolved concatenation target.
 *
 * @note The public API is ABI-agnostic. Target ABI resolution and backend
 * realization are delegated to ABI libraries through `target_abi` and ADL.
 *
 * @pre `target_abi<Ts...>` is well-formed and visible.
 */
DPL_EXPORT inline constexpr internal::concat_t concat{};
}
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
