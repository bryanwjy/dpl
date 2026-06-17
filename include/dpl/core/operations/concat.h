// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/operations/abi_promotion.h"
#include "dpl/core/operations/pack_mask.h"

#if !DPL_MODULES
#  include "dpl/core/basic/initialize.h"
#  include "dpl/core/basic/internal/abi.h"
#  include "dpl/core/basic/load.h"
#  include "dpl/core/basic/store.h"
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/operation/primitive.h"
#  include "dpl/core/operations/internal/array_for.h"
#  include "dpl/core/type_traits/common_abi.h"
#  include "dpl/core/type_traits/rebind_simd.h"
#  include "dpl/core/type_traits/simd_element_type.h"
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

template <typename T, typename... Ts>
concept concatable =
    (fixed_width_simd_type<T> && ... && fixed_width_simd_type<Ts>) &&
    (... && same_as<simd_element_type_t<T>, simd_element_type_t<Ts>>) &&
    all_common_abi<simd_abi_type_t<T>, simd_abi_type_t<Ts>...> &&
    requires {
        typename common_abi_t<simd_abi_type_t<T>, simd_abi_type_t<Ts>...>;
        typename promote_abi_t<
            common_abi_t<simd_abi_type_t<T>, simd_abi_type_t<Ts>...>>;
        typename concat_target_t<simd_abi_type_t<T>, simd_abi_type_t<Ts>...>;
    } &&
    concat_target_t<simd_abi_type_t<T>>::size ==
        (T::abi_type::size + ... + Ts::abi_type::size);

struct concat_t : private primitive_operation_base<concat_t> {
    using primitive_operation_base<concat_t>::operator();
};

template <>
struct operation_signature<concat_t> {
    template <fixed_width_vector... Ts>
    requires (sizeof...(Ts) > 1)
    static consteval void operator()(Ts&&...) noexcept {}
    template <fixed_width_mask... Ts>
    requires (sizeof...(Ts) > 1)
    static consteval void operator()(Ts&&...) noexcept {}
};

template <>
struct fallback_impl<concat_t> {
    template <typename E, fixed_width_abi A, fixed_width_abi... As>
    requires (simd_element_for<E, A> && ... && simd_element_for<E, As>) &&
        concatable<basic_vector<E, A>, basic_vector<E, As>...>
    static consteval auto operator()(
        basic_vector<E, A> arg, basic_vector<E, As>... args) noexcept {
        using ToA = concat_target_t<A, As...>;
        array_for<E, ToA> buffer{};
        auto* ptr = buffer.data;
#if (DPL_HAS_CXX26_EXTENSIONS || DPL_CXX26) && \
    __cpp_expansion_statements >= 202506L
        template for (auto const& arg : {args...}) {
            dx::store(arg, ptr);
            ptr += arg.size();
        }
#else
        auto append = [&ptr](auto const& val) {
            dx::store(val, ptr);
            ptr += val.size();
        };

        (append(arg), ..., append(args));
#endif
        return dx::load<E, ToA>(buffer.data);
    }

    template <typename E, fixed_width_abi A, fixed_width_abi... As>
    requires (simd_element_for<E, A> && ... && simd_element_for<E, As>) &&
        concatable<basic_mask<E, A>, basic_mask<E, As>...> &&
        (cpo_invocable<pack_mask_t, basic_mask<E, A>> && ... &&
            cpo_invocable<pack_mask_t, basic_mask<E, As>>)
    static consteval auto operator()(
        basic_mask<E, A> arg, basic_mask<E, As>... args) noexcept {
        using ToA = concat_target_t<A, As...>;
        return dx::initialize<E, ToA>(
            bitset(dx::pack_mask(arg), dx::pack_mask(args)...));
    }
};

template <typename T, typename... Ts>
concept unqualified_canonical_concat_to = requires(T arg, Ts... args) {
    concat<concat_target_t<simd_abi_type_t<T>, simd_abi_type_t<Ts>...>>(
        internal::abi<common_abi_t<simd_abi_type_t<T>, simd_abi_type_t<Ts>...>>,
        arg, args...);
};

template <typename T, typename... Ts>
concept unqualified_canonical_concat_from = requires(T arg, Ts... args) {
    concat<common_abi_t<simd_abi_type_t<T>, simd_abi_type_t<Ts>...>>(
        internal::abi<
            concat_target_t<simd_abi_type_t<T>, simd_abi_type_t<Ts>...>>,
        arg, args...);
};

template <typename T, typename... Ts>
concept unqualified_canonical_concat =
    unqualified_canonical_concat_to<T, Ts...> ||
    unqualified_canonical_concat_from<T, Ts...>;

template <>
struct canonical_impl<concat_t> {
private:
    template <typename E, typename A, typename... As>
    using vector_result DPL_NODEBUG =
        basic_vector<E, concat_target_t<A, As...>>;

    template <typename E, typename A, typename... As>
    using mask_result DPL_NODEBUG = basic_mask<E, concat_target_t<A, As...>>;

public:
    template <typename E, fixed_width_abi A, fixed_width_abi... As>
    requires (simd_element_for<E, A> && ... && simd_element_for<E, As>) &&
        concatable<basic_vector<E, A>, basic_vector<E, As>...> &&
        unqualified_canonical_concat<basic_vector<E, A>, basic_vector<E, As>...>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr vector_result<E, A, As...>
        DPL_VECTORCALL operator()(
            basic_vector<E, A> arg, basic_vector<E, As>... args) noexcept {
        using To = concat_target_t<A, As...>;
        using From = common_abi_t<A, As...>;
        if constexpr (unqualified_canonical_concat_to<basic_vector<E, A>,
                          basic_vector<E, As>...>) {
            return concat<To>(internal::abi<From>, arg, args...);
        } else {
            return concat<From>(internal::abi<To>, arg, args...);
        }
    }

    template <typename E, fixed_width_abi A, fixed_width_abi... As>
    requires (simd_element_for<E, A> && ... && simd_element_for<E, As>) &&
        concatable<basic_mask<E, A>, basic_mask<E, As>...> &&
        unqualified_canonical_concat<basic_mask<E, A>, basic_mask<E, As>...>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr mask_result<E, A, As...>
        DPL_VECTORCALL operator()(
            basic_mask<E, A> arg, basic_mask<E, As>... args) noexcept {
        using To = concat_target_t<A, As...>;
        using From = common_abi_t<A, As...>;
        if constexpr (unqualified_canonical_concat_to<basic_mask<E, A>,
                          basic_mask<E, As>...>) {
            return concat<To>(internal::abi<From>, arg, args...);
        } else {
            return concat<From>(internal::abi<To>, arg, args...);
        }
    }
};

template <typename T, typename... Ts>
concept unqualified_extended_concat = requires {
    {
        concat<concat_target_t<simd_abi_type_t<T>, simd_abi_type_t<Ts>...>>(
            internal::declarg<T>(), internal::declarg<Ts>()...)
    } -> equivalent_simd_type_with<rebind_simd_t<T, simd_element_type_t<T>,
        concat_target_t<simd_abi_type_t<T>, simd_abi_type_t<Ts>...>>>;
};

template <>
struct extended_impl<concat_t> {
    template <fixed_width_simd_type T, common_simd_type_with<T>... Ts>
    requires (extended_simd_type<T> || ... || extended_simd_type<Ts>) &&
        concatable<T, Ts...> && unqualified_extended_concat<T, Ts...>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        T&& arg, Ts&&... args) noexcept {
        using To = concat_target_t<simd_abi_type_t<T>, simd_abi_type_t<Ts>...>;
        return concat<To>(__DPL forward<T>(arg), __DPL forward<Ts>(args)...);
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
