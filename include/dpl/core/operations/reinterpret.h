// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/operations/evaluate.h"

#if !DPL_MODULES
#  include "dpl/core/basic/initialize.h"
#  include "dpl/core/basic/load.h"
#  include "dpl/core/basic/store.h"
#  include "dpl/core/basic/to_canonical.h"
#  include "dpl/core/concepts/basic_type.h"
#  include "dpl/core/concepts/common_size_with.h"
#  include "dpl/core/concepts/decayable.h"
#  include "dpl/core/concepts/operation_category.h"
#  include "dpl/core/concepts/simd_abi_traits.h"
#  include "dpl/core/concepts/simd_equivalence.h"
#  include "dpl/core/type_traits/array_for.h"
#  include "dpl/core/type_traits/iota_sequence.h"
#  include "dpl/core/type_traits/simd_abi_type.h"
#  include "dpl/core/type_traits/simd_expression_result.h"
#  include "dpl/std/bit/bit_cast.h"
#  include "dpl/std/concepts/invocable.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

template <typename>
void reinterpret(...) noexcept = delete;

template <typename>
struct reinterpret_t;

template <typename T, typename ToE, typename A>
concept extended_reinterpreted_vector =
    simd_vector<T> && same_as<simd_lane_type_t<T>, ToE> &&
    common_abi_with<A, typename T::abi_type>;

template <typename T, typename ToE, typename A>
concept extended_reinterpreted_mask =
    simd_mask<T> && same_as<simd_lane_type_t<T>, ToE> &&
    common_abi_with<A, typename T::abi_type>;

template <typename T, typename ToE, typename A>
concept canonical_reinterpreted_vector =
    extended_reinterpreted_vector<T, ToE, A> &&
    same_as<A, typename T::abi_type>;

template <typename T, typename ToE, typename A>
concept canonical_reinterpreted_mask =
    extended_reinterpreted_mask<T, ToE, A> && same_as<A, typename T::abi_type>;

template <typename From, typename To>
concept unqualified_canonical_vector_reinterpret = requires(From from) {
    {
        reinterpret<To>(internal::abi<From>, from)
    } -> canonical_reinterpreted_vector<To, simd_abi_type_t<From>>;
};

template <typename From, typename To>
concept unqualified_canonical_mask_reinterpret = requires(From from) {
    {
        reinterpret<To>(internal::abi<From>, from)
    } -> canonical_reinterpreted_mask<To, simd_abi_type_t<From>>;
};

template <typename From, typename To>
concept unqualified_extended_vector_reinterpret = requires(From from) {
    {
        reinterpret<To>(from)
    } -> extended_reinterpreted_vector<To, simd_abi_type_t<From>>;
};

template <typename From, typename To>
concept unqualified_extended_mask_reinterpret = requires(From from) {
    {
        reinterpret<To>(from)
    } -> extended_reinterpreted_mask<To, simd_abi_type_t<From>>;
};
template <typename From, typename To>
concept unqualified_extended_reinterpret =
    unqualified_extended_vector_reinterpret<From, To> ||
    unqualified_extended_mask_reinterpret<From, To>;

template <typename From, typename To>
concept expression_reinterpret = simd_expression<From> &&
    invocable<reinterpret_t<To>, simd_expression_result_t<From>>;

template <typename From, typename To>
inline constexpr auto reinterpret_lane_policy =
    sizeof(simd_lane_type_t<From>) == sizeof(To)
    ? operation_category::lane_agnostic
    : (operation_category::structural_transformation |
          operation_category::lane_agnostic);

template <typename From, typename To>
concept decayable_reinterpret =
    decayable_simd_for<From, reinterpret_lane_policy<From, To>> &&
    regular_invocable<reinterpret_t<To>, canonical_type_t<From>>;

template <typename From, typename To>
concept extended_reinterpret = unqualified_extended_reinterpret<From, To> ||
    expression_reinterpret<From, To> || decayable_reinterpret<From, To>;

template <typename ToE>
struct reinterpret_t {
private:
    template <typename From>
    static constexpr auto policy = []() {
        if constexpr (sizeof(simd_lane_type_t<From>) == sizeof(ToE)) {
            return operation_category::lane_agnostic;
        } else {
            return operation_category::structural_transformation |
                operation_category::lane_agnostic;
        }
    }();

    template <fixed_width_abi A, simd_element_for<A> FromE>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_vector<ToE, A> fallback(
        basic_vector<FromE, A> from) noexcept {
        using to_vector = typename simd_abi_traits<ToE, A>::native_vector;
        using from_vector = typename simd_abi_traits<FromE, A>::native_vector;
        if constexpr (same_as<to_vector, from_vector>) {
            return +from;
        } else if consteval {
            // Workaround MSVC's unions
            array_for<FromE, A> buffer;
            dx::store(from, buffer.data);
            return dx::load<A>(__DPL bit_cast<array_for<ToE, A>>(buffer).data);
        } else {
            return __DPL bit_cast<basic_vector<ToE, A>>(from);
        }
    }

    template <fixed_width_abi A, simd_element_for<A> FromE>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_mask<ToE, A> fallback(
        basic_mask<FromE, A> from) noexcept {
        using to_mask = typename simd_abi_traits<ToE, A>::native_mask;
        using from_mask = typename simd_abi_traits<FromE, A>::native_mask;
        if constexpr (same_as<to_mask, from_mask>) {
            return +from;
        } else if consteval {
            return [&]<size_t... Is>(index_sequence<Is...>) {
                return dx::initialize<ToE, A>(from[Is]...);
            }(iota_sequence<FromE, A>);
        } else {
            return __DPL bit_cast<basic_mask<ToE, A>>(from);
        }
    }

public:
    template <simd_abi A, simd_element_for<A> FromE>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<ToE, A> operator()(
        basic_vector<FromE, A> val) noexcept {
        if constexpr (same_as<FromE, ToE>) {
            return val;
        } else if constexpr (unqualified_canonical_vector_reinterpret<
                                 basic_vector<FromE, A>, ToE>) {
            if consteval {
                return fallback(val);
            } else {
                return reinterpret<ToE>(internal::abi<A>, val);
            }
        } else {
            return fallback(val);
        }
    }

    template <simd_abi A, simd_element_for<A> FromE>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<ToE, A> operator()(
        basic_mask<FromE, A> val) noexcept {
        if constexpr (same_as<FromE, ToE>) {
            return val;
        } else if constexpr (unqualified_canonical_mask_reinterpret<
                                 basic_mask<FromE, A>, ToE>) {
            if consteval {
                return fallback(val);
            } else {
                return reinterpret<ToE>(internal::abi<A>, val);
            }
        } else {
            return fallback(val);
        }
    }

    template <extended_class From>
    requires (simd_vector<From> ||
                 common_size_with<simd_lane_type_t<From>, ToE>) &&
        extended_reinterpret<From, ToE>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(From val) noexcept {
        if constexpr (unqualified_extended_reinterpret<From, ToE>) {
            return reinterpret<ToE>(val);
        } else if constexpr (expression_reinterpret<From, ToE>) {
            return operator()(dx::evaluate(val));
        } else {
            return operator()(dx::to_canonical(val));
        }
    }
};

template <simd_abi A, simd_element_for<A> ToE>
struct reinterpret_t<basic_vector<ToE, A>> {
public:
    template <simd_element_for<A> FromE>
    requires regular_invocable<reinterpret_t<ToE>, basic_vector<FromE, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<ToE, A> operator()(
        basic_vector<FromE, A> val) noexcept {
        return reinterpret_t<ToE>::operator()(val);
    }

    template <extended_vector From>
    requires regular_invocable<reinterpret_t<ToE>, From> &&
        same_as<invoke_result_t<reinterpret_t<ToE>, From>, basic_vector<ToE, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(From val) noexcept {
        return reinterpret_t<ToE>::operator()(val);
    }
};

template <simd_abi A, simd_element_for<A> ToE>
struct reinterpret_t<basic_mask<ToE, A>> {
public:
    template <simd_element_for<A> FromE>
    requires regular_invocable<reinterpret_t<ToE>, basic_mask<FromE, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<ToE, A> operator()(
        basic_mask<FromE, A> val) noexcept {
        return reinterpret_t<ToE>::operator()(val);
    }

    template <extended_mask From>
    requires regular_invocable<reinterpret_t<ToE>, From> &&
        same_as<invoke_result_t<reinterpret_t<ToE>, From>, basic_mask<ToE, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(From val) noexcept {
        return reinterpret_t<ToE>::operator()(val);
    }
};

template <extended_class To>
struct reinterpret_t<To> {
    using ToA DPL_NODEBUG = typename To::abi_type;
    using ToE DPL_NODEBUG = simd_lane_type_t<To>;

public:
    template <simd_with_abi<ToA> From>
    requires regular_invocable<reinterpret_t<ToE>, From> &&
        same_as<invoke_result_t<reinterpret_t<ToE>, From>, To>
        DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
        static constexpr To operator()(From val) noexcept
    requires simd_vector<To>
    {
        return reinterpret_t<ToE>::operator()(val);
    }

    template <extended_mask From>
    requires regular_invocable<reinterpret_t<ToE>, From> &&
        same_as<invoke_result_t<reinterpret_t<ToE>, From>, To>
        DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
        static constexpr auto operator()(From val) noexcept
    requires simd_mask<To>
    {
        return reinterpret_t<ToE>::operator()(val);
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT template <typename To>
inline constexpr internal::reinterpret_t<To> reinterpret{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
