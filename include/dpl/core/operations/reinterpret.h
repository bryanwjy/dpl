// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES

#  include "dpl/core/basic/to_basic_type.h"
#  include "dpl/core/concepts/basic_type.h"
#  include "dpl/core/concepts/common_abi_with.h"
#  include "dpl/core/concepts/common_size_with.h"
#  include "dpl/core/concepts/simd_class.h"
#  include "dpl/core/concepts/simd_equivalence.h"
#  include "dpl/std/concepts/invocable.h"
#  include "dpl/std/utility/as_const.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

template <typename>
void reinterpret(...) noexcept = delete;

template <typename>
struct reinterpret_t {};

template <typename From, typename To>
concept unqualified_reinterpretable_to = requires(From from) {
    {
        reinterpret<To>(internal::abi<From>, from)
    } -> simd_with<To, typename From::abi_type>;
};

template <typename From, typename To>
concept unqualified_mreinterpretable_to = requires(From from) {
    {
        reinterpret<To>(internal::abi<From>, from)
    } -> mask_with<To, typename From::abi_type>;
};

template <simd_element ToE>
struct reinterpret_t<ToE> {
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
    static constexpr basic_simd<ToE, A> fallback(
        basic_simd<FromE, A> from) noexcept {
        using to_vector = typename simd_abi_traits<ToE, A>::native_type;
        using from_vector = typename simd_abi_traits<FromE, A>::native_type;
        if constexpr (same_as<to_vector, from_vector>) {
            return +from;
        } else if consteval {
            // Workaround MSVC's unions
            array_for<FromE, A> buffer;
            dx::store(from, buffer.data);
            return dx::load<A>(__DPL bit_cast<array_for<ToE, A>>(buffer).data);
        } else {
            return __DPL bit_cast<basic_simd<ToE, A>>(from);
        }
    }

    template <fixed_width_abi A, simd_element_for<A> FromE>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_simd_mask<ToE, A> fallback(
        basic_simd_mask<FromE, A> from) noexcept {
        using to_mask = typename simd_abi_traits<ToE, A>::native_mask;
        using from_mask = typename simd_abi_traits<FromE, A>::native_mask;
        if constexpr (same_as<to_mask, from_mask>) {
            return +from;
        } else if consteval {
            return [&]<size_t... Is>(index_sequence<Is...>) {
                return dx::initialize<ToE, A>(from[Is]...);
            }(iota_sequence<FromE, A>);
        } else {
            return __DPL bit_cast<basic_simd_mask<ToE, A>>(from);
        }
    }

public:
    template <simd_abi A, simd_element_for<A> FromE>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_simd<ToE, A> operator()(
        basic_simd<FromE, A> val) noexcept {
        if constexpr (same_as<FromE, ToE>) {
            return val;
        } else if constexpr (unqualified_reinterpretable_to<
                                 basic_simd<FromE, A>, ToE>) {
            if consteval {
                return fallback(val);
            } else {
                return reinterpret<ToE>(internal::abi<A>, val);
            }
        } else {
            return fallback(val);
        }
    }

    template <extended_simd From>
    requires same_as<typename From::value_type, ToE> ||
        unqualified_reinterpretable_to<From, ToE> ||
        (decayable_vector_for<From, reinterpret_t::policy<From>> &&
            unqualified_reinterpretable_to<basic_type_t<From>, ToE>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(From val) noexcept {
        if constexpr (same_as<typename From::value_type, ToE>) {
            return val;
        } else if constexpr (unqualified_reinterpretable_to<From, ToE>) {
            return reinterpret<ToE>(internal::abi<From>, val);
        } else {
            return operator()(dx::to_basic_type(val));
        }
    }

    template <simd_abi A, simd_element_for<A> FromE>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_simd_mask<ToE, A> operator()(
        basic_simd_mask<FromE, A> val) noexcept {
        if constexpr (same_as<FromE, ToE>) {
            return val;
        } else if constexpr (unqualified_mreinterpretable_to<
                                 basic_simd_mask<FromE, A>, ToE>) {
            if consteval {
                return fallback(val);
            } else {
                return reinterpret<ToE>(internal::abi<A>, val);
            }
        } else {
            return fallback(val);
        }
    }

    template <extended_mask From>
    requires common_size_with<simd_lane_type_t<From>, ToE> &&
        (same_as<simd_lane_type_t<From>, ToE> ||
            unqualified_mreinterpretable_to<From, ToE> ||
            (decayable_mask_for<From, operation_category::lane_agnostic> &&
                unqualified_mreinterpretable_to<basic_type_t<From>, ToE>))
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(From val) noexcept {
        if constexpr (same_as<simd_lane_type_t<From>, ToE>) {
            return val;
        } else if constexpr (unqualified_mreinterpretable_to<From, ToE>) {
            return reinterpret<ToE>(internal::abi<From>, val);
        } else {
            return operator()(dx::to_basic_type(val));
        }
    }
};

template <simd_abi A, simd_element_for<A> ToE>
struct reinterpret_t<basic_simd<ToE, A>> {
public:
    template <simd_element_for<A> FromE>
    requires regular_invocable<reinterpret_t<ToE>, basic_simd<FromE, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_simd<ToE, A> operator()(
        basic_simd<FromE, A> val) noexcept {
        return reinterpret_t<ToE>::operator()(val);
    }

    template <extended_simd From>
    requires regular_invocable<reinterpret_t<ToE>, From> &&
        same_as<invoke_result_t<reinterpret_t<ToE>, From>, basic_simd<ToE, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(From val) noexcept {
        return reinterpret_t<ToE>::operator()(val);
    }
};

template <simd_abi A, simd_element_for<A> ToE>
struct reinterpret_t<basic_simd_mask<ToE, A>> {
public:
    template <simd_element_for<A> FromE>
    requires regular_invocable<reinterpret_t<ToE>, basic_simd_mask<FromE, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_simd_mask<ToE, A> operator()(
        basic_simd_mask<FromE, A> val) noexcept {
        return reinterpret_t<ToE>::operator()(val);
    }

    template <extended_mask From>
    requires regular_invocable<reinterpret_t<ToE>, From> &&
        same_as<invoke_result_t<reinterpret_t<ToE>, From>,
            basic_simd_mask<ToE, A>>
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
    requires simd_type<To>
    {
        return reinterpret_t<ToE>::operator()(val);
    }

    template <extended_mask From>
    requires regular_invocable<reinterpret_t<ToE>, From> &&
        same_as<invoke_result_t<reinterpret_t<ToE>, From>, To>
        DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
        static constexpr auto operator()(From val) noexcept
    requires simd_mask_type<To>
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
