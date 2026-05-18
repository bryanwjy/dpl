// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/operations/bitwise.h"

#if !DPL_MODULES
#  include "dpl/core/basic/load.h"
#  include "dpl/core/basic/to_canonical.h"
#  include "dpl/core/concepts/decayable.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_equivalence.h"
#  include "dpl/core/constants/max_value.h"
#  include "dpl/core/constants/min_value.h"
#  include "dpl/core/constants/zero.h"
#  include "dpl/core/type_traits/array_for.h"
#  include "dpl/core/type_traits/basic_type.h"
#  include "dpl/core/type_traits/iota_sequence.h"
#  include "dpl/core/type_traits/rebind_simd.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

template <typename>
struct element_cast_t {};
template <typename>
struct abi_cast_t {};
template <typename>
struct simd_cast_t {};

template <typename>
void element_cast(...) noexcept = delete;
template <typename>
void abi_cast(...) noexcept = delete;
void abi_cast(...) noexcept = delete;

template <typename From, typename To>
concept unqualified_element_castable_to = simd_vector<From> &&
    simd_element_for<To, typename From::abi_type> && requires(From arg) {
        {
            element_cast<To>(internal::abi<From>, arg)
        } -> simd_with<To, typename From::abi_type>;
    };

template <simd_element To>
struct element_cast_t<To> {
private:
    template <typename FromE>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr To safe_cast(FromE val) noexcept {
        if consteval {
            // Do we need this?
            if constexpr (floating_point<FromE> && integral<To>) {
                auto const min = static_cast<FromE>(min_value_v<To>);
                auto const max = static_cast<FromE>(max_value_v<To>);
                auto const lt = val < min;
                auto const gt = val > max;
                if (lt || gt || !(val <= max && val >= min)) {
                    return dx::msb;
                }
            }
        }

        if constexpr (convertible_to<FromE, To>) {
            return static_cast<To>(val);
        } else {
            static_assert(floating_point<To> && floating_point<FromE>);
            static_assert(dx::digits_v<float> >= dx::digits_v<FromE>);
            return static_cast<To>(static_cast<float>(val));
        }
    }

    template <typename E>
    static constexpr auto policy = []() {
        if constexpr (sizeof(E) != sizeof(To)) {
            return operation_category::lane_conversion;
        } else {
            return operation_category::lane_agnostic;
        }
    }();
    template <typename FromE, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_vector<To, A>
        DPL_VECTORCALL fallback(basic_vector<FromE, A> arg) noexcept {
        using src_t = basic_vector<FromE, A>;
        using dst_t = basic_vector<To, A>;
        if constexpr (same_as<src_t, dst_t>) {
            return arg;
        } else {
            return []<size_t... Is>(src_t arg, index_sequence<Is...>) {
                constexpr auto extent =
                    simd_abi_traits<src_t>::size < simd_abi_traits<dst_t>::size
                    ? simd_abi_traits<src_t>::size
                    : simd_abi_traits<dst_t>::size;
                array_for<To, A> buffer{
                    (Is < extent ? safe_cast(arg[Is]) : dx::zero_v<To>)...};
                return dx::load<A>(aligned, buffer.data);
            }(arg, iota_sequence<To, A>);
        }
    }

public:
    template <simd_element FromE, fixed_width_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<To, A> operator()(
        basic_vector<FromE, A> val) noexcept {
        using src_t = basic_vector<FromE, A>;
        if constexpr (same_as<To, FromE>) {
            return val;
        } else if constexpr (unqualified_element_castable_to<src_t, To>) {
            if consteval {
                return fallback(val);
            } else {
                return element_cast<To>(internal::abi<A>, val);
            }
        } else {
            return fallback(val);
        }
    }

    template <simd_element FromE, scalable_vector A>
    requires same_as<To, FromE> ||
        unqualified_element_castable_to<basic_vector<FromE, A>, To>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<To, A> operator()(
        basic_vector<FromE, A> val) noexcept {
        if constexpr (same_as<To, FromE>) {
            return val;
        } else {
            return element_cast<To>(internal::abi<A>, val);
        }
    }

    template <extended_vector From>
    requires same_as<typename From::value_type, To> ||
        unqualified_element_castable_to<From, To> ||
        (decayable_vector_for<From,
             element_cast_t::policy<typename From::value_type>> &&
            unqualified_element_castable_to<canonical_type_t<From>, To>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(From val) noexcept {
        if constexpr (same_as<typename From::value_type, To>) {
            return val;
        } else if constexpr (unqualified_element_castable_to<From, To>) {
            return element_cast<To>(internal::abi<From>, val);
        } else {
            return operator()(dx::to_canonical(val));
        }
    }
};

template <typename From, typename To>
concept unqualified_abi_target_castable = simd_vector<From> && simd_abi<To> &&
    simd_element_for<typename From::value_type, To> && requires(From arg) {
        {
            abi_cast<To>(internal::abi<From>, arg)
        } -> simd_with<typename From::value_type, To>;
    };

template <typename From, typename To>
concept unqualified_abi_source_castable = simd_vector<From> && simd_abi<To> &&
    simd_element_for<typename From::value_type, To> && requires(From arg) {
        {
            abi_cast(internal::abi<To>, arg)
        } -> simd_with<typename From::value_type, To>;
    };

template <typename From, typename To>
concept unqualified_abi_castable = unqualified_abi_target_castable<From, To> ||
    unqualified_abi_source_castable<From, To>;

template <simd_abi To>
struct abi_cast_t<To> {
private:
    template <simd_abi A>
    static constexpr auto policy = []() {
        if constexpr (simd_abi_traits<A>::size == simd_abi_traits<A>::size) {
            return operation_category::lane_agnostic;
        } else {
            return operation_category::structural_transformation;
        }
    }();

    template <typename E, typename From>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_vector<E, To>
        DPL_VECTORCALL fallback(basic_vector<E, From> arg) noexcept {
        if constexpr (same_as<To, From>) {
            return arg;
        } else {
            using src_t = basic_vector<E, From>;
            using dst_t = basic_vector<E, To>;
            if constexpr (From::size > To::size) {
                array_for<E, From> buffer DPL_INDETERMINATE;
                dx::store(arg, buffer.data);
                return dx::load<To>(buffer.data);
            } else {
                array_for<E, To> buffer DPL_INDETERMINATE;
                dx::store(arg, buffer.data);
                return dx::load<To>(buffer.data);
            }
        }
    }

    template <typename E, typename From>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_mask<E, To>
        DPL_VECTORCALL fallback(basic_mask<E, From> arg) noexcept {
        using src_t = basic_mask<E, From>;
        using dst_t = basic_mask<E, To>;
        if constexpr (same_as<To, From>) {
            return arg;
        } else if constexpr (From::size >= To::size) {
            return []<size_t... Is>(src_t arg, index_sequence<Is...>) {
                return dx::initialize<To>(arg[Is]...);
            }(arg, iota_sequence<E, To>);
        } else {
            return []<size_t... Is>(src_t arg, index_sequence<Is...>) {
                constexpr auto extent = simd_abi_traits<To>::size;
                return dx::initialize<To>((Is < extent ? arg[Is] : false)...);
            }(arg, iota_sequence<E, To>);
        }
    }

public:
    template <simd_element_for<To> E, fixed_width_abi FromA>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, To> operator()(
        basic_vector<E, FromA> val) noexcept
    requires fixed_width_abi<To> && common_abi_with<To, FromA>
    {
        using src_type = basic_vector<E, FromA>;
        if constexpr (same_as<To, FromA>) {
            return val;
        } else if constexpr (unqualified_abi_target_castable<src_type, To>) {
            if consteval {
                return fallback(val);
            } else {
                return abi_cast<To>(internal::abi<FromA>, val);
            }
        } else if constexpr (unqualified_abi_source_castable<src_type, To>) {
            if consteval {
                return fallback(val);
            } else {
                return abi_cast(internal::abi<To>, val);
            }
        } else {
            return fallback(val);
        }
    }

    template <simd_element_for<To> E, simd_abi FromA>
    requires (scalable_abi<To> || scalable_abi<FromA>) &&
        (same_as<To, FromA> ||
            unqualified_abi_castable<basic_vector<E, FromA>, To>)
        DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
        static constexpr basic_vector<E, To> operator()(
            basic_vector<E, FromA> val) noexcept
    requires common_abi_with<To, FromA>
    {
        using src_type = basic_vector<E, FromA>;
        if constexpr (same_as<To, FromA>) {
            return val;
        } else if constexpr (unqualified_abi_target_castable<src_type, To>) {
            return abi_cast<To>(internal::abi<FromA>, val);
        } else {
            return abi_cast(internal::abi<To>, val);
        }
    }

    template <simd_element_for<To> E, extended_vector From>
    requires same_as<typename From::abi_type, To> ||
        unqualified_abi_castable<From, To> ||
        (decayable_vector_for<From,
             abi_cast_t::policy<typename From::abi_type>> &&
            unqualified_abi_castable<canonical_type_t<From>, To>)
        DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
        static constexpr auto operator()(From val) noexcept
    requires common_abi_with<To, typename From::abi_type>
    {
        if constexpr (same_as<typename From::abi_type, To>) {
            return val;
        } else if constexpr (unqualified_abi_castable<From, To>) {
            if constexpr (unqualified_abi_target_castable<From, To>) {
                return abi_cast<To>(internal::abi<From>, val);
            } else {
                return abi_cast(internal::abi<To>, val);
            }
        } else {
            return operator()(dx::to_canonical(val));
        }
    }

    template <simd_element_for<To> E, fixed_width_abi FromA>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<E, To> operator()(
        basic_mask<E, FromA> val) noexcept
    requires fixed_width_abi<To> && common_abi_with<To, FromA>
    {
        using src_type = basic_mask<E, FromA>;
        if constexpr (same_as<FromA, To>) {
            return val;
        } else if constexpr (unqualified_abi_target_castable<src_type, To>) {
            if consteval {
                return fallback(val);
            } else {
                return abi_cast<To>(internal::abi<FromA>, val);
            }
        } else if constexpr (unqualified_abi_source_castable<src_type, To>) {
            if consteval {
                return fallback(val);
            } else {
                return abi_cast(internal::abi<To>, val);
            }
        } else {
            return fallback(val);
        }
    }

    template <simd_element_for<To> E, simd_abi FromA>
    requires (scalable_abi<To> || scalable_abi<FromA>) &&
        (same_as<To, FromA> ||
            unqualified_abi_castable<basic_mask<E, FromA>, To>)
        DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
        static constexpr basic_mask<E, To> operator()(
            basic_mask<E, FromA> val) noexcept
    requires common_abi_with<To, FromA>
    {
        using src_type = basic_mask<E, FromA>;
        if constexpr (same_as<To, FromA>) {
            return val;
        } else if constexpr (unqualified_abi_target_castable<src_type, To>) {
            return abi_cast<To>(internal::abi<FromA>, val);
        } else {
            return abi_cast(internal::abi<To>, val);
        }
    }

    template <simd_element_for<To> E, extended_mask From>
    requires same_as<typename From::abi_type, To> ||
        unqualified_abi_castable<From, To> ||
        (decayable_mask_for<From,
             abi_cast_t::policy<typename From::abi_type>> &&
            unqualified_abi_castable<canonical_type_t<From>, To>)
        DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
        static constexpr auto operator()(From val) noexcept
    requires common_abi_with<To, typename From::abi_type>
    {
        if constexpr (same_as<typename From::abi_type, To>) {
            return val;
        } else if constexpr (unqualified_abi_castable<From, To>) {
            if constexpr (unqualified_abi_target_castable<From, To>) {
                return abi_cast<To>(internal::abi<From>, val);
            } else {
                return abi_cast(internal::abi<To>, val);
            }
        } else {
            return operator()(dx::to_canonical(val));
        }
    }
};

template <simd_abi ToA, simd_element_for<ToA> ToE>
struct simd_cast_t<basic_vector<ToE, ToA>> {
private:
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(basic_vector<ToE, ToA> from) noexcept {
        return from;
    }

    template <simd_abi A>
    requires regular_invocable<abi_cast_t<ToA>, basic_vector<ToE, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(basic_vector<ToE, A> from) noexcept {
        return abi_cast_t<ToA>::operator()(from);
    }

    template <simd_element_for<ToA> E>
    requires regular_invocable<element_cast_t<ToE>, basic_vector<E, ToA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(basic_vector<E, ToA> from) noexcept {
        return element_cast_t<ToE>::operator()(from);
    }

    template <extended_vector From>
    requires explicitly_convertible_to<From, basic_vector<ToE, ToA>> ||
        simd_with<From, ToE, ToA> ||
        (same_as<typename From::value_type, ToE> &&
            regular_invocable<abi_cast_t<ToA>, From> &&
            same_as<invoke_result_t<abi_cast_t<ToA>, From>,
                basic_vector<ToE, ToA>>) ||
        (same_as<typename From::abi_type, ToA> &&
            regular_invocable<element_cast_t<ToE>,
                basic_vector<typename From::value_type, ToA>> &&
            same_as<invoke_result_t<element_cast_t<ToE>, From>,
                basic_vector<ToE, ToA>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(From from) noexcept {
        if constexpr (explicitly_convertible_to<From, basic_vector<ToE, ToA>>) {
            return static_cast<basic_vector<ToE, ToA>>(from);
        } else if constexpr (simd_with<From, ToE, ToA>) {
            return dx::to_canonical(from);
        } else if constexpr (same_as<typename From::value_type, ToE>) {
            return abi_cast_t<ToA>::operator()(from);
        } else {
            return element_cast_t<ToE>::operator()(from);
        }
    }
};

template <extended_vector To>
struct simd_cast_t<To> {
    using ToA DPL_NODEBUG = typename To::abi_type;
    using ToE DPL_NODEBUG = typename To::value_type;

public:
    template <simd_vector From>
    requires explicitly_convertible_to<From, To> ||
        (same_as<typename From::value_type, ToE> &&
            regular_invocable<abi_cast_t<ToA>, From> &&
            same_as<invoke_result_t<abi_cast_t<ToA>, From>, To>) ||
        (same_as<typename From::abi_type, ToA> &&
            regular_invocable<element_cast_t<ToE>,
                basic_vector<typename From::value_type, ToA>> &&
            same_as<invoke_result_t<element_cast_t<ToE>, From>, To>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr To operator()(From from) noexcept {
        if constexpr (explicitly_convertible_to<From, To>) {
            return static_cast<To>(from);
        } else if constexpr (same_as<typename From::value_type, ToE>) {
            return abi_cast_t<ToA>::operator()(from);
        } else {
            return element_cast_t<ToE>::operator()(from);
        }
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT template <typename To>
inline constexpr internal::abi_cast_t<To> abi_cast{};
DPL_EXPORT template <typename To>
inline constexpr internal::element_cast_t<To> element_cast{};
DPL_EXPORT template <typename To>
inline constexpr internal::simd_cast_t<To> simd_cast{};
} // namespace cpo
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
