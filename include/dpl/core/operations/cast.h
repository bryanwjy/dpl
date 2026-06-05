// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/operations/evaluate.h"
#include "dpl/core/operations/internal/array_for.h"

#if !DPL_MODULES
#  include "dpl/core/basic/initialize.h"
#  include "dpl/core/basic/internal/abi.h"
#  include "dpl/core/basic/internal/iota_sequence.h"
#  include "dpl/core/basic/load.h"
#  include "dpl/core/basic/store.h"
#  include "dpl/core/basic/to_canonical.h"
#  include "dpl/core/concepts/decayable.h"
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_expression.h"
#  include "dpl/core/constants/max_value.h"
#  include "dpl/core/constants/min_value.h"
#  include "dpl/core/constants/zero.h"
#  include "dpl/core/type_traits/canonical_type.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

template <typename>
struct element_cast_t;
template <typename>
struct abi_cast_t {};
template <typename>
struct simd_cast_t {};

template <typename>
void element_cast(...) noexcept = delete;
template <typename>
void abi_cast(...) noexcept = delete;
void abi_cast(...) noexcept = delete;
template <typename>
void simd_cast(...) noexcept = delete;

template <typename T, typename E, typename A>
concept extended_ecast_vector =
    simd_vector<T> && same_as<simd_element_type_t<T>, E> &&
    common_abi_with<simd_abi_type_t<T>, A>;
template <typename T, typename E, typename A>
concept canonical_ecast_vector =
    extended_ecast_vector<T, E, A> && same_as<simd_abi_type_t<T>, A>;

template <typename From, typename To>
concept unqualified_canonical_ecastable = simd_vector<From> &&
    simd_element_for<To, typename From::abi_type> && requires(From arg) {
        {
            element_cast<To>(internal::abi<From>, arg)
        } -> canonical_ecast_vector<To, typename From::abi_type>;
    };

template <typename From, typename To>
concept unqualified_extended_ecastable = simd_vector<From> &&
    simd_element_for<To, typename From::abi_type> && requires(From arg) {
        {
            element_cast<To>(arg)
        } -> extended_ecast_vector<To, typename From::abi_type>;
    };

template <typename FE, typename TE>
inline constexpr operation_category ecast_lane_policy =
    sizeof(FE) != sizeof(TE) ? operation_category::lane_conversion
                             : operation_category::lane_agnostic;

template <typename From, typename To>
concept expression_ecastable = vector_expression<From> &&
    invocable<element_cast_t<To>, simd_expression_result_t<From>>;

template <typename From, typename To>
concept decayable_ecastable =
    decayable_vector_for<From,
        ecast_lane_policy<simd_element_type_t<From>, To>> &&
    regular_invocable<element_cast_t<To>, canonical_type_t<From>>;

template <typename From, typename To>
concept extended_ecastable = unqualified_extended_ecastable<From, To> ||
    expression_ecastable<From, To> || decayable_ecastable<From, To>;

template <typename To>
struct element_cast_t {
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
    template <fixed_width_abi A, simd_element_for<A> FromE>
    requires simd_element_for<To, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<To, A> operator()(
        basic_vector<FromE, A> val) noexcept {
        if constexpr (same_as<To, FromE>) {
            return val;
        } else if constexpr (unqualified_canonical_ecastable<
                                 basic_vector<FromE, A>, To>) {
            if consteval {
                return fallback(val);
            } else {
                return element_cast<To>(internal::abi<A>, val);
            }
        } else {
            return fallback(val);
        }
    }

    template <scalable_abi A, simd_element_for<A> FromE>
    requires same_as<To, FromE> ||
        (simd_element_for<To, A> &&
            unqualified_canonical_ecastable<basic_vector<FromE, A>, To>)
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
    requires extended_ecastable<From, To>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(From val) {
        if constexpr (unqualified_extended_ecastable<From, To>) {
            return element_cast<To>(val);
        } else if constexpr (expression_ecastable<From, To>) {
            return operator()(dx::evaluate(val));
        } else {
            return operator()(dx::to_canonical(val));
        }
    }
};

template <typename From, typename To>
concept canonical_vector_target_acastable = simd_vector<From> && simd_abi<To> &&
    simd_element_for<simd_element_type_t<From>, To> && requires(From arg) {
        {
            abi_cast<To>(internal::abi<From>, arg)
        } -> vector_with<simd_element_type_t<From>, To>;
    };

template <typename From, typename To>
concept canonical_vector_source_acastable = simd_vector<From> && simd_abi<To> &&
    simd_element_for<simd_element_type_t<From>, To> && requires(From arg) {
        {
            abi_cast(internal::abi<To>, arg)
        } -> vector_with<simd_element_type_t<From>, To>;
    };

template <typename From, typename To>
concept canonical_mask_target_acastable = simd_mask<From> && simd_abi<To> &&
    simd_element_for<simd_element_type_t<From>, To> && requires(From arg) {
        {
            abi_cast<To>(internal::abi<From>, arg)
        } -> mask_with<simd_element_type_t<From>, To>;
    };

template <typename From, typename To>
concept canonical_mask_source_acastable = simd_mask<From> && simd_abi<To> &&
    simd_element_for<simd_element_type_t<From>, To> && requires(From arg) {
        {
            abi_cast(internal::abi<To>, arg)
        } -> mask_with<simd_element_type_t<From>, To>;
    };

template <typename From, typename To>
concept unqualified_extended_vector_acastable =
    simd_vector<From> && simd_abi<To> &&
    simd_element_for<simd_element_type_t<From>, To> && requires(From arg) {
        { abi_cast<To>(arg) } -> vector_with_abi<To>;
    };

template <typename From, typename To>
concept unqualified_extended_mask_acastable = simd_mask<From> && simd_abi<To> &&
    simd_element_for<simd_element_type_t<From>, To> && requires(From arg) {
        { abi_cast<To>(arg) } -> mask_with_abi<To>;
    };

template <typename From, typename To>
concept unqualified_extended_acastable =
    unqualified_extended_vector_acastable<From, To> ||
    unqualified_extended_mask_acastable<From, To>;

template <typename From, typename To>
concept expression_acastable = simd_expression<From> && simd_abi<To> &&
    invocable<abi_cast_t<To>, simd_expression_result_t<From>>;

template <typename FA, typename TA>
inline constexpr operation_category abi_lane_policy =
    simd_abi_traits<FA>::size == simd_abi_traits<TA>::size
    ? operation_category::lane_agnostic
    : operation_category::structural_transformation;

template <typename From, typename To>
concept decayable_acastable = simd_expression<From> && simd_abi<To> &&
    decayable_simd_for<From, abi_lane_policy<simd_abi_type_t<From>, To>> &&
    invocable<abi_cast_t<To>, canonical_type_t<From>>;

template <typename From, typename To>
concept extended_acastable = unqualified_extended_acastable<From, To> ||
    expression_acastable<From, To> || decayable_acastable<From, To>;

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

    template <typename E, typename FromA>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_mask<E, To>
        DPL_VECTORCALL fallback(basic_mask<E, FromA> arg) noexcept {
        using src_t = basic_mask<E, FromA>;
        using dst_t = basic_mask<E, To>;
        if constexpr (same_as<To, FromA>) {
            return arg;
        } else if constexpr (FromA::size >= To::size) {
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
        using From = basic_mask<E, FromA>;
        if constexpr (canonical_vector_target_acastable<From, To>) {
            if consteval {
                return fallback(val);
            } else {
                return abi_cast<To>(internal::abi<FromA>, val);
            }
        } else if constexpr (canonical_vector_source_acastable<From, To>) {
            if consteval {
                return fallback(val);
            } else {
                return abi_cast(internal::abi<To>, val);
            }
        } else {
            return fallback(val);
        }
    }

    template <simd_element_for<To> E, common_abi_with<To> FromA>
    requires (scalable_abi<To> || scalable_abi<FromA>) &&
        (canonical_vector_target_acastable<basic_vector<E, FromA>, To> ||
            canonical_vector_source_acastable<basic_vector<E, FromA>, To>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, To> operator()(
        basic_vector<E, FromA> val) noexcept {
        using From = basic_vector<E, FromA>;
        if constexpr (canonical_vector_target_acastable<From, To>) {
            return abi_cast<To>(internal::abi<FromA>, val);
        } else {
            return abi_cast(internal::abi<To>, val);
        }
    }

    ///
    template <simd_element_for<To> E, fixed_width_abi FromA>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<E, To> operator()(
        basic_mask<E, FromA> val) noexcept
    requires fixed_width_abi<To> && common_abi_with<To, FromA>
    {
        using From = basic_mask<E, FromA>;
        if constexpr (same_as<To, FromA>) {
            return val;
        } else if constexpr (canonical_mask_target_acastable<From, To>) {
            if consteval {
                return fallback(val);
            } else {
                return abi_cast<To>(internal::abi<FromA>, val);
            }
        } else if constexpr (canonical_mask_source_acastable<From, To>) {
            if consteval {
                return fallback(val);
            } else {
                return abi_cast(internal::abi<To>, val);
            }
        } else {
            return fallback(val);
        }
    }

    template <simd_element_for<To> E, common_abi_with<To> FromA>
    requires (scalable_abi<To> || scalable_abi<FromA>) &&
        (canonical_mask_target_acastable<basic_mask<E, FromA>, To> ||
            canonical_mask_source_acastable<basic_mask<E, FromA>, To>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<E, To> operator()(
        basic_mask<E, FromA> val) noexcept {
        using From = basic_mask<E, FromA>;
        if constexpr (canonical_mask_target_acastable<From, To>) {
            return abi_cast<To>(internal::abi<FromA>, val);
        } else {
            return abi_cast(internal::abi<To>, val);
        }
    }
    ///
    template <extended_simd_type From>
    requires extended_acastable<From, To>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(From val) noexcept
    requires common_abi_with<To, simd_abi_type_t<To>>
    {
        if constexpr (unqualified_extended_acastable<From, To>) {
            return abi_cast<To>(val);
        } else if constexpr (expression_acastable<From, To>) {
            return operator()(dx::evaluate(val));
        } else {
            return operator()(dx::to_canonical(val));
        }
    }
};

template <typename From, typename To>
concept unqualified_canonical_castable_to =
    common_simd_type_with<From, To> && requires(From from) {
        { simd_cast<To>(internal::abi<From>, from) } -> same_as<To>;
    };

template <typename From, typename To>
concept unqualified_canonical_castable_from =
    common_simd_type_with<From, To> && requires(From from) {
        { simd_cast<To>(internal::abi<To>, from) } -> same_as<To>;
    };
template <typename From, typename To>
concept unqualified_canonical_castable =
    unqualified_canonical_castable_to<From, To> ||
    unqualified_canonical_castable_from<From, To>;

template <typename From, typename To>
concept expression_castable = simd_expression<From> &&
    invocable<simd_cast_t<To>, simd_expression_result_t<From>>;

template <typename From, typename To>
concept extended_castable =
    explicitly_convertible_to<From, To> || expression_castable<From, To>;

template <canonical_simd_type To>
struct simd_cast_t<To> {
private:
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr To operator()(To from) noexcept {
        return from;
    }

    template <canonical_simd_type From>
    requires unqualified_canonical_castable<From, To>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr To operator()(From from) noexcept {
        if constexpr (unqualified_canonical_castable_to<From, To>) {
            return simd_cast<To>(internal::abi<From>, from);
        } else {
            return simd_cast<To>(internal::abi<To>, from);
        }
    }

    template <extended_simd_type From>
    requires same_as<canonical_type_t<From>, To>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr To operator()(From from) noexcept {
        if constexpr (explicitly_convertible_to<From, To>) {
            return static_cast<To>(from);
        } else {
            return dx::to_canonical(from);
        }
    }

    template <extended_simd_type From>
    requires (!same_as<canonical_type_t<From>, To>) &&
        extended_castable<From, To>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr To operator()(From from) noexcept {
        if constexpr (explicitly_convertible_to<From, To>) {
            return static_cast<To>(from);
        } else {
            return operator()(dx::evaluate(from));
        }
    }

    template <simd_type From>
    requires (!same_as<canonical_type_t<From>, To> &&
                 !extended_castable<From, To> &&
                 !(canonical_simd_type<From> &&
                     unqualified_canonical_castable<From, To>)) &&
        regular_invocable<abi_cast_t<simd_abi_type_t<To>>, From> &&
        same_as<invoke_result_t<abi_cast_t<simd_abi_type_t<To>>, From>, To>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr To operator()(From from) noexcept {
        return abi_cast_t<simd_abi_type_t<To>>::operator()(from);
    }

    template <simd_type From>
    requires (!same_as<canonical_type_t<From>, To> &&
                 !extended_castable<From, To> &&
                 !(canonical_simd_type<From> &&
                     unqualified_canonical_castable<From, To>)) &&
        regular_invocable<element_cast_t<simd_element_type_t<To>>, From> &&
        same_as<invoke_result_t<element_cast_t<simd_element_type_t<To>>, From>,
            To>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr To operator()(From from) noexcept {
        return element_cast_t<simd_element_type_t<To>>::operator()(from);
    }
};

template <extended_simd_type To>
struct simd_cast_t<To> {
public:
    template <simd_type From>
    requires (!explicitly_convertible_to<From, To>) &&
        regular_invocable<abi_cast_t<simd_abi_type_t<To>>, From> &&
        same_as<invoke_result_t<abi_cast_t<simd_abi_type_t<To>>, From>, To>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr To operator()(From from) noexcept {
        return abi_cast_t<simd_abi_type_t<To>>::operator()(from);
    }

    template <simd_type From>
    requires (!explicitly_convertible_to<From, To>) &&
        regular_invocable<element_cast_t<simd_element_type_t<To>>, From> &&
        same_as<invoke_result_t<element_cast_t<simd_element_type_t<To>>, From>,
            To>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr To operator()(From from) noexcept {
        return element_cast_t<simd_element_type_t<To>>::operator()(from);
    }

    template <simd_type From>
    requires explicitly_convertible_to<From, To>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr To operator()(From from) noexcept {
        return static_cast<To>(from);
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
