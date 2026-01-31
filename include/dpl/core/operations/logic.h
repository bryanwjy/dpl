// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/operations/abs.h"
#include "dpl/core/operations/bitwise.h"
#include "dpl/core/operations/negate.h"
#include "dpl/core/operations/permute.h"
#include "dpl/core/operations/select.h"

#if !DPL_MODULES
#  include "dpl/core/basic/load.h"
#  include "dpl/core/basic/reinterpret.h"
#  include "dpl/core/basic/to_basic_type.h"
#  include "dpl/core/concepts/basic_type.h"
#  include "dpl/core/concepts/common_size_with.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_equivalence.h"
#  include "dpl/core/concepts/simd_mask_type.h"
#  include "dpl/core/constants/exponent_bias.h"
#  include "dpl/core/constants/exponent_bits.h"
#  include "dpl/core/constants/max_value.h"
#  include "dpl/core/constants/min_value.h"
#  include "dpl/core/constants/zero.h"
#  include "dpl/core/type_traits/array_for.h"
#  include "dpl/core/type_traits/basic_element.h"
#  include "dpl/core/type_traits/basic_type.h"
#  include "dpl/core/type_traits/iota_sequence.h"
#  include "dpl/core/type_traits/rebind_simd.h"
#  include "dpl/std/concepts/convertible_to.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::internal {

template <typename T>
concept unqualified_all_of = simd_mask_type<T> && requires(T mask) {
    { all_of(internal::abi<T>, mask) } -> core_convertible_to<bool>;
};

struct all_of_t {
private:
    template <basic_simd_element E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr bool DPL_VECTORCALL
        fallback(simd_mask<E, A> mask) noexcept {
        return []<size_t... Is>(auto mask, index_sequence<Is...>) {
            return (... && mask[Is]);
        }(mask, iota_sequence<E, A>);
    }

public:
    template <basic_simd_mask_type T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr bool DPL_VECTORCALL operator()(T mask) noexcept {
        if constexpr (unqualified_all_of<T>) {
            if consteval {
                return fallback(mask);
            } else {
                return all_of(internal::abi<T>, mask);
            }
        } else {
            return fallback(mask);
        }
    }

    template <simd_mask_type T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr bool DPL_VECTORCALL operator()(T mask) noexcept {
        if constexpr (unqualified_all_of<T>) {
            return all_of(internal::abi<T>, mask);
        } else {
            return operator()(dx::to_basic_type(mask));
        }
    }

    template <size_t W, bit_type_t<W> V>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr bool operator()(basic_immediate_mask<W, V> val) noexcept {
        return all_of(val);
    }
};

template <typename T>
concept unqualified_any_of = simd_mask_type<T> && requires(T mask) {
    { any_of(internal::abi<T>, mask) } -> core_convertible_to<bool>;
};

struct any_of_t {
private:
    template <basic_simd_element E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr bool DPL_VECTORCALL
        fallback(simd_mask<E, A> mask) noexcept {
        return []<size_t... Is>(auto mask, index_sequence<Is...>) {
            return (... || mask[Is]);
        }(mask, iota_sequence<E, A>);
    }

public:
    template <basic_simd_mask_type T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr bool DPL_VECTORCALL operator()(T mask) noexcept {
        if constexpr (unqualified_any_of<T>) {
            if consteval {
                return fallback(mask);
            } else {
                return any_of(internal::abi<T>, mask);
            }
        } else {
            return fallback(mask);
        }
    }

    template <simd_mask_type T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr bool DPL_VECTORCALL operator()(T mask) noexcept {
        if constexpr (unqualified_any_of<T>) {
            return any_of(internal::abi<T>, mask);
        } else {
            return operator()(dx::to_basic_type(mask));
        }
    }

    template <size_t W, bit_type_t<W> V>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr bool operator()(basic_immediate_mask<W, V> val) noexcept {
        return any_of(val);
    }
};

template <typename T>
concept unqualified_none_of = simd_mask_type<T> && requires(T mask) {
    { none_of(internal::abi<T>, mask) } -> core_convertible_to<bool>;
};

struct none_of_t {
private:
    template <basic_simd_element E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr bool DPL_VECTORCALL
        fallback(simd_mask<E, A> mask) noexcept {
        return !any_of_t::operator()(mask);
    }

public:
    template <basic_simd_mask_type T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr bool DPL_VECTORCALL operator()(T mask) noexcept {
        if constexpr (unqualified_none_of<T>) {
            if consteval {
                return fallback(mask);
            } else {
                return none_of(internal::abi<T>, mask);
            }
        } else {
            return fallback(mask);
        }
    }

    template <simd_mask_type T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr bool DPL_VECTORCALL operator()(T mask) noexcept {
        if constexpr (unqualified_none_of<T>) {
            return none_of(internal::abi<T>, mask);
        } else {
            return operator()(dx::to_basic_type(mask));
        }
    }

    template <size_t W, bit_type_t<W> V>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr bool operator()(basic_immediate_mask<W, V> val) noexcept {
        return none_of(val);
    }
};

template <typename T>
concept unqualified_some_of = simd_mask_type<T> && requires(T mask) {
    { some_of(internal::abi<T>, mask) } -> core_convertible_to<bool>;
};

struct some_of_t {
private:
    template <basic_simd_element E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr bool DPL_VECTORCALL
        fallback(simd_mask<E, A> mask) noexcept {
        return any_of_t::operator()(mask) && !all_of_t::operator()(mask);
    }

public:
    template <basic_simd_mask_type T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr bool DPL_VECTORCALL operator()(T mask) noexcept {
        if constexpr (unqualified_some_of<T>) {
            if consteval {
                return fallback(mask);
            } else {
                return some_of(internal::abi<T>, mask);
            }
        } else {
            return fallback(mask);
        }
    }

    template <simd_mask_type T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr bool DPL_VECTORCALL operator()(T mask) noexcept {
        if constexpr (unqualified_some_of<T>) {
            return some_of(internal::abi<T>, mask);
        } else {
            return operator()(dx::to_basic_type(mask));
        }
    }

    template <size_t W, bit_type_t<W> V>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr bool operator()(basic_immediate_mask<W, V> val) noexcept {
        return some_of(val);
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::all_of_t all_of{};
DPL_EXPORT inline constexpr internal::any_of_t any_of{};
DPL_EXPORT inline constexpr internal::some_of_t some_of{};
DPL_EXPORT inline constexpr internal::none_of_t none_of{};
} // namespace cpo
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
