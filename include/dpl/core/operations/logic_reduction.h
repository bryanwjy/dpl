// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/basic/immediate_mask.h"
#  include "dpl/core/basic/to_canonical.h"
#  include "dpl/core/concepts/basic_type.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_mask.h"
#  include "dpl/core/type_traits/iota_sequence.h"
#  include "dpl/std/concepts/boolean_testable.h"
#  include "dpl/std/concepts/convertible_to.h"

#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::internal {

template <typename T>
concept unqualified_all_of = simd_mask<T> && requires(T mask) {
    { all_of(internal::abi<T>, mask) } -> core_convertible_to<bool>;
};

struct all_of_t {
private:
    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr bool DPL_VECTORCALL fallback(
        basic_mask<E, A> mask) noexcept {
        static_assert(
            fixed_width_abi<A>, "Scalable ABIs have no viable fallback");
        return []<size_t... Is>(auto mask, index_sequence<Is...>) {
            return (... && mask[Is]);
        }(mask, iota_sequence<E, A>);
    }

public:
    template <fixed_width_mask T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr bool operator()(T mask) noexcept {
        if constexpr (unqualified_all_of<T>) {
            if constexpr (canonical_mask<T>) {
                if consteval {
                    return fallback(mask);
                } else {
                    return all_of(internal::abi<T>, mask);
                }
            } else {
                return all_of(internal::abi<T>, mask);
            }
        } else if constexpr (canonical_mask<T>) {
            return fallback(mask);
        } else {
            return operator()(dx::to_canonical(mask));
        }
    }

    template <scalable_mask T>
    requires unqualified_all_of<T> || unqualified_all_of<canonical_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr bool operator()(T mask) noexcept {
        if constexpr (unqualified_all_of<T>) {
            return all_of(internal::abi<T>, mask);
        } else {
            return all_of(internal::abi<T>, dx::to_canonical(mask));
        }
    }

    template <immediate_mask_like T>
    static consteval bool operator()(T val) noexcept {
        if constexpr (requires {
                          { all_of(val) } -> boolean_testable;
                      }) {
            return all_of(val);
        } else {
            constexpr auto all =
                static_cast<typename T::value_type>((1ll << T::width) - 1);
            return (T::value & all) == all;
        }
    }
};

template <typename T>
concept unqualified_any_of = simd_mask<T> && requires(T mask) {
    { any_of(internal::abi<T>, mask) } -> core_convertible_to<bool>;
};

struct any_of_t {
private:
    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr bool DPL_VECTORCALL fallback(
        basic_mask<E, A> mask) noexcept {
        static_assert(
            fixed_width_abi<A>, "Scalable ABIs have no viable fallback");
        return []<size_t... Is>(auto mask, index_sequence<Is...>) {
            return (... || mask[Is]);
        }(mask, iota_sequence<E, A>);
    }

public:
    template <fixed_width_mask T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr bool operator()(T mask) noexcept {
        if constexpr (unqualified_any_of<T>) {
            if constexpr (canonical_mask<T>) {
                if consteval {
                    return fallback(mask);
                } else {
                    return any_of(internal::abi<T>, mask);
                }
            } else {
                return any_of(internal::abi<T>, mask);
            }
        } else if constexpr (canonical_mask<T>) {
            return fallback(mask);
        } else {
            return operator()(dx::to_canonical(mask));
        }
    }

    template <scalable_mask T>
    requires unqualified_any_of<T> || unqualified_any_of<canonical_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr bool operator()(T mask) noexcept {
        if constexpr (unqualified_any_of<T>) {
            return any_of(internal::abi<T>, mask);
        } else {
            return any_of(internal::abi<T>, dx::to_canonical(mask));
        }
    }

    template <immediate_mask_like T>
    static consteval bool operator()(T val) noexcept {
        if constexpr (requires {
                          { any_of(val) } -> boolean_testable;
                      }) {
            return any_of(val);
        } else {
            constexpr auto all =
                static_cast<typename T::value_type>((1ll << T::width) - 1);
            return (T::value & all) != 0;
        }
    }
};

template <typename T>
concept unqualified_none_of = simd_mask<T> && requires(T mask) {
    { none_of(internal::abi<T>, mask) } -> core_convertible_to<bool>;
};

struct none_of_t {
private:
    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr bool DPL_VECTORCALL fallback(
        basic_mask<E, A> mask) noexcept {
        return !any_of_t::operator()(mask);
    }

public:
    template <fixed_width_mask T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr bool DPL_VECTORCALL operator()(T mask) noexcept {
        if constexpr (unqualified_none_of<T>) {
            if constexpr (canonical_mask<T>) {
                if consteval {
                    return fallback(mask);
                } else {
                    return none_of(internal::abi<T>, mask);
                }
            } else {
                return none_of(internal::abi<T>, mask);
            }
        } else if constexpr (canonical_mask<T>) {
            return fallback(mask);
        } else {
            return operator()(dx::to_canonical(mask));
        }
    }

    template <scalable_mask T>
    requires unqualified_none_of<T> || unqualified_none_of<canonical_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr bool operator()(T mask) noexcept {
        if constexpr (unqualified_none_of<T>) {
            return none_of(internal::abi<T>, mask);
        } else {
            return none_of(internal::abi<T>, dx::to_canonical(mask));
        }
    }

    template <immediate_mask_like T>
    static consteval bool operator()(T val) noexcept {
        if constexpr (requires {
                          { none_of(val) } -> boolean_testable;
                      }) {
            return none_of(val);
        } else {
            constexpr auto all =
                static_cast<typename T::value_type>((1ll << T::width) - 1);
            return (T::value & all) == 0;
        }
    }
};

template <typename T>
concept unqualified_some_of = simd_mask<T> && requires(T mask) {
    { some_of(internal::abi<T>, mask) } -> core_convertible_to<bool>;
};

struct some_of_t {
private:
    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr bool DPL_VECTORCALL fallback(
        basic_mask<E, A> mask) noexcept {
        return any_of_t::operator()(mask) && !all_of_t::operator()(mask);
    }

public:
    template <fixed_width_mask T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr bool operator()(T mask) noexcept {
        if constexpr (unqualified_some_of<T>) {
            if constexpr (canonical_mask<T>) {
                if consteval {
                    return fallback(mask);
                } else {
                    return some_of(internal::abi<T>, mask);
                }
            } else {
                return some_of(internal::abi<T>, mask);
            }
        } else if constexpr (canonical_mask<T>) {
            return fallback(mask);
        } else {
            return operator()(dx::to_canonical(mask));
        }
    }

    template <scalable_mask T>
    requires unqualified_some_of<T> || unqualified_some_of<canonical_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr bool operator()(T mask) noexcept {
        if constexpr (unqualified_some_of<T>) {
            return some_of(internal::abi<T>, mask);
        } else {
            return some_of(internal::abi<T>, dx::to_canonical(mask));
        }
    }

    template <immediate_mask_like T>
    static consteval bool operator()(T val) noexcept {
        if constexpr (requires {
                          { some_of(val) } -> boolean_testable;
                      }) {
            return some_of(val);
        } else {
            return internal::any_of_t::operator()(val) &&
                !internal::all_of_t::operator()(val);
        }
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::all_of_t all_of{};
DPL_EXPORT inline constexpr internal::any_of_t any_of{};
DPL_EXPORT inline constexpr internal::none_of_t none_of{};
DPL_EXPORT inline constexpr internal::some_of_t some_of{};
} // namespace cpo
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
