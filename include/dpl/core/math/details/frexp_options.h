// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/std/concepts/different_from.h"
#  include "dpl/std/type_traits/constants.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::fmath {

enum class frexp_sign : uint8 {
    src,  // DEFAULT (copies from source)
    zero, // (Always positive)
    nan,  // (NAN if negative)
};

enum class frexp_interval : uint8 {
    standard, // [0.5,1) // DEFAULT
    binade,   // [1,2)
    extended, // [0.5,2)
    reduced   // [0.75,1.5)
};

template <different_from<frexp_sign> R>
consteval bool operator==(frexp_sign, R) noexcept {
    return false;
}

template <different_from<frexp_interval> R>
consteval bool operator==(frexp_interval, R) noexcept {
    return false;
}

template <typename T>
class frexp_option_base {};

template <frexp_sign S, frexp_interval I>
struct frexp_options_t;
template <frexp_interval I>
class frexp_interval_t;

template <frexp_sign S>
class DPL_EMPTY_BASES frexp_sign_t :
    public frexp_option_base<frexp_sign_t<S>>,
    public integral_constant<frexp_sign, S> {
public:
    template <frexp_interval V>
    consteval frexp_options_t<S, V> operator|(
        this frexp_sign_t, frexp_interval_t<V>) noexcept {
        return frexp_options_t<S, V>{};
    }

    static consteval bool contains(frexp_interval val) noexcept {
        return val == frexp_interval::standard;
    }

    static consteval bool contains(frexp_sign val) noexcept { //
        return val == S;
    }
};

template <frexp_interval I>
class DPL_EMPTY_BASES frexp_interval_t :
    public frexp_option_base<frexp_interval_t<I>>,
    public integral_constant<frexp_interval, I> {
public:
    template <frexp_sign V>
    consteval frexp_options_t<V, I> operator|(
        this frexp_interval_t, frexp_sign_t<V>) noexcept {
        return frexp_options_t<V, I>{};
    }

    static consteval bool contains(frexp_interval val) noexcept {
        return val == I;
    }

    static consteval bool contains(frexp_sign val) noexcept {
        return val == frexp_sign::src;
    }
};

template <frexp_sign S, frexp_interval I>
class frexp_options_t : public frexp_option_base<frexp_options_t<S, I>> {
public:
    explicit consteval frexp_options_t() noexcept = default;

    consteval bool operator==(frexp_options_t) noexcept { return true; }
    consteval bool operator!=(frexp_options_t) noexcept { return false; }

    static consteval bool contains(frexp_interval val) noexcept {
        return val == S;
    }

    static consteval bool contains(frexp_sign val) noexcept { //
        return val == I;
    }
};

} // namespace datapar::fmath

namespace datapar {
template <typename T>
concept frexp_options_type = derived_from<T, fmath::frexp_option_base<T>>;
} // namespace datapar

__DPL_DEFAULT_NAMESPACE_END
