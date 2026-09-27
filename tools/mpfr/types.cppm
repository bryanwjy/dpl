// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

export module mpfr:types;
import :utils;
import dpl;

export namespace mpfr {
template <size_t N>
struct type_name : static_string<N> {
private:
    using static_string<N>::data;

public:
    using static_string<N>::static_string;
};

template <size_t N>
type_name(char const (&data)[N]) -> type_name<N - 1>;

template <uint64 V>
struct type_hash : dpl::integral_constant<uint64, V> {

    using dpl::integral_constant<uint64, V>::integral_constant;
    friend consteval auto to_type(type_hash) noexcept;
};

template <typename T>
struct type_tag : dpl::type_identity<T> {
    friend consteval decltype(auto) to_type_name(type_tag) noexcept;
};

template <auto name, typename T>
struct type_map {
    using name_type = dpl::decay_t<decltype(name)>;
    using hash_type = type_hash<string_hash(name)>;
    friend consteval auto to_type(hash_type) noexcept { return type_tag<T>{}; }

    friend consteval decltype(auto) to_type_name(type_tag<T>) noexcept {
        return (name);
    }
};

inline namespace literals {
template <type_name S>
consteval auto operator""_tp_name() noexcept {
    return S;
}
} // namespace literals

} // namespace mpfr
