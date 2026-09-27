// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

export module mpfr:cli;
import :utils;
import dpl;

export namespace mpfr {
template <size_t N>
struct cli_option_name : static_string<N> {
private:
    using static_string<N>::data;

public:
    using static_string<N>::static_string;
};

template <size_t N>
cli_option_name(char const (&data)[N]) -> cli_option_name<N - 1>;

template <auto name>
struct cli_option {
    static constexpr auto value = name;
    friend consteval auto to_short_option(cli_option opt) noexcept;
};

template <char C>
struct cli_short_option : dpl::integral_constant<char, C> {
    friend constexpr auto to_long_option(cli_short_option opt) noexcept;
};

template <uint64 V>
struct cli_option_hash : dpl::integral_constant<uint64, V> {

    using dpl::integral_constant<uint64, V>::integral_constant;
    friend consteval auto to_cli_option(cli_option_hash opt) noexcept;
};

template <auto name, char C>
struct cli_option_map {
    using hash_type = cli_option_hash<string_hash(name)>;
    friend consteval auto to_short_option(cli_option<name>) noexcept {
        return C;
    }
    friend constexpr auto to_long_option(cli_short_option<C>) noexcept {
        return name;
    }
    friend consteval auto to_cli_option(hash_type hash) noexcept {
        return name;
    }
};

inline namespace literals {
template <cli_option_name S>
consteval auto operator""_cli_opt() noexcept {
    return S;
}

template <char C>
constexpr auto to_long_option() noexcept
requires requires { to_long_option(cli_short_option<C>{}); }
{
    return to_long_option(cli_short_option<C>{});
}

} // namespace literals

} // namespace mpfr
