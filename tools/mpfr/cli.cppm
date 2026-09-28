// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <string_view>

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

template <cli_option_name name>
struct cli_option {
    static constexpr auto value = name;
    friend consteval auto to_short_option(cli_option opt) noexcept;
};

template <char C>
struct cli_short_option : dpl::integral_constant<char, C> {
    friend constexpr decltype(auto) to_long_option(
        cli_short_option opt) noexcept;
};

template <uint64 V>
struct cli_option_hash : dpl::integral_constant<uint64, V> {

    using dpl::integral_constant<uint64, V>::integral_constant;
    friend consteval decltype(auto) to_cli_option(cli_option_hash opt) noexcept;
};

template <cli_option_name name, char C>
struct cli_option_map {
    using hash_type = cli_option_hash<string_hash(name)>;
    friend consteval auto to_short_option(cli_option<name>) noexcept {
        return C;
    }
    friend constexpr decltype(auto) to_long_option(
        cli_short_option<C>) noexcept {
        return (name);
    }
    friend consteval decltype(auto) to_cli_option(hash_type hash) noexcept {
        return (name);
    }
};

inline namespace literals {
template <cli_option_name S>
consteval decltype(auto) operator""_cli_opt() noexcept {
    return (S);
}
} // namespace literals

template <uint64 V>
requires requires(cli_option_hash<V> hash) { to_cli_option(hash); }
consteval auto to_cli_option(dpl::integral_constant<uint64, V> = {}) noexcept {
    constexpr cli_option_hash<V> hash;
    return to_cli_option(hash);
}

template <typename T, T name>
requires requires { typename cli_option<name>; } &&
    requires(cli_option<name> opt) { to_short_option(opt); }
consteval auto to_short_option(dpl::integral_constant<T, name> = {}) noexcept {
    constexpr cli_option<name> opt;
    return to_short_option(opt);
}

template <char C>
consteval decltype(auto) to_long_option(
    dpl::integral_constant<char, C> = {}) noexcept
requires requires { to_long_option(cli_short_option<C>{}); }
{
    return to_long_option(cli_short_option<C>{});
}

#define DEFINE_CLI_OPT(NAME, C)                              \
    [] {                                                     \
        constexpr auto opt = DPL_CONCAT(#NAME, _cli_opt);    \
        static_assert(sizeof(mpfr::cli_option_map<opt, C>)); \
        return opt;                                          \
    }()

constexpr dpl::constant_type_pack< //
    DEFINE_CLI_OPT(output, 'o'),   //
    DEFINE_CLI_OPT(count, 'n'),    //
    DEFINE_CLI_OPT(seed, 's'),     //
    DEFINE_CLI_OPT(input, 'i'),    //
    DEFINE_CLI_OPT(append, 'a'),   //
    DEFINE_CLI_OPT(help, 'h')>
    cli_options = {};

#undef DEFINE_CLI_OPT

constexpr std::string_view find_long_option(char val) noexcept {
    constexpr auto jump_table = __DPL apply(
        [](auto... types) {
            return mpfr::make_jump_table<mpfr::to_short_option(types)...>();
        },
        cli_options);

    return jump_table(
        [](auto key) {
            if constexpr (requires { mpfr::to_long_option(key); }) {
                return +mpfr::to_long_option(key);
            } else {
                return std::string_view();
            }
        },
        val);
}

} // namespace mpfr
