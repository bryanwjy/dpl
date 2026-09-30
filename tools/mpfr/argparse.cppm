// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <algorithm>
#include <charconv>
#include <expected>
#include <ranges>
#include <string_view>
#include <variant>
#include <vector>

export module mpfr:argparse;
import :utils;
import :cli;
import :typeconv;
import dpl;

namespace tpc = mpfr::typeconv;

export namespace mpfr {

template <cli_option_name name>
struct argument_tag {
    friend consteval auto to_type_identity(argument_tag) noexcept;
};

template <cli_option_name name, tpc::convertible T>
struct argument_type_map {
    friend consteval auto to_type_identity(argument_tag<name>) noexcept {
        return dpl::type_identity<T>{};
    }
};

template <cli_option_name name>
requires requires(argument_tag<name> opt) { to_type_identity(opt); }
using argument_type_t =
    typename decltype(to_type_identity(argument_tag<name>{}))::type;

template <cli_option_name name, auto... V>
struct argument_default_map {};

template <cli_option_name name,
    dpl::convertible_to<tpc::storage_type_t<argument_type_t<name>>> auto V>
struct argument_default_map<name, V> {
    using value_type = tpc::storage_type_t<argument_type_t<name>>;

    friend consteval value_type get_default(cli_option<name>) noexcept {
        return V;
    }
};

template <cli_option_name name>
requires requires(argument_tag<name> tag) { get_default(tag); }
consteval auto get_default() noexcept {
    constexpr argument_tag<name> tag;
    return get_default(tag);
}

template <cli_option_name opt, char C, tpc::convertible T, auto... I>
requires (sizeof...(I) <= 1)
consteval auto define_cli_option() noexcept {
    constexpr cli_option_map<opt, C> map1;
    constexpr argument_type_map<opt, T> map2;
    constexpr argument_default_map<opt, I...> map3;
    return opt;
}

constexpr dpl::constant_type_pack< //
    define_cli_option<"output"_cli_opt, 'o', tpc::string>(),
    define_cli_option<"count"_cli_opt, 'n', dpl::uint32, 512u>(),
    define_cli_option<"seed"_cli_opt, 's', dpl::uint32, 2813478177u>(),
    // each 'input' argument tuple will append to a list
    // all argument tuple of 'input' must have the same size
    define_cli_option<"input"_cli_opt, 'i',
        tpc::list<tpc::tuple<tpc::unbounded<tpc::string>>>>(),
    define_cli_option<"append"_cli_opt, 'a', bool>(),
    define_cli_option<"help"_cli_opt, 'h', bool>(),
    // each 'range' argument set will replace the previous one
    define_cli_option<"range"_cli_opt, 'r',
        tpc::tuple<tpc::string, tpc::unbounded<tpc::string>>>()>
    cli_options = {};

constexpr std::string_view find_long_option(char val) noexcept {
    constexpr auto jump_table = dpl::apply(
        [](auto... types) {
            return make_jump_table<to_short_option(types)...>();
        },
        cli_options);

    return jump_table(
        [](auto key) {
            if constexpr (requires { to_long_option(key); }) {
                return +to_long_option(key);
            } else {
                return std::string_view();
            }
        },
        val);
}

template <cli_option_name O>
class argument_storage : typeconv::storage<argument_type_t<O>> {
private:
    using base_type = typeconv::storage<argument_type_t<O>>;

public:
    constexpr argument_storage() noexcept = default;
    constexpr argument_storage() noexcept
    requires requires { mpfr::get_default<O>(); }
        : base_type(mpfr::get_default<O>()) {}

    using base_type::extent;
    using base_type::get;
    using base_type::update;
};

consteval auto arguments_tuple() noexcept {
    return dpl::apply(
        [](auto... args) { return std::tuple<argument_storage<args()>...>(); },
        cli_options);
}

enum class parse_error {
    invalid_argument,
    unrecognized_argument,
    io_error,
};

class argument_parser {
    using argument_view = std::span<char const* const>;
    static constexpr auto arg_chunker =
        std::views::chunk_by([](std::string_view, std::string_view next) {
            return !(next.starts_with('@') || next.starts_with("--") ||
                (next.size() == 2 && next[0] == '-' &&
                    !mpfr::find_long_option(next[1]).empty()));
        });

    using parse_result = std::expected<void, parse_error>;

public:
    argument_parser(int argc, char const** argv) noexcept
        : cli_args_(argv, argv + argc)
        , positionals_([&] {
            auto const args = cli_args_.subspan(1);
            auto const end =
                std::ranges::find_if(args, [](std::string_view arg) {
                    return arg.starts_with('@') || arg.starts_with("--") ||
                        (arg.size() == 2 && arg[0] == '-' &&
                            !mpfr::find_long_option(arg[1]).empty());
                });
            return std::span{args.begin(), end};
        }()) {}

    constexpr size_t argc() const noexcept { return cli_args_.size(); }
    constexpr argument_view argv() const noexcept { return cli_args_; }
    constexpr std::string_view program() const noexcept {
        return cli_args_.front();
    }
    constexpr std::string_view positional(size_t idx) const noexcept {
        return idx < positionals_.size() ? positionals_[idx]
                                         : std::string_view();
    }
    constexpr argument_view positionals() const noexcept {
        return positionals_;
    }

    parse_result parse() {
        for (auto chunk : cli_args_.subspan(1 + positionals_.size()) |
                std::views::transform([](char const* ptr) {
                    return mpfr::strip(std::string_view(ptr));
                }) |
                arg_chunker) {
            if (std::ranges::empty(chunk)) {
                continue;
            }

            auto const current = chunk.front();
            if (!current.starts_with('@')) {
                chunk.advance(1);
                if (auto result = parse_arg(current, dpl::move(chunk));
                    !result) {
                    return result;
                }
            } else if (auto result = parse_response_file(current.substr(1));
                !result) {
                return result;
            }
        }

        return {};
    }

    template <cli_option_name O, typename S>
    constexpr decltype(auto) value(this S&& self) {
        return dpl::forward_like<S>(
            std::get<argument_storage<O>>(self.args_).get().value());
    }

    template <cli_option_name O>
    constexpr bool has_value() const noexcept {
        return std::get<argument_storage<O>>(args_).get().has_value();
    }

    template <cli_option_name O>
    constexpr size_t extent() const noexcept {
        return std::get<argument_storage<O>>(args_).extent();
    }

    template <cli_option_name O, typename S,
        typename V = tpc::storage_type_t<argument_type_t<O>>>
    constexpr auto value_or(this S&& self, V&& val) noexcept {
        return dpl::forward_like<S>(
            std::get<argument_storage<O>>(self.args_).get())
            .value_or(dpl::forward<V>(val));
    }

private:
    parse_result parse_response_file(std::string_view filepath) {
        auto const& file =
            response_files_.emplace_back(mpfr::tmp_string(filepath).c_str());
        if (!file) {
            fprintf(stderr, "Unable to open response file '%.*s'\n",
                static_cast<int>(filepath.size()), filepath.data());
            return std::unexpected(parse_error::io_error);
        }

        auto const fileview = std::string_view(
            reinterpret_cast<char const*>(file.data()), file.size());

        for (auto chunk : fileview | std::views::split('\n') |
                std::views::transform([](auto line) {
                    return mpfr::strip(std::string_view(
                        std::ranges::begin(line), std::ranges::end(line)));
                }) |
                std::views::filter(
                    [](auto line) { return !line.starts_with('#'); }) |
                arg_chunker) {

            if (std::ranges::empty(chunk)) {
                continue;
            }

            auto const current = chunk.front();
            if (auto const result = parse_arg(current, chunk.advance(1));
                !result) {
                return result;
            }
        }

        return {};
    }

    parse_result parse_arg(std::string_view opt, auto chunk) {
        auto const error_path = [opt]() {
            fprintf(stderr, "Unrecognized option: '%.*s'\n",
                static_cast<int>(opt.size()), opt.data());
            return std::unexpected(parse_error::unrecognized_argument);
        };

        auto const opt_offset = opt.find_first_not_of('-');
        if (opt_offset > 2zu) {
            return {};
        }

        opt.remove_prefix(opt_offset);
        if (opt_offset == 1) {
            auto const longopt = opt.size() > 2
                ? std::string_view()
                : mpfr::find_long_option(opt.front());
            if (longopt.empty()) {
                return error_path();
            }

            opt = longopt;
        }

        constexpr auto jump_table = dpl::apply(
            [](auto... types) {
                return mpfr::make_jump_table<mpfr::string_hash(types())...>();
            },
            mpfr::cli_options);

        return jump_table(
            [this, &error_path](auto hash, auto chunk) -> parse_result {
                if constexpr (requires { mpfr::to_cli_option(hash); }) {
                    constexpr auto& const_opt = mpfr::to_cli_option(hash);
                    using T = argument_storage<const_opt>;
                    auto const size =
                        static_cast<size_t>(std::ranges::distance(chunk));
                    return std::get<T>(args_)
                        .update(dpl::move(chunk))
                        .transform_error([&](typeconv::errc code) {
                            switch (code) {
                            case typeconv::errc::parsing_error:
                                fprintf(stderr,
                                    "Encountered parsing error during "
                                    "processing of "
                                    "option '%s'\n",
                                    const_opt.c_str());
                                break;
                            case typeconv::errc::extent_mismatch: {
                                if (auto extent = std::get<T>(args_).extent();
                                    extent != tpc::uninitialized_extent) {
                                    fprintf(stderr,
                                        "Encountered extent mismatch during "
                                        "processing "
                                        "of option '%s': Expecting %zu "
                                        "arguments but "
                                        "received %zu\n",
                                        const_opt.c_str(), extent, size);
                                } else {
                                    using tpc_type = argument_type_t<const_opt>;
                                    fprintf(stderr,
                                        "Encountered extent mismatch during "
                                        "processing "
                                        "of option '%s': Expecting argument "
                                        "count "
                                        "(%zu) to be greater than or equal to "
                                        "the "
                                        "minimum extent (%zu)\n",
                                        const_opt.c_str(), size,
                                        tpc::minimum_extent<tpc_type>);
                                }
                                break;
                            }
                            default:
                                // uninitialized will never be encountered here
                                dpl::unreachable();
                            };
                            return parse_error::invalid_argument;
                        });
                } else {
                    return error_path();
                }
            },
            mpfr::string_hash(opt), dpl::move(chunk));
    }

    using arguments_tuple_t = decltype(arguments_tuple());

    std::span<char const* const> cli_args_;
    std::span<char const* const> positionals_;
    arguments_tuple_t args_;
    std::vector<mpfr::mapped_memory> response_files_;
};

} // namespace mpfr
