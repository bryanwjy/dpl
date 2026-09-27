// Copyright 2026 Bryan Wong
// offline MPFR reference-data generator for DPL math CPO tests.
//
// Usage:  dpl_mpfr_gen <func> <type> [options] [@file ...]      (see --help)
//
// Each record's expected value is computed by MPFR directly in the target
// format (target precision and exponent range, subnormals via
// mpfr_subnormalize), so it is correctly rounded.
//
// Formats are handled as raw bit patterns; no compiler support for f16/bf16 is
// needed. Every value of every supported format is exactly representable as a
// double, which is used as the carrier between MPFR and the bit encoding.
//

#include "dpl/config.h"

#include <ctype.h>
#include <mpfr.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <expected>
#include <filesystem>
#include <numeric>
#include <ranges>
#include <string_view>
#include <system_error>
#include <tuple>
#include <vector>

import mpfr;
import dpl;
import dpl.test.support;

namespace {

using mpfr::literals::operator""_func;
using mpfr::literals::operator""_tp_name;
using mpfr::literals::operator""_cli_opt;

#define _CONCAT(X, Y) X##Y
#define CONCAT(X, Y) _CONCAT(X, Y)

#define DEFINE_FUNCTION(NAME, FUNC)                                 \
    [] {                                                            \
        constexpr auto name = CONCAT(#NAME, _func);                 \
        static_assert(sizeof(mpfr::function_map<name, &FUNC>) > 0); \
        return name;                                                \
    }()

constexpr std::tuple functions = {
    // DEFINE_FUNCTION(abs, mpfr_abs),
    DEFINE_FUNCTION(acos, mpfr_acos),
    DEFINE_FUNCTION(acosh, mpfr_acosh),
    DEFINE_FUNCTION(asin, mpfr_asin),
    DEFINE_FUNCTION(asinh, mpfr_asinh),
    DEFINE_FUNCTION(atan, mpfr_atan),
    DEFINE_FUNCTION(atan2, mpfr_atan2),
    DEFINE_FUNCTION(atanh, mpfr_atanh),
    DEFINE_FUNCTION(beta, mpfr_beta),
    DEFINE_FUNCTION(cbrt, mpfr_cbrt),
    DEFINE_FUNCTION(ceil, mpfr_ceil),
    // DEFINE_FUNCTION(copysign, mpfr_copysign),
    DEFINE_FUNCTION(cos, mpfr_cos),
    DEFINE_FUNCTION(cosh, mpfr_cosh),
    DEFINE_FUNCTION(cot, mpfr_cot),
    DEFINE_FUNCTION(coth, mpfr_coth),
    DEFINE_FUNCTION(csc, mpfr_csc),
    DEFINE_FUNCTION(csch, mpfr_csch),
    DEFINE_FUNCTION(erf, mpfr_erf),
    DEFINE_FUNCTION(erfc, mpfr_erfc),
    DEFINE_FUNCTION(exp, mpfr_exp),
    DEFINE_FUNCTION(exp10, mpfr_exp10),
    DEFINE_FUNCTION(exp2, mpfr_exp2),
    DEFINE_FUNCTION(expm1, mpfr_expm1),
    DEFINE_FUNCTION(floor, mpfr_floor),
    DEFINE_FUNCTION(fmod, mpfr_fmod),
    DEFINE_FUNCTION(hypot, mpfr_hypot),
    DEFINE_FUNCTION(tgamma, mpfr_gamma),
    DEFINE_FUNCTION(lgamma, mpfr_lngamma),
    DEFINE_FUNCTION(log, mpfr_log),
    DEFINE_FUNCTION(log10, mpfr_log10),
    DEFINE_FUNCTION(log1p, mpfr_log1p),
    DEFINE_FUNCTION(log2, mpfr_log2),
    // DEFINE_FUNCTION(modf, mpfr_modf),
    // DEFINE_FUNCTION(nextabove, mpfr_nextabove),
    // DEFINE_FUNCTION(nextbelow, mpfr_nextbelow),
    DEFINE_FUNCTION(nexttoward, mpfr_nexttoward),
    DEFINE_FUNCTION(pow, mpfr_pow),
    // DEFINE_FUNCTION(ldexp, mpfr_mul_2si), // TODO
    DEFINE_FUNCTION(remainder, mpfr_remainder),
    // DEFINE_FUNCTION(remquo, mpfr_remquo),
    DEFINE_FUNCTION(round, mpfr_round),         // CMATH
    DEFINE_FUNCTION(roundeven, mpfr_roundeven), // IEEE
    DEFINE_FUNCTION(sec, mpfr_sec),
    DEFINE_FUNCTION(sech, mpfr_sech),
    DEFINE_FUNCTION(sin, mpfr_sin),
    DEFINE_FUNCTION(sinh, mpfr_sinh),
    DEFINE_FUNCTION(sqrt, mpfr_sqrt),
    DEFINE_FUNCTION(tan, mpfr_tan),
    DEFINE_FUNCTION(tanh, mpfr_tanh),
    DEFINE_FUNCTION(trunc, mpfr_trunc),
    DEFINE_FUNCTION(zeta, mpfr_zeta),
};
#undef DEFINE_FUNCTION

#define DEFINE_FLOAT_TYPE(NAME, TP)                          \
    [] {                                                     \
        constexpr auto name = CONCAT(#NAME, _tp_name);       \
        static_assert(sizeof(mpfr::type_map<name, TP>) > 0); \
        return name;                                         \
    }()

constexpr std::tuple float_types = {
    DEFINE_FLOAT_TYPE(f16, dpl::ext::float16),
    DEFINE_FLOAT_TYPE(bf16, dpl::ext::bfloat16),
    DEFINE_FLOAT_TYPE(f32, float),
    DEFINE_FLOAT_TYPE(f64, double),
};

#undef DEFINE_FLOAT_TYPE

#define DEFINE_CLI_OPT(NAME, C)                              \
    [] {                                                     \
        constexpr auto opt = CONCAT(#NAME, _cli_opt);        \
        static_assert(sizeof(mpfr::cli_option_map<opt, C>)); \
        return mpfr::cli_option<opt>{};                      \
    }()

constexpr std::tuple cli_options = {
    DEFINE_CLI_OPT(output, 'o'),
    DEFINE_CLI_OPT(count, 'n'),
    DEFINE_CLI_OPT(seed, 's'),
    DEFINE_CLI_OPT(input, 'i'),
    DEFINE_CLI_OPT(append, 'a'),
    DEFINE_CLI_OPT(help, 'h'),
};

#undef DEFINE_CLI_OPT

inline std::string_view find_long_option(char val) noexcept {
    constexpr dpl::make_index_sequence<
        std::tuple_size_v<dpl::decay_t<decltype(cli_options)>>>
        seq;
    return [&]<size_t... Is>(dpl::index_sequence<Is...>) {
        constexpr mpfr::jump_table<char,
            to_short_option(std::get<Is>(cli_options))...>
            table = {};
        return table(
            []<typename U>(U) {
                if constexpr (requires {
                                  U::value;
                                  mpfr::to_long_option<U::value>();
                              }) {
                    static constexpr auto opt =
                        mpfr::to_long_option<U::value>();
                    return std::string_view(opt);
                } else {
                    return std::string_view();
                }
            },
            val);
    }(seq);
}

template <mpfr::function_name_type auto name>
constexpr auto is_exact_v = false;

template <>
constexpr auto is_exact_v<"nexttoward"_func> = true;
template <>
constexpr auto is_exact_v<"ceil"_func> = true;
template <>
constexpr auto is_exact_v<"floor"_func> = true;
template <>
constexpr auto is_exact_v<"trunc"_func> = true;
template <>
constexpr auto is_exact_v<"roundeven"_func> = true;
template <>
constexpr auto is_exact_v<"round"_func> = true;

template <mpfr::function_name_type auto name>
constexpr auto argument_count_v = 1zu;
template <>
constexpr auto argument_count_v<"pow"_func> = 2zu;
template <>
constexpr auto argument_count_v<"hypot"_func> = 2zu;
template <>
constexpr auto argument_count_v<"fmod"_func> = 2zu;
template <>
constexpr auto argument_count_v<"tan2"_func> = 2zu;
template <>
constexpr auto argument_count_v<"atan2"_func> = 2zu;
template <>
constexpr auto argument_count_v<"beta"_func> = 2zu;
template <>
constexpr auto argument_count_v<"remainder"_func> = 2zu;

template <dpl::floating_point_like T>
constexpr int precision_v = dpl::floating_point_traits<T>::digits + 32;

enum class range_type {
    linear,
    urand
};

struct configuration {
    std::span<std::string_view const> inputs;
    range_type range_type = range_type::urand;
    std::string_view ranges_spec;
    dpl::uint32 range_count = 100;
    dpl::uint32 seed = 2813478177u;
    std::string_view output;
    bool append;
};

template <dpl::floating_point_like T>
constexpr auto expectation_type_v = static_cast<mpfr::expectation_type>(0xff);
template <>
constexpr auto expectation_type_v<dpl::ext::float16> =
    mpfr::expectation_type::f16;
template <>
constexpr auto expectation_type_v<dpl::ext::bfloat16> =
    mpfr::expectation_type::bf16;
template <>
constexpr auto expectation_type_v<float> = mpfr::expectation_type::f32;
template <>
constexpr auto expectation_type_v<double> = mpfr::expectation_type::f64;

template <dpl::floating_point_like T, size_t N>
class input_generator {
public:
    auto process_inputs(configuration const& config) noexcept {
        inputs_.reserve(config.inputs.size() * N + config.range_count);
        dpl::test::mt19937 rng(config.seed);
        return generate_ranges(
            rng, config.range_type, config.ranges_spec, config.range_count)
            .and_then([&]() { return generate_inputs(config.inputs); })
            .and_then([this]() {
                using result_type = std::span<std::array<T, N> const>;
                using expected_type =
                    std::expected<result_type, std::error_code>;
                return expected_type(std::in_place, std::span{inputs_});
            });
    }

private:
    std::expected<void, std::error_code> generate_ranges(
        dpl::test::mt19937& rng, range_type type, std::string_view bounds,
        dpl::uint32 count) {
        if (type == range_type::linear) {
            auto const result = generate_linear(bounds, count);
            return result ? std::unexpected(result)
                          : std::expected<void, std::error_code>(std::in_place);
        } else {
            auto const result = generate_random(rng, bounds, count);
            return result ? std::unexpected(result)
                          : std::expected<void, std::error_code>(std::in_place);
        }
    }

    std::expected<void, std::error_code> generate_inputs(
        std::span<std::string_view const> inputs) {
        std::array<T, N> vals;
        auto it = vals.begin();
        for (auto const str : inputs) {
            for (auto const val : str | std::views::split(',') |
                    std::views::transform([](auto chunk) {
                        return mpfr::strip(std::string_view(chunk));
                    }) |
                    std::views::take(vals.size())) {
                if (auto result = parse_float(*it++, str)) {
                    fprintf(stderr,
                        "Linear range parsing failure of value: '%.*s'\n",
                        static_cast<int>(str.size()), str.data());
                    return std::unexpected(
                        std::make_error_code(std::errc::invalid_argument));
                }
            }

            if (it != vals.end()) {
                fprintf(stderr,
                    "Insufficient input argument specified, expected %zu, "
                    "parsed "
                    "%ld\n",
                    N,
                    static_cast<long>(std::ranges::distance(vals.begin(), it)));
                return std::unexpected(
                    std::make_error_code(std::errc::invalid_argument));
            }

            inputs_.emplace_back(vals);
        }

        return std::expected<void, std::error_code>(std::in_place);
    }

    std::error_code generate_random(
        dpl::test::mt19937& rng, std::string_view bounds, dpl::uint32 count) {
        if (bounds.empty()) {
            dpl::test::scalar_generator<T> const generator;
            std::array<T, N> vals;
            for (auto const _ : std::views::iota(0u, count)) {
                for (auto& val : vals) {
                    val = generator(rng);
                }
                inputs_.emplace_back(vals);
            }

            return std::error_code();
        }

        std::array<std::pair<T, T>, N> pairs;
        auto it = pairs.begin();
        for (auto const pair : bounds | std::views::split(',') |
                std::views::transform([](auto chunk) {
                    return mpfr::strip(std::string_view(
                        std::ranges::begin(chunk), std::ranges::end(chunk)));
                })) {

            auto const mid = pair.find(':');
            if (mid >= pair.size()) {
                fprintf(stderr, "Invalid linear range encountered: '%.*s'\n",
                    static_cast<int>(pair.size()), pair.data());
                return std::make_error_code(std::errc::invalid_argument);
            }
            using std::string_view_literals::operator""sv;

            auto const start =
                mid == 0zu ? "-infinity"sv : pair.substr(0zu, mid);
            auto const end =
                mid + 1 == pair.size() ? "infinity"sv : pair.substr(mid + 1);

            if (auto result = parse_float(it->first, start)) {
                fprintf(stderr,
                    "Linear range parsing failure of value: '%.*s'\n",
                    static_cast<int>(start.size()), start.data());
                return std::make_error_code(std::errc::invalid_argument);
            }

            if (auto result = parse_float(it->second, end)) {
                fprintf(stderr,
                    "Linear range parsing failure of value: '%.*s'\n",
                    static_cast<int>(start.size()), start.data());
                return std::make_error_code(std::errc::invalid_argument);
            }

            it++;
        }

        if (it != pairs.end()) {
            fprintf(stderr,
                "Insufficient range argument specified, expected %zu, parsed "
                "%ld\n",
                N, static_cast<long>(std::ranges::distance(pairs.begin(), it)));
            return std::make_error_code(std::errc::invalid_argument);
        }

        return [&]<size_t... Is>(dpl::index_sequence<Is...>) {
            std::array const gens{dpl::test::scalar_generator<T>(
                std::min(pairs[Is].first, pairs[Is].second),
                std::max(pairs[Is].first, pairs[Is].second))...};

            for (auto const _ : std::views::iota(0u, count)) {
                inputs_.emplace_back(std::array{gens[Is](rng)...});
            }

            return std::error_code();
        }(dpl::make_index_sequence<N>{});
    }

    std::error_code generate_linear(
        std::string_view bounds, dpl::uint32 count) {
        std::array<std::pair<T, T>, N> pairs;
        auto it = pairs.begin();
        for (auto const pair : bounds | std::views::split(',') |
                std::views::transform([](auto chunk) {
                    return mpfr::strip(std::string_view(
                        std::ranges::begin(chunk), std::ranges::end(chunk)));
                }) |
                std::views::take(pairs.size())) {
            auto const mid = pair.find(':');
            if (mid >= pair.size()) {
                fprintf(stderr, "Invalid linear range encountered: '%.*s'\n",
                    static_cast<int>(pair.size()), pair.data());
                return std::make_error_code(std::errc::invalid_argument);
            }

            auto const start = pair.substr(0zu, mid);
            auto const end = pair.substr(mid + 1);

            if (auto result = parse_float(it->first, start)) {
                fprintf(stderr,
                    "Linear range parsing failure of value: '%.*s'\n",
                    static_cast<int>(start.size()), start.data());
                return std::make_error_code(std::errc::invalid_argument);
            }

            if (auto result = parse_float(it->second, end)) {
                fprintf(stderr,
                    "Linear range parsing failure of value: '%.*s'\n",
                    static_cast<int>(start.size()), start.data());
                return std::make_error_code(std::errc::invalid_argument);
            }

            it++;
        }
        if (it != pairs.end()) {
            fprintf(stderr,
                "Insufficient range argument specified, expected %zu, parsed "
                "%ld\n",
                N, std::ranges::distance(pairs.begin(), it));
            return std::make_error_code(std::errc::invalid_argument);
        }

        return [&]<size_t... Is>(dpl::index_sequence<Is...>) {
            std::array vals{pairs[Is].first...};
            std::array const steps{
                ((pairs[Is].second - pairs[Is].first) / count)...};
            for (auto const _ : std::views::iota(0u, count)) {
                (..., (vals[Is] += steps[Is]));
                inputs_.emplace_back(vals);
            }

            return std::error_code();
        }(dpl::make_index_sequence<N>{});
    }

    std::error_code parse_float(T& val, std::string_view str) noexcept {
        auto const negative = str.starts_with('-');
        str.remove_prefix(negative ? 1zu : 0zu);

        constexpr auto lowercase = [](char val) { return ::tolower(val); };
        if (std::ranges::equal(
                str, "nan", std::ranges::equal_to{}, lowercase)) {
            constexpr auto nanbits = ~dpl::bitset<dpl::type_bit_v<T>>();
            val = dpl::bit_cast<T>(nanbits >> !negative);
            return std::error_code();
        }

        auto const result = [&] {
            if (std::ranges::equal(
                    str, "infinity", std::ranges::equal_to{}, lowercase)) {
                val = dpl::bit_cast<T>(
                    dpl::floating_point_traits<T>::exponent_mask);
                return static_cast<std::errc>(0);
            }

            auto const ishex = str.starts_with("0x") || str.starts_with("0X");
            auto const format =
                ishex ? std::chars_format::hex : std::chars_format::general;
            str.remove_prefix(ishex ? 2zu : 0zu);
            if constexpr (sizeof(T) < sizeof(float)) {
                float tmp = 0;
                auto result = std::from_chars(
                    str.data(), str.data() + str.size(), tmp, format)
                                  .ec;
                val = tmp;
                return result;
            } else {
                return std::from_chars(
                    str.data(), str.data() + str.size(), val, format)
                    .ec;
            }
        }();

        if (negative) {
            val = -val;
        }

        return std::make_error_code(result);
    }

    std::vector<std::array<T, N>> inputs_;
};

/**
 * Brought out of generator to make it faster to compile
 */
inline std::error_code verify_append(FILE* stream, std::string_view filepath,
    mpfr::latest_file_header const& header, mpfr::expectation_type type,
    std::string_view function, size_t arity) {

    constexpr auto header_meta = mpfr::latest_file_header{}.base;
    if (memcmp(&header_meta, &header.base, sizeof(header_meta)) != 0) {
        fprintf(stderr, "Cannot append to unrecognized datafile: '%.*s'\n",
            static_cast<int>(filepath.size()), filepath.data());
        return std::make_error_code(std::errc::illegal_byte_sequence);
    }

    if (header.expectation_type.get() != type) {
        fprintf(stderr, "Mismatch expectation type in datafile: '%.*s'\n",
            static_cast<int>(filepath.size()), filepath.data());
        return std::make_error_code(std::errc::illegal_byte_sequence);
    }

    if (header.function_name_length.get() != function.size() ||
        header.function_arity.get() != arity) {
        fprintf(stderr, "Mismatch function in datafile: '%.*s'\n",
            static_cast<int>(filepath.size()), filepath.data());
        return std::make_error_code(std::errc::illegal_byte_sequence);
    }

    struct filepos {
        FILE* file;
        long offset;
        ~filepos() {
            if (file != nullptr && offset >= 0) {
                fseek(file, offset, SEEK_SET);
            }
        }
    } const pos{.file = stream, .offset = ftell(stream)};

    if (fseek(stream, header.function_name_offset.get(), SEEK_SET) != 0) {
        return std::make_error_code(static_cast<std::errc>(errno));
    }

    mpfr::tmp_string name_buffer(function);
    if (fread(name_buffer.data(), 1zu, function.size(), stream) <
        function.size()) {
        return std::make_error_code(static_cast<std::errc>(errno));
    }

    if (function != name_buffer.c_str()) {
        fprintf(stderr, "Mismatch function in datafile: '%.*s'\n",
            static_cast<int>(filepath.size()), filepath.data());
        return std::make_error_code(std::errc::illegal_byte_sequence);
    }

    return std::error_code();
}

template <dpl::floating_point_like T, auto func>
class generator {
    using input_type = std::array<T, argument_count_v<func>>;

public:
    explicit generator(configuration config) : config_(config) {
        if (config_.output.empty()) {
            static constexpr auto func_name = +func;
            static constexpr auto tp_name_v = to_type_name(mpfr::type_tag<T>{});
            static constexpr auto tp_name = std::string_view(tp_name_v);
            static constexpr std::string_view extension = ".dat";
            constexpr auto length =
                func_name.size() + tp_name.size() + 1 + extension.size();
            static char default_name[length + 1] = {};
            auto cursor = 0zu;
            cursor += func_name.copy(default_name, sizeof(default_name));
            default_name[func_name.size()] = '-';
            ++cursor;
            cursor += tp_name.copy(
                default_name + cursor, sizeof(default_name) - cursor);
            extension.copy(
                default_name + cursor, sizeof(default_name) - cursor);
            config_.output = std::string_view(default_name);
        }
    }

    std::error_code execute() {
        static constexpr auto tp = to_type_name(mpfr::type_tag<T>{});
        input_generator<T, argument_count_v<func>> gen;

        auto const inputs = gen.process_inputs(config_);
        if (!inputs) {
            return inputs.error();
        }

        std::vector<mpfr::data_entry<T, argument_count_v<func>>> rows;
        for (auto const input : inputs.value()) {
            auto& current = rows.emplace_back();
            auto const expectation = calculate(input);
            if constexpr (argument_count_v<func> == 1) {
                current.input = input[0];
            } else {
                current.input =
                    *reinterpret_cast<T const(*)[argument_count_v<func>]>(
                        &input);
            }

            current.value = expectation.value;
            current.residual = expectation.residual;
            current.flags = expectation.flags;
        }

        return write_results(rows);
    }

    ~generator() { mpfr_free_cache(); }

private:
    static mpfr::expectation<T> calculate(
        std::array<T, argument_count_v<func>> const& input) noexcept {
        using value_type = mpfr::value<precision_v<T>>;
        using ref_result_type = mpfr::result<precision_v<T>>;

        return [&]<dpl::size_t... Is>(dpl::index_sequence<Is...>) {
            std::array const args{
                value_type(mpfr::from_floating_point, input[Is])...};
            auto const any_nan = (... || mpfr::is_nan(args[Is]));
            auto ref_result = [&] {
                constexpr mpfr::function_hash<string_hash(func)> hash;
                if constexpr (is_exact_v<func>) {
                    return ref_result_type(
                        mpfr::from_generator,
                        [&](auto... vals) {
                            return mpfr::invoke<func>(vals...);
                        },
                        args[Is].get()...);
                } else {
                    return ref_result_type(
                        mpfr::from_generator,
                        [&](auto... vals) {
                            return mpfr::invoke<func>(vals...);
                        },
                        args[Is].get()..., MPFR_RNDN);
                }
            }();

            auto const result = mpfr::result_cast<T>(ref_result);
            mpfr::expectation<T> ret{
                .value = static_cast<T>(result),
                .residual = 0,
                .flags = mpfr::result_flag::none,
            };

            constexpr mpfr_exp_t emin = 2 -
                static_cast<int>(dpl::floating_point_traits<T>::exponent_bias);
            if (mpfr::is_nan(result) && !any_nan) {
                ret.flags = mpfr::result_flag::invalid;
                return ret;
            }

            if (!mpfr::is_exact(result)) {
                ret.flags |= mpfr::result_flag::inexact;
                if (mpfr::is_inf(result)) {
                    ret.flags |= mpfr::result_flag::overflow;
                } else if (mpfr::is_number(result) &&
                    (!mpfr::is_regular(ref_result) ||
                        mpfr::get_exp(ref_result) < emin)) {
                    ret.flags |= mpfr::result_flag::underflow;
                }
            }

            if (!mpfr::is_number(result)) {
                ret.residual = mpfr::is_inf(result) && mpfr::is_exact(result)
                    ? 0.0f
                    : dpl::bit_cast<float>(-1);
            } else {
                auto const exp = mpfr::is_zero(result)
                    ? emin
                    : dpl::datapar::max(mpfr::get_exp(result), emin);
                auto const digits = static_cast<mpfr_exp_t>(
                    dpl::floating_point_traits<T>::digits);
                mpfr_sub(+ref_result, +ref_result, +result, MPFR_RNDN);
                mpfr_mul_2si(+ref_result, +ref_result, digits - exp, MPFR_RNDN);
                ret.residual = static_cast<float>(ref_result);
            }
            return ret;
        }(dpl::make_index_sequence<argument_count_v<func>>{});
    }

    std::error_code write_results(
        std::span<mpfr::data_entry<T, argument_count_v<func>> const> entries)
        const {

        auto const tmp = mpfr::tmp_string(config_.output);
        auto const* filepath = tmp.c_str();

        if (!filepath) {
            return std::make_error_code(
                std::errc::resource_unavailable_try_again);
        }

        static constexpr auto tp_name_v = to_type_name(mpfr::type_tag<T>{});
        static constexpr auto tp_name = std::string_view(tp_name_v);
        static constexpr auto func_name = std::string_view(func);

        struct make_tmpname_t {
            char buffer[64];
            make_tmpname_t() noexcept {
                pid_t const pid = getpid();
                snprintf(buffer, sizeof(buffer), "~%.*s-%.*s-%d.dat",
                    static_cast<int>(func_name.size()), func_name.data(),
                    static_cast<int>(tp_name.size()), tp_name.data(), pid);
            }
            char const* c_str() const noexcept { return buffer; }
        } const tmpname;

        auto const file_exists = mpfr::is_regular_file(filepath);
        auto const appending = config_.append && file_exists;

        struct filestream_t {
            FILE* ptr;
            ~filestream_t() {
                if (ptr) {
                    fclose(ptr);
                }
            }
        } const stream{
            .ptr = appending ? mpfr::create_copy(tmpname.c_str(), filepath)
                             : ::fopen(tmpname.c_str(), "wb"),
        };

        mpfr::latest_file_header header{};
        if (appending) {
            if (stream.ptr == nullptr ||
                fread(&header, sizeof(header), 1zu, stream.ptr) < 1zu) {
                auto code = std::make_error_code(static_cast<std::errc>(errno));
                fprintf(stderr, "Failed to read file header: %s\n",
                    code.message().c_str());
                return code;
            }

            if (auto verification = verify_append(stream.ptr, filepath, header,
                    expectation_type_v<T>, func, argument_count_v<func>)) {
                return verification;
            }

            if (fseek(stream.ptr,
                    offsetof(mpfr::latest_file_header, record_count),
                    SEEK_SET) != 0) {
                auto code = std::make_error_code(static_cast<std::errc>(errno));
                fprintf(stderr, "Failed to read metadata: %s\n",
                    code.message().c_str());
                return code;
            }
            auto const old_count = header.record_count.get();
            auto const new_count = old_count + entries.size();
            // Update record count
            if (fwrite(&new_count, sizeof(new_count), 1zu, stream.ptr) < 1zu) {
                auto code = std::make_error_code(static_cast<std::errc>(errno));
                fprintf(stderr, "Failed to update metadata: %s\n",
                    code.message().c_str());
                return code;
            }

            // Seek to end of old records
            auto const append_offset = header.record_offset.get() +
                old_count * sizeof(mpfr::data_entry<T, argument_count_v<func>>);
            if (fseek(stream.ptr, append_offset, SEEK_SET) != 0) {
                auto code = std::make_error_code(static_cast<std::errc>(errno));
                fprintf(stderr, "Failed to seek file: %s\n",
                    code.message().c_str());
                return code;
            }
        } else {
            header.expectation_type = expectation_type_v<T>;
            header.record_count = entries.size();
            header.record_offset =
                header.base.header_size.get() + func_name.size();
            header.function_name_offset = header.base.header_size.get();
            header.function_name_length = func_name.size();
            header.function_arity = argument_count_v<func>;
            if (stream.ptr == nullptr ||
                fwrite(&header, sizeof(header), 1zu, stream.ptr) < 1zu ||
                fwrite(func_name.data(), 1zu, func_name.size(), stream.ptr) <
                    func_name.size()) {
                auto code = std::make_error_code(static_cast<std::errc>(errno));
                fprintf(stderr, "Failed to prefix metadata: %s\n",
                    code.message().c_str());
                return code;
            }
        }

        if (fwrite(entries.data(),
                sizeof(mpfr::data_entry<T, argument_count_v<func>>),
                entries.size(), stream.ptr) < entries.size()) {
            auto code = std::make_error_code(static_cast<std::errc>(errno));
            fprintf(stderr, "Failed to write entries: %s\n",
                code.message().c_str());
            return code;
        }

        if (file_exists && ::remove(filepath) != 0) {
            auto code = std::make_error_code(static_cast<std::errc>(errno));
            fprintf(
                stderr, "Failed to delete file: %s\n", code.message().c_str());
            return code;
        }

        if (::rename(tmpname.c_str(), filepath) != 0) {
            auto code = std::make_error_code(static_cast<std::errc>(errno));
            fprintf(
                stderr, "Failed to rename file: %s\n", code.message().c_str());
            return code;
        }

        return std::error_code();
    }

    configuration config_;
};

template <dpl::floating_point_like T>
std::error_code execute_generator(
    std::string_view func, configuration config) noexcept {
    constexpr dpl::make_index_sequence<
        std::tuple_size_v<dpl::decay_t<decltype(functions)>>>
        seq;
    return [&]<size_t... Is>(dpl::index_sequence<Is...>) {
        constexpr mpfr::jump_table<dpl::uint64,
            mpfr::string_hash(std::get<Is>(functions))...>
            table = {};
        return table(
            []<typename U>(
                U hash, std::string_view name, configuration const& config) {
                if constexpr (requires {
                                  requires !dpl::integral<U>;
                                  to_function(mpfr::function_hash<U::value>{});
                              }) {
                    constexpr auto func =
                        to_function(mpfr::function_hash<U::value>{});
                    return generator<T, func>(config).execute();
                } else {
                    fprintf(stderr, "Unrecognized function: '%.*s'\n",
                        static_cast<int>(name.size()), name.data());
                    return std::make_error_code(std::errc::invalid_argument);
                }
            },
            mpfr::string_hash(func), func, config);
    }(seq);
}

class config_parser {
    static constexpr auto arg_chunker =
        std::views::chunk_by([](std::string_view, std::string_view next) {
            return !next.starts_with('-') && !next.starts_with('@');
        });

public:
    config_parser(int argc, char const** argv)
        : cli_args_{argv + 3, argv + argc} {
        inputs_.reserve(static_cast<size_t>(argc));
        arg_files_.reserve(static_cast<size_t>(argc));
    }

    std::expected<configuration, std::error_code> operator()() {
        for (auto chunk :
            cli_args_ | std::views::transform([](char const* ptr) {
                return mpfr::strip(std::string_view(ptr));
            }) | arg_chunker) {
            if (std::ranges::empty(chunk)) {
                continue;
            }
            auto const current = chunk.front();
            chunk.advance(1);
            if (current.starts_with('@')) {
                if (auto const code = parse_response_file(current.substr(1))) {
                    return std::unexpected(code);
                }
            } else if (current.starts_with('-')) {
                if (auto const code = parse_arg(current, chunk)) {
                    return std::unexpected(code);
                }
            }
        }

        config_.inputs = std::span{inputs_};
        return std::expected<configuration, std::error_code>(
            std::in_place, config_);
    }

private:
    std::error_code parse_response_file(std::string_view filepath) {
        auto const& file =
            arg_files_.emplace_back(mpfr::tmp_string(filepath).c_str());
        if (!file) {
            fprintf(stderr, "Unable to open response file '%.*s'\n",
                static_cast<int>(filepath.size()), filepath.data());
            return std::make_error_code(std::errc::no_such_file_or_directory);
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
            // no recursive response file allowed
            if (!current.starts_with('-')) {
                continue;
            }

            if (auto const code = parse_arg(current, chunk.advance(1))) {
                return code;
            }
        }

        return std::error_code();
    }

    std::error_code parse_arg(std::string_view opt, auto chunk);

    template <dpl::integral U>
    static std::error_code parse_int(U& val, std::string_view str) noexcept {
        auto const negative = str.starts_with('-');
        str.remove_prefix(negative ? 1zu : 0zu);

        auto const result = [&] {
            if (!str.starts_with('0') || str.size() <= 1) {
                return std::from_chars(
                    str.data(), str.data() + str.size(), val, 10)
                    .ec;
            }

            str.remove_prefix(1zu);
            if (str.starts_with('x') || str.starts_with('X')) {
                str.remove_prefix(1zu);
                return std::from_chars(
                    str.data(), str.data() + str.size(), val, 16)
                    .ec;
            }

            if (!str.starts_with('.')) {
                return std::from_chars(
                    str.data(), str.data() + str.size(), val, 8)
                    .ec;
            }

            val = 0;
            return static_cast<std::errc>(0);
        }();

        if (negative) {
            val = -val;
        }

        return std::make_error_code(result);
    }

    std::span<char const*> cli_args_;
    std::vector<std::string_view> inputs_;
    std::vector<mpfr::mapped_memory> arg_files_;
    configuration config_;

    template <auto opt>
    struct store_arg_t;
    template <auto opt>
    static constexpr store_arg_t<opt> store_arg;
};

template <>
struct config_parser::store_arg_t<"output"_cli_opt> {
    static std::error_code operator()(
        config_parser& parser, auto chunk) noexcept {
        if (chunk.empty()) {
            return std::make_error_code(std::errc::invalid_argument);
        }

        parser.config_.output = chunk.front();
        return std::error_code();
    }
};

template <>
struct config_parser::store_arg_t<"seed"_cli_opt> {
    static std::error_code operator()(
        config_parser& parser, auto chunk) noexcept {
        if (chunk.empty()) {
            return std::make_error_code(std::errc::invalid_argument);
        }

        return parse_int(parser.config_.seed, chunk.front());
    }
};

template <>
struct config_parser::store_arg_t<"count"_cli_opt> {
    static std::error_code operator()(
        config_parser& parser, auto chunk) noexcept {
        if (chunk.empty()) {
            return std::make_error_code(std::errc::invalid_argument);
        }

        return parse_int(parser.config_.range_count, chunk.front());
        return std::error_code();
    }
};

template <>
struct config_parser::store_arg_t<"urand"_cli_opt> {
    static std::error_code operator()(
        config_parser& parser, auto chunk) noexcept {
        if (chunk.empty()) {
            return std::make_error_code(std::errc::invalid_argument);
        }

        parser.config_.range_type = range_type::urand;
        parser.config_.ranges_spec = chunk.front();
        return std::error_code();
    }
};

template <>
struct config_parser::store_arg_t<"linear"_cli_opt> {
    static std::error_code operator()(
        config_parser& parser, auto chunk) noexcept {
        if (chunk.empty()) {
            return std::make_error_code(std::errc::invalid_argument);
        }

        parser.config_.range_type = range_type::linear;
        parser.config_.ranges_spec = chunk.front();
        return std::error_code();
    }
};

template <>
struct config_parser::store_arg_t<"append"_cli_opt> {
    static std::error_code operator()(config_parser& parser, auto) noexcept {
        parser.config_.append = true;
        return std::error_code();
    }
};

template <>
struct config_parser::store_arg_t<"input"_cli_opt> {
    static std::error_code operator()(
        config_parser& parser, auto chunk) noexcept {
        if (chunk.empty()) {
            return std::make_error_code(std::errc::invalid_argument);
        }

        parser.inputs_.emplace_back(chunk.front());
        return std::error_code();
    }
};

void print_usage() noexcept;

template <>
struct config_parser::store_arg_t<"help"_cli_opt> {

    static std::error_code operator()(config_parser&, auto chunk) noexcept {
        print_usage();
        return std::make_error_code(std::errc::operation_canceled);
    }
};

std::error_code config_parser::parse_arg(std::string_view opt, auto chunk) {
    auto const error_path = [opt]() {
        fprintf(stderr, "Unrecognized option: -'%.*s'\n",
            static_cast<int>(opt.size()), opt.data());
        return std::make_error_code(std::errc::invalid_argument);
    };

    auto const opt_offset = opt.find_first_not_of('-');
    if (opt_offset > 2zu) {
        return error_path();
    }

    opt.remove_prefix(opt_offset);
    if (opt_offset == 1) {
        auto const longopt =
            opt.size() > 2 ? std::string_view() : find_long_option(opt.front());
        if (longopt.empty()) {
            return error_path();
        }

        opt = longopt;
    }

    constexpr dpl::make_index_sequence<
        std::tuple_size_v<dpl::decay_t<decltype(cli_options)>>>
        seq;
    return [&]<size_t... Is>(dpl::index_sequence<Is...>) {
        constexpr mpfr::jump_table<dpl::uint64,
            mpfr::string_hash(std::get<Is>(cli_options).value)...>
            table = {};
        return table(
            [&]<typename U>(U hash, std::string_view opt, auto chunk) {
                if constexpr (requires {
                                  requires !dpl::integral<U>;
                                  to_cli_option(
                                      mpfr::cli_option_hash<U::value>{});
                              }) {
                    return store_arg<to_cli_option(
                        mpfr::cli_option_hash<U::value>{})>(*this, chunk);
                } else {
                    return error_path();
                }
            },
            mpfr::string_hash(opt), opt, chunk);
    }(seq);
}

void print_usage() noexcept {
    fputs( //
        "usage: generate <func> <type> [options] [@file ...]\n"
        "\n"
        "Generates correctly rounded reference results with MPFR\n"
        "\n"
        "  <func>                function, e.g.: exp\n"
        "  <type>                element type: f16 | bf16 | f32 | f64\n"
        "\n"
        "options:\n"
        "  -o, --output PATH     output file (default: ./<func>-<type>.dat)\n"
        "      --urand R[,R...]  every representable value in range R is "
        "equally likely (default: unbounded)\n"
        "      --linear R[,R...] values spread evenly across range R; both "
        "bounds must be finite\n"
        "  -n, --count N         random inputs to generate, excluding -i "
        "inputs (default: 100)\n"
        "  -s, --seed N          random seed, decimal or 0x-hex\n"
        "  -i, --input V[,V...]  add one call's arguments; "
        "repeat for more calls\n"
        "  -a, --append          append to an existing data file"
        "  -h, --help            show this help\n"
        "  @file                 read arguments from a file: "
        "newline-separated, # starts a comment, nested @paths are ignored\n"
        "\n"
        "ranges:\n"
        "  One [min,max) per argument, written as min:max. 'min:' and "
        "':max' leave one end unbounded; ':' or an empty entry leaves that "
        "argument unbounded.\n"
        "\n"
        "values (for -i and range bounds):\n"
        "  decimal (1.5, -2e-3), hex float (0x1.8p-3), inf, -inf, nan, -0, "
        "or raw bits (bits:0x3c00, -i only). Values are rounded to "
        "nearest-even in <type>.\n"
        "\n"
        "Random inputs are never subnormal, infinite or NaN.\n",
        stderr);
}
} // namespace

int main(int argc, char const* argv[]) {
    if (argc < 3) {
        print_usage();
        return -1;
    }

    constexpr dpl::make_index_sequence<
        std::tuple_size_v<dpl::decay_t<decltype(float_types)>>>
        seq;
    auto const result = [&]<size_t... Is>(dpl::index_sequence<Is...>) {
        config_parser parser(argc, argv);
        constexpr mpfr::jump_table<dpl::uint64,
            mpfr::string_hash(std::get<Is>(float_types))...>
            table = {};
        auto const name = std::string_view(argv[2]);
        return table(
            [&]<typename U>(U) {
                if constexpr (requires {
                                  requires !dpl::integral<U>;
                                  to_type(mpfr::type_hash<U::value>{});
                              }) {
                    auto const config = parser();
                    if (!config) {
                        return config.error();
                    }

                    using fptype = typename decltype(to_type(
                        mpfr::type_hash<U::value>{}))::type;
                    return execute_generator<fptype>(
                        std::string_view(argv[1]), config.value());
                } else {
                    fprintf(stderr, "Unrecognized type: '%.*s'\n",
                        static_cast<int>(name.size()), name.data());
                    return std::make_error_code(std::errc::invalid_argument);
                }
            },
            mpfr::string_hash(name));
    }(seq);

    return result.value();
}
