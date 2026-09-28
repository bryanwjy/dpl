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

using mpfr::literals::operator""_cli_opt;

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

template <dpl::floating_point_like T, size_t N>
class input_processor {
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

template <dpl::floating_point_like T>
mpfr::expectation<T> make_expectation(
    mpfr::result<precision_v<T>>&& ref_result, bool nan_arguments) noexcept {
    auto const result = mpfr::result_cast<T>(dpl::as_const(ref_result));
    mpfr::expectation<T> expectation{
        .value = static_cast<T>(result),
        .residual = 0,
        .flags = mpfr::result_flag::none,
    };

    constexpr mpfr_exp_t emin =
        2 - static_cast<int>(dpl::floating_point_traits<T>::exponent_bias);
    if (mpfr::is_nan(result) && !nan_arguments) {
        expectation.flags = mpfr::result_flag::invalid;
        return expectation;
    }

    if (!mpfr::is_exact(result)) {
        expectation.flags |= mpfr::result_flag::inexact;
        if (mpfr::is_inf(result)) {
            expectation.flags |= mpfr::result_flag::overflow;
        } else if (mpfr::is_number(result) &&
            (!mpfr::is_regular(ref_result) ||
                mpfr::get_exp(ref_result) < emin)) {
            expectation.flags |= mpfr::result_flag::underflow;
        }
    }

    if (!mpfr::is_number(result)) {
        expectation.residual = mpfr::is_inf(result) && mpfr::is_exact(result)
            ? 0.0f
            : dpl::bit_cast<float>(-1);
    } else {
        auto const exp = mpfr::is_zero(result)
            ? emin
            : dpl::datapar::max(mpfr::get_exp(result), emin);
        auto const digits =
            static_cast<mpfr_exp_t>(dpl::floating_point_traits<T>::digits);
        mpfr_sub(+ref_result, +ref_result, +result, MPFR_RNDN);
        mpfr_mul_2si(+ref_result, +ref_result, digits - exp, MPFR_RNDN);
        expectation.residual = static_cast<float>(ref_result);
    }

    return expectation;
}

struct append_request {
    mpfr::expectation_type type;
    std::string_view function;
    size_t arity;
};

enum class append_error {
    header_format_mismatch,
    expectation_type_mismatch,
    function_mismatch,
};

inline std::expected<void, append_error> verify_append(
    mpfr::latest_file_header const& header, std::string_view function,
    append_request const& request) noexcept {
    constexpr auto header_meta = mpfr::latest_file_header{}.base;

    using result_type = std::expected<void, append_error>;

    if (memcmp(&header_meta, &header.base, sizeof(header_meta)) != 0) {
        return result_type(std::unexpect, append_error::header_format_mismatch);
    }

    if (header.expectation_type.get() != request.type) {
        return result_type(
            std::unexpect, append_error::expectation_type_mismatch);
    }

    if (function != request.function ||
        header.function_arity.get() != request.arity) {
        return result_type(std::unexpect, append_error::function_mismatch);
    }

    return result_type(std::in_place);
}

inline std::expected<mpfr::tmp_string, std::error_code> read_function_name(
    FILE* file, size_t offset, size_t length) noexcept {
    struct filepos {
        FILE* file;
        long offset;
        ~filepos() {
            if (file != nullptr && offset >= 0) {
                fseek(file, offset, SEEK_SET);
            }
        }
    } const pos{.file = file, .offset = ftell(file)};

    if (fseek(file, offset, SEEK_SET) != 0) {
        return std::unexpected(
            std::make_error_code(static_cast<std::errc>(errno)));
    }

    std::expected<mpfr::tmp_string, std::error_code> result(
        std::in_place, length);
    if (fread(result->data(), 1zu, length, file) < length) {
        fprintf(stderr,
            "Could not read function name of length %zu from offset %zu\n",
            length, offset);
        return std::unexpected(
            std::make_error_code(static_cast<std::errc>(errno)));
    }

    return result;
}

std::error_code translate_append_error(append_error val, std::string_view func,
    char const* filepath, char const* file_func) noexcept {
    switch (val) {
    case append_error::header_format_mismatch:
        fprintf(stderr, "File metadata format mismatch in %s\n", filepath);
        break;
    case append_error::function_mismatch:
        fprintf(stderr,
            "Function mismatch in data file, "
            "expected "
            "'%.*s', but file %s contains data for "
            "'%s'\n",
            static_cast<int>(func.size()), func.data(), filepath, file_func);
        break;
    case append_error::expectation_type_mismatch:
        fprintf(stderr,
            "Function type mismatch in data file "
            "%s",
            filepath);
        break;
    }
    return std::make_error_code(std::errc::illegal_byte_sequence);
}

template <dpl::floating_point_like T, auto func>
class generator {
public:
    explicit generator(configuration config) : config_(config) {
        if (config_.output.empty()) {
            static constexpr std::string_view extension = ".dat";
            static constexpr auto func_name = +func;
            static constexpr auto tp_name = +mpfr::to_type_name<T>();
            constexpr auto length =
                func_name.size() + tp_name.size() + 1 + extension.size();
            static char default_name[length + 1] = {};
            snprintf(default_name, length + 1, "%s-%s.dat", func_name.data(),
                tp_name.data());
            config_.output = std::string_view(default_name);
        }
    }

    std::error_code execute() {
        static constexpr auto tp = to_type_name(mpfr::type_tag<T>{});
        input_processor<T, mpfr::arity_v<func>> proc;

        auto const inputs = proc.process_inputs(config_);
        if (!inputs) {
            return inputs.error();
        }

        std::vector<mpfr::data_entry<T, mpfr::arity_v<func>>> rows;
        for (auto const input : inputs.value()) {
            auto& current = rows.emplace_back();
            auto const expectation = calculate(input);
            if constexpr (mpfr::arity_v<func> == 1) {
                current.input = input[0];
            } else {
                current.input =
                    *reinterpret_cast<T const(*)[mpfr::arity_v<func>]>(&input);
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
        std::array<T, mpfr::arity_v<func>> const& inputs) noexcept {
        using value_type = mpfr::value<precision_v<T>>;
        using result_type = mpfr::result<precision_v<T>>;
        auto const args = dpl::apply(
            [](auto... inputs) {
                return std::array{
                    value_type(mpfr::from_floating_point, inputs)...};
            },
            inputs);

        auto result = dpl::apply(
            [](auto... args) {
                return result_type(
                    mpfr::from_generator, mpfr::invoke<func>, args.get()...);
            },
            args);

        auto const nan_arguments = dpl::pack::any_of(
            [](value_type const& arg) { return mpfr::is_nan(arg); }, args);

        return make_expectation<T>(dpl::move(result), nan_arguments);
    }

    std::error_code write_results(
        std::span<mpfr::data_entry<T, mpfr::arity_v<func>> const> entries)
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

            auto const verified =
                read_function_name(stream.ptr,
                    header.function_name_offset.get(),
                    header.function_name_length.get())
                    .and_then([&](mpfr::tmp_string const& file_function) {
                        return verify_append(header, file_function,
                            append_request{
                                .type = mpfr::expectation_type_v<T>,
                                .function = +func,
                                .arity = mpfr::arity_v<func>,
                            })
                            .transform_error([&](append_error val) {
                                return translate_append_error(
                                    val, func, filepath, file_function.c_str());
                            });
                    });

            if (!verified) {
                return verified.error();
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
                old_count * sizeof(mpfr::data_entry<T, mpfr::arity_v<func>>);
            if (fseek(stream.ptr, append_offset, SEEK_SET) != 0) {
                auto code = std::make_error_code(static_cast<std::errc>(errno));
                fprintf(stderr, "Failed to seek file: %s\n",
                    code.message().c_str());
                return code;
            }
        } else {
            header.expectation_type = mpfr::expectation_type_v<T>;
            header.record_count = entries.size();
            header.record_offset =
                header.base.header_size.get() + func_name.size();
            header.function_name_offset = header.base.header_size.get();
            header.function_name_length = func_name.size();
            header.function_arity = mpfr::arity_v<func>;
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
                sizeof(mpfr::data_entry<T, mpfr::arity_v<func>>),
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
std::expected<void, std::error_code> execute_generator(
    std::string_view func, configuration config) noexcept {

    constexpr auto jump_table = dpl::apply(
        [](auto... funcs) {
            return mpfr::make_jump_table<mpfr::string_hash(funcs())...>();
        },
        mpfr::functions);

    auto const code = jump_table(
        [](auto constant, std::string_view name, configuration config) {
            if constexpr (requires { mpfr::to_function_name(constant); }) {
                constexpr auto& const_name = mpfr::to_function_name(constant);
                constexpr auto func_name = +const_name;
                constexpr auto tp_name = +mpfr::to_type_name<T>();
                constexpr auto length =
                    func_name.size() + tp_name.size() + 1 + 4;
                mpfr::tmp_string default_name(length + 1);

                if (config.output.empty()) {
                    snprintf(default_name.data(), length + 1, "%s-%s.dat",
                        func_name.data(), tp_name.data());
                    config.output = std::string_view(default_name);
                }

                return generator<T, const_name>(config).execute();
            } else {
                fprintf(stderr, "Unrecognized function: '%.*s'\n",
                    static_cast<int>(name.size()), name.data());
                return std::make_error_code(std::errc::invalid_argument);
            }
        },
        mpfr::string_hash(func), func, dpl::move(config));

    return std::unexpected(code);
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

    template <mpfr::cli_option_name opt>
    struct store_arg_t {
        static std::error_code operator()(
            config_parser& parser, auto chunk) noexcept {
            constexpr auto opt_str = +opt;
            fprintf(stderr, "Missing argument handler for option '%.*s'\n",
                static_cast<int>(opt_str.size()), opt_str.data());
            return std::make_error_code(std::errc::invalid_argument);
        }
    };

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
        auto const longopt = opt.size() > 2
            ? std::string_view()
            : mpfr::find_long_option(opt.front());
        if (longopt.empty()) {
            return error_path();
        }

        opt = longopt;
    }

    constexpr auto jump_table = __DPL apply(
        [](auto... types) {
            return mpfr::make_jump_table<mpfr::string_hash(types())...>();
        },
        mpfr::cli_options);

    return jump_table(
        [this, &error_path](auto hash, auto chunk) {
            if constexpr (requires { mpfr::to_cli_option(hash); }) {
                constexpr auto const_opt = mpfr::to_cli_option(hash);
                return store_arg<const_opt>(*this, chunk);
            } else {
                return error_path();
            }
        },
        mpfr::string_hash(opt), dpl::move(chunk));
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

    constexpr auto jump_table = __DPL apply(
        [](auto... types) {
            return mpfr::make_jump_table<mpfr::string_hash(types())...>();
        },
        mpfr::float_types);

    config_parser parser(argc, argv);
    auto const function = std::string_view(argv[1]);
    auto const type = std::string_view(argv[2]);

    return jump_table(
        [&function, &parser](auto hash, std::string_view type) {
            if constexpr (requires { mpfr::to_type(hash); }) {
                using float_t = typename decltype(mpfr::to_type(hash))::type;
                return parser()
                    .and_then([&](configuration&& config) {
                        return execute_generator<float_t>(
                            function, dpl::move(config));
                    })
                    .error_or(std::error_code());
            } else {
                fprintf(stderr, "Unrecognized type: '%.*s'\n",
                    static_cast<int>(type.size()), type.data());
                return std::make_error_code(std::errc::invalid_argument);
            }
        },
        mpfr::string_hash(type), type)
        .value();
}
