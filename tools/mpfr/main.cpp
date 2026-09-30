// Copyright 2026 Bryan Wong
// offline MPFR reference-data generator for DPL math CPO tests.
//
// Usage:  mpfr <func> <type> [options] [@file ...]      (see --help)
//
// Each record's expected value is computed by MPFR directly in the target
// format (target precision and exponent range, subnormals via
// mpfr_subnormalize), so it is correctly rounded.

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

constexpr auto input_error = std::unexpected(std::errc::invalid_argument);

enum class range_type {
    linear,
    urand
};

template <dpl::floating_point_like T, size_t N>
class input_processor {
    using input_span =
        decltype(std::span<std::string_view const>{} | std::views::chunk(N));
    using result_type = std::vector<std::array<T, N>>;

public:
    std::expected<result_type, std::errc> process(dpl::test::mt19937& rng,
        dpl::uint32 count, range_type range,
        std::span<std::string_view const, N> bounds, input_span inputs) {
        result_.reserve(std::ranges::size(inputs) + count);
        return generate_ranges(rng, range, bounds, count)
            .and_then([&] { return generate_inputs(dpl::move(inputs)); })
            .and_then([&] {
                using expected_type = std::expected<result_type, std::errc>;
                result_type vec;
                vec.swap(result_);
                return expected_type(std::in_place, dpl::move(vec));
            });
    }

private:
    std::expected<void, std::errc> generate_ranges(dpl::test::mt19937& rng,
        range_type type, std::span<std::string_view const, N> bounds,
        dpl::uint32 count) {
        if (type == range_type::linear) {
            return generate_linear(bounds, count);
        } else {
            return generate_random(rng, bounds, count);
        }
    }

    std::expected<void, std::errc> generate_inputs(input_span inputs) {
        std::array<T, N> vals;
        auto it = vals.begin();
        for (auto const row : inputs) {
            for (auto const [idx, str] : row | std::views::enumerate) {
                if (!mpfr::parse<T>(str).transform(
                        [&](T val) { vals[idx] = val; })) {
                    fprintf(stderr,
                        "Linear range parsing failure of value: '%.*s'\n",
                        static_cast<int>(str.size()), str.data());
                    return input_error;
                }
            }

            result_.emplace_back(vals);
        }

        return {};
    }

    std::expected<void, std::errc> generate_random(dpl::test::mt19937& rng,
        std::span<std::string_view const, N> bounds, dpl::uint32 count) {
        std::array<std::pair<T, T>, N> pairs;
        auto it = pairs.begin();
        for (auto bound : bounds) {
            bound = mpfr::strip(bound);
            auto const mid = bound.find(':');
            if (mid >= bound.size()) {
                fprintf(stderr, "Invalid linear range encountered: '%.*s'\n",
                    static_cast<int>(bound.size()), bound.data());
                return input_error;
            }

            using std::string_view_literals::operator""sv;
            auto const start =
                mid == 0zu ? "-infinity"sv : bound.substr(0zu, mid);
            auto const end =
                mid + 1 == bound.size() ? "infinity"sv : bound.substr(mid + 1);

            if (!mpfr::parse<T>(start).transform(
                    [&](T val) { it->first = val; })) {
                fprintf(stderr,
                    "urand range parsing failure of value: '%.*s'\n",
                    static_cast<int>(start.size()), start.data());
                return input_error;
            }

            if (!mpfr::parse<T>(end).transform(
                    [&](T val) { it->second = val; })) {
                fprintf(stderr,
                    "urand range parsing failure of value: '%.*s'\n",
                    static_cast<int>(end.size()), end.data());
                return input_error;
            }

            it++;
        }

        [&]<size_t... Is>(dpl::index_sequence<Is...>) {
            std::array const gens{dpl::test::scalar_generator<T>(
                std::min(pairs[Is].first, pairs[Is].second),
                std::max(pairs[Is].first, pairs[Is].second))...};

            for (auto const _ : std::views::iota(0u, count)) {
                result_.emplace_back(std::array{gens[Is](rng)...});
            }
        }(dpl::make_index_sequence<N>{});

        return {};
    }

    std::expected<void, std::errc> generate_linear(
        std::span<std::string_view const, N> bounds, dpl::uint32 count) {
        std::array<std::pair<T, T>, N> pairs;
        auto it = pairs.begin();
        for (auto bound : bounds) {
            bound = mpfr::strip(bound);
            auto const mid = bound.find(':');
            if (mid >= bound.size()) {
                fprintf(stderr, "Invalid linear range encountered: '%.*s'\n",
                    static_cast<int>(bound.size()), bound.data());
                return input_error;
            }

            auto const start = bound.substr(0zu, mid);
            auto const end = bound.substr(mid + 1);

            if (!mpfr::parse<T>(start).transform(
                    [&](T val) { it->first = val; })) {
                fprintf(stderr,
                    "Linear range parsing failure of value: '%.*s'\n",
                    static_cast<int>(start.size()), start.data());
                return input_error;
            }

            if (!mpfr::parse<T>(end).transform(
                    [&](T val) { it->second = val; })) {
                fprintf(stderr,
                    "Linear range parsing failure of value: '%.*s'\n",
                    static_cast<int>(end.size()), end.data());
                return input_error;
            }

            it++;
        }

        [&]<size_t... Is>(dpl::index_sequence<Is...>) {
            std::array vals{pairs[Is].first...};
            std::array const steps{
                ((pairs[Is].second - pairs[Is].first) / count)...};
            for (auto const _ : std::views::iota(0u, count)) {
                (..., (vals[Is] += steps[Is]));
                result_.emplace_back(vals);
            }
        }(dpl::make_index_sequence<N>{});

        return {};
    }

    result_type result_;
};

template <dpl::floating_point_like T, size_t N>
std::expected<std::vector<std::array<T, N>>, std::error_condition>
process_inputs(mpfr::argument_parser const& config) {
    static constexpr auto default_bounds = []<size_t... Is>(
                                               dpl::index_sequence<Is...>) {
        using std::string_view_literals::operator""sv;
        constexpr auto colon = ":"sv;
        return std::array{"urand"sv, (Is < N ? colon : colon)...};
    }(dpl::make_index_sequence<N>{});

    if (auto const extent = config.extent<"input"_cli_opt>(); extent != N) {
        fprintf(stderr,
            "Input extent (%zu) does not match function arity (%zu)", extent,
            N);
        return input_error;
    }

    if (auto const extent = config.extent<"range"_cli_opt>();
        extent != mpfr::typeconv::uninitialized_extent && extent != N + 1) {
        fprintf(stderr,
            "Range extent (%zu) does not match function arity (%zu)", extent,
            N);
        return input_error;
    }

    dpl::test::mt19937 rng(config.value<"seed"_cli_opt>());
    auto const count = config.value<"count"_cli_opt>();
    auto const [range, bounds] = [&] {
        auto const range = config.has_value<"range"_cli_opt>()
            ? std::span(default_bounds)
            : std::span<std::string_view const, N + 1>(
                  config.value<"range"_cli_opt>());
        return std::pair{
            range.front() == "linear" ? range_type::linear : range_type::urand,
            range.template subspan<1>(),
        };
    }();

    auto const inputs = config.has_value<"input"_cli_opt>()
        ? std::span<std::string_view const>(config.value<"input"_cli_opt>())
        : std::span<std::string_view const>();

    return input_processor<T, N>{}.process(
        rng, count, range, bounds, inputs | std::views::chunk(N));
}

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

inline std::expected<mpfr::tmp_string, std::error_condition> read_function_name(
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
        return std::unexpected(static_cast<std::errc>(errno));
    }

    std::expected<mpfr::tmp_string, std::error_condition> result(
        std::in_place, length);
    if (fread(result->data(), 1zu, length, file) < length) {
        fprintf(stderr,
            "Could not read function name of length %zu from offset %zu\n",
            length, offset);
        return std::unexpected(static_cast<std::errc>(errno));
    }

    return result;
}

std::error_condition translate_append_error(append_error val,
    std::string_view func, char const* filepath,
    char const* file_func) noexcept {
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
    return std::errc::invalid_argument;
}

template <dpl::floating_point_like T>
mpfr::tmp_string make_tmpname(std::string_view function) noexcept {
    constexpr auto& tp_name = mpfr::to_type_name<T>();
    pid_t const pid = getpid();
    auto const size = snprintf(nullptr, 0zu, "~%.*s-%s-%d.dat",
        static_cast<int>(function.size()), function.data(), tp_name.c_str(),
        pid);
    auto buffer = mpfr::tmp_string(size);
    snprintf(buffer.data(), size, "~%.*s-%s-%d.dat",
        static_cast<int>(function.size()), function.data(), tp_name.c_str(),
        pid);
    return buffer;
}

std::expected<void, std::error_condition> replace_file(
    char const* from, char const* to) noexcept {
    if (mpfr::is_regular_file(to) && ::remove(to) != 0) {
        auto condition =
            std::make_error_condition(static_cast<std::errc>(errno));
        fprintf(stderr, "Failed to delete file '%s': %s\n", to,
            condition.message().c_str());
        return std::unexpected(condition);
    }

    if (::rename(from, to) != 0) {
        auto condition =
            std::make_error_condition(static_cast<std::errc>(errno));
        fprintf(stderr, "Failed to rename file from '%s' to '%s': %s\n", from,
            to, condition.message().c_str());
        return std::unexpected(condition);
    }

    return {};
}

template <dpl::floating_point_like T, size_t N>
std::expected<void, std::error_condition> write_results(
    std::string_view function,
    std::vector<mpfr::data_entry<T, N>> const& entries, std::string_view file) {
    auto const cfile = mpfr::tmp_string(file);
    auto const* filepath = cfile.c_str();
    auto const tmpname = make_tmpname<T>(function);
    struct filestream_t {
        FILE* ptr;
        ~filestream_t() {
            if (ptr) {
                fclose(ptr);
            }
        }
    } const stream{.ptr = ::fopen(tmpname.c_str(), "wb")};
    mpfr::latest_file_header header{};
    header.expectation_type = mpfr::expectation_type_v<T>;
    header.record_count = entries.size();
    header.record_offset = header.base.header_size.get() + function.size();
    header.function_name_offset = header.base.header_size.get();
    header.function_name_length = function.size();
    header.function_arity = N;
    if (stream.ptr == nullptr ||
        fwrite(&header, sizeof(header), 1zu, stream.ptr) < 1zu ||
        fwrite(function.data(), 1zu, function.size(), stream.ptr) <
            function.size()) {
        auto condition =
            std::make_error_condition(static_cast<std::errc>(errno));
        fprintf(stderr, "Failed to prefix metadata: %s\n",
            condition.message().c_str());
        return std::unexpected(condition);
    }

    if (fwrite(entries.data(), sizeof(mpfr::data_entry<T, N>), entries.size(),
            stream.ptr) < entries.size()) {
        auto condition =
            std::make_error_condition(static_cast<std::errc>(errno));
        fprintf(stderr, "Failed to write entries: %s\n",
            condition.message().c_str());
        return std::unexpected(condition);
    }

    return replace_file(tmpname.c_str(), filepath);
}

template <dpl::floating_point_like T, size_t N>
std::expected<void, std::error_condition> append_results(
    std::string_view function,
    std::vector<mpfr::data_entry<T, N>> const& entries, std::string_view file) {
    auto const cfile = mpfr::tmp_string(file);
    auto const* filepath = cfile.c_str();
    auto const tmpname = make_tmpname<T>(function);
    if (!mpfr::is_regular_file(filepath)) {
        return write_results(function, entries, file);
    }
    struct filestream_t {
        FILE* ptr;
        ~filestream_t() {
            if (ptr) {
                fclose(ptr);
            }
        }
    } const stream{.ptr = mpfr::create_copy(tmpname.c_str(), filepath)};

    mpfr::latest_file_header header{};
    if (stream.ptr == nullptr ||
        fread(&header, sizeof(header), 1zu, stream.ptr) < 1zu) {
        auto condition = std::error_condition(static_cast<std::errc>(errno));
        fprintf(stderr, "Failed to read file header: %s\n",
            condition.message().c_str());
        return std::unexpected(condition);
    }

    return read_function_name(stream.ptr, header.function_name_offset.get(),
        header.function_name_length.get())
        .and_then([&](mpfr::tmp_string const& file_function) {
            return verify_append(header, file_function,
                append_request{
                    .type = mpfr::expectation_type_v<T>,
                    .function = function,
                    .arity = N,
                })
                .transform_error([&](append_error val) {
                    return translate_append_error(
                        val, function, filepath, file_function.c_str());
                });
        })
        .and_then([&] -> std::expected<void, std::error_condition> {
            if (fseek(stream.ptr,
                    offsetof(mpfr::latest_file_header, record_count),
                    SEEK_SET) != 0) {
                auto condition =
                    std::make_error_condition(static_cast<std::errc>(errno));
                fprintf(stderr, "Failed to seek to record count: %s\n",
                    condition.message().c_str());
                return std::unexpected(condition);
            }

            auto const old_count = header.record_count.get();
            auto const new_count = old_count + entries.size();
            // Update record count
            if (fwrite(&new_count, sizeof(new_count), 1zu, stream.ptr) < 1zu) {
                auto condition =
                    std::make_error_condition(static_cast<std::errc>(errno));
                fprintf(stderr,
                    "Failed to update record count in metadata: %s\n",
                    condition.message().c_str());
                return std::unexpected(condition);
            }

            auto const append_offset = header.record_offset.get() +
                old_count * sizeof(mpfr::data_entry<T, N>);
            if (fseek(stream.ptr, append_offset, SEEK_SET) != 0) {
                auto condition =
                    std::make_error_condition(static_cast<std::errc>(errno));
                fprintf(stderr, "Failed to seek file: %s\n",
                    condition.message().c_str());
                return std::unexpected(condition);
            }

            if (fwrite(entries.data(), sizeof(mpfr::data_entry<T, N>),
                    entries.size(), stream.ptr) < entries.size()) {
                auto condition =
                    std::make_error_condition(static_cast<std::errc>(errno));
                fprintf(stderr, "Failed to write entries: %s\n",
                    condition.message().c_str());
                return std::unexpected(condition);
            }

            return replace_file(tmpname.c_str(), filepath);
        });
}

template <mpfr::function_name func, dpl::floating_point_like T>
mpfr::expectation<T> calculate(
    std::array<T, mpfr::arity_v<func>> const& inputs) noexcept {
    using value_type = mpfr::value<precision_v<T>>;
    using result_type = mpfr::result<precision_v<T>>;
    auto const args = dpl::apply(
        [](auto... inputs) {
            return std::array{value_type(mpfr::from_floating_point, inputs)...};
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

template <dpl::floating_point_like T, mpfr::function_name func>
bool generate(mpfr::argument_parser const& parser) {
    static constexpr auto arity = mpfr::arity_v<func>;
    using row_entry = mpfr::data_entry<T, arity>;
    using parsed_inputs = std::vector<std::array<T, arity>>;
    using argument_set = std::array<T, arity>;
    struct shutdown {
        ~shutdown() { mpfr_free_cache(); }
    } const _;

    return process_inputs<T, arity>(parser)
        .and_then([&](parsed_inputs const& inputs) {
            std::vector<row_entry> rows;
            rows.reserve(inputs.size());
            for (argument_set const& args : inputs) {
                auto& current = rows.emplace_back();
                auto const expectation = calculate<func>(args);
                if constexpr (arity == 1) {
                    current.input = args[0];
                } else {
                    current.input = *reinterpret_cast<T const(*)[arity]>(&args);
                }

                current.value = expectation.value;
                current.residual = expectation.residual;
                current.flags = expectation.flags;
            }

            auto const output = parser.value_or<"output"_cli_opt>([] {
                using std::string_view_literals::operator""sv;
                static constexpr auto extension = ".dat"sv;
                static constexpr auto tp_name = mpfr::to_type_name<T>();
                constexpr auto length =
                    func.size() + tp_name.size() + 1 + extension.size();
                static char default_name[length + 1] = {};
                snprintf(default_name, sizeof(default_name), "%s-%s.dat",
                    func.c_str(), tp_name.c_str());
                return std::string_view(default_name);
            }());

            if (parser.value<"append"_cli_opt>()) {
                return append_results(func, rows, output);
            } else {
                return write_results(func, rows, output);
            }
        })
        .has_value();
}

template <mpfr::function_name func>
bool generate(mpfr::argument_parser const& parser) {
    auto const type = parser.positional(1zu);
    if (type.empty()) {
        fprintf(
            stderr, "Missing type argument for function '%s'\n", func.c_str());
        return false;
    }

    constexpr auto jump_table = dpl::apply(
        [](auto... funcs) {
            return mpfr::make_jump_table<mpfr::string_hash(funcs())...>();
        },
        mpfr::float_types);

    return jump_table(
        [](auto constant, std::string_view type,
            mpfr::argument_parser const& parser) {
            if constexpr (requires { mpfr::to_type(constant); }) {
                using float_t =
                    typename decltype(mpfr::to_type(constant))::type;
                return generate<float_t, func>(parser);
            } else {
                fprintf(stderr, "Unrecognized type '%.*s' for function '%s'\n",
                    static_cast<int>(type.size()), type.data(), func.c_str());
                return false;
            }
        },
        mpfr::string_hash(type), type, parser);
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
        "':max' leave one end unbounded; ':' entry leaves that "
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

    constexpr auto jump_table = dpl::apply(
        [](auto... funcs) {
            return mpfr::make_jump_table<mpfr::string_hash(funcs())...>();
        },
        mpfr::functions);

    mpfr::argument_parser parser(argc, argv);
    auto const function = parser.positional(0zu);
    if (function.empty()) {
        print_usage();
        return EXIT_FAILURE;
    }

    return parser.parse()
        .transform_error([](mpfr::parse_error err) -> int {
            if (err == mpfr::parse_error::unrecognized_argument) {
                print_usage();
            }

            return EXIT_FAILURE;
        })
        .and_then([&] -> std::expected<void, int> {
            if (parser.value<"help"_cli_opt>()) {
                print_usage();
                return {};
            }

            auto const succeeded = jump_table(
                [](auto constant, std::string_view name,
                    mpfr::argument_parser const& args) {
                    if constexpr (requires {
                                      mpfr::to_function_name(constant);
                                  }) {
                        constexpr auto cfunc_name =
                            mpfr::to_function_name(constant);
                        return generate<cfunc_name>(args);
                    } else {
                        fprintf(stderr, "Unrecognized function: '%.*s'\n",
                            static_cast<int>(name.size()), name.data());
                        return false;
                    }
                },
                mpfr::string_hash(function), function, parser);

            if (succeeded) {
                return {};
            } else {
                return std::unexpected(EXIT_FAILURE);
            }
        })
        .error_or(EXIT_SUCCESS);
}
