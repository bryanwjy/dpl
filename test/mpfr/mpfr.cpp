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
        dpl::uint32 count, std::span<std::string_view const> ranges,
        input_span inputs) {
        auto const ranges_count = ranges.size() / (N + 1);
        result_.reserve(std::ranges::size(inputs) + count * ranges_count);
        return generate_ranges(rng, ranges, count)
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
        std::span<std::string_view const> ranges, dpl::uint32 count) {

        using tuple_span = std::span<std::string_view const, N + 1>;
        auto const generate = [&](tuple_span info) {
            if (info.front() == "linear") {
                return generate_linear(info.template subspan<1>(), count);
            } else {
                return generate_random(rng, info.template subspan<1>(), count);
            }
        };

        for (auto const& info : ranges | std::views::chunk(N + 1)) {
            auto result = generate(tuple_span(info.begin(), info.end()));
            if (!result) {
                return result;
            }
        }

        return {};
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

    auto const has_inputs = config.has_value<"input"_cli_opt>();
    if (has_inputs) {
        if (auto const extent = config.extent<"input"_cli_opt>(); extent != N) {
            fprintf(stderr,
                "Input extent (%zu) does not match function arity (%zu)\n",
                extent, N);
            return input_error;
        }
    }

    if (auto const extent = config.extent<"range"_cli_opt>();
        extent != mpfr::typeconv::uninitialized_extent && extent != N + 1) {
        fprintf(stderr,
            "Range extent (%zu) does not match function arity (%zu)\n", extent,
            N);
        return input_error;
    }

    auto const count = config.value<"count"_cli_opt>();
    dpl::test::mt19937 rng(config.value<"seed"_cli_opt>());
    auto const range_info = !config.has_value<"range"_cli_opt>()
        ? std::span<std::string_view const>(default_bounds)
        : std::span<std::string_view const>(config.value<"range"_cli_opt>());

    auto const inputs = has_inputs
        ? std::span<std::string_view const>(config.value<"input"_cli_opt>())
        : std::span<std::string_view const>();

    return input_processor<T, N>{}.process(
        rng, count, range_info, inputs | std::views::chunk(N));
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

std::expected<mpfr::tmp_string, std::error_condition> read_function_name(
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

template <dpl::floating_point_like T>
constexpr std::string_view type_name_v = "";
template <>
constexpr std::string_view type_name_v<float> = "float";
template <>
constexpr std::string_view type_name_v<double> = "double";
template <>
constexpr std::string_view type_name_v<dpl::ext::float16> =
    "::dpl::ext::float16";
template <>
constexpr std::string_view type_name_v<dpl::ext::bfloat16> =
    "::dpl::ext::bfloat16";

template <dpl::floating_point_like T>
constexpr std::string_view suffix_v = "";
template <>
constexpr std::string_view suffix_v<float> = "f";
template <>
constexpr std::string_view suffix_v<double> = "";
template <>
constexpr std::string_view suffix_v<dpl::ext::float16> = "_f16";
template <>
constexpr std::string_view suffix_v<dpl::ext::bfloat16> = "_bf16";

template <dpl::floating_point_like T>
constexpr size_t max_hex_float_chars = [] {
    constexpr auto const bias = dpl::floating_point_traits<T>::exponent_bias;
    constexpr auto const frac_bits =
        dpl::countr_zero(dpl::floating_point_traits<T>::exponent_mask);
    constexpr auto const subnormal_expmin = 1 - bias - frac_bits;
    constexpr auto const exp =
        bias > -subnormal_expmin ? bias : -subnormal_expmin;
    constexpr auto log10p1 = [](int val) {
        int d = 1;
        while (val >= 10) {
            val /= 10;
            ++d;
        }
        return d;
    };
    auto const val = 1 + 2 + 2 +
        (dpl::floating_point_traits<T>::digits - 1 + 3) / 4 + 2 + log10p1(exp);
    using std::string_view_literals::operator""sv;
    constexpr auto inf_str = "-dpp::infinity"sv;
    auto const max_suffix = 5; // "_bf16"
    return val + max_suffix < inf_str.size() ? inf_str.size()
                                             : val + max_suffix;
}();

template <dpl::floating_point_like T>
using float_str = std::array<char, max_hex_float_chars<T> + 1>;

template <dpl::floating_point_like T>
constexpr float_str<T> to_string(T const& val) noexcept {
    // use a reference here, certain ISAs ftz on copy ctor
    using std::string_view_literals::operator""sv;
    float_str<T> buffer{};
    if (val != val) {
        "dpp::nan"sv.copy(buffer.data(), buffer.size());
        return buffer;
    }
    if (val == dpl::datapar::infinity_v<T>) {
        "dpp::infinity"sv.copy(buffer.data(), buffer.size());
        return buffer;
    }
    if (val == -dpl::datapar::infinity_v<T>) {
        "-dpp::infinity"sv.copy(buffer.data(), buffer.size());
        return buffer;
    }

    using traits = dpl::floating_point_traits<T>;
    auto const valbits = dpl::to_bit_representation(val);

    static constexpr auto shift = dpl::countr_zero(traits::exponent_mask);
    auto const uexp = (valbits & traits::exponent_mask) >> shift;
    auto const exp = (uexp ? dpl::to_signed(dpl::to_underlying(uexp)) : 1) -
        traits::exponent_bias;
    auto const prefix = ((valbits & ~traits::signbit) >> shift) ? 1 : 0;
    static constexpr auto mantissa_shift = (4 - (traits::digits - 1) % 4) % 4;
    auto const mantissa_hex = (valbits & traits::mantissa_mask)
        << mantissa_shift;
    auto const negative = static_cast<bool>(valbits & traits::signbit);
    // There is a weird behaviour with !(valbits & ~traits::signbit)
    // where the compiler will return zero if val is subnormal if the type has
    // flush-to-zero behaviour, e.g. bf16 on x86
    auto const iszero =
        dpl::countr_zero(valbits) == dpl::countr_zero(traits::signbit);
    auto ptr = buffer.data();
    *ptr++ = negative ? '-' : '+';
    *ptr++ = '0';

    constexpr auto write_unsigned = [](char* out, dpl::uint64 n) {
        constexpr char digits[] = "0123456789";
        char tmp[24];
        int len = 0;
        do {
            tmp[len++] = digits[n % 10];
            n /= 10;
        } while (n != 0);
        while (len > 0)
            *out++ = tmp[--len];
        return out;
    };

    constexpr auto write_mantissa = [](char* out, dpl::uint64 n) {
        constexpr char digits[] = "0123456789abcdef";
        constexpr auto leading = (shift + mantissa_shift) / 4;
        char tmp[24];
        int len = 0;
        do {
            tmp[len++] = digits[n % 16];
            n /= 16;
        } while (n != 0);
        while (len < leading) {
            tmp[len++] = '0';
        }
        while (len > 0)
            *out++ = tmp[--len];
        return out;
    };

    if (iszero) {
        *ptr++ = '.';
        *ptr++ = '0';
    } else {
        *ptr++ = 'x';
        *ptr++ = prefix ? '1' : '0';
        *ptr++ = '.';
        ptr = write_mantissa(ptr, dpl::to_underlying(mantissa_hex));
        *ptr++ = 'p';
        *ptr++ = exp < 0 ? '-' : '+';
        ptr =
            write_unsigned(ptr, static_cast<dpl::uint64>(exp < 0 ? -exp : exp));
    }

    if constexpr (dpl::is_same_v<T, float>) {
        *ptr++ = 'f';
    } else if constexpr (!dpl::is_same_v<T, double>) {
        *ptr++ = '_';
        std::ranges::copy(mpfr::to_type_name<T>().c_str(),
            mpfr::to_type_name<T>().c_str() + mpfr::to_type_name<T>().size(),
            ptr);
    }

    return buffer;
}

template <dpl::floating_point_like T, size_t N>
std::expected<void, std::error_condition> write_results(dpl::uint32 seed,
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
    } const stream{.ptr = ::fopen(tmpname.c_str(), "w")};
    static constexpr char comment[] = //
        "// Copyright 2026 Bryan Wong\n"
        "// DO NOT EDIT: This file is auto-generated\n";
    static constexpr char prefix[] = //
        "#pragma once\n\n"
        "#include \"dpl/config.h\"\n\n"
        "// clang-format off\n"
        "namespace testdata {\n"
        "namespace dpp = dpl::datapar;\n";

    if (stream.ptr == nullptr ||
        fwrite(comment, 1zu, sizeof(comment) - 1, stream.ptr) <
            sizeof(comment) - 1 ||
        fprintf(stream.ptr, "// Seed: %u\n", seed) < 0 ||
        fwrite(prefix, 1zu, sizeof(prefix) - 1, stream.ptr) <
            sizeof(prefix) - 1) {
        auto condition =
            std::make_error_condition(static_cast<std::errc>(errno));
        fprintf(
            stderr, "Failed to prefix file: %s\n", condition.message().c_str());
        return std::unexpected(condition);
    }

    if (fprintf(stream.ptr,
            "template <> inline constexpr mpfr::data_entry<%s, %zuzu> %s<%s>[] "
            "= {\n",
            mpfr::reflect_name<T>().c_str(), N,
            mpfr::tmp_string(function).c_str(),
            mpfr::reflect_name<T>().c_str()) < 0) {
        auto condition =
            std::make_error_condition(static_cast<std::errc>(errno));
        fprintf(stderr, "Failed to declare data struct for '%s': %s\n",
            mpfr::tmp_string(function).c_str(), condition.message().c_str());
        return std::unexpected(condition);
    }

    auto const error_path = [&] {
        auto condition =
            std::make_error_condition(static_cast<std::errc>(errno));
        fprintf(stderr, "Failed to write data for '%s': %s\n",
            mpfr::tmp_string(function).c_str(), condition.message().c_str());
        return std::unexpected(condition);
    };

    for (auto const& entry : entries) {
        if (fwrite("\t{ ", 1zu, 3zu, stream.ptr) < 3zu) {
            return error_path();
        }

        if constexpr (N > 1) {
            if (fwrite("{ ", 1zu, 2zu, stream.ptr) < 2zu ||
                dpl::pack::any_of(
                    [&](auto idx) {
                        auto const str = to_string(entry.input[idx]);
                        auto const view = std::string_view(str.data());
                        if constexpr (idx < N - 1) {
                            return fwrite(view.data(), 1zu, view.size(),
                                       stream.ptr) < view.size() ||
                                fputs(", ", stream.ptr) < 0;
                        } else {
                            return fwrite(view.data(), 1zu, view.size(),
                                       stream.ptr) < view.size();
                        }
                    },
                    dpl::make_index_sequence<N>{}) ||
                fwrite("},", 1zu, 2zu, stream.ptr) < 2zu) {
                return error_path();
            }
        } else {
            auto const str = to_string(entry.input);
            auto const view = std::string_view(str.data());
            if (fwrite(view.data(), 1zu, view.size(), stream.ptr) <
                    view.size() ||
                fwrite(",", 1zu, 1zu, stream.ptr) < 1zu) {
                return error_path();
            }
        }

        auto const str = to_string(entry.value);
        auto const view = std::string_view(str.data());
        if (fwrite(view.data(), 1zu, view.size(), stream.ptr) < view.size() ||
            fwrite(",", 1zu, 1zu, stream.ptr) < 1zu) {
            return error_path();
        }

        auto const nan_residual = entry.residual != entry.residual;
        if (nan_residual) {
            if (fprintf(stream.ptr,
                    "dpp::nan,static_cast<mpfr::result_flag>(%u) },\n",
                    0u + dpl::to_underlying(entry.flags)) < 0) {
                return error_path();
            }
        } else if (fprintf(stream.ptr,
                       "%g,static_cast<mpfr::result_flag>(%u) },\n",
                       entry.residual,
                       0u + dpl::to_underlying(entry.flags)) < 0) {
            return error_path();
        }
    }

    constexpr char suffix[] = "};\n}\n// clang-format on\n";
    if (fwrite(suffix, 1zu, sizeof(suffix) - 1, stream.ptr) <
        sizeof(suffix) - 1) {
        auto condition =
            std::make_error_condition(static_cast<std::errc>(errno));
        fprintf(stderr, "Failed to end datafile '%s': %s\n",
            mpfr::tmp_string(function).c_str(), condition.message().c_str());
        return std::unexpected(condition);
    }

    return replace_file(tmpname.c_str(), filepath);
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

    auto expectation = make_expectation<T>(dpl::move(result), nan_arguments);
    if constexpr (mpfr::exact_v<func>) {
        expectation.residual = 0;
        expectation.flags = mpfr::result_flag::none;
    }
    return expectation;
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
                    memcpy(current.input, args.data(), sizeof(current.input));
                }

                current.value = expectation.value;
                current.residual = expectation.residual;
                current.flags = expectation.flags;
            }

            auto const output = parser.value_or<"output"_cli_opt>([] {
                using std::string_view_literals::operator""sv;
                static constexpr auto extension = ".h"sv;
                static constexpr auto tp_name = mpfr::to_type_name<T>();
                constexpr auto length =
                    func.size() + tp_name.size() + 1 + extension.size();
                static char default_name[length + 1] = {};
                snprintf(default_name, sizeof(default_name), "%s-%s.h",
                    func.c_str(), tp_name.c_str());
                return std::string_view(default_name);
            }());

            // Shuffle
            [&](std::span<row_entry> entries) {
                dpl::test::mt19937 rng(parser.value<"seed"_cli_opt>());
                while (!entries.empty()) {
                    auto const idx = rng() % entries.size();
                    dpl::ranges::swap(entries[idx], entries.back());
                    entries = entries.subspan(0zu, entries.size() - 1zu);
                }
            }(std::span{rows});

            return write_results(
                parser.value<"seed"_cli_opt>(), func, rows, output);
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
        "  -o, --output PATH     output file (default: "
        "./<func>-<type>.dat)\n"
        "      --urand R[,R...]  every representable value in range R "
        "is "
        "equally likely (default: unbounded)\n"
        "      --linear R[,R...] values spread evenly across range R; "
        "both "
        "bounds must be finite\n"
        "  -n, --count N         random inputs to generate, excluding "
        "-i "
        "inputs (default: 100)\n"
        "  -s, --seed N          random seed, decimal or 0x-hex\n"
        "  -i, --input V[,V...]  add one call's arguments; "
        "repeat for more calls\n"
        "  -h, --help            show this help\n"
        "  @file                 read arguments from a file: "
        "newline-separated, # starts a comment, nested @paths are "
        "ignored\n"
        "\n"
        "ranges:\n"
        "  One [min,max) per argument, written as min:max. 'min:' and "
        "':max' leave one end unbounded; ':' entry leaves that "
        "argument unbounded.\n"
        "\n"
        "values (for -i and range bounds):\n"
        "  decimal (1.5, -2e-3), hex float (0x1.8p-3), inf, -inf, nan, "
        "-0, "
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
