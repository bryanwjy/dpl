#include <cassert>

import dpl.xmm;

namespace xmm = dpl::datapar::xmm;
namespace dpp = dpl::datapar;

template <typename T, dpl::size_t N>
struct array {
    T data[N];

    friend constexpr auto begin(array const& src) noexcept { return src.data; }

    friend constexpr auto end(array const& src) noexcept {
        return src.data + N;
    }
};

template <typename... Args>
array(Args...)
    -> array<dpl::decay_t<dpl::common_type_t<Args...>>, sizeof...(Args)>;

template <dpl::integral I, dpl::floating_point F>
constexpr void general_int_to_fp() noexcept {
    auto const inputs = [](auto... vals) {
        constexpr I zero = 0;
        if constexpr (dpl::unsigned_integral<I>) {
            return array{
                (vals > dpp::max_value ? zero : static_cast<I>(vals))...};
        } else {
            return array{
                (vals > dpp::max_value_v<I> ? zero : static_cast<I>(vals))...,
                (-vals < dpp::min_value_v<I> ? zero
                                             : static_cast<I>(-vals))...};
        }
    }(0, 1, 2, 3, 7, 15, 255, 1023, 65535, 1048575);
    auto const make_expected = [](I val) {
        constexpr auto to_keep = sizeof(I) > sizeof(F)
            ? dpp::element_count<F, xmm::abi_tag> -
                sizeof(I) / sizeof(F) // NOLINT(bugprone-sizeof-expression)
            : dpp::element_count<F, xmm::abi_tag>;
        constexpr auto mask = dpp::imm<(1 << to_keep) - 1>;
        auto const result =
            dpp::broadcast<F, xmm::abi_tag>(static_cast<F>(val));
        return dpp::bit_keep(mask, result);
    };

    for (auto const val : inputs) {
        auto const in = dpp::broadcast<I, xmm::abi_tag>(val);
        auto const expected = make_expected(val);
        assert(dpp::all_of(dpp::cast<F>(in) == expected));
    }
}

constexpr bool general() noexcept {
    general_int_to_fp<dpl::int64, float>();
    general_int_to_fp<dpl::int32, float>();
    general_int_to_fp<dpl::int16, float>();
    general_int_to_fp<dpl::int8, float>();

    general_int_to_fp<dpl::int64, double>();
    general_int_to_fp<dpl::int32, double>();
    general_int_to_fp<dpl::int16, double>();
    general_int_to_fp<dpl::int8, double>();

    return true;
}

int main() {
    static_assert(general());
    assert(general());
    return 0;
}
