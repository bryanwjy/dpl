// Copyright 2025-2026 Bryan Wong
#include "dpl/config.h"

#include <cassert>

import dpl.xmm;

namespace {
namespace xmm = dpl::datapar::xmm;
namespace dpp = dpl::datapar;
using abi = xmm::abi_tag;

constexpr auto min(auto lhs, auto rhs) noexcept {
    return lhs < rhs ? lhs : rhs;
}

template <dpl::floating_point To, dpl::floating_point From,
    typename Pred = decltype(dpp::cmpeq)>
requires dpl::different_from<From, To>
constexpr bool round_trip(From val, Pred pred = dpp::cmpeq) noexcept {
    constexpr auto count =
        min(dpp::element_count<From, abi>, dpp::element_count<To, abi>);
    constexpr auto keep_mask = dpp::imm<(1 << count) - 1>;
    auto const src = dpp::bit_keep(keep_mask, dpp::broadcast<From, abi>(val));
    auto const dst =
        dpp::bit_keep(keep_mask, dpp::broadcast<To, abi>(static_cast<To>(val)));
    auto const actual_dst = dpp::cast<To>(src);
    auto const dstequal = pred(actual_dst, dst);
    auto const actual_castback = dpp::cast<From>(actual_dst);
    auto const isequal = pred(actual_castback, src);
    return dpp::all_of(isequal) && dpp::all_of(dstequal);
}

constexpr auto nextafter(auto val) noexcept {
    using type = decltype(val);
    using rep = dpp::signed_representation_t<type>;
    return dpl::bit_cast<type>(static_cast<rep>(dpl::bit_cast<rep>(val) + 1));
}

constexpr auto nextbefore(auto val) noexcept {
    using type = decltype(val);
    using rep = dpp::signed_representation_t<type>;
    return dpl::bit_cast<type>(static_cast<rep>(dpl::bit_cast<rep>(val) - 1));
}

inline constexpr struct nancmp_t {
    constexpr auto operator()(auto lhs, auto rhs) noexcept {
        return dpp::isnan(lhs) == dpp::isnan(rhs);
    }
} nancmp{};

inline constexpr struct bitcmp_t {
    template <typename T>
    constexpr auto operator()(T lhs, T rhs) noexcept {
        using type = typename T::value_type;
        return dpp::reinterpret<dpp::signed_representation_t<type>>(lhs) ==
            dpp::reinterpret<dpp::signed_representation_t<type>>(rhs);
    }
} bitcmp{};

template <dpl::floating_point To, dpl::floating_point From,
    typename Pred = decltype(dpp::cmpeq)>
requires (dpp::digits_v<To> < dpp::digits_v<From>)
constexpr bool precision_loss(From val, Pred pred = dpp::cmpeq) noexcept {
    constexpr auto count =
        min(dpp::element_count<From, abi>, dpp::element_count<To, abi>);
    constexpr auto keep_mask = dpp::imm<(1 << count) - 1>;
    auto const src = dpp::bit_keep(keep_mask, dpp::broadcast<From, abi>(val));
    auto const dst =
        dpp::bit_keep(keep_mask, dpp::broadcast<To, abi>(static_cast<To>(val)));
    auto const actual_dst = dpp::cast<To>(src);
    auto const dstequal = pred(actual_dst, dst);
    return dpp::all_of(dstequal);
}

constexpr bool test() {

    assert(round_trip<double>(0.0f));
    assert(round_trip<double>(-0.0f, bitcmp));
    assert(round_trip<double>(1.0f));
    assert(round_trip<double>(nextafter(1.0f)));
    assert(round_trip<double>(nextbefore(1.0f)));
    assert(round_trip<double>(0x1.p24f));
    assert(round_trip<double>(dpl::bit_cast<float>(1 << 23)));
    assert(round_trip<double>(dpl::bit_cast<float>(1)));
    assert(round_trip<double>(1e-40f));
    assert(round_trip<double>(dpp::max_value_v<float>));
    assert(round_trip<double>(dpp::infinity_v<float>));
    assert(round_trip<double>(-dpp::infinity_v<float>));

    // qnan
    assert(round_trip<double>(dpl::bit_cast<float>(0x7fc00000), nancmp));
    // snan
    assert(round_trip<double>(dpl::bit_cast<float>(0x7fa00000), nancmp));

    assert(round_trip<float>(0.0));
    assert(round_trip<float>(-0.0, bitcmp));
    assert(round_trip<float>(1.0));
    assert(round_trip<float>(0x1.p24));
    assert(round_trip<float>(dpp::infinity_v<double>));
    assert(round_trip<float>(-dpp::infinity_v<double>));
    // qnan
    assert(round_trip<float>(dpl::bit_cast<double>(0x7ff8ll << 48), nancmp));
    // snan
    assert(round_trip<float>(dpl::bit_cast<double>(0x7ff4ll << 48), nancmp));

    assert(precision_loss<float>(nextafter(1.0)));
    assert(precision_loss<float>(nextbefore(1.0)));
    assert(precision_loss<float>(dpl::bit_cast<double>(1ll << 52)));
    assert(precision_loss<float>(dpl::bit_cast<double>(1ll)));

    assert(precision_loss<float>(0x1.p24 + 1.0));
    assert(precision_loss<float>(0x1.p10 + 0x1.p-14));
    assert(precision_loss<float>(1.0 + 0x1.p-24));

#if DPL_SUPPORTS_FLOAT16
    assert(round_trip<float>(0.0f16));
    assert(round_trip<float>(-0.0f16, bitcmp));
    assert(round_trip<float>(1.0f16));
    assert(round_trip<float>(nextafter(1.0f16)));
    assert(round_trip<float>(nextbefore(1.0f16)));
    assert(round_trip<float>(0x1.p12f16));
    assert(round_trip<float>(
        dpl::bit_cast<dpl::float16>(static_cast<short>(1 << 11))));
    assert(
        round_trip<float>(dpl::bit_cast<dpl::float16>(static_cast<short>(1))));
    assert(round_trip<float>(1e-6f16));
    assert(round_trip<float>(dpp::max_value_v<dpl::float16>));
    assert(round_trip<float>(dpp::infinity_v<dpl::float16>));
    assert(round_trip<float>(-dpp::infinity_v<dpl::float16>));
    assert(precision_loss<dpl::float16>(nextafter(1.0f)));
    assert(precision_loss<dpl::float16>(nextbefore(1.0f)));
    assert(precision_loss<dpl::float16>(dpl::bit_cast<float>(1 << 23)));
    assert(precision_loss<dpl::float16>(dpl::bit_cast<float>(1)));
    assert(precision_loss<dpl::float16>(0x1.p11f + 1.0f));
    assert(precision_loss<dpl::float16>(0x1.p5f + 0x1.p-6f));
    assert(precision_loss<dpl::float16>(1.0f + 0x1.p-11f));

    // qnan
    assert(round_trip<float>(
        dpl::bit_cast<dpl::float16>(static_cast<short>(0x7e00)), nancmp));
    // snan
    assert(round_trip<float>(
        dpl::bit_cast<dpl::float16>(static_cast<short>(0x7c01)), nancmp));

    assert(round_trip<double>(0.0f16));
    assert(round_trip<double>(-0.0f16, bitcmp));
    assert(round_trip<double>(1.0f16));
    assert(round_trip<double>(nextafter(1.0f16)));
    assert(round_trip<double>(nextbefore(1.0f16)));
    assert(round_trip<double>(0x1.p12f16));
    assert(round_trip<double>(
        dpl::bit_cast<dpl::float16>(static_cast<short>(1 << 11))));
    assert(
        round_trip<double>(dpl::bit_cast<dpl::float16>(static_cast<short>(1))));
    assert(round_trip<double>(1e-6f16));
    assert(round_trip<double>(dpp::max_value_v<dpl::float16>));
    assert(round_trip<double>(dpp::infinity_v<dpl::float16>));
    assert(round_trip<double>(-dpp::infinity_v<dpl::float16>));
    // qnan
    assert(round_trip<double>(
        dpl::bit_cast<dpl::float16>(static_cast<short>(0x7e00)), nancmp));
    // snan
    assert(round_trip<double>(
        dpl::bit_cast<dpl::float16>(static_cast<short>(0x7c01)), nancmp));

    assert(precision_loss<dpl::float16>(nextafter(1.0)));
    assert(precision_loss<dpl::float16>(nextbefore(1.0)));
    assert(precision_loss<dpl::float16>(dpl::bit_cast<double>(1ll << 52)));
    assert(precision_loss<dpl::float16>(dpl::bit_cast<double>(1ll)));
    assert(precision_loss<dpl::float16>(0x1.p11 + 1.0));
    assert(precision_loss<dpl::float16>(0x1.p5 + 0x1.p-6));
    assert(precision_loss<dpl::float16>(1.0 + 0x1.p-11));
#endif

    return true;
}
} // namespace

int main() {
    static_assert(test());
    assert(test());
    return 0;
}
