// Copyright 2025-2026 Bryan Wong
#include "dpl/config.h"

#include <cassert>

import dpl.xmm;

namespace {
namespace xmm = dpl::datapar::xmm;
namespace dpp = dpl::datapar;
using abi = xmm::abi_tag;
template <typename... Ts>
constexpr auto element_count = dpp::simd_abi_traits<Ts...>::size;

using dpl::ext_literals::operator""_bf16;

constexpr auto min(auto lhs, auto rhs) noexcept {
    return lhs < rhs ? lhs : rhs;
}

template <dpl::floating_point_like To, dpl::floating_point_like From>
constexpr To float_cast(From val) noexcept {
    static_assert(dpl::explicitly_convertible_to<From, To>);
    return static_cast<To>(val);
}

template <dpl::floating_point_like To, dpl::floating_point_like From,
    typename Pred = decltype(dpp::cmpeq)>
requires dpl::different_from<From, To>
constexpr bool round_trip(From val, Pred pred = dpp::cmpeq) noexcept {
    constexpr auto count =
        min(element_count<From, abi>, element_count<To, abi>);
    constexpr auto M = (1 << count) - 1;
    auto const src = dpp::selecti<M>(dpp::broadcast<From, abi>(val), dpp::zero);
    static_assert(dpl::same_as<From,
        typename dpl::remove_const_t<decltype(src)>::value_type>);
    auto const dst = dpp::selecti<M>(
        dpp::broadcast<To, abi>(float_cast<To>(val)), dpp::zero);
    auto const actual_dst = dpp::element_cast<To>(src);
    auto const dstequal = pred(actual_dst, dst);
    auto const actual_castback = dpp::element_cast<From>(actual_dst);
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

template <dpl::floating_point_like To, dpl::floating_point_like From,
    typename Pred = decltype(dpp::cmpeq)>
requires (dpp::digits_v<To> != dpp::digits_v<From>)
constexpr bool one_way(From val, Pred pred = dpp::cmpeq) noexcept {
    constexpr auto count =
        min(element_count<From, abi>, element_count<To, abi>);
    constexpr auto M = (1 << count) - 1;
    auto const src = dpp::selecti<M>(dpp::broadcast<From, abi>(val), dpp::zero);
    auto const dst = dpp::selecti<M>(
        dpp::broadcast<To, abi>(float_cast<To>(val)), dpp::zero);
    auto const actual_dst = dpp::element_cast<To>(src);
    auto const dstequal = pred(actual_dst, dst);
    auto const actual_castback = dpp::element_cast<From>(actual_dst);
    auto const isequal = pred(actual_castback, src);
    return dpp::all_of(dstequal) && dpp::none_of(isequal);
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

    assert(one_way<float>(nextafter(1.0)));
    assert(one_way<float>(nextbefore(1.0)));
    assert(one_way<float>(dpl::bit_cast<double>(1ll << 52)));
    assert(one_way<float>(dpl::bit_cast<double>(1ll)));

    // Three values where there is insufficient precision
    assert(one_way<float>(0x1.p24 + 1.0));
    assert(one_way<float>(0x1.p10 + 0x1.p-14));
    assert(one_way<float>(1.0 + 0x1.p-24));

    return true;
}

constexpr bool test_fp16() noexcept {
#if DPL_SUPPORTS_FLOAT16
    assert(round_trip<float>(0.0f16));
    assert(round_trip<float>(-0.0f16, bitcmp));
    assert(round_trip<float>(1.0f16));
    assert(round_trip<float>(nextafter(1.0f16)));
    assert(round_trip<float>(nextbefore(1.0f16)));
    assert(round_trip<float>(0x1.p11f16));
    assert(round_trip<float>(
        dpl::bit_cast<dpl::float16>(static_cast<short>(1 << 11))));
    assert(
        round_trip<float>(dpl::bit_cast<dpl::float16>(static_cast<short>(1))));
    assert(round_trip<float>(1e-6f16));
    assert(round_trip<float>(dpp::max_value_v<dpl::float16>));
    assert(round_trip<float>(dpp::infinity_v<dpl::float16>));
    assert(round_trip<float>(-dpp::infinity_v<dpl::float16>));
    assert(one_way<dpl::float16>(nextafter(1.0f)));
    assert(one_way<dpl::float16>(nextbefore(1.0f)));
    assert(one_way<dpl::float16>(dpl::bit_cast<float>(1 << 23)));
    assert(one_way<dpl::float16>(dpl::bit_cast<float>(1)));
    assert(one_way<dpl::float16>(0x1.p11f + 1.0f));
    assert(one_way<dpl::float16>(0x1.p5f + 0x1.p-6f));
    assert(one_way<dpl::float16>(1.0f + 0x1.p-11f));

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
    assert(round_trip<double>(0x1.p11f16));
    assert(round_trip<double>(
        dpl::bit_cast<dpl::float16>(static_cast<short>(1 << 10))));
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

    assert(one_way<dpl::float16>(nextafter(1.0)));
    assert(one_way<dpl::float16>(nextbefore(1.0)));
    assert(one_way<dpl::float16>(dpl::bit_cast<double>(1ll << 52)));
    assert(one_way<dpl::float16>(dpl::bit_cast<double>(1ll)));
    assert(one_way<dpl::float16>(0x1.p11 + 1.0));
    assert(one_way<dpl::float16>(0x1.p5 + 0x1.p-6));
    assert(one_way<dpl::float16>(1.0 + 0x1.p-11));

#  if DPL_SUPPORTS_BFLOAT16
    assert(round_trip<dpl::bfloat16>(0.0f16));
    assert(round_trip<dpl::bfloat16>(-0.0f16, bitcmp));
    assert(round_trip<dpl::bfloat16>(1.0f16));
    assert(one_way<dpl::bfloat16>(nextafter(1.0f16)));
    assert(one_way<dpl::bfloat16>(nextbefore(1.0f16)));
    assert(round_trip<dpl::bfloat16>(0x1.p8f16));
    assert(round_trip<dpl::bfloat16>(dpp::infinity_v<dpl::float16>));
    assert(round_trip<dpl::bfloat16>(-dpp::infinity_v<dpl::float16>));
    assert(round_trip<dpl::bfloat16>(1e-6f16));
    assert(one_way<dpl::bfloat16>(dpp::max_value_v<dpl::float16>));
    assert(round_trip<dpl::bfloat16>(
        dpl::bit_cast<dpl::float16>(static_cast<unsigned short>(1 << 10))));
    assert(round_trip<dpl::bfloat16>(
        dpl::bit_cast<dpl::float16>(static_cast<unsigned short>(1))));
#  endif
#endif

    return true;
}

constexpr bool test_bf16() noexcept {
#if DPL_SUPPORTS_BFLOAT16
    constexpr auto min_normal =
        static_cast<dpl::bfloat16>(dpl::bit_cast<float>(1 << 23));
    constexpr auto min_subnormal =
        static_cast<dpl::bfloat16>(dpl::bit_cast<float>(1 << 16));

    assert(round_trip<float>(BF16(0.0)));
    assert(round_trip<float>(BF16(-0.0), bitcmp));
    assert(round_trip<float>(BF16(1.0)));
    assert(round_trip<float>(nextafter(BF16(1.0))));
    assert(round_trip<float>(nextbefore(BF16(1.0))));
    assert(round_trip<float>(BF16(0x1.p8)));
    assert(round_trip<float>(BF16(1e-40)));
    assert(round_trip<float>(dpp::max_value_v<dpl::bfloat16>));
    assert(round_trip<float>(dpp::infinity_v<dpl::bfloat16>));
    assert(round_trip<float>(-dpp::infinity_v<dpl::bfloat16>));
    assert(round_trip<float>(min_normal));
    assert(round_trip<float>(min_subnormal));
    assert(one_way<dpl::bfloat16>(nextafter(1.0f)));
    assert(one_way<dpl::bfloat16>(nextbefore(1.0f)));
    assert(one_way<dpl::bfloat16>(dpl::bit_cast<float>(1)));
    assert(one_way<dpl::bfloat16>(0x1.p8f + 1.0f));
    assert(one_way<dpl::bfloat16>(0x1.p4f + 0x1.p-4f));
    assert(one_way<dpl::bfloat16>(1.0f + 0x1.p-8f));

    assert(round_trip<double>(BF16(0.0)));
    assert(round_trip<double>(BF16(-0.0), bitcmp));
    assert(round_trip<double>(BF16(1.0)));
    assert(round_trip<double>(nextafter(BF16(1.0))));
    assert(round_trip<double>(nextbefore(BF16(1.0))));
    assert(round_trip<double>(BF16(0x1.p8)));
    assert(round_trip<double>(BF16(1e-40)));
    assert(round_trip<double>(dpp::max_value_v<dpl::bfloat16>));
    assert(round_trip<double>(dpp::infinity_v<dpl::bfloat16>));
    assert(round_trip<double>(-dpp::infinity_v<dpl::bfloat16>));
    assert(round_trip<double>(min_normal));
    assert(round_trip<double>(min_subnormal));
    assert(one_way<dpl::bfloat16>(nextafter(1.0)));
    assert(one_way<dpl::bfloat16>(nextbefore(1.0)));
    assert(one_way<dpl::bfloat16>(dpl::bit_cast<double>(1ll << 52)));
    assert(one_way<dpl::bfloat16>(dpl::bit_cast<double>(1ll)));
    assert(one_way<dpl::bfloat16>(0x1.p8 + 1.0));
    assert(one_way<dpl::bfloat16>(0x1.p4 + 0x1.p-4));
    assert(one_way<dpl::bfloat16>(1.0 + 0x1.p-8));

#  if DPL_SUPPORTS_FLOAT16
    assert(round_trip<dpl::float16>(BF16(0.0)));
    assert(round_trip<dpl::float16>(BF16(-0.0), bitcmp));
    assert(round_trip<dpl::float16>(BF16(1.0)));
    assert(round_trip<dpl::float16>(nextafter(BF16(1.0))));
    assert(round_trip<dpl::float16>(nextbefore(BF16(1.0))));
    assert(round_trip<dpl::float16>(BF16(0x1.p8)));
    assert(round_trip<dpl::float16>(dpp::infinity_v<dpl::bfloat16>));
    assert(round_trip<dpl::float16>(-dpp::infinity_v<dpl::bfloat16>));
    assert(one_way<dpl::float16>(BF16(1e-40)));
    assert(one_way<dpl::float16>(dpp::max_value_v<dpl::bfloat16>));
    assert(one_way<dpl::float16>(min_normal));
    assert(one_way<dpl::float16>(min_subnormal));
#  endif
#endif

    return true;
}

} // namespace

int main() {
    static_assert(test());
    assert(test());

    static_assert(test_fp16());
    assert(test_fp16());

    static_assert(test_bf16());
    assert(test_bf16());
    return 0;
}
