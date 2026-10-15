// Generated with Claude
#include "../common.h"

import dpl.xmm;

// Tests for to_bitset and from_bitset, CPO and xmm convenience forms only.
//
// to_bitset forms:
//   (1) dpp::to_bitset(mask)     -- CPO
//   (2) xmm::to_bitset(mask)     -- xmm convenience
//
// from_bitset forms:
//   (3) dpp::from_bitset<E, A>(b)     -- E and A both explicit
//   (4) dpp::from_bitset<A, E>(b)     -- A and E reordered
//   (5) dpp::from_bitset<mask_t<E>>(b) -- mask type T
//   (6) dpp::from_bitset<A>(b)         -- A only, E deduced from bitset width
//                                          via unsigned_integral_type_t;
//                                          unsigned types only
//   (7) xmm::from_bitset<E>(b)         -- xmm convenience
//
// Test structure:
//   - from_bitset single-bit: build a bitset with only bit i set, verify
//     from_bitset produces a mask with exactly lane i true via dpp::extract.
//   - to_bitset round-trip: to_bitset(from_bitset(b)) == b, exercised for
//     all-zero, all-ones, alternating, and each single-bit pattern.
//   - All forms produce the correct return type (static_assert).

namespace {

namespace dpp = dpl::datapar;
namespace xmm = dpl::datapar::xmm;

using abi_t = xmm::abi_tag;

template <typename E>
using vec_t = xmm::vector<E>;

template <typename E>
using mask_t = xmm::mask<E>;

template <typename E>
using abi_traits = dpp::simd_abi_traits<abi_t, E>;

// ---- from_bitset -----------------------------------------------------------

template <typename E>
constexpr void test_from_bitset() {
    constexpr auto lanes = abi_traits<E>::size();
    using bitset_t = dpl::bitset<lanes>;

    // Return type checks
    static_assert(
        dpl::is_same_v<decltype(dpp::from_bitset<E, abi_t>(bitset_t{})),
            mask_t<E>>);
    static_assert(
        dpl::is_same_v<decltype(dpp::from_bitset<abi_t, E>(bitset_t{})),
            mask_t<E>>);
    static_assert(
        dpl::is_same_v<decltype(dpp::from_bitset<mask_t<E>>(bitset_t{})),
            mask_t<E>>);
    static_assert(
        dpl::is_same_v<decltype(xmm::from_bitset<E>(bitset_t{})), mask_t<E>>);

    // All-zero bitset: every lane false
    {
        bitset_t z;
        auto check = [&](mask_t<E> m) {
            for (auto i = 0zu; i < lanes; ++i)
                assert(dpp::extract(m, i) == false);
        };
        check(dpp::from_bitset<E, abi_t>(z));  // (3)
        check(dpp::from_bitset<abi_t, E>(z));  // (4)
        check(dpp::from_bitset<mask_t<E>>(z)); // (5)
        check(xmm::from_bitset<E>(z));         // (7)
    }

    // All-ones bitset: every lane true
    {
        bitset_t all;
        all.invert();
        auto check = [&](mask_t<E> m) {
            for (auto i = 0zu; i < lanes; ++i)
                assert(dpp::extract(m, i) == true);
        };
        check(dpp::from_bitset<E, abi_t>(all));  // (3)
        check(dpp::from_bitset<abi_t, E>(all));  // (4)
        check(dpp::from_bitset<mask_t<E>>(all)); // (5)
        check(xmm::from_bitset<E>(all));         // (7)
    }

    // Single-bit: bit i set -> only lane i true
    for (auto i = 0zu; i < lanes; ++i) {
        bitset_t b;
        b.set(i);
        auto check = [&](mask_t<E> m) {
            for (auto j = 0zu; j < lanes; ++j)
                assert(dpp::extract(m, j) == (j == i));
        };
        check(dpp::from_bitset<E, abi_t>(b));  // (3)
        check(dpp::from_bitset<abi_t, E>(b));  // (4)
        check(dpp::from_bitset<mask_t<E>>(b)); // (5)
        check(xmm::from_bitset<E>(b));         // (7)
    }
}

// from_bitset<A> only: E is deduced as unsigned_integral_type_t<register_bits /
// W>, always an unsigned integer type. Tested separately for each deduced type.
template <typename E>
constexpr void test_from_bitset_abi_deduced() {
    static_assert(dpl::integral<E>);
    constexpr auto lanes = abi_traits<E>::size();
    using bitset_t = dpl::bitset<lanes>;

    static_assert(dpl::is_same_v<decltype(dpp::from_bitset<abi_t>(bitset_t{})),
        mask_t<E>>);

    bitset_t z;
    auto m_false = dpp::from_bitset<abi_t>(z);
    for (auto i = 0zu; i < lanes; ++i)
        assert(dpp::extract(m_false, i) == false);

    bitset_t all;
    all.invert();
    auto m_true = dpp::from_bitset<abi_t>(all);
    for (auto i = 0zu; i < lanes; ++i)
        assert(dpp::extract(m_true, i) == true);

    for (auto i = 0zu; i < lanes; ++i) {
        bitset_t b;
        b.set(i);
        auto m = dpp::from_bitset<abi_t>(b);
        for (auto j = 0zu; j < lanes; ++j)
            assert(dpp::extract(m, j) == (j == i));
    }
}

// ---- to_bitset -------------------------------------------------------------

template <typename E>
constexpr void test_to_bitset() {
    constexpr auto lanes = abi_traits<E>::size();
    using bitset_t = dpl::bitset<lanes>;

    // Return type checks
    mask_t<E> dummy = dpp::broadcast<E, abi_t>(false);
    static_assert(dpl::is_same_v<decltype(dpp::to_bitset(dummy)), bitset_t>);
    static_assert(dpl::is_same_v<decltype(xmm::to_bitset(dummy)), bitset_t>);

    // All-false -> all zero bits
    {
        mask_t<E> m = dpp::broadcast<E, abi_t>(false);
        auto b1 = dpp::to_bitset(m); // (1)
        auto b2 = xmm::to_bitset(m); // (2)
        assert(dpl::popcount(b1) == 0);
        assert(dpl::popcount(b2) == 0);
    }

    // All-true -> all lane bits set
    {
        mask_t<E> m = dpp::broadcast<E, abi_t>(true);
        auto b1 = dpp::to_bitset(m);
        auto b2 = xmm::to_bitset(m);
        assert(dpl::popcount(b1) == static_cast<int>(lanes));
        assert(dpl::popcount(b2) == static_cast<int>(lanes));
    }

    // Round-trip to_bitset(from_bitset(b)) == b for several patterns
    auto round_trip = [&](bitset_t b) {
        assert(dpp::to_bitset(dpp::from_bitset<E, abi_t>(b)) == b); // (1)
        assert(xmm::to_bitset(dpp::from_bitset<E, abi_t>(b)) == b); // (2)
    };

    // All zeros
    round_trip(bitset_t{});

    // All ones
    {
        bitset_t all;
        all.invert();
        round_trip(all);
    }

    // Alternating bits
    {
        bitset_t alt;
        for (auto i = 0zu; i < lanes; i += 2zu)
            alt.set(i);
        round_trip(alt);
    }

    // Each single-bit pattern
    for (auto i = 0zu; i < lanes; ++i) {
        bitset_t b;
        b.set(i);
        round_trip(b);
    }
}

constexpr bool run_all() {
    test_from_bitset<dpl::int8>();
    test_from_bitset<dpl::uint8>();
    test_from_bitset<dpl::int16>();
    test_from_bitset<dpl::uint16>();
    test_from_bitset<dpl::int32>();
    test_from_bitset<dpl::uint32>();
    test_from_bitset<dpl::int64>();
    test_from_bitset<dpl::uint64>();
    test_from_bitset<float>();
    test_from_bitset<double>();

    // from_bitset<abi_t>: E deduced as unsigned type per bitset width
    test_from_bitset_abi_deduced<dpl::uint8>();  // bitset<16>
    test_from_bitset_abi_deduced<dpl::uint16>(); // bitset<8>
    test_from_bitset_abi_deduced<dpl::uint32>(); // bitset<4>
    test_from_bitset_abi_deduced<dpl::uint64>(); // bitset<2>

    test_to_bitset<dpl::int8>();
    test_to_bitset<dpl::uint8>();
    test_to_bitset<dpl::int16>();
    test_to_bitset<dpl::uint16>();
    test_to_bitset<dpl::int32>();
    test_to_bitset<dpl::uint32>();
    test_to_bitset<dpl::int64>();
    test_to_bitset<dpl::uint64>();
    test_to_bitset<float>();
    test_to_bitset<double>();

    return true;
}

} // namespace

int main() {
    static_assert(run_all());
    assert(run_all());
    return 0;
}
