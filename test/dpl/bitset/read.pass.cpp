// Copyright 2025-2026 Bryan Wong
#include "dpl/config.h"

#include <cassert>

import dpl;

namespace {

// test(idx)
template <dpl::size_t W>
constexpr void test_test_vs_subscript() {
    // build a pattern with alternating bits
    dpl::bitset<W> b;
    for (auto i = 0zu; i < W; i += 2zu) {
        b.set(i);
    }

    // non-const object
    for (auto i = 0zu; i < W; ++i) {
        assert(b.test(i) == b[i]);
        assert(b.test(i) == (i % 2zu == 0));
    }

    // const object
    dpl::bitset<W> const& cb = b;
    for (auto i = 0zu; i < W; ++i) {
        assert(cb.test(i) == cb[i]);
        assert(cb.test(i) == (i % 2zu == 0));
    }
}

// For array-backed types, also explicitly probe chunk-boundary positions
// (indices 63, 64, 127, 128) since those are where per-word bit addressing
// is most likely to have an off-by-one.
template <dpl::size_t W>
constexpr void test_test_chunk_boundaries() {
    static_assert(W > 128);
    dpl::bitset<W> b;

    constexpr dpl::size_t boundaries[] = {0, 63, 64, 127, 128};
    for (auto idx : boundaries) {
        if (idx >= W)
            continue;
        b.set(idx);
        assert(b.test(idx) == true);
        b.clear(idx);
        assert(b.test(idx) == false);
    }
}

// operator== / operator!= (same width)

template <dpl::size_t W>
constexpr void test_eq_same_width() {
    dpl::bitset<W> a, b;

    // default-constructed: reflexive and symmetric
    assert(a == b);
    assert(b == a);
    assert(!(a != b));

    // reflexivity on a non-zero value
    a.set(0);
    assert(a == a);
    assert(!(a != a));

    // a != b after modifying one
    assert(!(a == b));
    assert(a != b);

    // equal after making them the same
    b.set(0);
    assert(a == b);
    assert(b == a);
    assert(!(a != b));

    // all-ones vs all-ones
    dpl::bitset<W> c, d;
    c.invert();
    d.invert();
    assert(c == d);
    assert(!(c != d));

    // all-ones vs all-zeros
    a.clear(0);
    assert(!(c == a));
    assert(c != a);
}

// operator== / operator!= (different width, same storage kind)

// integral vs integral: value_ is compared with arithmetic promotion so
// a bitset<5> with value 31 (0b11111) should equal a bitset<8> with the
// same value 31.
constexpr void test_eq_integral_cross_width() {
    // both zero
    dpl::bitset<5> z5;
    dpl::bitset<8> z8;
    dpl::bitset<32> z32;
    assert(z5 == z8);
    assert(z8 == z5);
    assert(z5 == z32);
    assert(z32 == z5);

    // same non-zero value
    dpl::bitset<5> v5(true); // bit 0 only
    dpl::bitset<8> v8(true);
    dpl::bitset<32> v32(true);
    assert(v5 == v8);
    assert(v8 == v5);
    assert(v5 == v32);
    assert(v32 == v5);

    // different values
    dpl::bitset<5> a5;
    a5.set(1);
    assert(!(v5 == a5));
    assert(v5 != a5);

    // a value that fits in W1 but has bits beyond W2's width (no such bits
    // exist after reinitialize() but confirm cross-width != still fires)
    dpl::bitset<8> b8;
    b8.set(7); // bit 7 does not exist in bitset<5>
    assert(!(z5 == b8));
    assert(z5 != b8);
}

// large vs large: word-by-word prefix compare with zero-check on excess words
constexpr void test_eq_large_cross_width() {
    // both zero
    dpl::bitset<129> z129;
    dpl::bitset<192> z192;
    dpl::bitset<256> z256;
    assert(z129 == z192);
    assert(z192 == z129);
    assert(z129 == z256);
    assert(z256 == z129);

    // same non-zero value (bit 0 only)
    dpl::bitset<129> v129;
    dpl::bitset<192> v192;
    dpl::bitset<256> v256;
    v129.set(0);
    v192.set(0);
    v256.set(0);
    assert(v129 == v192);
    assert(v192 == v129);
    assert(v129 == v256);
    assert(v256 == v129);

    // a bit set only in the wider operand's excess words means not equal to
    // the narrower operand
    dpl::bitset<256> hi256;
    hi256.set(191); // beyond bitset<192>'s W but within bitset<256>'s W
    assert(!(z192 == hi256));
    assert(z192 != hi256);

    // same prefix, excess bit set: not equal
    dpl::bitset<192> w192;
    dpl::bitset<256> w256;
    w192.set(0);
    w256.set(0);
    w256.set(200);
    assert(!(w192 == w256));
    assert(w192 != w256);
}

// operator== / operator!= (cross-storage-kind)

// integral (small) vs large: the large side delegates by reversing the call,
// routing to integral_bitset::operator== with the large as `other`.
constexpr void test_eq_cross_storage() {
    // both zero
    dpl::bitset<64> z64;
    dpl::bitset<129> z129;
    assert(z64 == z129);
    assert(z129 == z64);
    assert(!(z64 != z129));

    // same low bit set
    dpl::bitset<64> v64;
    dpl::bitset<129> v129;
    v64.set(0);
    v129.set(0);
    assert(v64 == v129);
    assert(v129 == v64);
    assert(!(v64 != v129));

    // different values
    dpl::bitset<64> a64;
    dpl::bitset<129> a129;
    a64.set(0);
    // a129 is all-zero
    assert(!(a64 == a129));
    assert(a64 != a129);
    assert(!(a129 == a64));
    assert(a129 != a64);

    // bit only set within the large type's range beyond the small type's
    // capacity -- they cannot be equal
    dpl::bitset<129> hi129;
    hi129.set(128);
    assert(!(z64 == hi129));
    assert(z64 != hi129);
}

constexpr bool run_all() {
    // test(idx) -- integral-backed
    test_test_vs_subscript<1>();
    test_test_vs_subscript<5>();
    test_test_vs_subscript<8>();
    test_test_vs_subscript<16>();
    test_test_vs_subscript<32>();
    test_test_vs_subscript<50>();
    test_test_vs_subscript<64>();

    // test(idx) -- array-backed
    test_test_vs_subscript<129>();
    test_test_vs_subscript<150>();
    test_test_vs_subscript<192>();
    test_test_vs_subscript<256>();

    // chunk-boundary probing (array-backed only)
    test_test_chunk_boundaries<129>();
    test_test_chunk_boundaries<192>();
    test_test_chunk_boundaries<256>();

    // operator== / != same width, integral-backed
    test_eq_same_width<1>();
    test_eq_same_width<5>();
    test_eq_same_width<8>();
    test_eq_same_width<16>();
    test_eq_same_width<32>();
    test_eq_same_width<50>();
    test_eq_same_width<64>();

    // operator== / != same width, array-backed
    test_eq_same_width<129>();
    test_eq_same_width<150>();
    test_eq_same_width<192>();
    test_eq_same_width<256>();

    // operator== / != cross-width, same storage kind
    test_eq_integral_cross_width();
    test_eq_large_cross_width();

    // operator== / != cross-storage-kind
    test_eq_cross_storage();

    return true;
}

} // namespace

int main() {
    static_assert(run_all());
    assert(run_all());
    return 0;
}
