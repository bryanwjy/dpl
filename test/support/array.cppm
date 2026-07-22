// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

export module dpl.test:support.array;
import :support.span;
import dpl;

namespace dpl::test {

export template <dpl::semiregular E, size_t N>
class array {
    static_assert(dpl::is_same_v<E, dpl::decay_t<E>>);

public:
    using pointer = E*;
    using const_pointer = E const*;
    using reference = E&;
    using const_reference = E const&;
    using size_type = size_t;
    using value_type = E;
    using difference_type = ptrdiff_t;

    constexpr reference operator[](difference_type idx) noexcept {
        return data_[idx];
    }

    constexpr const_reference operator[](difference_type idx) const noexcept {
        return data_[idx];
    }

    constexpr pointer data() noexcept { return data_; }
    constexpr const_pointer data() const noexcept { return data_; }
    constexpr const_pointer cdata() const noexcept { return data_; }
    constexpr pointer begin() noexcept { return data_; }
    constexpr pointer end() noexcept { return data_ + N; }
    constexpr const_pointer begin() const noexcept { return data_; }
    constexpr const_pointer end() const noexcept { return data_ + N; }
    constexpr const_pointer cbegin() const noexcept { return data_; }
    constexpr const_pointer cend() const noexcept { return data_ + N; }
    constexpr size_type size() const noexcept { return N; }
    constexpr bool empty() const noexcept { return false; }
    constexpr operator span<E, N>() noexcept { return span<E, N>(data_); }
    constexpr operator span<E const, N>() const noexcept {
        return span<E const, N>(data_);
    }

    E data_[N];
};

template <typename E, size_t N>
span(array<E, N>&) -> span<E, N>;
template <typename E, size_t N>
span(array<E, N> const&) -> span<E const, N>;

template <typename E0, typename... Es>
array(E0, Es...) -> array<E0, sizeof...(Es) + 1>;

} // namespace dpl::test
