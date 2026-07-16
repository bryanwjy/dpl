// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <cassert>

export module dpl.test:utils.span;

import dpl;

namespace dpl::test {

inline constexpr size_t dynamic_extent = -1zu;

template <typename E, size_t N = dynamic_extent>
class span;

template <typename E>
class span<E, dynamic_extent> {
public:
    using pointer = E*;
    using const_pointer = E const*;
    using reference = E&;
    using const_reference = E const&;
    using size_type = size_t;
    using element_type = E;
    using value_type = remove_cv_t<E>;
    using difference_type = ptrdiff_t;

    static constexpr size_type extent = dynamic_extent;

    constexpr pointer data() const noexcept { return data_; }

    constexpr size_type size() const noexcept { return size_; }
    constexpr size_type size_bytes() const noexcept {
        return size_ * sizeof(E);
    }

    constexpr bool empty() const noexcept { return size() == 0; }
    constexpr pointer begin() const noexcept { return data_; }
    constexpr pointer end() const noexcept { return data_ + size_; }
    template <size_t N>
    requires (N != dynamic_extent)
    constexpr span<E, N> first() const noexcept {
        return span(data(), N);
    }
    template <size_t N>
    requires (N != dynamic_extent)
    constexpr span<E, N> last() const noexcept {
        return span(end() - N, N);
    }
    constexpr span first(size_type count) const noexcept {
        return span(data(), count);
    }
    constexpr span last(size_type count) const noexcept {
        return span(end() - count, count);
    }

    constexpr reference operator[](difference_type idx) const noexcept {
        return data_[idx];
    }

    constexpr span subspan(
        size_t offset, size_type count = dynamic_extent) const noexcept {
        offset = dpl::datapar::min(offset, size());
        count = dpl::datapar::min(size() - offset, count);
        return span(data() + offset, count);
    }

    constexpr span() noexcept = default;
    constexpr span(span const&) noexcept = default;
    constexpr span& operator=(span const&) noexcept = default;
    constexpr ~span() noexcept = default;

    template <typename TE, size_t N>
    requires dpl::same_as<dpl::copy_cv_t<element_type, TE>, element_type>
    constexpr span(span<TE, N> src) noexcept : span(src.data(), src.size()) {}

    template <size_t N>
    constexpr span(dpl::type_identity_t<element_type> (&data)[N]) noexcept
        : span(data, N) {}

    explicit constexpr span(pointer data, size_type size) noexcept
        : data_(data)
        , size_(size) {}

private:
    pointer data_ = nullptr;
    size_type size_ = 0zu;
};

template <typename E>
class span<E, 0zu> {
public:
    using pointer = E*;
    using const_pointer = E const*;
    using reference = E&;
    using const_reference = E const&;
    using size_type = size_t;
    using element_type = E;
    using value_type = remove_cv_t<E>;
    using difference_type = ptrdiff_t;

    static constexpr size_type extent = 0zu;

    constexpr pointer data() const noexcept { return data_; }
    constexpr size_type size() const noexcept { return 0zu; }
    constexpr size_type size_bytes() const noexcept { return 0zu; }
    constexpr bool empty() const noexcept { return true; }
    constexpr pointer begin() const noexcept { return data_; }
    constexpr pointer end() const noexcept { return data_; }
    template <size_t N = 0zu>
    requires (N != dynamic_extent)
    constexpr span first() const noexcept {
        static_assert(N == 0);
        return *this;
    }
    template <size_t N = 0zu>
    requires (N != dynamic_extent)
    constexpr span last() const noexcept {
        static_assert(N == 0);
        return *this;
    }

    template <size_t I, size_t N = dynamic_extent>
    constexpr auto subspan() const noexcept {
        if constexpr (N == dynamic_extent) {
            static_assert(I == 0);
            return span<E>(data(), 0zu);
        } else {
            static_assert(I == 0zu);
            static_assert(N == 0zu);
            return *this;
        }
    }

    constexpr span() noexcept = default;
    constexpr span(span const&) noexcept = default;
    constexpr span& operator=(span const&) noexcept = default;
    constexpr ~span() noexcept = default;

    template <typename TE, size_t TN>
    requires dpl::same_as<dpl::copy_cv_t<element_type, TE>, element_type> &&
        (TN == extent || TN == dynamic_extent)
    explicit(TN == dynamic_extent) constexpr span(span<TE, TN> src) noexcept
        : span(src.data(), src.size()) {}

    explicit constexpr span(pointer data, size_type size = 0zu) noexcept
        : data_(data)
        , size_(ensure_size(size)) {}

    constexpr reference operator[](difference_type idx) const noexcept = delete;

private:
    static constexpr size_type ensure_size(size_type size) noexcept {
        assert(size == 0zu);
        return 0zu;
    }
    pointer data_ = nullptr;
    size_type size_ = 0zu;
};

template <typename E, size_t N>
class span {
public:
    using pointer = E*;
    using const_pointer = E const*;
    using reference = E&;
    using const_reference = E const&;
    using size_type = size_t;
    using element_type = E;
    using value_type = remove_cv_t<E>;
    using difference_type = ptrdiff_t;

    static constexpr size_type extent = N;

    constexpr pointer data() const noexcept { return data_; }
    constexpr size_type size() const noexcept { return N; }
    constexpr size_type size_bytes() const noexcept {
        return N * sizeof(value_type);
    }
    constexpr bool empty() const noexcept { return false; }
    constexpr pointer begin() const noexcept { return data_; }
    constexpr pointer end() const noexcept { return data_ + N; }

    template <size_t TN>
    requires (TN != dynamic_extent && TN <= extent)
    constexpr span<E, TN> first() const noexcept {
        return span<E, TN>(data(), N);
    }

    template <size_t TN>
    requires (TN != dynamic_extent && TN <= extent)
    constexpr span<E, TN> last() const noexcept {
        static_assert(N == 0);
        return span<E, TN>(end() - TN, TN);
    }

    template <size_t I, size_t TN = dynamic_extent>
    requires (TN == dynamic_extent || I <= extent && TN <= extent - I)
    constexpr span<E, TN> subspan() const noexcept {
        if constexpr (TN == dynamic_extent) {
            return span<E>(data() + I, extent - I);
        } else {
            return span<E>(data() + I, TN);
        }
    }

    constexpr span() noexcept = delete;
    constexpr span(span const&) noexcept = default;
    constexpr span& operator=(span const&) noexcept = default;
    constexpr ~span() noexcept = default;

    template <typename TE, size_t TN>
    requires dpl::same_as<dpl::copy_cv_t<element_type, TE>, element_type> &&
        (TN == extent || TN == dynamic_extent)
    explicit(TN == dynamic_extent) constexpr span(span<TE, TN> src) noexcept
        : span(src.data(), src.size()) {}

    constexpr span(dpl::type_identity_t<element_type> (&data)[N]) noexcept
        : data_(data) {}

    explicit constexpr span(pointer data, size_type size) noexcept
        : data_(data) {
        assert(size == extent);
    }

    constexpr reference operator[](difference_type idx) const noexcept {
        return data_[idx];
    }

private:
    pointer data_;
};

template <typename P, dpl::integral_constant_like E>
span(P*, E) -> span<P, E::value>;
template <typename P>
span(P*, size_t) -> span<P>;
template <typename T, size_t N>
span(T (&)[N]) -> span<T, N>;
} // namespace dpl::test
