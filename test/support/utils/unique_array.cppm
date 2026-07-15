// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

export module dpl.test:utils.unique_array;
import :utils.span;
import dpl;

namespace dpl::test {

template <dpl::semiregular E>
class unique_array {
    static_assert(dpl::is_same_v<E, dpl::decay_t<E>>);

public:
    using pointer = E*;
    using const_pointer = E const*;
    using reference = E&;
    using const_reference = E const&;
    using size_type = size_t;
    using value_type = E;
    using difference_type = ptrdiff_t;

    constexpr unique_array() noexcept = default;
    constexpr ~unique_array() noexcept { destroy(); }

    explicit constexpr unique_array(size_t count)
        : begin_(new E[count])
        , end_(begin_ + count) {}

    explicit constexpr unique_array(span<E const> data)
        : begin_(new E[data.size()])
        , end_(begin_ + data.size()) {
        for (auto* ptr = begin_; auto const& val : data) {
            *ptr++ = val;
        }
    }

    unique_array(unique_array const&) = delete;
    unique_array& operator=(unique_array const&) = delete;

    constexpr unique_array(unique_array&& other) noexcept
        : begin_(dpl::exchange(other.begin_, nullptr))
        , end_(dpl::exchange(other.end_, nullptr)) {}

    constexpr unique_array& operator=(unique_array&& other) noexcept {
        destroy();
        begin_ = dpl::exchange(other.begin_, nullptr);
        end_ = dpl::exchange(other.end_, nullptr);
        return *this;
    }

    constexpr reference operator[](difference_type idx) noexcept {
        return begin_[idx];
    }

    constexpr const_reference operator[](difference_type idx) const noexcept {
        return begin_[idx];
    }

    constexpr pointer data() noexcept { return begin_; }
    constexpr const_pointer data() const noexcept { return begin_; }
    constexpr const_pointer cdata() const noexcept { return begin_; }
    constexpr pointer begin() noexcept { return begin_; }
    constexpr pointer end() noexcept { return end_; }
    constexpr const_pointer begin() const noexcept { return begin_; }
    constexpr const_pointer end() const noexcept { return end_; }
    constexpr const_pointer cbegin() const noexcept { return begin_; }
    constexpr const_pointer cend() const noexcept { return end_; }
    constexpr size_type size() const noexcept {
        return static_cast<size_type>(end_ - begin_);
    }
    constexpr bool empty() const noexcept { return size() == 0; }

    constexpr operator span<E>() noexcept { return span<E>(data(), size()); }

    constexpr operator span<E const>() const noexcept {
        return span<E const>(data(), size());
    }

private:
    constexpr void destroy() {
        auto* ptr = dpl::exchange(begin_, nullptr);
        end_ = nullptr;
        if (ptr != nullptr) {
            delete[] ptr;
        }
    }

    pointer begin_;
    pointer end_;
};

template <typename E>
span(unique_array<E>&) -> span<E>;
template <typename E>
span(unique_array<E> const&) -> span<E const>;

} // namespace dpl::test
