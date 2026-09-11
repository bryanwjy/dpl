// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

export module dpl.test.support:linear_counter;
import dpl;

export namespace dpl::test {

namespace dpp = dpl::datapar;
inline namespace support {

class linear_counter {
    class iterator;
    struct begin_t {};
    struct end_t {};

public:
    constexpr linear_counter(size_t count) noexcept : count_(count) {}

    template <typename T>
    requires dpl::is_class_v<dpl::decay_t<T>> && requires(T&& range) {
        { dpl::forward<T>(range).size() } -> dpl::core_convertible_to<size_t>;
    }
    constexpr linear_counter(T&& range) noexcept
        : linear_counter(dpl::forward<T>(range).size()) {}
    template <typename T, size_t N>
    constexpr linear_counter(T const (&&)[N]) noexcept : linear_counter(N) {}
    template <typename T, size_t N>
    constexpr linear_counter(T const (&)[N]) noexcept : linear_counter(N) {}

    constexpr explicit operator size_t(this linear_counter self) {
        return self.count_;
    }

    constexpr iterator begin(this linear_counter self) noexcept {
        return iterator(begin_t{}, self.count_);
    }

    constexpr iterator end(this linear_counter self) noexcept {
        return iterator(end_t{}, self.count_);
    }

    constexpr size_t size() const noexcept { return count_; }

private:
    class iterator {
    public:
        explicit constexpr iterator(begin_t, size_t count) noexcept
            : current_(0zu)
            , end_(count) {}

        explicit constexpr iterator(end_t, size_t count) noexcept
            : current_(count)
            , end_(count) {}

        constexpr iterator() noexcept = default;

        constexpr size_t operator*() const noexcept { return current_; }

        constexpr iterator& operator++(this iterator& self) noexcept {
            ++self.current_;
            return self;
        }

        constexpr iterator operator++(this iterator& self, int) noexcept {
            auto copy = self;
            ++self.current_;
            return copy;
        }

        constexpr bool operator==(
            this iterator const& self, iterator const& other) noexcept {
            return self.end_ == other.end_ && self.current_ == other.current_;
        }

        constexpr bool operator!=(
            this iterator const&, iterator const&) noexcept = default;

    private:
        size_t current_;
        size_t end_;
    };

    size_t count_ = 0zu;
};

} // namespace support
} // namespace dpl::test
