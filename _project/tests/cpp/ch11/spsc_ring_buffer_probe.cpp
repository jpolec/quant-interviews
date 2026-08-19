#define main ch11_spsc_example_main
#include "../../../../_book/cpp/ch11/spsc_ring_buffer.cpp"
#undef main

#include <cassert>

struct Token {
    int value;
    Token() = delete;
    explicit Token(int input) noexcept : value(input) {}
    Token(Token&&) noexcept = default;
    Token& operator=(Token&&) noexcept = default;
    Token(const Token&) = delete;
    Token& operator=(const Token&) = delete;
};

int main() {
    SpscRingBuffer<Token, 4> queue;
    assert(queue.push(Token{1}));
    assert(queue.push(Token{2}));
    assert(queue.push(Token{3}));
    assert(!queue.push(Token{99}));
    auto one = queue.pop();
    auto two = queue.pop();
    assert(one && one->value == 1);
    assert(two && two->value == 2);
    assert(queue.push(Token{4}));
    assert(queue.push(Token{5}));
    for (int expected : {3, 4, 5}) {
        auto observed = queue.pop();
        assert(observed && observed->value == expected);
    }
    assert(!queue.pop());
}
