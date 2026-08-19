#include <algorithm>
#include <cstddef>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <utility>

class Buffer {
public:
    explicit Buffer(std::size_t size)
        : size_(size), data_(std::make_unique<double[]>(size)) {}

    Buffer(const Buffer& other)
        : Buffer(other.size_) {
        std::copy(other.data_.get(), other.data_.get() + size_, data_.get());
    }

    Buffer& operator=(const Buffer& other) {
        if (this == &other) return *this;
        Buffer copy(other);
        swap(copy);
        return *this;
    }

    Buffer(Buffer&& other) noexcept
        : size_(std::exchange(other.size_, 0)),
          data_(std::move(other.data_)) {}

    Buffer& operator=(Buffer&& other) noexcept {
        if (this != &other) {
            data_ = std::move(other.data_);
            size_ = std::exchange(other.size_, 0);
        }
        return *this;
    }
    ~Buffer() = default;

    void swap(Buffer& other) noexcept {
        std::swap(size_, other.size_);
        data_.swap(other.data_);
    }

    double& operator[](std::size_t index) {
        if (index >= size_) throw std::out_of_range("buffer index");
        return data_[index];
    }
    const double& operator[](std::size_t index) const {
        if (index >= size_) throw std::out_of_range("buffer index");
        return data_[index];
    }
    std::size_t size() const noexcept { return size_; }

private:
    std::size_t size_{};
    std::unique_ptr<double[]> data_;
};

int main() {
    Buffer a(3);
    a[0] = 1.0; a[1] = 2.0; a[2] = 3.0;
    Buffer b = a;             // deep copy
    Buffer c = std::move(b);  // ownership transfer
    if (b.size() != 0) return 1;
    std::cout << "size=" << c.size() << " sum=" << c[0] + c[1] + c[2] << "\n";
}
