#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string_view>

class FileHandle {
public:
    FileHandle(const char* path, const char* mode) : file_(std::fopen(path, mode)) {
        if (file_ == nullptr) {
            throw std::runtime_error("failed to open file");
        }
    }

    FileHandle(const FileHandle&) = delete;
    FileHandle& operator=(const FileHandle&) = delete;

    FileHandle(FileHandle&& other) noexcept : file_(other.file_) {
        other.file_ = nullptr;
    }

    FileHandle& operator=(FileHandle&& other) noexcept {
        if (this != &other) {
            close();
            file_ = other.file_;
            other.file_ = nullptr;
        }
        return *this;
    }

    ~FileHandle() { close(); }

    void write(std::string_view text) {
        if (file_ == nullptr) {
            throw std::logic_error("write on a moved-from or closed handle");
        }
        if (std::fwrite(text.data(), 1, text.size(), file_) != text.size()) {
            throw std::runtime_error("short write");
        }
    }

    void commit() {
        if (file_ == nullptr) throw std::logic_error("handle is closed");
        if (std::fflush(file_) != 0) throw std::runtime_error("flush failed");
        if (std::fclose(file_) != 0) {
            file_ = nullptr;
            throw std::runtime_error("close failed");
        }
        file_ = nullptr;
    }

private:
    void close() noexcept {
        if (file_ != nullptr) {
            std::fclose(file_);
            file_ = nullptr;
        }
    }

    std::FILE* file_{};
};

int main() {
    FileHandle output("/tmp/ch11_raii_example.txt", "wb");
    output.write("deterministic-resource-ownership\n");
    output.commit();
    std::cout << "OK\n";
}
