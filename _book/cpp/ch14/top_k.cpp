#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

class UpdateableTopK {
public:
    void update(std::string identifier, double score) {
        if (identifier.empty() || !std::isfinite(score)) {
            throw std::invalid_argument("identifier must be non-empty and score finite");
        }
        const auto found = scores_.find(identifier);
        if (found != scores_.end()) {
            ordered_.erase({found->second, identifier});
        }
        ordered_.emplace(score, identifier);
        scores_.insert_or_assign(std::move(identifier), score);
    }

    bool erase(const std::string& identifier) {
        const auto found = scores_.find(identifier);
        if (found == scores_.end()) {
            return false;
        }
        ordered_.erase({found->second, identifier});
        scores_.erase(found);
        return true;
    }

    [[nodiscard]] std::vector<std::pair<double, std::string>> top_k(
        std::size_t k
    ) const {
        if (k == 0) {
            throw std::invalid_argument("k must be positive");
        }
        std::vector<std::pair<double, std::string>> result;
        result.reserve(std::min(k, ordered_.size()));
        for (auto iterator = ordered_.rbegin();
             iterator != ordered_.rend() && result.size() < k;
             ++iterator) {
            result.push_back(*iterator);
        }
        return result;
    }

private:
    std::set<std::pair<double, std::string>> ordered_;
    std::unordered_map<std::string, double> scores_;
};

int main() {
    UpdateableTopK book;
    book.update("A", 5.0);
    book.update("B", 4.0);
    book.update("C", 3.0);
    book.update("A", 1.0);
    for (const auto& [score, identifier] : book.top_k(2)) {
        std::cout << identifier << ":" << score << "\n";
    }
}
