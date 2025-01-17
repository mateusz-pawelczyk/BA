#pragma once
#include "core/ransac.hpp"
#include <iostream>
/**
 * @brief Gathers the top models from the global heap and sorts them by error.
 */
template <typename Comparator>
void RANSAC::gatherTopModels(
    std::priority_queue<
        FlatModelEntry, 
        std::vector<FlatModelEntry>,
        Comparator
    > &heap,
    std::vector<std::unique_ptr<FlatModel>> &models, 
    std::vector<double> &errors
) const {
    // Temporary container to store the heap elements
    std::vector<FlatModelEntry> temp;
    temp.reserve(heap.size());

    // Extract everything from the heap
    while (!heap.empty()) {
        auto topVal = std::move(const_cast<FlatModelEntry&>(heap.top()));
        heap.pop();
        temp.push_back(std::move(topVal));
    }

    // Sort by error (ascending)
    std::sort(temp.begin(), temp.end(), 
              [](const auto &a, const auto &b) { return a.first < b.first; });

    // Move into the vectors
    for (auto &entry : temp) {
        errors.push_back(entry.first);
        models.push_back(std::move(entry.second));
    }
};
