#include "disjoint_set.h"

#include <algorithm>

namespace inseg_lib {

DisjointSet::DisjointSet(const int size) {
    Resize(size);
}

void DisjointSet::Resize(const int size) {
    parents_.assign(size, -1);
    num_elements_ = size;
}

int DisjointSet::Find(const int x) {
    if (x < 0 || x >= static_cast<int>(parents_.size())) {
        return 0;
    }
    int& parent = parents_[x];
    if (parent < 0) {
        return x;
    }
    parent = Find(parent);
    return parent;
}

bool DisjointSet::FindSet(const int x, const int y) {
    return Find(x) == Find(y);
}

bool DisjointSet::UnionSet(const int x, const int y) {
    int root_x = Find(x);
    int root_y = Find(y);
    if (root_x == root_y) {
        return false;
    }

    if (parents_[root_x] > parents_[root_y]) {
        std::swap(root_x, root_y);
    }

    parents_[root_x] += parents_[root_y];
    parents_[root_y] = root_x;
    --num_elements_;
    return true;
}

int DisjointSet::GetSize(const int x) {
    const int root = Find(x);
    if (root < 0 || root >= static_cast<int>(parents_.size())) {
        return 0;
    }
    return -parents_[root];
}

int DisjointSet::GetTotalSize() {
    return num_elements_;
}

}  // namespace inseg_lib
