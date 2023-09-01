#include <deque>
#include <map>
#include <optional>

typedef int64_t index_t;

template <typename key_t, typename distance_t> class Metric {
    public:
    static distance_t distance(const key_t &a, const key_t &b);
    static std::optional<distance_t> threshold_distance(const key_t &a, const key_t &b, distance_t threshold);
};

template <typename key_t, typename distance_t> class BKNode {
    public:
    key_t key;
    index_t index;
    std::map<distance_t, BKNode<key_t, distance_t> *> leaves;
    std::optional<distance_t> max_distance;

    BKNode(key_t key, index_t index) : key(key), index(index) {}

    void add_leaf(distance_t distance, key_t key, index_t index) {
        leaves[distance] = new BKNode<key_t, distance_t>(key, index);
        max_distance = std::max<distance_t>(distance, max_distance.value_or(0));
    }

    void clear() {
        for (auto [d, bknode] : leaves) {
            delete bknode;
        }
        leaves.clear();
    }

    virtual ~BKNode() { clear(); }
};

template <typename key_t, typename distance_t, typename metric> class BKTree {
    BKNode<key_t, distance_t> *root;
    size_t index;

    public:
    BKTree() : root(nullptr), index(0) {}

    void add(key_t key) {
        if (root == nullptr) {
            root = new BKNode<key_t, distance_t>(key, index);
        } else {
            BKNode<key_t, distance_t> *node = root;
            distance_t dist;

            while (true) {
                dist = metric::distance(node->key, key);
                auto next_it = node->leaves.find(dist);
                if (next_it == node->leaves.end()) {
                    break;
                }
                node = next_it->second;
            }

            node->add_leaf(dist, key, index);
        }
        ++index;
    }

    void update(std::vector<key_t> keys) {
        for (auto const &key : keys) {
            add(key);
        }
    }

    std::tuple<std::vector<index_t>, std::vector<distance_t>, std::vector<key_t>> find(key_t key, distance_t threshold) {
        static_assert(std::is_signed<distance_t>::value, "Arithmetic required signed distances");

        BKNode<key_t, distance_t> *node = root;
        std::vector<index_t> indices;
        std::vector<distance_t> distances;
        std::vector<key_t> keys;

        if (node == nullptr) {
            return std::make_tuple(indices, distances, keys);
        }

        std::deque<BKNode<key_t, distance_t> *> candidates = {node};
        distance_t distance_cutoff, dist, lower, upper;
        BKNode<key_t, distance_t> *candidate;
        std::optional<distance_t> dist_opt;

        while (!candidates.empty()) {
            candidate = candidates.front();
            candidates.pop_front();
            distance_cutoff = candidate->max_distance.value_or(0) + threshold;
            dist_opt = metric::threshold_distance(key, candidate->key, distance_cutoff);

            if (!dist_opt.has_value()) {
                continue;
            }

            dist = dist_opt.value();

            if (dist <= threshold) {
                indices.push_back(candidate->index);
                distances.push_back(dist);
                keys.push_back(candidate->key);
            }

            lower = dist - threshold;
            upper = dist + threshold;
            for (auto [d, bknode] : candidate->leaves) {
                if (lower <= d && d <= upper) {
                    candidates.push_back(bknode);
                }
            }
        }
        return std::make_tuple(indices, distances, keys);
    }

    std::tuple<std::vector<std::vector<index_t>>, std::vector<std::vector<distance_t>>, std::vector<std::vector<key_t>>>
    find_batch(const std::vector<key_t> &keys, distance_t threshold) {
        std::vector<std::vector<index_t>> indices_out(keys.size());
        std::vector<std::vector<distance_t>> distances_out(keys.size());
        std::vector<std::vector<key_t>> keys_out(keys.size());

#if (ENABLE_OMP_PARALLEL)
#pragma omp parallel for schedule(static, 1)
#endif
        // i should be size_t, however msvc requires signed integral loop variables (except with -openmp:llvm)
        for (int i = 0; i < static_cast<int>(keys.size()); ++i) {
            auto &&[indices_res, distances_res, keys_res] = find(keys[i], threshold);
            indices_out[i] = std::move(indices_res);
            distances_out[i] = std::move(distances_res);
            keys_out[i] = std::move(keys_res);
        }

        return std::make_tuple(indices_out, distances_out, keys_out);
    }

    bool empty() { return root == nullptr; }

    static void _values(BKNode<key_t, distance_t> *node, std::vector<key_t> &out) {
        out.push_back(node->key);

        for (auto [d, bknode] : node->leaves) {
            _values(bknode, out);
        }
    }

    std::vector<key_t> values() {
        std::vector<key_t> out;

        if (root != nullptr) {
            _values(root, out);
        }
        return out;
    }

    void clear() {
        if (root != nullptr) {
            delete root;
        }
        root = nullptr;
    }

    size_t size() { return static_cast<size_t>(index); }

    virtual ~BKTree() { clear(); }
};
