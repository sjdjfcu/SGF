#ifndef _H_INC_INSEG_MAP_SEGMENTS_IMPL_
#define _H_INC_INSEG_MAP_SEGMENTS_IMPL_
#include <unordered_set>
#include <unordered_map>
#include <functional>
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include "config.h"
#include <inseg_lib/surfel.h>
#include <inseg_lib/map_segments.h>
#include "node.h"
#include "edge.h"
#include <mutex>
#include <ORUtils/thread_pool.hpp>

namespace PSLAM {

struct SpatialCellKey {
    int x = 0;
    int y = 0;
    int z = 0;

    bool operator==(const SpatialCellKey &other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct SpatialCellKeyHash {
    size_t operator()(const SpatialCellKey &key) const {
        size_t seed = 0;
        const auto hash_combine = [&seed](int value) {
            seed ^= std::hash<int>{}(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        };
        hash_combine(key.x);
        hash_combine(key.y);
        hash_combine(key.z);
        return seed;
    }
};

class Graph: public inseg_lib::SegmentsInterface {
public:
    ~Graph() {}
    Graph(const ConfigPSLAM *configPslam, bool useThread);

    int Add(const SurfelPtr &surfel) override;
    /// Merge segment idx_from to idx_to
    void Merge(const int seg_idx_to, const int seg_idx_from) override;
    void Clear() override;
    void Update(const SurfelPtr surfel,
                const Eigen::Vector3f& pos,
                const Eigen::Vector3f& normal) override;
    void CheckConnectivity(float margin);
    /**
     * Update the label of a surfel
     * Remove surfel from node[surfel->label_old], chagne surfel->label to label and add it to node[label]
     * @param surfel target surfel
     * @param label new label
     * @return the index of the surfel in the graph
     */
    int UpdateLabel(const SurfelPtr surfel, const int label) override;
    int AddEdge(int from, int to, const std::shared_ptr<Edge> &edge);
    int AddEdge(const std::shared_ptr<Edge> &edge);

    void
    UpdateSelectedNodes(const std::unordered_set<int> &filtered_selected_nodes, const size_t time, const bool force);

    void Wait();

    // ===
    // thread
    // ===
    std::map<int, NodePtr> nodes;
    std::map<std::pair<int,int>, EdgePtr > edges;
    std::set<int> nodes_to_update;
    std::mutex mMutNode; // when access nodes
    std::mutex mMutEdge; // when access edges
    std::mutex mMutThis;
private:
    const ConfigPSLAM *mConfigPslam;
    void RemoveSurfelFromNode(const SurfelPtr &surfel);
    void RemoveNode(int idx);
    void RemoveEdge(const std::pair<int,int> &pair);
    void RemoveNodeFromSpatialIndex(int idx);
    void SyncSpatialEdge(const std::pair<int, int> &pair, bool connected);
    std::vector<SpatialCellKey> GetSpatialKeysForNode(const NodePtr &node, float margin) const;
    std::unordered_set<int> UpdateSpatialIndexAndCollectCandidates(int idx, const NodePtr &node, float margin);
    bool mbThread;
    float mSpatialCellSize = 0.0f;
    std::mutex mMutSpatial;
    std::unordered_map<int, std::vector<SpatialCellKey>> mNodeSpatialKeys;
    std::unordered_map<SpatialCellKey, std::unordered_set<int>, SpatialCellKeyHash> mSpatialBuckets;

    std::unique_ptr<tools::TaskThreadPool> mPools;
};

}  // namespace inseg_lib

#endif  // _H_INC_INSEG_MAP_SEGMENTS_IMPL_
