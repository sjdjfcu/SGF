#include "node.h"

#include <ORUtils/Logging.h>
#include <random>
#include <algorithm>
#include <set>
using namespace PSLAM;

Node::Node(int label):
idx(label), instance_idx(0), mDebug(false), time_stamp(0),lastUpdatePropertySize(0),
mbNeedUpdateNodeFeature(false),mIdxCounter(0) {
    last_class_predicted = Unknown();
    instance_idx=idx;
    surfels.reserve(256);
    edges.reserve(16);
    neighbors.reserve(16);
    selected_surfels.reserve(256);
    centroid.setZero();
    pos_sum.setZero();
    bbox_max.setZero();
    bbox_min.setZero();
}

int Node::Add(const SurfelPtr &surfel) {
    assert(surfel->label == this->idx);
    /*Find Idx*/
    int idx_;
    {
        Lock lock(mMutNode);
        if(!mqFreeIndices.empty()) {
            idx_ = mqFreeIndices.front();
            mqFreeIndices.pop();
        } else {
            idx_ = mIdxCounter++;
        }
    }

    size_t size;
    {
        Lock lock(mMutSurfel);
        surfels.emplace(idx_, surfel);
        size = surfels.size();
    }

    /*Update Properties*/
    {
        const Eigen::Vector3f& pos = surfel->pos;
        Lock lock(mMutNode);
        pos_sum += pos;
        centroid = pos_sum / size;
        if(size == 1){
            bNeedCheckNeighbor = true;
            bbox_min = bbox_max = pos;
        } else {
            if (pos.x() > bbox_max.x()) {
                bNeedCheckNeighbor = true;
                bbox_max.x() = pos.x();
            }
            if (pos.y() > bbox_max.y()) {
                bNeedCheckNeighbor = true;
                bbox_max.y() = pos.y();
            }
            if (pos.z() > bbox_max.z()) {
                bNeedCheckNeighbor = true;
                bbox_max.z() = pos.z();
            }
            if (pos.x() < bbox_min.x()) {
                bNeedCheckNeighbor = true;
                bbox_min.x() = pos.x();
            }
            if (pos.y() < bbox_min.y()) {
                bNeedCheckNeighbor = true;
                bbox_min.y() = pos.y();
            }
            if (pos.z() < bbox_min.z()) {
                bNeedCheckNeighbor = true;
                bbox_min.z() = pos.z();
            }
        }
    }
    return idx_;
}

void Node::Remove(const int index_in_graph_old) {
    SurfelPtr surfel;
    size_t remaining_size = 0;
    {
        Lock lock_surfel(mMutSurfel);
        auto surfel_it = surfels.find(index_in_graph_old);
        if(surfel_it == surfels.end())
            throw std::runtime_error("Trying to remove surfel but it doesn't exist.");
        assert(surfel_it->second->label == this->idx);
        surfel = surfel_it->second;
        surfels.erase(surfel_it);
        remaining_size = surfels.size();
    }

    Lock lock_node(mMutNode);
    pos_sum -= surfel->pos;
    if (remaining_size == 0) {
        centroid.setZero();
    } else {
        centroid = pos_sum / static_cast<float>(remaining_size);
    }
    bool need_recal_bbox = surfel->pos==bbox_min || surfel->pos ==bbox_max;

    mqFreeIndices.emplace(index_in_graph_old);
    if(remaining_size == 0) {
        bbox_min.setZero();
        bbox_max.setZero();
        bNeedCheckNeighbor = true;
    } else if(need_recal_bbox) {
        bool initialized = false;
        Lock lock_surfel(mMutSurfel);
        for (const auto &pair : surfels) {
            const auto &surfel_ = pair.second;
            if (!surfel_) continue;
            if (!initialized) {
                bbox_max = bbox_min = surfel_->pos;
                initialized = true;
                continue;
            }
            for (size_t j = 0; j < 3; ++j) {
                if (surfel_->pos[j] > bbox_max[j]) bbox_max[j] = surfel_->pos[j];
                if (surfel_->pos[j] < bbox_min[j]) bbox_min[j] = surfel_->pos[j];
            }
        }
        if (!initialized) {
            bbox_min.setZero();
            bbox_max.setZero();
        }
        bNeedCheckNeighbor = true;
    }
}

void Node::Update(const int label,
                   const int index,
                   const Eigen::Vector3f& pos,
                   const Eigen::Vector3f& normal,
                   const std::array<unsigned char, 3>& color) {
    throw std::runtime_error("not implemented.");
}

void Node::RemoveEdge(const EdgePtr &edge){
    Lock lock(mMutEdge);
    if (edges.find(edge) == edges.end()) return;
    edges.erase(edge);
}

void Node::UpdatePrediction(const std::map<std::string, float> &pd, const std::map<std::string,std::pair<size_t, size_t>> &sizeAndEdge) {
    Lock lock(mMutPred);
    for(const auto &pair:pd) {
        auto name = pair.first;
        auto value = pair.second;

        if(mClsProb.find(name) == mClsProb.end()) {
            mClsProb[name] = value;
            if (sizeAndEdge.find(name) != sizeAndEdge.end()) {
                mSizeAndEdge[name] = sizeAndEdge.at(name);
            }
        } else {
            mClsProb[name] = value;
        }
    }
    // find the newest with maximum prob
    if(mClsProb.empty()) return;
    {
        float max_v=mClsProb.begin()->second;
        std::string max_label = GetLabel();
        for(const auto& pair:mClsProb){
            if(pair.second>=max_v){
                max_v = pair.second;
                max_label = pair.first;
            }
        }
        Lock lock_pd (mMutPDLabel);
        last_class_predicted = max_label;
    }
}

void Node::UpdateSemanticObservation(const std::map<std::string, float> &observation, float alpha) {
    if (observation.empty()) return;
    alpha = std::clamp(alpha, 0.0f, 1.0f);
    if (alpha <= 0.0f) return;

    std::map<std::string, float> normalized_observation = observation;
    float obs_sum = 0.0f;
    for (const auto &pair : normalized_observation) {
        obs_sum += std::max(0.0f, pair.second);
    }
    if (obs_sum <= 0.0f) return;
    for (auto &pair : normalized_observation) {
        pair.second = std::max(0.0f, pair.second) / obs_sum;
    }

    Lock lock(mMutPred);
    if (mClsProb.empty()) {
        mClsProb = normalized_observation;
    } else {
        std::set<std::string> labels;
        for (const auto &pair : mClsProb) labels.insert(pair.first);
        for (const auto &pair : normalized_observation) labels.insert(pair.first);

        std::map<std::string, float> updated_probs;
        for (const auto &label : labels) {
            const float old_prob = mClsProb.count(label) ? mClsProb.at(label) : 0.0f;
            const float obs_prob = normalized_observation.count(label) ? normalized_observation.at(label) : 0.0f;
            const float new_prob = (1.0f - alpha) * old_prob + alpha * obs_prob;
            if (new_prob > 1e-6f) {
                updated_probs[label] = new_prob;
            }
        }

        float prob_sum = 0.0f;
        for (const auto &pair : updated_probs) prob_sum += pair.second;
        if (prob_sum > 0.0f) {
            for (auto &pair : updated_probs) pair.second /= prob_sum;
        }
        mClsProb = std::move(updated_probs);
    }

    if (mClsProb.empty()) return;
    auto max_iter = std::max_element(
        mClsProb.begin(), mClsProb.end(),
        [](const auto &lhs, const auto &rhs) { return lhs.second < rhs.second; });
    Lock lock_pd(mMutPDLabel);
    last_class_predicted = max_iter == mClsProb.end() ? Unknown() : max_iter->first;
}

bool Node::CheckConnectivity(Node *nodeP, float margin, bool modify){
    int idx_s = this->idx;
    int idx_t = nodeP->idx;
    if(idx_s == idx_t) return true;

    bool has_out = false;
    {
        std::scoped_lock<std::mutex, std::mutex> lock_this(this->mMutNode, nodeP->mMutNode);
        for(size_t m=0;m<3;++m){
            if (bbox_min[m]-margin > nodeP->bbox_max[m]+margin) has_out = true;
            if (nodeP->bbox_min[m]-margin > bbox_max[m]+margin) has_out = true;
        }
        if(modify) this->bNeedCheckNeighbor = false;
    }
    if(!modify) return !has_out;

    std::scoped_lock<std::mutex, std::mutex> lock(mMutNN, nodeP->mMutNN);
    if (has_out) {
        if(neighbors.find(idx_t) != neighbors.end()) neighbors.erase(idx_t);
        if(nodeP->neighbors.find(idx_s) != nodeP->neighbors.end())
            nodeP->neighbors.erase(idx_s);
    } else {
        neighbors.insert(idx_t);
        nodeP->neighbors.insert(idx_s);
    }
    return !has_out;
}

int Node::GetPointSize() {
    Lock lock (mMutSurfel);
    return surfels.size();
}

void Node::UpdateSelectedNode(const size_t time, const size_t filter_size, const size_t num_pts, bool force) {
    if(mbNeedUpdateNodeFeature) return;
    const auto& last_size = lastUpdatePropertySize; // this will be updated in UpdatePrediction
    size_t new_size;
    {
        Lock lock (mMutSurfel);
        new_size = surfels.size();
    }
    static float threshold_size = 0.1;
    static float threshold_time = 60;
    const float changes = abs(float(new_size)-float(last_size))/ float(new_size);
    bool should_update = false;
    should_update |= changes >= threshold_size;
    should_update |= (time - time_stamp) > threshold_time;
    if(force) should_update = true;
    if( !should_update) return;

    Lock lock(mMutSelected);
    std::random_device rd; // obtain a random number from hardware
    std::mt19937 gen(rd()); // seed the generator

    selected_surfels.clear();
    selected_surfels.reserve(num_pts);

    if (this->surfels.size() < filter_size) {
        return;
    } else {
        size_t valid_pts = 0;
        if (!mDebug) {
            std::vector<size_t> indices(this->surfels.size(), 0);
            std::iota(indices.begin(), indices.end(), 0);
            std::shuffle(indices.begin(), indices.end(), gen);
            for (const auto &i_idx : indices) {
                if (surfels.find(i_idx) == surfels.end()) continue;
                if (!surfels.at(i_idx)->is_stable || !surfels.at(i_idx)->is_valid) continue;
                valid_pts++;
                if (selected_surfels.size() < num_pts)
                    this->selected_surfels.emplace_back(surfels.at(i_idx));
                if (valid_pts >= filter_size && selected_surfels.size() == num_pts) break;
            }
        } else {
            for (const auto &surfel:surfels) {//TODO: remove me
                if (!surfel.second->is_stable || !surfel.second->is_valid) continue;
                valid_pts++;
                if (selected_surfels.size() < num_pts)
                    this->selected_surfels.emplace_back(surfel.second);
                if (valid_pts >= filter_size && selected_surfels.size() == num_pts) break;
            }
        }
        if(valid_pts < filter_size || selected_surfels.size() != num_pts){
            selected_surfels.clear();
            return;
        }
    }
    lock.unlock();

    if (mDebug)
        for(size_t i=0;i<1;++i) {//TODO: remove me
            Eigen::Vector3f color;
            color.x() = float(selected_surfels.at(i)->color[2])/255*2.f-1;
            color.y() = float(selected_surfels.at(i)->color[1])/255*2.f-1;
            color.z() = float(selected_surfels.at(i)->color[0])/255*2.f-1;
            SCLOG(DEBUG) << selected_surfels.at(i)->pos.transpose() << " " << color.transpose() << " " << selected_surfels.at(i)->normal.transpose();
        }

    mCentroid.setZero();
    mStd.setZero();
    for(size_t i=0;i<selected_surfels.size();++i){
        const auto &surfel = selected_surfels.at(i);
        mCentroid+=surfel->pos;

        if(i==0){
            mBBox_max=mBBox_min=surfel->pos;
        } else {
            for(size_t j=0;j<3;++j) {
                if (surfel->pos[j] > mBBox_max[j]) {
                    mBBox_max[j] = surfel->pos[j];
                }
                if (surfel->pos[j] < mBBox_min[j]) {
                    mBBox_min[j] = surfel->pos[j];
                }
            }
        }
    }
    mCentroid/=selected_surfels.size();

    for(auto & selected_surfel : selected_surfels) {
        for (size_t d = 0; d < 3; ++d)
            mStd[d] += pow(selected_surfel->pos[d] - mCentroid[d], 2);
    }
    for (size_t d = 0; d < 3; ++d)
        mStd[d] = std::sqrt(mStd[d] / (selected_surfels.size() - 1.f) );

    //TODO: remove me
    if (mDebug)
    {
        float volume = 1;
        auto dim = (mBBox_max-mBBox_min)*1e-3;
        for (size_t d = 0; d < 3; ++d) volume *= dim[d];
        float length = *std::max_element(dim.begin(), dim.end());
        size_t valid_surfels = 0;
        for(const auto& surfel:surfels)
            if(surfel.second->is_valid && surfel.second->is_stable) valid_surfels++;
    }

    mbNeedUpdateNodeFeature=true;
    lastUpdatePropertySize = new_size;
    time_stamp = time;
}

const std::string Node::GetLabel() const {
    Lock lock(mMutPDLabel);
    return last_class_predicted;
}
