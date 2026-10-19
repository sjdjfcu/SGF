#include "edge.h"
using namespace PSLAM;

Edge::Edge(Edge &edge) {
    this->nodeFrom = edge.nodeFrom;
    this->nodeTo = edge.nodeTo;
    this->labelProp = edge.labelProp;
    this->label = edge.label;
#ifdef COMPILE_WITH_GRAPHPRED
    this->mFeatures = edge.mFeatures;
#endif
}

std::string Edge::GetLabel() const {
    std::unique_lock<std::mutex> lock(mMutLabel);
    return label;
}

void Edge::UpdatePrediction(const std::map<std::string, float> &prop){
    std::unique_lock<std::mutex> lock(mMutex);
    if(prop.empty()) return;

    for (const auto &pair:prop) {
        const auto &name = pair.first;
        auto new_v = pair.second;
        labelProp[name] = new_v;
    }

    if (labelProp.empty())return;
    float max_prop = labelProp.begin()->second;
    std::string ll;
    for(const auto &pair:labelProp){
        if(pair.second>=max_prop){
            max_prop = pair.second;
            ll = pair.first;
        }
    }
    std::unique_lock<std::mutex> lock_label(mMutLabel);
    label = ll;
}
