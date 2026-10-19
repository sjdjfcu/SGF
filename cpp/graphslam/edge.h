#pragma once
#ifdef COMPILE_WITH_GRAPHPRED
#include "graphPredictor/MemoryBlock.h"
#endif
#include <string>
#include <memory>
#include <map>
#include <mutex>
namespace PSLAM {
    class Edge {
    public:
        inline static std::string None() {return "none";}
        inline static std::string Same() {return "same part";}
        inline static std::string Spatial() {return "spatial";}
        Edge() = default;
        Edge(Edge &edge);

        int nodeFrom{}, nodeTo{};
        /// Shared
        std::map<std::string, float> labelProp;
        /// Thread_SG
#ifdef COMPILE_WITH_GRAPHPRED
        std::map<std::string, std::shared_ptr<MemoryBlock>> mFeatures;
#endif
        mutable std::mutex mMutex, mMutLabel;

        std::string GetLabel() const;
        void UpdatePrediction(const std::map<std::string, float> &prop);

        std::string label = None();
    };
    typedef std::shared_ptr<Edge> EdgePtr;
    static inline EdgePtr MakeEdgePtr(){return std::make_shared<Edge>();}
}
