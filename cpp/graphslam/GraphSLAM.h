//
// Created by sc on 8/21/20.
//

#ifndef GRAPHSLAM_GRAPHSLAM_H
#define GRAPHSLAM_GRAPHSLAM_H
#include "config.h"
#include "disjSet.h"
#include "graph.h"
#include "navigation_map.h"
#include "YoloSegBridge.h"
#include "../camera/CameraParameters.h"
#include <inseg_lib/lib.h>
#include <ORUtils/Logging.h>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <future>

#ifdef COMPILE_WITH_GRAPHPRED
#include "graphPredictor/GraphPredictor.h"
#else
#include <ORUtils/JsonUtil.h>
#endif
namespace PSLAM {
    class GraphSLAM {
    public:
        struct RuntimeStats {
            double inseg_ms = 0.0;
            double connectivity_ms = 0.0;
            double yolo_ms = 0.0;
            double navigation_free_ms = 0.0;
            double navigation_occupied_ms = 0.0;
        };

        explicit GraphSLAM(ConfigPSLAM *config, const CameraParameters &camParamD);
        ~GraphSLAM();

        //////////
        /// Frameprocessing
        //////////
        bool Initialize(const CameraParameters &camParamD);

        void ProcessFrame(int idx, const cv::Mat &colorImage, const cv::Mat &depthImage, const Eigen::Matrix4f *pose);

        bool &UseThread(){return mConfig->use_thread;}

        //////////
        /// Access
        //////////
        inseg_lib::InSegLib *GetInSeg() { return inseg_.get(); }

        Graph *GetGraph() { return mGraph.get(); }

        ConfigPSLAM *GetConfig(){return mConfig;}
        bool HasSemanticModel() const { return !mSemanticNameToId.empty(); }
        unsigned short GetNodeSemanticLabelId(const NodePtr &node) const;
        cv::Vec3b GetNodeSemanticColor(const NodePtr &node) const;
        const RuntimeStats &GetLastRuntimeStats() const { return mLastRuntimeStats; }
        void SetNavigationMapConfig(const NavigationMapConfig &config);
        void ResetNavigationMap();
        NavigationMap *GetNavigationMap() { return &mNavigationMap; }
        const NavigationMap *GetNavigationMap() const { return &mNavigationMap; }
        void BeginProvisionalNavigationMap();
        bool HasProvisionalNavigationMap() const { return static_cast<bool>(mProvisionalNavigationMap); }
        bool DropProvisionalNavigationMap();
        bool CommitProvisionalNavigationMap();
        bool NavigationGraphOccupancyEnabled() const { return mNavigationGraphOccupancyEnabled; }


#ifdef COMPILE_WITH_GRAPHPRED
        GraphPredictor *GetGraphPred() { return mpGraphPredictor.get(); }
#endif

#ifdef COMPILE_WITH_JSON
        json11::Json GetSceneGraph(bool full);
#endif

        //////////
        /// IO
        //////////
        void SaveModel(const std::string &output_folder) const;

        void SaveGraph(const std::string &output_folder, bool fullProb);

        void SaveSurfelsToPLY(int segment_filter, const std::string &output_folder, const std::string &output_name, bool binary);

        enum SAVECOLORMODE {
            SAVECOLORMODE_RGB, SAVECOLORMODE_SEGMENT, SAVECOLORMODE_INSTANCE, SAVECOLORMODE_SEMANTIC, SAVECOLORMODEL_PANOPTIC
        };

        void SaveNodesToPLY(int segment_filter, const std::string &output_folder, SAVECOLORMODE saveColorMode, bool binary=false);

        void SaveSurfelsToPLY(const std::string &output_folder, const std::string &output_name,
                              const std::vector<std::shared_ptr<inseg_lib::Surfel>> &surfels, bool binary);

        //////////
        /// Graph
        //////////
        void RunFullPrediction();
        /// Start backend
        void Start();
        /// Stop backend
        void Stop();

        void AddSelectedNodeToUpdate(int idx);

        bool LoadPredictModel();

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        std::set<int> mLastUpdatedSegments{};

        // Per-detection summary, matched or not, from mask pixels with valid depth
        struct DetectionSummary {
            std::string class_name;
            float confidence = 0.f;
            int bbox_x = 0, bbox_y = 0, bbox_w = 0, bbox_h = 0;  // full-resolution pixels
            int mask_area = 0;                                   // label-map pixels
            int matched_node_id = 0;
            float best_iou = 0.f;
            int n_points = 0;
            Eigen::Vector3f centroid_world_mm = Eigen::Vector3f::Zero();
            Eigen::Vector3f min_world_mm = Eigen::Vector3f::Zero();
            Eigen::Vector3f max_world_mm = Eigen::Vector3f::Zero();
            std::vector<Eigen::Vector3f> sample_points_world_mm;
        };
        const std::vector<DetectionSummary> &LastDetectionSummaries() const { return mLastDetectionSummaries; }
    private:
        std::vector<DetectionSummary> mLastDetectionSummaries;
        // system
        size_t mTimeStamp=0;
        bool mbInitMap = false;

        Eigen::Matrix4f pose_;
        std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> trajectory_{};
        CameraParameters mCamera;

        // Inseg Standard configuration.
        std::shared_ptr<inseg_lib::InSegLib> inseg_;

        // Graph
        ConfigPSLAM *mConfig;
        std::shared_ptr<Graph> mGraph;
#ifdef COMPILE_WITH_GRAPHPRED
        GraphPredictorPtr mpGraphPredictor;
#endif

        std::vector<std::shared_ptr<inseg_lib::Surfel>> GetUpdatedSurfels();

        std::vector<std::shared_ptr<inseg_lib::Surfel>> FilterSegment(
                int segment_filter, const std::vector<std::shared_ptr<inseg_lib::Surfel>> &surfels);

        void UpdateNodeSemanticsWithYolo(const cv::Mat &colorImage, const std::vector<YoloSegBridge::Detection> &detections);
        unsigned short GetSemanticLabelId(const NodePtr &node) const;
        cv::Vec3b GetSemanticColor(const NodePtr &node) const;

        std::unique_ptr<YoloSegBridge> mpYoloSegBridge;
        std::map<std::string, int> mSemanticNameToId;
        RuntimeStats mLastRuntimeStats;
        NavigationMap mNavigationMap;
        std::unique_ptr<NavigationMap> mProvisionalNavigationMap;
        bool mNavigationGraphOccupancyEnabled = true;



    };
}

#endif //GRAPHSLAM_GRAPHSLAM_H
