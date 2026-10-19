#include <fstream>
#include <cstdlib>
#include <unistd.h>
#include <future>
//
// Created by sc on 8/21/20.
//

#include "GraphSLAM.h"
#include <ORUtils/LogUtil.h>
#ifdef COMPILE_WITH_TINYPLY
#include <tinyply.h>
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <stdexcept>

using namespace PSLAM;

namespace {

struct YoloNodeMatch {
    int node_id = 0;
    float iou = 0.0f;
    float alpha = 0.0f;
    float confidence = 0.0f;
    int class_id = -1;
    std::string class_name;
};

inline bool IsFiniteVec3(const cv::Vec3f &value) {
    return std::isfinite(value[0]) && std::isfinite(value[1]) && std::isfinite(value[2]);
}

inline bool UseAccurateMaskMatching(const ConfigPSLAM *config) {
    return config != nullptr && config->yolo_match_mode == "accurate_mask";
}

}  // namespace

GraphSLAM::GraphSLAM(ConfigPSLAM *config, const CameraParameters &camParamD):mConfig(config){
    mGraph = std::make_shared<Graph>(config, config->use_thread);
    inseg_ = std::make_shared<inseg_lib::InSegLib>(mGraph, config->inseg_config, config->main_config, config->map_config, config->segmentation_config);
    pose_.setIdentity();
    if (mConfig->use_yolo_semantic) {
        mpYoloSegBridge = std::make_unique<YoloSegBridge>(
            mConfig->pth_yolo_model,
            mConfig->python_executable,
            mConfig->yolo_classes
        );
        if (!mpYoloSegBridge->IsReady()) {
            const bool configured_queries_required = !mConfig->yolo_classes.empty();
            mpYoloSegBridge.reset();
            mConfig->use_yolo_semantic = false;
            if (configured_queries_required) {
                throw std::runtime_error(
                    "YOLO-seg bridge failed to configure the requested class queries."
                );
            }
            SCLOG(WARNING) << "YOLO-seg bridge is unavailable. Semantic EMA updates are disabled.";
        } else {
            mSemanticNameToId = mpYoloSegBridge->GetClassNameToId();
        }
    }
    if(mConfig->graph_predict && !mConfig->use_yolo_semantic) {
#ifdef COMPILE_WITH_GRAPHPRED
        if(LoadPredictModel())
            mpGraphPredictor->RUN(mGraph.get());
        else
            mConfig->graph_predict = false;
#else
        SCLOG(WARNING) << "Did not compile with graph prediction.";
#endif
    }
    Initialize(camParamD);
};
GraphSLAM::~GraphSLAM(){
#ifdef COMPILE_WITH_GRAPHPRED
    if(mpGraphPredictor)mpGraphPredictor->Stop();
#endif
}

bool GraphSLAM::Initialize(const CameraParameters &camParamD){
    mbInitMap = false;
    mCamera = camParamD;
    ResetNavigationMap();
    Eigen::Matrix3d intrinsics;
    intrinsics(0,0) = camParamD.fx;
    intrinsics(1,1) = camParamD.fy;
    intrinsics(0,2) = camParamD.cx;
    intrinsics(1,2) = camParamD.cy;
    inseg_->InitializeCamera(intrinsics, camParamD.width, camParamD.height);
    return true;
}

void GraphSLAM::SetNavigationMapConfig(const NavigationMapConfig &config) {
    mNavigationMap.SetConfig(config);
    if (mProvisionalNavigationMap) {
        mProvisionalNavigationMap->SetConfig(config);
    }
}

void GraphSLAM::ResetNavigationMap() {
    mNavigationMap.Reset();
    mProvisionalNavigationMap.reset();
    mNavigationGraphOccupancyEnabled = true;
}

void GraphSLAM::BeginProvisionalNavigationMap() {
    auto replacement = std::make_unique<NavigationMap>(mNavigationMap.Config());
    mProvisionalNavigationMap.swap(replacement);
}

bool GraphSLAM::DropProvisionalNavigationMap() {
    const bool had_provisional_map = static_cast<bool>(mProvisionalNavigationMap);
    mProvisionalNavigationMap.reset();
    return had_provisional_map;
}

bool GraphSLAM::CommitProvisionalNavigationMap() {
    if (!mProvisionalNavigationMap) {
        return false;
    }
    std::swap(mNavigationMap, *mProvisionalNavigationMap);
    mProvisionalNavigationMap.reset();
    mNavigationGraphOccupancyEnabled = false;
    return true;
}

bool GraphSLAM::LoadPredictModel() {
#ifdef COMPILE_WITH_GRAPHPRED
    try {
        mpGraphPredictor = MakeGraphPredictor(mConfig);
        return true;
    } catch (...) {
        SCLOG(WARNING) << "Loading model from " << mConfig->pth_model << " failed.";
    }
#else
    SCLOG(WARNING) << "Did not compile with graph prediction.";
#endif
    return false;
}

void GraphSLAM::SaveModel(const std::string &output_folder) const {
    if (!inseg_) {
        printf("unable to save model. inseg is not initialized.\n");
        return ;
    }
    inseg_->map().SaveModel(output_folder+"/_inseg.map");
}

void GraphSLAM::ProcessFrame(int idx, const cv::Mat &colorImage, const cv::Mat &depthImage,const Eigen::Matrix4f *pose) {
    if(pose){
        pose_ = *pose;
        inseg_->set_pose(pose_);
    }

    mTimeStamp = idx;

    std::future<std::vector<YoloSegBridge::Detection>> yolo_future;
    const bool use_accurate_mask = UseAccurateMaskMatching(mConfig);
    if (mConfig->use_yolo_semantic && mpYoloSegBridge && mpYoloSegBridge->IsReady()) {
        float conf_t = mConfig->yolo_conf_threshold;
        float iou_t = mConfig->yolo_iou_threshold;
        cv::Mat cimg = colorImage.clone();
        yolo_future = std::async(std::launch::async, [this, cimg, conf_t, iou_t, use_accurate_mask]() {
            return mpYoloSegBridge->Infer(cimg, conf_t, iou_t, use_accurate_mask);
        });
    }
    CTICK("[SLAM][ProcessFrame]1.ProcessFrame");
    const auto inseg_begin = std::chrono::steady_clock::now();
    inseg_->ProcessFrame(depthImage, colorImage);
    mLastRuntimeStats.inseg_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - inseg_begin).count();
    CTOCK("[SLAM][ProcessFrame]1.ProcessFrame");
    pose_ = inseg_->pose();
    mLastRuntimeStats.navigation_free_ms = 0.0;
    mLastRuntimeStats.navigation_occupied_ms = 0.0;
    if (mConfig->use_navigation_map) {
        const auto navigation_free_begin = std::chrono::steady_clock::now();
        if (mProvisionalNavigationMap) {
            mProvisionalNavigationMap->UpdateFreeSpace(
                depthImage, pose_, mCamera
            );
        } else {
            mNavigationMap.UpdateFreeSpace(depthImage, pose_, mCamera);
        }
        mLastRuntimeStats.navigation_free_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - navigation_free_begin).count();
    }

    if (!mbInitMap) {
        inseg_->InitializeMap(pose_);
        mbInitMap = true;
    }

    mLastRuntimeStats.connectivity_ms = 0.0;
    if (!mGraph->nodes_to_update.empty()) {
        CTICK("[SLAM][ProcessFrame]2.ChechConnectivity");
        const auto connectivity_begin = std::chrono::steady_clock::now();
        mGraph->CheckConnectivity(mConfig->neighbor_margin);
        mLastRuntimeStats.connectivity_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - connectivity_begin).count();
        mLastUpdatedSegments = mGraph->nodes_to_update;
        CTOCK("[SLAM][ProcessFrame]2.ChechConnectivity");
    } else {
        mLastUpdatedSegments.clear();
    }

    mLastRuntimeStats.yolo_ms = 0.0;
    if (mConfig->use_yolo_semantic) {
        CTICK("[SLAM][ProcessFrame]3.YoloSemantic");
        const auto yolo_begin = std::chrono::steady_clock::now();
        std::vector<YoloSegBridge::Detection> detections;
        if (yolo_future.valid()) {
            detections = yolo_future.get();
        }
        UpdateNodeSemanticsWithYolo(colorImage, detections);
        mLastRuntimeStats.yolo_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - yolo_begin).count();
        CTOCK("[SLAM][ProcessFrame]3.YoloSemantic");
    }

    if(mConfig->graph_predict && !mConfig->use_yolo_semantic) {
        CTICK("[SLAM][ProcessFrame]4.SSCPrediction");
        AddSelectedNodeToUpdate(idx);
        CTOCK("[SLAM][ProcessFrame]4.SSCPrediction");

#ifdef COMPILE_WITH_GRAPHPRED
        if(mpGraphPredictor->Pin()){
            SCLOG(ERROR) << "prediction thread is dead.";
        }
#endif
    }

    if (
        mConfig->use_navigation_map
        && mNavigationGraphOccupancyEnabled
        && !mProvisionalNavigationMap
    ) {
        const auto navigation_occupied_begin = std::chrono::steady_clock::now();
        mNavigationMap.UpdateOccupiedFromGraph(mGraph.get());
        mLastRuntimeStats.navigation_occupied_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - navigation_occupied_begin).count();
    }

    mGraph->nodes_to_update.clear();
}

void GraphSLAM::UpdateNodeSemanticsWithYolo(const cv::Mat &colorImage, const std::vector<YoloSegBridge::Detection> &detections) {
    mLastDetectionSummaries.clear();
    if (!mpYoloSegBridge || !mpYoloSegBridge->IsReady()) return;

    const auto &frame_label_map = inseg_->GetFrameLabelMap();
    if (frame_label_map.empty() || colorImage.empty()) return;

    cv::Mat label_map;
    if (frame_label_map.type() == CV_32SC1) {
        label_map = frame_label_map;
    } else {
        frame_label_map.convertTo(label_map, CV_32SC1);
    }

    const bool use_accurate_mask = UseAccurateMaskMatching(mConfig);
    if (detections.empty()) return;
    mSemanticNameToId = mpYoloSegBridge->GetClassNameToId();

    cv::Mat visible_mask;
    cv::compare(label_map, 0, visible_mask, cv::CMP_GT);
    const bool any_visible = cv::countNonZero(visible_mask) > 0;

    // Vertex map at the label map's resolution, else the finest map rescaled
    const cv::Mat *geometry_cloud = nullptr;
    for (int level = 0; level < 4 && !geometry_cloud; ++level) {
        const cv::Mat &candidate = inseg_->image_pyramid().GetFramePointCloud(level);
        if (candidate.empty()) break;
        if (candidate.size() == label_map.size() && candidate.type() == CV_32FC3) geometry_cloud = &candidate;
    }
    const cv::Mat &finest_cloud = inseg_->image_pyramid().GetFramePointCloud(0);
    const bool finest_usable = !finest_cloud.empty() && finest_cloud.type() == CV_32FC3;
    const float geometry_sx = finest_usable ? static_cast<float>(finest_cloud.cols) / static_cast<float>(label_map.cols) : 1.f;
    const float geometry_sy = finest_usable ? static_cast<float>(finest_cloud.rows) / static_cast<float>(label_map.rows) : 1.f;
    const Eigen::Matrix4f world_from_camera = pose_.inverse();
    auto summarise_geometry = [&](const cv::Mat &full_mask, DetectionSummary &summary) {
        std::vector<Eigen::Vector3f> points;
        for (int row = 0; row < full_mask.rows; ++row) {
            const uint8_t *mask_ptr = full_mask.ptr<uint8_t>(row);
            for (int col = 0; col < full_mask.cols; ++col) {
                if (!mask_ptr[col]) continue;
                cv::Vec3f point;
                if (geometry_cloud) {
                    point = geometry_cloud->ptr<cv::Vec3f>(row)[col];
                } else if (finest_usable) {
                    const int yy = std::min(finest_cloud.rows - 1, static_cast<int>(row * geometry_sy));
                    const int xx = std::min(finest_cloud.cols - 1, static_cast<int>(col * geometry_sx));
                    point = finest_cloud.ptr<cv::Vec3f>(yy)[xx];
                } else {
                    continue;
                }
                if (!IsFiniteVec3(point) || cv::norm(point) <= 1e-6f) continue;
                const Eigen::Vector4f world = world_from_camera * Eigen::Vector4f(point[0], point[1], point[2], 1.f);
                points.emplace_back(world.x(), world.y(), world.z());
            }
        }
        summary.n_points = static_cast<int>(points.size());
        if (points.empty()) return;
        Eigen::Vector3f sum = Eigen::Vector3f::Zero();
        Eigen::Vector3f lo = points.front(), hi = points.front();
        for (const auto &pt : points) { sum += pt; lo = lo.cwiseMin(pt); hi = hi.cwiseMax(pt); }
        summary.centroid_world_mm = sum / static_cast<float>(points.size());
        summary.min_world_mm = lo; summary.max_world_mm = hi;
        const size_t max_samples = 48;
        const size_t stride = std::max<size_t>(1, points.size() / max_samples);
        for (size_t i = 0; i < points.size(); i += stride) summary.sample_points_world_mm.push_back(points[i]);
    };

    int max_node_id = 0;
    {
        std::unique_lock<std::mutex> lock(mGraph->mMutNode);
        if (!mGraph->nodes.empty()) {
            max_node_id = mGraph->nodes.rbegin()->first;
        }
    }
    std::vector<int> node_visible_area(static_cast<size_t>(max_node_id + 1), 0);
    for (int row = 0; row < label_map.rows; ++row) {
        const int *label_ptr = label_map.ptr<int>(row);
        for (int col = 0; col < label_map.cols; ++col) {
            const int node_id = label_ptr[col];
            if (node_id > 0 && node_id <= max_node_id) node_visible_area[static_cast<size_t>(node_id)] += 1;
        }
    }

    std::vector<YoloNodeMatch> matches;
    matches.reserve(detections.size());

    const auto &point_cloud = inseg_->image_pyramid().GetFramePointCloud(0);
    const auto &normal_map = inseg_->image_pyramid().GetFrameNormalMap(0);
    const bool can_score_view = use_accurate_mask &&
                                point_cloud.size() == label_map.size() && normal_map.size() == label_map.size() &&
                                point_cloud.type() == CV_32FC3 && normal_map.type() == CV_32FC3;

    for (const auto &detection : detections) {
        if (detection.class_name.empty()) continue;
        std::vector<cv::Point> overlap_pixels;
        int visible_area_in_detection = 0;
        cv::Rect roi = detection.bbox & cv::Rect(0, 0, label_map.cols, label_map.rows);
        cv::Mat label_map_roi;

        mLastDetectionSummaries.emplace_back();
        DetectionSummary &summary = mLastDetectionSummaries.back();
        summary.class_name = detection.class_name;
        summary.confidence = detection.confidence;
        summary.bbox_x = detection.bbox.x; summary.bbox_y = detection.bbox.y;
        summary.bbox_w = detection.bbox.width; summary.bbox_h = detection.bbox.height;

        if (use_accurate_mask) {
            cv::Mat mask = detection.mask;
            if (mask.empty()) continue;
            if (mask.type() != CV_8UC1) {
                mask.convertTo(mask, CV_8UC1);
            }
            if (mask.size() != label_map.size()) {
                cv::resize(mask, mask, label_map.size(), 0, 0, cv::INTER_NEAREST);
            }
            cv::threshold(mask, mask, 0, 255, cv::THRESH_BINARY);

            const int mask_area = cv::countNonZero(mask);
            summary.mask_area = mask_area;
            if (mask_area == 0) continue;
            summarise_geometry(mask, summary);
            if (!any_visible) continue;

            cv::Mat overlap_mask;
            cv::bitwise_and(mask, visible_mask, overlap_mask);
            cv::findNonZero(overlap_mask, overlap_pixels);
            if (overlap_pixels.empty()) continue;
            visible_area_in_detection = mask_area;
        } else {
            if (roi.width <= 0 || roi.height <= 0) continue;
            {
                cv::Mat rect_mask = cv::Mat::zeros(label_map.size(), CV_8UC1);
                rect_mask(roi).setTo(255);
                summary.mask_area = roi.area();
                summarise_geometry(rect_mask, summary);
            }
            if (!any_visible) continue;
            label_map_roi = label_map(roi);
            const cv::Mat visible_mask_roi = visible_mask(roi);
            cv::findNonZero(visible_mask_roi, overlap_pixels);
            if (overlap_pixels.empty()) continue;
            visible_area_in_detection = static_cast<int>(overlap_pixels.size());
        }

        std::vector<int> intersections(static_cast<size_t>(max_node_id + 1), 0);
        std::vector<int> touched_nodes;
        for (const auto &pixel : overlap_pixels) {
            const int node_id = use_accurate_mask ? label_map.ptr<int>(pixel.y)[pixel.x]
                                                  : label_map_roi.ptr<int>(pixel.y)[pixel.x];
            if (node_id <= 0 || node_id > max_node_id) continue;
            const auto index = static_cast<size_t>(node_id);
            if (intersections[index] == 0) touched_nodes.push_back(node_id);
            intersections[index] += 1;
        }

        int best_node_id = 0;
        float best_iou = 0.0f;
        for (const auto node_id : touched_nodes) {
            const auto index = static_cast<size_t>(node_id);
            const int intersection_count = intersections[index];
            const int visible_area = node_visible_area[index];
            const float intersection = static_cast<float>(intersection_count);
            const float union_area = static_cast<float>(visible_area_in_detection + visible_area - intersection_count);
            if (union_area <= 0.0f) continue;
            const float iou = intersection / union_area;
            if (iou > best_iou) {
                best_iou = iou;
                best_node_id = node_id;
            }
        }

        // Debug dump of match outcomes (env SGF_DEBUG_YOLO_MATCH=<dir>)
        {
            static const char *debug_dir = std::getenv("SGF_DEBUG_YOLO_MATCH");
            if (debug_dir && *debug_dir) {
                static std::ofstream debug_stream;
                if (!debug_stream.is_open()) {
                    debug_stream.open(std::string(debug_dir) + "/" + std::to_string(::getpid()) + ".tsv", std::ios::app);
                }
                if (debug_stream.is_open()) {
                    double sx = 0.0, sy = 0.0, sz = 0.0; int sn = 0;
                    const cv::Mat *cloud = nullptr; int cloud_level = -1;
                    for (int level = 0; level < 4 && !cloud; ++level) {
                        const cv::Mat &candidate = inseg_->image_pyramid().GetFramePointCloud(level);
                        if (candidate.empty()) break;
                        if (candidate.size() == label_map.size() && candidate.type() == CV_32FC3) { cloud = &candidate; cloud_level = level; }
                    }
                    static bool header_written = false;
                    if (!header_written) {
                        header_written = true;
                        const cv::Mat &c0 = inseg_->image_pyramid().GetFramePointCloud(0);
                        debug_stream << "#label_map " << label_map.cols << "x" << label_map.rows << " cloud0 " << c0.cols << "x" << c0.rows << " type " << c0.type() << " chosen_level " << cloud_level << '\n';
                    }
                    if (cloud) {
                        for (const auto &pixel : overlap_pixels) {
                            const cv::Vec3f &point = cloud->ptr<cv::Vec3f>(pixel.y)[pixel.x];
                            if (!IsFiniteVec3(point) || cv::norm(point) <= 1e-6f) continue;
                            sx += point[0]; sy += point[1]; sz += point[2]; ++sn;
                        }
                    } else {
                        const cv::Mat &c0 = inseg_->image_pyramid().GetFramePointCloud(0);
                        if (!c0.empty() && c0.type() == CV_32FC3) {
                            const float fx_scale = static_cast<float>(c0.cols) / static_cast<float>(label_map.cols);
                            const float fy_scale = static_cast<float>(c0.rows) / static_cast<float>(label_map.rows);
                            for (const auto &pixel : overlap_pixels) {
                                const int yy = std::min(c0.rows - 1, static_cast<int>(pixel.y * fy_scale));
                                const int xx = std::min(c0.cols - 1, static_cast<int>(pixel.x * fx_scale));
                                const cv::Vec3f &point = c0.ptr<cv::Vec3f>(yy)[xx];
                                if (!IsFiniteVec3(point) || cv::norm(point) <= 1e-6f) continue;
                                sx += point[0]; sy += point[1]; sz += point[2]; ++sn;
                            }
                        }
                    }
                    Eigen::Vector4f world(0.f, 0.f, 0.f, 1.f);
                    if (sn > 0) {
                        world = pose_.inverse() * Eigen::Vector4f(static_cast<float>(sx / sn), static_cast<float>(sy / sn), static_cast<float>(sz / sn), 1.f);
                    }
                    const int best_intersection = best_node_id > 0 ? intersections[static_cast<size_t>(best_node_id)] : 0;
                    const int best_visible = best_node_id > 0 ? node_visible_area[static_cast<size_t>(best_node_id)] : 0;
                    debug_stream << mTimeStamp << '\t' << detection.class_name << '\t' << detection.confidence << '\t'
                                 << visible_area_in_detection << '\t' << overlap_pixels.size() << '\t' << best_node_id << '\t' << best_iou << '\t'
                                 << best_intersection << '\t' << best_visible << '\t' << touched_nodes.size() << '\t'
                                 << (best_node_id != 0 && best_iou >= mConfig->yolo_match_iou_threshold ? 1 : 0) << '\t'
                                 << sn << '\t' << world.x() << '\t' << world.y() << '\t' << world.z() << '\n';
                    debug_stream.flush();
                }
            }
        }
        summary.best_iou = best_iou;
        if (best_node_id == 0 || best_iou < mConfig->yolo_match_iou_threshold) continue;
        summary.matched_node_id = best_node_id;

        const float confidence = std::clamp(detection.confidence, 0.0f, 1.0f);
        float view_factor = 1.0f;
        if (can_score_view) {
            double frontal_sum = 0.0;
            int frontal_count = 0;
            for (const auto &pixel : overlap_pixels) {
                if (label_map.ptr<int>(pixel.y)[pixel.x] != best_node_id) continue;

                const cv::Vec3f &point = point_cloud.ptr<cv::Vec3f>(pixel.y)[pixel.x];
                const cv::Vec3f &normal = normal_map.ptr<cv::Vec3f>(pixel.y)[pixel.x];
                if (!IsFiniteVec3(point) || !IsFiniteVec3(normal)) continue;

                const float point_norm = cv::norm(point);
                const float normal_norm = cv::norm(normal);
                if (point_norm <= 1e-6f || normal_norm <= 1e-6f) continue;

                const cv::Vec3f view_direction = -point / point_norm;
                const cv::Vec3f unit_normal = normal / normal_norm;
                frontal_sum += std::clamp(std::abs(unit_normal.dot(view_direction)), 0.0f, 1.0f);
                frontal_count += 1;
            }
            if (frontal_count > 0) {
                view_factor = static_cast<float>(frontal_sum / frontal_count);
            }
        }
        const float alpha = std::clamp(
            mConfig->ema_alpha_min +
                (mConfig->ema_alpha_max - mConfig->ema_alpha_min) * confidence * view_factor,
            mConfig->ema_alpha_min,
            mConfig->ema_alpha_max);

        matches.push_back({best_node_id, best_iou, alpha, confidence, detection.class_id, detection.class_name});
    }

    std::sort(matches.begin(), matches.end(), [](const auto &lhs, const auto &rhs) {
        if (lhs.iou == rhs.iou) return lhs.confidence > rhs.confidence;
        return lhs.iou > rhs.iou;
    });

    std::unordered_set<int> assigned_nodes;
    for (const auto &match : matches) {
        if (!assigned_nodes.insert(match.node_id).second) continue;
        mSemanticNameToId[match.class_name] = match.class_id;

        NodePtr node;
        {
            std::unique_lock<std::mutex> lock(mGraph->mMutNode);
            if (mGraph->nodes.find(match.node_id) == mGraph->nodes.end()) continue;
            node = mGraph->nodes.at(match.node_id);
        }
        node->UpdateSemanticObservation({{match.class_name, 1.0f}}, match.alpha);
    }
}

unsigned short GraphSLAM::GetSemanticLabelId(const NodePtr &node) const {
    if (!node) return 0;
    const auto label = node->GetLabel();
    if (label == Node::Unknown()) return 0;
    const auto iter = mSemanticNameToId.find(label);
    if (iter == mSemanticNameToId.end()) return 0;
    return static_cast<unsigned short>(iter->second + 1);
}

cv::Vec3b GraphSLAM::GetSemanticColor(const NodePtr &node) const {
    const auto semantic_id = GetSemanticLabelId(node);
    if (semantic_id == 0) return {0, 0, 0};
    return inseg_lib::CalculateLabelColor(semantic_id);
}

unsigned short GraphSLAM::GetNodeSemanticLabelId(const NodePtr &node) const {
    return GetSemanticLabelId(node);
}

cv::Vec3b GraphSLAM::GetNodeSemanticColor(const NodePtr &node) const {
    return GetSemanticColor(node);
}

void GraphSLAM::AddSelectedNodeToUpdate(int idx){
#ifdef COMPILE_WITH_ONNX
    if (!mpGraphPredictor) {
        SCLOG(WARNING) << "predictor is not initialized!";
        return;
    }
    SCLOG(VERBOSE) << "=== filter and extract graph ===";
    CTICK("[SLAM][SemanticClassification]1.DataPreparation");
    std::unordered_set<int> filtered_selected_nodes;
    for (auto node_idx : mGraph->nodes_to_update)
        if (node_idx != 0 && int(mGraph->nodes.at(node_idx)->surfels.size()) > mConfig->filter_num_node)
            filtered_selected_nodes.insert(node_idx);
    SCLOG(VERBOSE) << "Update selected surfels.";
    mGraph->UpdateSelectedNodes(filtered_selected_nodes, idx, false);
    mGraph->nodes_to_update.clear();

    if(!mConfig->use_thread) mpGraphPredictor->RUN(mGraph.get()); // if not use thread, call it every run
#else
    SCLOG(WARNING) << "Did not compile with onnx!";
#endif
}

void GraphSLAM::RunFullPrediction(){
#ifdef COMPILE_WITH_GRAPHPRED
    if (!mpGraphPredictor) {
        SCLOG(WARNING) << "predictor is not initialized!";
        return;
    }
    mpGraphPredictor->Stop();

    std::unordered_set<int> selected_nodes;
    for(const auto &pair: mGraph->nodes)
        if(pair.second->surfels.size() > (size_t)mConfig->filter_num_node)
            selected_nodes.insert(pair.first);
    mGraph->UpdateSelectedNodes(selected_nodes, 0, true);
    mGraph->Wait();
    mpGraphPredictor->RUN(mGraph.get());
#endif
}

void GraphSLAM::Start() {
#ifdef COMPILE_WITH_GRAPHPRED
    if(!mpGraphPredictor) return;
    if(!mConfig->use_thread)return;
    mpGraphPredictor->RUN(mGraph.get());
#endif
}

void GraphSLAM::Stop() {
#ifdef COMPILE_WITH_GRAPHPRED
    if(!mpGraphPredictor) return;
    if(!mConfig->use_thread)return;
    mpGraphPredictor->Stop();
#endif
}

std::vector<std::shared_ptr<inseg_lib::Surfel>> GraphSLAM::GetUpdatedSurfels(){
    const auto& surfels = inseg_->map().surfels;
    const auto& updated_segment_ids = mGraph->nodes_to_update;
    std::vector<std::shared_ptr<inseg_lib::Surfel>> filtered_surfels;
    filtered_surfels.reserve(surfels.size());
    for(const auto& surfel : surfels){
        if ((!surfel->is_valid || !surfel->is_stable) || (surfel->label == 0))
            continue;
        if(updated_segment_ids.find(surfel->label) == updated_segment_ids.end())
            continue;
        filtered_surfels.push_back(surfel);
    }
    return filtered_surfels;
}

std::vector<std::shared_ptr<inseg_lib::Surfel>> GraphSLAM::FilterSegment(
        int segment_filter,  const std::vector<std::shared_ptr<inseg_lib::Surfel>> &surfels){
    if(segment_filter<=0) return surfels;
    std::map<unsigned short, size_t> counter;
    std::set<unsigned short> outliers;
    for (const auto &surfel:surfels){
        counter[surfel->label]++;
    }
    for(auto pair:counter){
        if (int(pair.second) < segment_filter) outliers.insert(pair.first);
    }
    std::vector<std::shared_ptr<inseg_lib::Surfel>> filtered;
    filtered.reserve(surfels.size());
    for (const auto &surfel:surfels){
        counter[surfel->label]++;
    }
    for (const auto &surfel:surfels){
        auto label = surfel->label;
        if(outliers.find(label) == outliers.end()){
            filtered.push_back(surfel);
        }
    }
    return filtered;
}

void GraphSLAM::SaveSurfelsToPLY(int segment_filter, const std::string &output_name, const std::string &output_folder, bool binary) {
    auto filtered_surfels = FilterSegment(segment_filter,inseg_->map().surfels);
    SaveSurfelsToPLY(output_folder, output_name, filtered_surfels, binary);
}

template<typename T, typename T2>
void write_binary (std::ostream& os, T2 v) {
    T vv = static_cast<T>(v);
    os.write((char*)&vv,sizeof(T));
}

void GraphSLAM::SaveNodesToPLY(int segment_filter, const std::string &output_folder, SAVECOLORMODE saveColorMode, bool binary) {
    size_t counter = 0;
    for (const auto &node : mGraph->nodes) {
        if (node.first == 0 ) continue;
        if (segment_filter > 0)
            if (node.second->surfels.size() < size_t(segment_filter)) continue;
        for(const auto &s : node.second->surfels)
            if(s.second->is_valid && s.second->is_stable) counter++;
    }
    if (counter == 0) {
        printf("No valid points found. return.\n");
        return;
    }

    std::string path = output_folder + "/node";
    switch (saveColorMode) {
        case GraphSLAM::SAVECOLORMODE_RGB:
            path += "_RGB";
            break;
        case GraphSLAM::SAVECOLORMODE_SEGMENT:
            path += "_segment";
            break;
        case GraphSLAM::SAVECOLORMODE_INSTANCE:
            path += "_instance";
            break;
        case GraphSLAM::SAVECOLORMODE_SEMANTIC:
            path += "_semantic";
            break;
        case GraphSLAM::SAVECOLORMODEL_PANOPTIC:
            path += "_panoptic";
            break;
    }
    path += ".ply";
    std::fstream file;
    if(!binary)
        file.open(path, std::ios::out);
    else
        file.open(path, std::ios::out | std::ios::binary);
    assert(file.is_open());
    file << "ply\n";
    if(!binary)
        file << "format ascii 1.0\n";
    else
        file << "format binary_little_endian 1.0\n";
    file << "element vertex " + std::to_string(counter) + "\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property ushort label\n"
            "property ushort semantic\n"
            "property ushort instance\n"
            "property float nx\n"
            "property float ny\n"
            "property float nz\n"
            "property float curvature\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "property uchar alpha\n"
            "property float quality\n"
            "property float radius\n"
            "end_header\n";

    for (const auto &node: mGraph->nodes) {
        if (node.first == 0 ) continue;
        if (segment_filter > 0)
            if (node.second->surfels.size() < size_t(segment_filter)) continue;

        for (const auto &pair :node.second->surfels) {
            if(!pair.second->is_valid || !pair.second->is_stable) continue;
            const auto &surfel = pair.second;
            // pos
            if(!binary) {
                for (size_t i = 0; i < 3; ++i) file << surfel->pos[i] / 1000.0f << " ";
            } else {
                for (size_t i = 0; i < 3; ++i) {
                    float v = surfel->pos[i] / 1000.0f;
                    file.write((char*)&v,sizeof(float));
                }
            }

            // label
            if(!binary) file << node.first << " ";
            else {
                unsigned short value = node.first;
                file.write((char*)&value, sizeof(unsigned short));
            }

            // semantic
            {

                const unsigned short value = GetSemanticLabelId(node.second);
                if(!binary) file << value << " ";
                else file.write((char*)&value,sizeof(unsigned short));
            }

            // instance
            if(!binary) file << node.second->instance_idx << " ";
            else {
                unsigned short value = node.second->instance_idx;
                file.write((char*)&value,sizeof(unsigned short));
            }

            // normal
            if(!binary) {
                for (size_t i = 0; i < 3; ++i) file << surfel->normal[i] << " ";
                file << "1 ";
            } else {
                for (size_t i = 0; i < 3; ++i) file.write((char*)&surfel->normal[i],sizeof(float));
                float value = 1.f;
                file.write((char*)&value,sizeof(float));
            }

            // color
            switch (saveColorMode) {
                case SAVECOLORMODE_RGB: {
                    if(!binary)
                        for (size_t i = 0; i < 3; ++i) file << int(surfel->color[2 - i]) << " ";
                    else {
                        for (size_t i = 0; i < 3; ++i) {
                            unsigned char value = surfel->color[2 - i];
                            file.write((char*)&value,sizeof(unsigned char));
                        }
                    }
                    break;
                }
                case SAVECOLORMODE_SEGMENT: {
                    const cv::Vec3b &color = inseg_lib::CalculateLabelColor(node.first);
                    if(!binary)
                        for (size_t i = 0; i < 3; ++i) file << int(color[i]) << " ";
                    else{
                        for (size_t i = 0; i < 3; ++i) {
                            unsigned char value = color[i];
                            file.write((char*)&value,sizeof(unsigned char));
                        }
                    }
                    break;
                }
                case SAVECOLORMODE_INSTANCE: {
                    const cv::Vec3b &color = inseg_lib::CalculateLabelColor(node.second->instance_idx);
                    if(!binary)
                        for (size_t i = 0; i < 3; ++i) file << int(color[i]) << " ";
                    else {
                        for (size_t i = 0; i < 3; ++i) {
                            unsigned char value = color[i];
                            file.write((char*)&value,sizeof(unsigned char));
                        }
                    }
                    break;
                }
                case SAVECOLORMODE_SEMANTIC: {
                    const auto color = GetSemanticColor(node.second);
                    if (!binary)
                        for (size_t i = 0; i < 3; ++i) file << int(color[i]) << " ";
                    else {
                        for (size_t i = 0; i < 3; ++i) {
                            const unsigned char value = color[i];
                            file.write((char*)&value,sizeof(unsigned char));
                        }
                    }
                    break;
                }

                case SAVECOLORMODEL_PANOPTIC: {
                    const auto &name = node.second->GetLabel();
                    cv::Vec3b color;
                    if (name != Node::Unknown() && (name == "wall" || name == "floor")) {
                        color = GetSemanticColor(node.second);
                    } else {
                        auto color_ = inseg_lib::CalculateLabelColor(node.second->instance_idx.load());
                        for(auto i=0;i<3;++i)
                            color[i] = color_[i];
                    }

                    if(!binary)
                        for (size_t i = 0; i < 3; ++i) file << int(color[i]) << " ";
                    else {
                        for (size_t i = 0; i < 3; ++i) {
                            unsigned char value = color[i];
                            file.write((char*)&value,sizeof(unsigned char));
                        }
                    }
                    break;
                }

            }
            if (!binary)
                file << "255 ";
            else {
                unsigned char value = 255;
                file.write((char*)&value,sizeof(unsigned char));
            }

            // quality
            if(!binary)
                file << surfel->confidence << " ";
            else
                file.write((char*)&surfel->confidence,sizeof(float));
            // radius
            if(!binary)
                file << surfel->radius << "\n";
            else
                file.write((char*)&surfel->radius,sizeof(float));

        }

    }

    file.close();
}

void GraphSLAM::SaveSurfelsToPLY(const std::string &output_folder, const std::string &output_name, const std::vector<std::shared_ptr<inseg_lib::Surfel>> &surfels,
        bool binary) {
#ifdef WITH_TINYPLY
    struct TINYPLY_SURFEL {
        std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> pts;
        std::vector<unsigned short> labels;
        std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f>> normals;
        std::vector<std::array<unsigned char, 4>> colors;
        std::vector<float> quality;
        std::vector<float> radius;
    };
    TINYPLY_SURFEL tinysurfels;
#endif

    // calculate labeled surfels
    size_t counter = 0;
    for(const auto &surfel:surfels)
        if(surfel->label>0 && surfel->is_valid && surfel->is_stable)counter++;

    std::fstream file;
    if(!binary)
        file.open(output_folder+"/"+output_name, std::ios::out);
    else
        file.open(output_folder+"/"+output_name, std::ios::out | std::ios::binary);
    assert(file.is_open());

    file << "ply\n";
    if(!binary)
        file << "format ascii 1.0\n";
    else
        file << "format binary_little_endian 1.0\n";

    file << "element vertex " + std::to_string(counter) + "\n"
                                                          "property float x\n"
                                                          "property float y\n"
                                                          "property float z\n"
                                                          "property ushort label\n"
                                                          "property float nx\n"
                                                          "property float ny\n"
                                                          "property float nz\n"
                                                          "property float curvature\n"
                                                          "property uchar red\n"
                                                          "property uchar green\n"
                                                          "property uchar blue\n"
                                                          "property uchar alpha\n"
                                                          "property float quality\n"
                                                          "property float radius\n"
                                                          "end_header\n";


    for (const auto &surfel:surfels){
        if(surfel->label==0 || !surfel->is_valid || !surfel->is_stable) continue;
        // pos
        if(!binary)
            for(size_t i=0;i<3;++i) file << surfel->pos[i]/1000.0f << " ";
        else {
            for(size_t i=0;i<3;++i) {
                float v = surfel->pos[i] / 1000.0f;
                write_binary<float>(file, v);
            }
        }
        // label
        if(!binary) file << surfel->label << " ";
        else write_binary<unsigned short>(file,surfel->label);
        // normal
        if(!binary) {
            for (size_t i = 0; i < 3; ++i) file << surfel->normal[i] << " ";
            file << "1 ";
        } else {
            for (size_t i = 0; i < 3; ++i) write_binary<float>(file, surfel->normal[i]);
            write_binary<float>(file,1.f);
        }
        // color
        const bool segment_color = false;
        if(!binary) {
            if (segment_color) {
                const cv::Vec3b &color = inseg_lib::CalculateLabelColor(surfel->label);
                for (size_t i = 0; i < 3; ++i) file << int(color[i]) << " ";
            } else {
                for (size_t i = 0; i < 3; ++i) file << int(surfel->color[2 - i]) << " ";
            }
            file << "255 ";
        } else {
            if (segment_color) {
                const cv::Vec3b &color = inseg_lib::CalculateLabelColor(surfel->label);
                for (size_t i = 0; i < 3; ++i) write_binary<unsigned char>(file,color[i]);
            } else {
                for (size_t i = 0; i < 3; ++i) write_binary<unsigned char>(file,surfel->color[2-i]);
            }
            write_binary<unsigned char>(file,255);
        }

        // quality
        if(!binary) file << surfel->confidence << " ";
        else write_binary<float>(file,surfel->confidence);
        // radius
        if(!binary)file << surfel->radius << "\n";
        else write_binary<float>(file,surfel->radius);

#ifdef WITH_TINYPLY
        tinysurfels.pts.push_back(surfel->pos);
        tinysurfels.labels.push_back(surfel->label);
        Eigen::Vector4f n;
        n.head<3>() = surfel->normal; n.w() =1;
        tinysurfels.normals.emplace_back(n);
        std::array<unsigned char,4> c = {surfel->color[2],surfel->color[1],surfel->color[0],255};
        tinysurfels.colors.push_back(c);
        tinysurfels.quality.push_back(surfel->confidence);
        tinysurfels.radius.push_back(surfel->radius);
#endif
    }

    file.close();

#ifdef WITH_TINYPLY
    std::filebuf fb_binary;
    auto filename = output_folder+"/inseg_tinyply.ply";
    fb_binary.open(filename, std::ios::out | std::ios::binary);
    std::ostream outstream_binary(&fb_binary);
    if (outstream_binary.fail()) throw std::runtime_error("failed to open " + filename);
    tinyply::PlyFile plyFile;
    plyFile.add_properties_to_element("vertex", { "x", "y", "z" },
                                      tinyply::Type::FLOAT32, tinysurfels.pts.size(), reinterpret_cast<uint8_t*>(tinysurfels.pts.data()), tinyply::Type::INVALID, 0);

    plyFile.add_properties_to_element("vertex", { "label" },
                                      tinyply::Type::UINT16, tinysurfels.labels.size(), reinterpret_cast<uint8_t*>(tinysurfels.labels.data()), tinyply::Type::INVALID, 0);

    plyFile.add_properties_to_element("vertex", { "nx", "ny", "nz", "curvature" },
                                      tinyply::Type::FLOAT32, tinysurfels.normals.size(), reinterpret_cast<uint8_t*>(tinysurfels.normals.data()), tinyply::Type::INVALID, 0);

    plyFile.add_properties_to_element("vertex", { "red", "green", "blue", "alpha" },
                                      tinyply::Type::UINT8, tinysurfels.colors.size(), reinterpret_cast<uint8_t*>(tinysurfels.colors.data()), tinyply::Type::INVALID, 0);

    plyFile.add_properties_to_element("vertex", { "quality" },
                                      tinyply::Type::FLOAT32, tinysurfels.quality.size(), reinterpret_cast<uint8_t*>(tinysurfels.quality.data()), tinyply::Type::INVALID, 0);

    plyFile.add_properties_to_element("vertex", { "radius" },
                                      tinyply::Type::FLOAT32, tinysurfels.radius.size(), reinterpret_cast<uint8_t*>(tinysurfels.radius.data()), tinyply::Type::INVALID, 0);
    plyFile.get_comments().push_back("generated by tinyply 2.3");

    plyFile.write(outstream_binary, true);
#endif

}

#ifdef COMPILE_WITH_JSON
json11::Json GraphSLAM::GetSceneGraph(bool full){
    std::unique_lock<std::mutex> lock(mGraph->mMutNode);
    json11::Json::object nodes;
    json11::Json::object edges;
    json11::Json::object colors;
    json11::Json::object kfs;
    json11::Json::object prediction;
    if(!full) {
        for (const auto &node : mGraph->nodes) {
            if (node.second->GetLabel() == Node::Unknown()) continue;
            nodes[std::to_string(node.first)] = node.second->GetLabel();
        }

        std::map<std::string, json11::Json::array> tmp_edges;
        for (const auto &edge : mGraph->edges){
            if (edge.second->GetLabel() == Edge::Same())
                if (mGraph->nodes.at(edge.second->nodeFrom)->GetLabel() !=
                    mGraph->nodes.at(edge.second->nodeTo)->GetLabel())
                    continue;
            std::string name = std::to_string(edge.second->nodeFrom) + "_" + std::to_string(edge.second->nodeTo);
            if (tmp_edges.find(name) == tmp_edges.end()) tmp_edges[name] = json11::Json::array();
            tmp_edges.at(name).push_back(edge.second->GetLabel());
        }
        for(auto &pair:tmp_edges){
            edges[pair.first] = pair.second;
        }
    } else {
        for (const auto &node : mGraph->nodes) {
            if (node.second->GetLabel() == Node::Unknown()) continue;
            json11::Json::object probs;
            if (node.second->mClsProb.empty()) continue;
            for(const auto &pair:node.second->mClsProb) {
                probs[pair.first] = pair.second==0?0 :log(pair.second); // softmax -> logsoftmax
            }
            nodes[std::to_string(node.first)] = probs;
        }

        for(const auto &edge : mGraph->edges) {
            if (edge.second->GetLabel() == Edge::Same())
                if (mGraph->nodes.at(edge.second->nodeFrom)->GetLabel() !=
                    mGraph->nodes.at(edge.second->nodeTo)->GetLabel())
                    continue;
            std::string name = std::to_string(edge.second->nodeFrom) + "_" + std::to_string(edge.second->nodeTo);
            json11::Json::object probs;
            if (edge.second->labelProp.empty()) continue;
            for(auto &pair:edge.second->labelProp) {
                probs[pair.first] = pair.second==0? 0 : log(pair.second);// softmax -> logsoftmax
            }
            edges[name] = probs;
        }
    }
    prediction["nodes"] = nodes;
    prediction["edges"] = edges;
    prediction["kfs"] = kfs;
    return prediction;
}
#endif

void GraphSLAM::SaveGraph(const std::string &output_folder, bool fullProb) {
#ifdef COMPILE_WITH_JSON
    auto prediction = GetSceneGraph(fullProb);
    ORUtils::JsonUtils::Dump(prediction,"./prediction.json");
#else
    SCLOG(WARNING) << "Did not compile with json.";
#endif
}
