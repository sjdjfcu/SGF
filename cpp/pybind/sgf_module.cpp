#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "hot_path_bindings.h"

#include "../camera/CameraParameters.h"
#include "../graphslam/GraphSLAM.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <mutex>
#include <limits>
#include <memory>
#include <optional>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace py = pybind11;

namespace {

Eigen::Vector3f ReadVec3Locked(const PSLAM::NodePtr &node, const Eigen::Vector3f PSLAM::Node::*member) {
    std::unique_lock<std::mutex> lock(node->mMutNode);
    return node.get()->*member;
}

float ReadMaxSemanticProbability(const PSLAM::NodePtr &node) {
    std::unique_lock<std::mutex> lock(node->mMutPred);
    float best = 0.0f;
    for (const auto &pair : node->mClsProb) {
        best = std::max(best, pair.second);
    }
    return best;
}

py::dict ReadSemanticProbabilities(const PSLAM::NodePtr &node) {
    py::dict probabilities;
    std::unique_lock<std::mutex> lock(node->mMutPred);
    for (const auto &pair : node->mClsProb) {
        probabilities[py::str(pair.first)] = pair.second;
    }
    return probabilities;
}

cv::Mat ColorArrayToMat(const py::array_t<uint8_t, py::array::c_style | py::array::forcecast> &color) {
    const auto info = color.request();
    if (info.ndim != 3) {
        throw std::runtime_error("color must be a HxWxC uint8 array");
    }
    if (info.shape[2] != 3 && info.shape[2] != 4) {
        throw std::runtime_error("color must have 3 or 4 channels");
    }

    const int rows = static_cast<int>(info.shape[0]);
    const int cols = static_cast<int>(info.shape[1]);
    cv::Mat mat(rows, cols, info.shape[2] == 4 ? CV_8UC4 : CV_8UC3, info.ptr);
    if (info.shape[2] == 4) {
        cv::Mat bgr;
        cv::cvtColor(mat, bgr, cv::COLOR_BGRA2BGR);
        return bgr.clone();
    }
    return mat.clone();
}

template <typename T>
cv::Mat FloatDepthToMat(const py::buffer_info &info) {
    const int rows = static_cast<int>(info.shape[0]);
    const int cols = static_cast<int>(info.shape[1]);
    cv::Mat depth_mm(rows, cols, CV_16UC1);
    const auto *src = static_cast<const T *>(info.ptr);
    for (int row = 0; row < rows; ++row) {
        auto *dst_ptr = depth_mm.ptr<uint16_t>(row);
        for (int col = 0; col < cols; ++col) {
            const auto value = static_cast<double>(src[row * cols + col]);
            if (!std::isfinite(value) || value <= 0.0) {
                dst_ptr[col] = 0;
                continue;
            }
            const auto clamped = std::clamp(value * 1000.0, 0.0, static_cast<double>(std::numeric_limits<uint16_t>::max()));
            dst_ptr[col] = static_cast<uint16_t>(std::llround(clamped));
        }
    }
    return depth_mm;
}

cv::Mat DepthArrayToMat(const py::array &depth) {
    const auto info = depth.request();
    if (info.ndim != 2) {
        throw std::runtime_error("depth must be a HxW array");
    }

    if (info.format == py::format_descriptor<uint16_t>::format()) {
        cv::Mat depth_mm(static_cast<int>(info.shape[0]), static_cast<int>(info.shape[1]), CV_16UC1, info.ptr);
        return depth_mm.clone();
    }
    if (info.format == py::format_descriptor<float>::format()) {
        return FloatDepthToMat<float>(info);
    }
    if (info.format == py::format_descriptor<double>::format()) {
        return FloatDepthToMat<double>(info);
    }
    throw std::runtime_error("depth must use uint16, float32 or float64 dtype");
}

Eigen::Matrix4f WorldFromCameraToCameraFromWorldMm(
    const py::array_t<float, py::array::c_style | py::array::forcecast> &world_from_camera_cv
) {
    const auto info = world_from_camera_cv.request();
    if (info.ndim != 2 || info.shape[0] != 4 || info.shape[1] != 4) {
        throw std::runtime_error("world_from_camera_cv must be a 4x4 float32 matrix");
    }

    Eigen::Matrix<float, 4, 4, Eigen::RowMajor> world_from_camera =
        Eigen::Map<const Eigen::Matrix<float, 4, 4, Eigen::RowMajor>>(static_cast<const float *>(info.ptr));
    Eigen::Matrix4f camera_from_world = world_from_camera.inverse();
    camera_from_world.block<3, 1>(0, 3) *= 1000.0f;
    return camera_from_world;
}

py::tuple Vec3ToTupleMeters(const Eigen::Vector3f &vector_mm) {
    return py::make_tuple(vector_mm.x() / 1000.0f, vector_mm.y() / 1000.0f, vector_mm.z() / 1000.0f);
}

py::tuple Vec3MetersToTuple(const Eigen::Vector3f &vector_m) {
    return py::make_tuple(vector_m.x(), vector_m.y(), vector_m.z());
}

Eigen::Vector3f ArrayToVec3Meters(const py::array_t<float, py::array::c_style | py::array::forcecast> &array) {
    const auto info = array.request();
    if (info.size != 3) {
        throw std::runtime_error("expected a 3-element float vector");
    }
    const auto *ptr = static_cast<const float *>(info.ptr);
    return Eigen::Vector3f(ptr[0], ptr[1], ptr[2]);
}

py::object OptionalVec3MetersToPython(const std::optional<Eigen::Vector3f> &value) {
    if (!value.has_value()) {
        return py::none();
    }
    return Vec3MetersToTuple(*value);
}

py::array_t<uint8_t> MatToUint8Array(const cv::Mat &mat) {
    if (mat.empty()) {
        return py::array_t<uint8_t>(std::vector<py::ssize_t>{0, 0});
    }
    if (mat.type() == CV_8UC1) {
        py::array_t<uint8_t> array({static_cast<py::ssize_t>(mat.rows), static_cast<py::ssize_t>(mat.cols)});
        std::memcpy(array.mutable_data(), mat.data, static_cast<size_t>(mat.total()));
        return array;
    }
    if (mat.type() == CV_8UC3) {
        py::array_t<uint8_t> array({
            static_cast<py::ssize_t>(mat.rows),
            static_cast<py::ssize_t>(mat.cols),
            static_cast<py::ssize_t>(3),
        });
        std::memcpy(array.mutable_data(), mat.data, static_cast<size_t>(mat.total() * mat.elemSize()));
        return array;
    }
    throw std::runtime_error("expected uint8 OpenCV Mat");
}

struct SemanticNodeSnapshot {
    int id = -1;
    std::string label;
    Eigen::Vector3f centroid_m = Eigen::Vector3f::Zero();
    Eigen::Vector3f bbox_min_m = Eigen::Vector3f::Zero();
    Eigen::Vector3f bbox_max_m = Eigen::Vector3f::Zero();
    int surfel_count = 0;
    float semantic_confidence = 0.0f;
    // Class posterior fused over frames (argmax gives label)
    std::map<std::string, float> class_probs;
};

bool LabelInQuerySet(
    const std::string &label,
    const std::vector<std::string> &queries
) {
    if (label.empty()) return false;
    return std::find(queries.begin(), queries.end(), label) != queries.end();
}

// Highest posterior probability among the query phrases
float QueryProbability(
    const SemanticNodeSnapshot &node,
    const std::vector<std::string> &queries,
    std::string *best_query = nullptr
) {
    float best = 0.0f;
    for (const auto &query : queries) {
        const auto found = node.class_probs.find(query);
        if (found == node.class_probs.end()) continue;
        if (found->second > best) {
            best = found->second;
            if (best_query) *best_query = query;
        }
    }
    return best;
}

float PlanarDistanceXZ(const Eigen::Vector3f &a, const Eigen::Vector3f &b) {
    return std::hypot(a.x() - b.x(), a.z() - b.z());
}

// 3-D distance, so failed goals on different floors stay distinct
float EuclideanDistanceXYZ(const Eigen::Vector3f &a, const Eigen::Vector3f &b) {
    return (a - b).norm();
}

bool IsFiniteVec3(const Eigen::Vector3f &value) {
    return std::isfinite(value.x()) && std::isfinite(value.y()) && std::isfinite(value.z());
}

std::vector<Eigen::Vector3f> ArrayToVec3List(const py::array_t<float, py::array::c_style | py::array::forcecast> &array) {
    const auto info = array.request();
    std::vector<Eigen::Vector3f> values;
    if (info.size == 0) return values;
    if (info.ndim == 1 && info.shape[0] == 3) {
        const auto *ptr = static_cast<const float *>(info.ptr);
        values.emplace_back(ptr[0], ptr[1], ptr[2]);
        return values;
    }
    if (info.ndim != 2 || info.shape[1] != 3) {
        throw std::runtime_error("expected an Nx3 float array");
    }
    const auto *ptr = static_cast<const float *>(info.ptr);
    values.reserve(static_cast<size_t>(info.shape[0]));
    for (py::ssize_t index = 0; index < info.shape[0]; ++index) {
        values.emplace_back(ptr[index * 3 + 0], ptr[index * 3 + 1], ptr[index * 3 + 2]);
    }
    return values;
}

bool IsRepeatedFailedGoal(
    const Eigen::Vector3f &goal,
    const std::vector<Eigen::Vector3f> &failed_goals,
    const std::vector<int> &failed_goal_counts,
    float failed_goal_radius_m,
    int failed_goal_repeat_penalty_count
) {
    for (size_t index = 0; index < failed_goals.size(); ++index) {
        if (EuclideanDistanceXYZ(goal, failed_goals[index]) >= failed_goal_radius_m) continue;
        const int count = index < failed_goal_counts.size() ? failed_goal_counts[index] : 1;
        if (count >= failed_goal_repeat_penalty_count) return true;
    }
    return false;
}

float ComputeFailedGoalPenalty(
    const Eigen::Vector3f &goal,
    const std::vector<Eigen::Vector3f> &failed_goals,
    const std::vector<float> &failed_goal_penalty_scales,
    float failed_goal_radius_m
) {
    float penalty = 0.0f;
    for (size_t index = 0; index < failed_goals.size(); ++index) {
        const float distance = EuclideanDistanceXYZ(goal, failed_goals[index]);
        if (distance >= failed_goal_radius_m) continue;
        const float scale = index < failed_goal_penalty_scales.size()
            ? std::max(1.0f, failed_goal_penalty_scales[index])
            : 1.0f;
        penalty += scale * (
            1.0f
            + (failed_goal_radius_m - distance) / std::max(1e-6f, failed_goal_radius_m)
        );
    }
    return penalty;
}

py::dict EvaluateFailedGoalPolicy(
    const py::array_t<float, py::array::c_style | py::array::forcecast> &goal_m,
    const py::array_t<float, py::array::c_style | py::array::forcecast> &failed_goals_m,
    const std::vector<int> &failed_goal_counts,
    const std::vector<float> &failed_goal_penalty_scales,
    float failed_goal_radius_m,
    int failed_goal_repeat_penalty_count
) {
    const Eigen::Vector3f goal = ArrayToVec3Meters(goal_m);
    const auto failed_goals = ArrayToVec3List(failed_goals_m);
    py::dict result;
    result["repeated"] = IsRepeatedFailedGoal(
        goal,
        failed_goals,
        failed_goal_counts,
        failed_goal_radius_m,
        failed_goal_repeat_penalty_count
    );
    result["penalty"] = ComputeFailedGoalPenalty(
        goal,
        failed_goals,
        failed_goal_penalty_scales,
        failed_goal_radius_m
    );
    return result;
}

std::unordered_set<int> ToIntSet(const std::vector<int> &values) {
    return std::unordered_set<int>(values.begin(), values.end());
}

bool NearExcludedPosition(
    const Eigen::Vector3f &point,
    const std::vector<Eigen::Vector3f> &excluded_positions,
    float exclusion_radius_m
) {
    for (const auto &excluded : excluded_positions) {
        if (PlanarDistanceXZ(point, excluded) <= exclusion_radius_m) return true;
    }
    return false;
}

float DistanceFromMap(const PSLAM::NavigationMap *navigation_map, const cv::Mat &distance_map, const Eigen::Vector3f &goal_m) {
    if (!navigation_map || distance_map.empty()) return std::numeric_limits<float>::infinity();
    const auto [row, col] = navigation_map->WorldToCell(goal_m);
    if (row < 0 || row >= distance_map.rows || col < 0 || col >= distance_map.cols) {
        return std::numeric_limits<float>::infinity();
    }
    const float value = distance_map.at<float>(row, col);
    return std::isfinite(value) ? value : std::numeric_limits<float>::infinity();
}

float MaxSemanticProbability(const SemanticNodeSnapshot &node) {
    return std::max(0.0f, node.semantic_confidence);
}

py::dict NodeSnapshotToDict(const SemanticNodeSnapshot &node) {
    py::dict node_dict;
    node_dict["id"] = node.id;
    node_dict["label"] = node.label;
    node_dict["centroid_m"] = Vec3MetersToTuple(node.centroid_m);
    node_dict["bbox_min_m"] = Vec3MetersToTuple(node.bbox_min_m);
    node_dict["bbox_max_m"] = Vec3MetersToTuple(node.bbox_max_m);
    node_dict["num_surfels"] = node.surfel_count;
    node_dict["semantic_confidence"] = node.semantic_confidence;
    return node_dict;
}

float ComputeGeometryScore(const SemanticNodeSnapshot &node, float agent_y) {
    const Eigen::Vector3f extents = (node.bbox_max_m - node.bbox_min_m).cwiseAbs();
    if (!IsFiniteVec3(extents) || extents.maxCoeff() <= 1e-6f) return 0.1f;
    const float min_horizontal = std::min(extents.x(), extents.z());
    const float max_horizontal = std::max(extents.x(), extents.z());
    const float vertical = extents.y();
    const float volume = extents.x() * extents.y() * extents.z();
    const float aspect = max_horizontal / std::max(min_horizontal, 1e-6f);

    float score = 0.82f;
    if (min_horizontal < 0.03f) score -= 0.35f;
    else if (min_horizontal < 0.06f) score -= 0.18f;
    if (max_horizontal > 3.0f) score -= 0.2f;
    if (vertical < 0.04f) score -= 0.28f;
    else if (vertical < 0.10f) score -= 0.12f;
    if (vertical > 2.6f) score -= 0.12f;
    if (volume > 4.0f) score -= 0.25f;
    else if (volume > 2.0f) score -= 0.1f;
    if (aspect > 12.0f) score -= 0.28f;
    else if (aspect > 6.0f) score -= 0.12f;
    const float base_height = node.bbox_min_m.y() - agent_y;
    if (std::isfinite(base_height)) {
        if (base_height > 1.6f) score -= 0.2f;
        else if (base_height < -1.0f) score -= 0.1f;
    }
    return std::clamp(score, 0.05f, 0.98f);
}

bool InViewCone(
    const Eigen::Vector2f &viewpoint_xz,
    const Eigen::Vector2f &direction_xz,
    const Eigen::Vector2f &target_xz,
    float hfov_deg,
    float max_range_m
) {
    const Eigen::Vector2f delta = target_xz - viewpoint_xz;
    const float distance = delta.norm();
    if (distance > std::max(0.0f, max_range_m)) return false;
    const float direction_norm = direction_xz.norm();
    if (direction_norm <= 1e-6f || hfov_deg >= 359.0f) return true;
    const Eigen::Vector2f unit_delta = delta / std::max(distance, 1e-6f);
    const Eigen::Vector2f unit_direction = direction_xz / direction_norm;
    constexpr float kPi = 3.14159265358979323846f;
    const float cos_half_fov = std::cos(kPi / 180.0f * std::max(1.0f, hfov_deg) * 0.5f);
    return unit_delta.dot(unit_direction) >= cos_half_fov;
}

int CountVisibleUnknownCells(
    const PSLAM::NavigationMap *navigation_map,
    const cv::Mat &unknown_mask,
    const Eigen::Vector3f &viewpoint,
    const Eigen::Vector2f &direction_xz,
    float hfov_deg,
    float max_range_m
) {
    if (!navigation_map || unknown_mask.empty()) return 0;
    const Eigen::Vector2f viewpoint_xz(viewpoint.x(), viewpoint.z());
    int count = 0;
    for (int row = 0; row < unknown_mask.rows; ++row) {
        const auto *mask = unknown_mask.ptr<uint8_t>(row);
        for (int col = 0; col < unknown_mask.cols; ++col) {
            if (mask[col] == 0) continue;
            const Eigen::Vector3f world = navigation_map->CellToWorld(row, col, viewpoint.y());
            if (InViewCone(viewpoint_xz, direction_xz, Eigen::Vector2f(world.x(), world.z()), hfov_deg, max_range_m)) {
                ++count;
            }
        }
    }
    return count;
}

cv::Mat ArrayToU8Mask(const py::array &array, int rows, int cols) {
    const auto info = array.request();
    if (info.size == 0) {
        return cv::Mat::zeros(rows, cols, CV_8UC1);
    }
    if (info.ndim != 2 || info.shape[0] != rows || info.shape[1] != cols) {
        throw std::runtime_error("mask shape must match the navigation map");
    }
    cv::Mat mask(rows, cols, CV_8UC1, cv::Scalar(0));
    if (info.format == py::format_descriptor<bool>::format()) {
        const auto *src = static_cast<const bool *>(info.ptr);
        for (int index = 0; index < rows * cols; ++index) {
            mask.data[index] = src[index] ? 255 : 0;
        }
        return mask;
    }
    if (info.format == py::format_descriptor<uint8_t>::format()) {
        const cv::Mat src(rows, cols, CV_8UC1, info.ptr);
        src.copyTo(mask);
        return mask;
    }
    const auto casted = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>(array);
    const auto casted_info = casted.request();
    const cv::Mat src(rows, cols, CV_8UC1, casted_info.ptr);
    src.copyTo(mask);
    return mask;
}

}  // namespace

class OnlineSGF {
public:
    OnlineSGF(
        int width,
        int height,
        float fx,
        float fy,
        float cx,
        float cy,
        bool enable_semantics = true,
        const std::string &yolo_model = "",
        const std::string &python_executable = "python3",
        const std::vector<std::string> &yolo_classes = {},
        const std::string &yolo_match_mode = "fast_bbox",
        int min_pyr_level = 2,
        int segment_filter = 512,
        float depth_edge_threshold = -1.0f,
        float yolo_conf_threshold = 0.25f,
        float yolo_iou_threshold = 0.5f,
        float yolo_match_iou_threshold = 0.15f,
        float ema_alpha_min = 0.05f,
        float ema_alpha_max = 0.60f,
        float navigation_resolution_m = 0.05f,
        int navigation_depth_stride = 4,
        float navigation_max_range_m = 5.0f,
        float navigation_camera_height_m = 1.5f,
        float navigation_surfel_min_confidence = 0.30f,
        int navigation_obstacle_confirm_hits = 2,
        int navigation_free_confirm_hits = 1,
        int navigation_obstacle_free_margin = 1
    ) {
        mCamera.Set(static_cast<unsigned int>(width), static_cast<unsigned int>(height), fx, fy, cx, cy);
        mConfig = std::make_unique<PSLAM::ConfigPSLAM>();
        mConfig->graph_predict = false;
        mConfig->use_yolo_semantic = enable_semantics;
        mConfig->use_navigation_map = true;
        mConfig->pth_yolo_model = yolo_model.empty() ? mConfig->pth_yolo_model : yolo_model;
        mConfig->python_executable = python_executable;
        mConfig->yolo_classes = yolo_classes;
        mConfig->yolo_match_mode = yolo_match_mode;
        mConfig->filter_num_node = segment_filter;
        mConfig->main_config.max_pyr_level = 4;
        mConfig->main_config.min_pyr_level = min_pyr_level;
        mConfig->yolo_conf_threshold = yolo_conf_threshold;
        mConfig->yolo_iou_threshold = yolo_iou_threshold;
        mConfig->yolo_match_iou_threshold = yolo_match_iou_threshold;
        mConfig->ema_alpha_min = ema_alpha_min;
        mConfig->ema_alpha_max = ema_alpha_max;
        mNavigationConfig.resolution_m = navigation_resolution_m;
        mNavigationConfig.depth_stride = navigation_depth_stride;
        mNavigationConfig.max_range_m = navigation_max_range_m;
        mNavigationConfig.camera_height_m = navigation_camera_height_m;
        mNavigationConfig.surfel_min_confidence = navigation_surfel_min_confidence;
        mNavigationConfig.obstacle_confirm_hits = navigation_obstacle_confirm_hits;
        mNavigationConfig.free_confirm_hits = navigation_free_confirm_hits;
        mNavigationConfig.obstacle_free_margin = navigation_obstacle_free_margin;

        if (depth_edge_threshold == -1.0f) {
            if (min_pyr_level == 1) mConfig->inseg_config.depth_edge_threshold = 0.99f;
            else if (min_pyr_level == 2) mConfig->inseg_config.depth_edge_threshold = 0.99f;
            else if (min_pyr_level == 3) mConfig->inseg_config.depth_edge_threshold = 0.98f;
        } else {
            mConfig->inseg_config.depth_edge_threshold = depth_edge_threshold;
        }

        Reset();
    }

    void Reset() {
        mFrameCounter = 0;
        mGraphSlam = std::make_unique<PSLAM::GraphSLAM>(mConfig.get(), mCamera);
        mGraphSlam->SetNavigationMapConfig(mNavigationConfig);
    }

    void ProcessFrame(
        const py::array_t<uint8_t, py::array::c_style | py::array::forcecast> &color_bgr,
        const py::array &depth,
        const py::array_t<float, py::array::c_style | py::array::forcecast> &world_from_camera_cv,
        int frame_index = -1
    ) {
        if (!mGraphSlam) {
            throw std::runtime_error("GraphSLAM is not initialized");
        }

        cv::Mat color = ColorArrayToMat(color_bgr);
        cv::Mat depth_mm = DepthArrayToMat(depth);
        if (color.rows != depth_mm.rows || color.cols != depth_mm.cols) {
            throw std::runtime_error("color and depth shapes must match");
        }

        const Eigen::Matrix4f camera_from_world_mm = WorldFromCameraToCameraFromWorldMm(world_from_camera_cv);
        const int effective_frame_index = frame_index >= 0 ? frame_index : mFrameCounter;
        mGraphSlam->ProcessFrame(effective_frame_index, color, depth_mm, &camera_from_world_mm);
        mFrameCounter = effective_frame_index + 1;
    }

    py::list GetNodes(bool include_unknown = false, int min_surfels = 0) const {
        py::list nodes;
        if (!mGraphSlam) {
            return nodes;
        }

        auto *graph = mGraphSlam->GetGraph();
        std::unique_lock<std::mutex> graph_lock(graph->mMutNode);
        for (const auto &pair : graph->nodes) {
            if (pair.first == 0 || !pair.second) continue;
            const auto &node = pair.second;

            std::string label = node->GetLabel();
            if (!include_unknown && label == PSLAM::Node::Unknown()) continue;

            int surfel_count = 0;
            {
                std::unique_lock<std::mutex> surfel_lock(node->mMutSurfel);
                surfel_count = static_cast<int>(node->surfels.size());
            }
            if (surfel_count < min_surfels) continue;

            const Eigen::Vector3f centroid = ReadVec3Locked(node, &PSLAM::Node::centroid);
            const Eigen::Vector3f bbox_min = ReadVec3Locked(node, &PSLAM::Node::bbox_min);
            const Eigen::Vector3f bbox_max = ReadVec3Locked(node, &PSLAM::Node::bbox_max);
            const float semantic_confidence = ReadMaxSemanticProbability(node);

            py::dict node_dict;
            node_dict["id"] = pair.first;
            node_dict["label"] = label;
            node_dict["centroid_m"] = Vec3ToTupleMeters(centroid);
            node_dict["bbox_min_m"] = Vec3ToTupleMeters(bbox_min);
            node_dict["bbox_max_m"] = Vec3ToTupleMeters(bbox_max);
            node_dict["num_surfels"] = surfel_count;
            node_dict["semantic_confidence"] = semantic_confidence;
            node_dict["semantic_probs"] = ReadSemanticProbabilities(node);
            nodes.append(node_dict);
        }
        return nodes;
    }

    py::dict GetSurfels(float min_confidence, bool stable_only) const {
        py::dict result;
        if (!mGraphSlam) {
            result["positions_m"] = py::array_t<float>(std::vector<py::ssize_t>{0, 3});
            result["labels"] = py::array_t<int32_t>(std::vector<py::ssize_t>{0});
            result["node_ids"] = py::array_t<int32_t>(std::vector<py::ssize_t>{0});
            result["confidences"] = py::array_t<float>(std::vector<py::ssize_t>{0});
            return result;
        }

        std::vector<Eigen::Vector3f> positions;
        std::vector<int32_t> labels;
        std::vector<int32_t> node_ids;
        std::vector<float> confidences;

        auto *graph = mGraphSlam->GetGraph();
        std::unique_lock<std::mutex> graph_lock(graph->mMutNode);
        for (const auto &pair : graph->nodes) {
            if (pair.first == 0 || !pair.second) continue;
            const auto &node = pair.second;
            std::unique_lock<std::mutex> surfel_lock(node->mMutSurfel);
            for (const auto &surfel_entry : node->surfels) {
                const auto &surfel = surfel_entry.second;
                if (!surfel || !surfel->is_valid) continue;
                if (stable_only && !surfel->is_stable) continue;
                if (surfel->confidence < min_confidence) continue;
                positions.emplace_back(surfel->pos);
                labels.emplace_back(surfel->label);
                node_ids.emplace_back(static_cast<int32_t>(pair.first));
                confidences.emplace_back(surfel->confidence);
            }
        }

        const py::ssize_t n = static_cast<py::ssize_t>(positions.size());
        py::array_t<float> positions_arr({n, static_cast<py::ssize_t>(3)});
        auto positions_buf = positions_arr.mutable_unchecked<2>();
        for (py::ssize_t i = 0; i < n; ++i) {
            positions_buf(i, 0) = positions[i].x() / 1000.0f;
            positions_buf(i, 1) = positions[i].y() / 1000.0f;
            positions_buf(i, 2) = positions[i].z() / 1000.0f;
        }

        py::array_t<int32_t> labels_arr(n);
        py::array_t<int32_t> node_ids_arr(n);
        py::array_t<float> confidences_arr(n);
        std::memcpy(labels_arr.mutable_data(), labels.data(), n * sizeof(int32_t));
        std::memcpy(node_ids_arr.mutable_data(), node_ids.data(), n * sizeof(int32_t));
        std::memcpy(confidences_arr.mutable_data(), confidences.data(), n * sizeof(float));

        result["positions_m"] = positions_arr;
        result["labels"] = labels_arr;
        result["node_ids"] = node_ids_arr;
        result["confidences"] = confidences_arr;
        return result;
    }

    py::list GetLastDetections() const {
        py::list out;
        if (!mGraphSlam) return out;
        for (const auto &d : mGraphSlam->LastDetectionSummaries()) {
            py::dict item;
            item["label"] = d.class_name;
            item["confidence"] = d.confidence;
            item["bbox_xyxy"] = py::make_tuple(d.bbox_x, d.bbox_y, d.bbox_x + d.bbox_w, d.bbox_y + d.bbox_h);
            item["mask_area"] = d.mask_area;
            item["matched_node_id"] = d.matched_node_id;
            item["best_iou"] = d.best_iou;
            item["n_points"] = d.n_points;
            item["centroid_m"] = Vec3ToTupleMeters(d.centroid_world_mm);
            item["bbox_min_m"] = Vec3ToTupleMeters(d.min_world_mm);
            item["bbox_max_m"] = Vec3ToTupleMeters(d.max_world_mm);
            py::array_t<float> points(std::vector<py::ssize_t>{static_cast<py::ssize_t>(d.sample_points_world_mm.size()), 3});
            auto buf = points.mutable_unchecked<2>();
            for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(d.sample_points_world_mm.size()); ++i) {
                buf(i, 0) = d.sample_points_world_mm[i].x() / 1000.0f;
                buf(i, 1) = d.sample_points_world_mm[i].y() / 1000.0f;
                buf(i, 2) = d.sample_points_world_mm[i].z() / 1000.0f;
            }
            item["points_m"] = points;
            out.append(item);
        }
        return out;
    }

    void SetQueryConditionalLabels(bool enabled) { mQueryConditionalLabels = enabled; }
    bool QueryConditionalLabels() const { return mQueryConditionalLabels; }

    // Ranking: geodesic for navigation_map, planar for the euclidean modes
    // Only euclidean admits candidates unreachable on the navigation map
    void SetTargetAssociation(const std::string &mode) {
        if (mode == "navigation_map") {
            mEuclideanTargetAssociation = false;
        } else if (mode == "euclidean" || mode == "reachable_euclidean") {
            mEuclideanTargetAssociation = true;
        } else {
            throw std::invalid_argument(
                "target association must be 'navigation_map', 'reachable_euclidean' or 'euclidean', got '" + mode + "'");
        }
        mRequireReachableTarget = mode != "euclidean";
    }
    std::string TargetAssociation() const {
        return mEuclideanTargetAssociation
            ? (mRequireReachableTarget ? "reachable_euclidean" : "euclidean") : "navigation_map";
    }

    // Query posterior if query-conditional labels are enabled, else argmax label
    bool NodeMatchesQueries(
        const SemanticNodeSnapshot &node,
        const std::vector<std::string> &queries,
        float *probability,
        std::string *label
    ) const {
        if (mQueryConditionalLabels) {
            std::string best_query;
            const float query_probability = QueryProbability(node, queries, &best_query);
            if (query_probability <= 0.0f) return false;
            if (probability) *probability = query_probability;
            if (label) *label = best_query;
            return true;
        }
        if (!LabelInQuerySet(node.label, queries)) return false;
        if (probability) *probability = node.semantic_confidence;
        if (label) *label = node.label;
        return true;
    }

    py::dict GetSemanticLabelSummary(const std::vector<std::string> &goal_phrases) const {
        struct LabelAggregate {
            int max_surfels = 0;
            float max_confidence = 0.0f;
            bool goal_like = false;
        };

        const auto nodes = CollectSemanticNodes(false, 0);
        std::map<std::string, LabelAggregate> aggregates;
        for (const auto &node : nodes) {
            if (node.label.empty()) continue;
            auto &aggregate = aggregates[node.label];
            aggregate.max_surfels = std::max(aggregate.max_surfels, node.surfel_count);
            aggregate.max_confidence = std::max(aggregate.max_confidence, node.semantic_confidence);
            aggregate.goal_like = aggregate.goal_like || NodeMatchesQueries(node, goal_phrases, nullptr, nullptr);
        }

        py::list labels;
        for (const auto &pair : aggregates) {
            py::dict entry;
            entry["label"] = pair.first;
            entry["max_surfels"] = pair.second.max_surfels;
            entry["max_confidence"] = pair.second.max_confidence;
            entry["goal_like"] = pair.second.goal_like;
            labels.append(entry);
        }

        py::dict summary;
        summary["node_count"] = static_cast<int>(nodes.size());
        summary["labels"] = labels;
        return summary;
    }

    py::tuple SelectTargetNode(
        const py::array_t<float, py::array::c_style | py::array::forcecast> &agent_position_m,
        const std::vector<std::string> &goal_phrases,
        int min_surfels,
        float min_confidence,
        const std::vector<int> &excluded_node_ids,
        const py::array_t<float, py::array::c_style | py::array::forcecast> &excluded_positions_m,
        float exclusion_radius_m
    ) const {
        if (!mGraphSlam) {
            return py::make_tuple(py::none(), py::dict());
        }
        const Eigen::Vector3f agent_position = ArrayToVec3Meters(agent_position_m);
        const auto excluded_ids = ToIntSet(excluded_node_ids);
        const auto excluded_positions = ArrayToVec3List(excluded_positions_m);
        auto *navigation_map = mGraphSlam->GetNavigationMap();
        const cv::Mat distance_map = navigation_map->GeodesicDistanceMap(agent_position);
        const auto nodes = CollectSemanticNodes(false, 0);

        std::map<std::string, int> diagnostics{{"association_unreachable_candidates", 0}};
        py::object best_node_py = py::none();
        float best_distance = std::numeric_limits<float>::infinity();
        py::list candidates_py;

        for (const auto &node : nodes) {
            if (excluded_ids.count(node.id) != 0) continue;
            float goal_probability = 0.0f;
            std::string goal_label;
            if (!NodeMatchesQueries(node, goal_phrases, &goal_probability, &goal_label)) continue;
            diagnostics["goal_label_nodes"] += 1;
            if (node.surfel_count < min_surfels) {
                diagnostics["rejected_min_surfels"] += 1;
                continue;
            }
            if (goal_probability < min_confidence) {
                diagnostics["rejected_min_confidence"] += 1;
                continue;
            }
            const Eigen::Vector3f target_floor(node.centroid_m.x(), agent_position.y(), node.centroid_m.z());
            const auto snapped = navigation_map->SnapToFree(target_floor);
            if (!snapped.has_value() || !IsFiniteVec3(*snapped)) {
                diagnostics["rejected_snap_failed"] += 1;
                continue;
            }
            if (NearExcludedPosition(*snapped, excluded_positions, exclusion_radius_m)) {
                diagnostics["rejected_excluded_position"] += 1;
                continue;
            }
            const float distance = DistanceFromMap(navigation_map, distance_map, *snapped);
            const bool reachable = std::isfinite(distance);
            if (!reachable) {
                diagnostics["association_unreachable_candidates"] += 1;
                if (mRequireReachableTarget) {
                    diagnostics["rejected_geodesic_invalid"] += 1;
                    continue;
                }
                diagnostics["euclidean_admitted_unreachable"] += 1;
            }
            const float association_distance = mEuclideanTargetAssociation
                ? PlanarDistanceXZ(agent_position, *snapped)
                : distance;
            py::dict candidate = NodeSnapshotToDict(node);
            candidate["label"] = goal_label;
            candidate["argmax_label"] = node.label;
            candidate["semantic_confidence"] = goal_probability;
            candidate["goal_position"] = Vec3MetersToTuple(*snapped);
            candidate["distance_m"] = distance;
            candidate["association_distance_m"] = association_distance;
            candidate["association"] = TargetAssociation();
            candidates_py.append(candidate);
            if (association_distance >= best_distance) continue;
            best_distance = association_distance;
            best_node_py = candidate;
        }
        if (!best_node_py.is_none()) {
            diagnostics["selected_target_node"] += 1;
        }
        py::dict diagnostic_dict;
        for (const auto &pair : diagnostics) {
            diagnostic_dict[py::str(pair.first)] = pair.second;
        }
        return py::make_tuple(best_node_py, diagnostic_dict, candidates_py);
    }

    py::object SelectHintNode(
        const py::array_t<float, py::array::c_style | py::array::forcecast> &agent_position_m,
        const std::vector<std::string> &goal_phrases,
        int min_surfels,
        float min_confidence,
        const std::vector<int> &excluded_node_ids,
        const py::array_t<float, py::array::c_style | py::array::forcecast> &excluded_positions_m,
        float exclusion_radius_m
    ) const {
        if (!mGraphSlam) {
            return py::none();
        }
        const Eigen::Vector3f agent_position = ArrayToVec3Meters(agent_position_m);
        const auto excluded_ids = ToIntSet(excluded_node_ids);
        const auto excluded_positions = ArrayToVec3List(excluded_positions_m);
        auto *navigation_map = mGraphSlam->GetNavigationMap();
        const cv::Mat distance_map = navigation_map->GeodesicDistanceMap(agent_position);
        const auto nodes = CollectSemanticNodes(false, 0);

        py::object best_node_py = py::none();
        float best_geometry_score = -std::numeric_limits<float>::infinity();
        float best_distance = std::numeric_limits<float>::infinity();

        for (const auto &node : nodes) {
            if (excluded_ids.count(node.id) != 0) continue;
            float goal_probability = 0.0f;
            std::string goal_label;
            if (!NodeMatchesQueries(node, goal_phrases, &goal_probability, &goal_label)) continue;
            if (node.surfel_count < min_surfels) continue;
            if (goal_probability < min_confidence) continue;

            const float geometry_score = ComputeGeometryScore(node, agent_position.y());
            if (geometry_score <= 0.0f) continue;
            const Eigen::Vector3f target_floor(node.centroid_m.x(), agent_position.y(), node.centroid_m.z());
            const auto snapped = navigation_map->SnapToFree(target_floor);
            if (!snapped.has_value() || !IsFiniteVec3(*snapped)) continue;
            if (NearExcludedPosition(*snapped, excluded_positions, exclusion_radius_m)) continue;
            const float distance = DistanceFromMap(navigation_map, distance_map, *snapped);
            if (!std::isfinite(distance)) continue;

            if (geometry_score < best_geometry_score) continue;
            if (geometry_score == best_geometry_score && distance >= best_distance) continue;
            best_geometry_score = geometry_score;
            best_distance = distance;
            py::dict candidate = NodeSnapshotToDict(node);
            candidate["label"] = goal_label;
            candidate["argmax_label"] = node.label;
            candidate["semantic_confidence"] = goal_probability;
            candidate["goal_position"] = Vec3MetersToTuple(*snapped);
            candidate["distance_m"] = distance;
            candidate["geometry_score"] = std::round(geometry_score * 10000.0f) / 10000.0f;
            candidate["hint_source"] = "target_query";
            best_node_py = std::move(candidate);
        }
        return best_node_py;
    }

    py::dict SnapToFreeMany(
        const py::array_t<float, py::array::c_style | py::array::forcecast> &xyzs_m,
        float max_radius_m = 2.0f
    ) const {
        if (!mGraphSlam) {
            throw std::runtime_error("GraphSLAM is not initialized");
        }
        const auto points = ArrayToVec3List(xyzs_m);
        py::array_t<float> snapped_array({static_cast<py::ssize_t>(points.size()), static_cast<py::ssize_t>(3)});
        py::array_t<uint8_t> valid_array(static_cast<py::ssize_t>(points.size()));
        auto snapped = snapped_array.mutable_unchecked<2>();
        auto valid = valid_array.mutable_unchecked<1>();
        auto *navigation_map = mGraphSlam->GetNavigationMap();
        for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(points.size()); ++i) {
            const auto result = navigation_map->SnapToFree(points[static_cast<size_t>(i)], max_radius_m);
            valid(i) = result.has_value() ? 1 : 0;
            const Eigen::Vector3f value = result.value_or(Eigen::Vector3f(
                std::numeric_limits<float>::quiet_NaN(),
                std::numeric_limits<float>::quiet_NaN(),
                std::numeric_limits<float>::quiet_NaN()
            ));
            snapped(i, 0) = value.x();
            snapped(i, 1) = value.y();
            snapped(i, 2) = value.z();
        }
        py::dict result;
        result["points_m"] = snapped_array;
        result["valid"] = valid_array;
        return result;
    }

    py::array_t<float> NavigationGeodesicDistances(
        const py::array_t<float, py::array::c_style | py::array::forcecast> &start_m,
        const py::array_t<float, py::array::c_style | py::array::forcecast> &goals_m
    ) const {
        if (!mGraphSlam) {
            throw std::runtime_error("GraphSLAM is not initialized");
        }
        const Eigen::Vector3f start = ArrayToVec3Meters(start_m);
        const auto goals = ArrayToVec3List(goals_m);
        const auto distances = mGraphSlam->GetNavigationMap()->GeodesicDistancesFrom(start, goals);
        py::array_t<float> array(static_cast<py::ssize_t>(distances.size()));
        std::memcpy(array.mutable_data(), distances.data(), distances.size() * sizeof(float));
        return array;
    }

    py::object NavigationUtilityFrontierGoal(
        const py::array_t<float, py::array::c_style | py::array::forcecast> &position_m,
        const py::array &visited_mask,
        const py::array_t<float, py::array::c_style | py::array::forcecast> &failed_goals_m,
        const std::vector<int> &failed_goal_counts,
        const std::vector<float> &failed_goal_penalty_scales,
        const std::vector<std::string> &goal_phrases,
        int visited_radius_cells,
        float min_goal_distance_m,
        int min_frontier_size,
        float failed_goal_radius_m,
        int failed_goal_repeat_penalty_count,
        float failed_goal_repeat_penalty_scale,
        float hfov_deg,
        float max_range_m,
        float info_gain_weight,
        float goal_likelihood_weight,
        float cost_weight,
        float revisit_weight,
        const std::vector<int> &excluded_node_ids
    ) const {
        if (!mGraphSlam) {
            return py::none();
        }
        auto *navigation_map = mGraphSlam->GetNavigationMap();
        const Eigen::Vector3f position = ArrayToVec3Meters(position_m);
        const auto failed_goals = ArrayToVec3List(failed_goals_m);
        const cv::Mat frontier = navigation_map->FrontierCells();
        if (cv::countNonZero(frontier) == 0) return py::none();
        const cv::Mat unknown = navigation_map->UnknownMask();
        const cv::Mat distance_map = navigation_map->GeodesicDistanceMap(position);
        const cv::Mat visited = ArrayToU8Mask(visited_mask, frontier.rows, frontier.cols);
        std::vector<SemanticNodeSnapshot> semantic_nodes = (goal_likelihood_weight != 0.0f && !goal_phrases.empty())
            ? CollectSemanticNodes(false, 0)
            : std::vector<SemanticNodeSnapshot>{};
        // Excluded nodes do not contribute to frontier goal likelihood
        if (!excluded_node_ids.empty() && !semantic_nodes.empty()) {
            const auto excluded_ids = ToIntSet(excluded_node_ids);
            semantic_nodes.erase(
                std::remove_if(
                    semantic_nodes.begin(),
                    semantic_nodes.end(),
                    [&excluded_ids](const SemanticNodeSnapshot &node) {
                        return excluded_ids.count(node.id) != 0;
                    }
                ),
                semantic_nodes.end()
            );
        }

        cv::Mat labels, stats, centroids;
        const int label_count = cv::connectedComponentsWithStats(frontier, labels, stats, centroids, 8, CV_32S);
        struct FrontierCandidate {
            Eigen::Vector3f goal;
            float utility = -std::numeric_limits<float>::infinity();
            float information_gain = 0.0f;
            float goal_likelihood = 0.0f;
            float cost = 0.0f;
            float revisit_penalty = 0.0f;
            int area = 0;
        };
        const auto better_frontier = [](const FrontierCandidate &lhs, const FrontierCandidate &rhs) {
            if (lhs.utility != rhs.utility) return lhs.utility > rhs.utility;
            if (lhs.information_gain != rhs.information_gain) return lhs.information_gain > rhs.information_gain;
            if (lhs.goal_likelihood != rhs.goal_likelihood) return lhs.goal_likelihood > rhs.goal_likelihood;
            if (lhs.area != rhs.area) return lhs.area > rhs.area;
            return lhs.cost < rhs.cost;
        };
        std::vector<FrontierCandidate> candidates;
        candidates.reserve(std::max(0, label_count - 1));
        std::optional<FrontierCandidate> best;
        int best_index = -1;
        for (int label = 1; label < label_count; ++label) {
            const int area = stats.at<int>(label, cv::CC_STAT_AREA);
            if (area < min_frontier_size) continue;
            const double centroid_col = centroids.at<double>(label, 0);
            const double centroid_row = centroids.at<double>(label, 1);
            const Eigen::Vector3f candidate_world = navigation_map->CellToWorld(
                static_cast<int>(std::lround(centroid_row)),
                static_cast<int>(std::lround(centroid_col)),
                position.y()
            );
            const auto snapped = navigation_map->SnapToFree(candidate_world);
            if (!snapped.has_value() || !IsFiniteVec3(*snapped)) continue;
            if (PlanarDistanceXZ(*snapped, position) < min_goal_distance_m) continue;
            if (IsRepeatedFailedGoal(
                    *snapped,
                    failed_goals,
                    failed_goal_counts,
                    failed_goal_radius_m,
                    failed_goal_repeat_penalty_count
                )) {
                continue;
            }
            const float cost = DistanceFromMap(navigation_map, distance_map, *snapped);
            if (!std::isfinite(cost)) continue;

            const Eigen::Vector2f direction_xz(snapped->x() - position.x(), snapped->z() - position.z());
            const float information_gain = static_cast<float>(
                CountVisibleUnknownCells(navigation_map, unknown, *snapped, direction_xz, hfov_deg, max_range_m)
            );
            const float goal_likelihood = ComputeSparseGoalLikelihood(
                navigation_map,
                semantic_nodes,
                *snapped,
                direction_xz,
                hfov_deg,
                max_range_m,
                goal_phrases,
                mQueryConditionalLabels
            );
            const float revisit_penalty = ComputeRevisitPenalty(
                navigation_map,
                visited,
                *snapped,
                failed_goals,
                failed_goal_penalty_scales,
                visited_radius_cells,
                failed_goal_radius_m
            );
            const float utility = info_gain_weight * information_gain
                + goal_likelihood_weight * goal_likelihood
                - cost_weight * cost
                - revisit_weight * revisit_penalty;
            FrontierCandidate candidate{*snapped, utility, information_gain, goal_likelihood, cost, revisit_penalty, area};
            candidates.push_back(candidate);
            if (!best.has_value() || better_frontier(candidate, *best)) {
                best = candidate;
                best_index = static_cast<int>(candidates.size()) - 1;
            }
        }
        if (!best.has_value()) return py::none();

        const auto candidate_dict = [](const FrontierCandidate &candidate) {
            py::dict item;
            item["goal"] = Vec3MetersToTuple(candidate.goal);
            item["utility"] = candidate.utility;
            item["information_gain"] = candidate.information_gain;
            item["goal_likelihood"] = candidate.goal_likelihood;
            item["cost"] = candidate.cost;
            item["revisit_penalty"] = candidate.revisit_penalty;
            item["area"] = candidate.area;
            return item;
        };
        py::dict result;
        result["goal"] = Vec3MetersToTuple(best->goal);
        result["utility"] = best->utility;
        result["information_gain"] = best->information_gain;
        result["goal_likelihood"] = best->goal_likelihood;
        result["cost"] = best->cost;
        result["revisit_penalty"] = best->revisit_penalty;
        result["area"] = best->area;
        py::list candidate_results;
        for (const auto &candidate : candidates) {
            candidate_results.append(candidate_dict(candidate));
        }
        result["candidates"] = candidate_results;
        result["best_index"] = best_index;
        return result;
    }

    py::object SelectShortTermGoal(
        const py::array_t<float, py::array::c_style | py::array::forcecast> &agent_position_m,
        const py::array_t<float, py::array::c_style | py::array::forcecast> &forward_xz,
        int agent_row,
        int agent_col,
        const py::array_t<float, py::array::c_style | py::array::forcecast> &fmm_dist,
        const py::array &traversible_mask,
        int step_cells
    ) const {
        if (!mGraphSlam) {
            return py::none();
        }
        const auto dist_info = fmm_dist.request();
        if (dist_info.ndim != 2) {
            throw std::runtime_error("fmm_dist must be a 2D float array");
        }
        const int rows = static_cast<int>(dist_info.shape[0]);
        const int cols = static_cast<int>(dist_info.shape[1]);
        if (agent_row < 0 || agent_row >= rows || agent_col < 0 || agent_col >= cols) {
            return py::none();
        }
        const auto *dist_ptr = static_cast<const float *>(dist_info.ptr);
        const cv::Mat traversible = ArrayToU8Mask(traversible_mask, rows, cols);
        if (traversible.at<uint8_t>(agent_row, agent_col) == 0) {
            return py::none();
        }
        const float current_distance = dist_ptr[agent_row * cols + agent_col];
        if (!std::isfinite(current_distance)) {
            return py::none();
        }
        const Eigen::Vector3f agent_position = ArrayToVec3Meters(agent_position_m);
        const auto forward_info = forward_xz.request();
        if (forward_info.size < 2) {
            throw std::runtime_error("forward_xz must contain at least two floats");
        }
        const auto *forward_ptr = static_cast<const float *>(forward_info.ptr);
        Eigen::Vector2f forward(forward_ptr[0], forward_ptr[1]);
        const float forward_norm = forward.norm();
        if (forward_norm <= 1e-6f) {
            forward = Eigen::Vector2f(0.0f, -1.0f);
        } else {
            forward /= forward_norm;
        }

        const int safe_step_cells = std::max(1, step_cells);
        const int max_radius_cells = std::max(safe_step_cells + 1, safe_step_cells * 3);
        bool have_best = false;
        float best_score = -std::numeric_limits<float>::infinity();
        float best_drop = -std::numeric_limits<float>::infinity();
        float best_neg_abs_heading = -std::numeric_limits<float>::infinity();
        float best_heading = 0.0f;
        Eigen::Vector3f best_waypoint = Eigen::Vector3f::Zero();

        for (int delta_row = -max_radius_cells; delta_row <= max_radius_cells; ++delta_row) {
            for (int delta_col = -max_radius_cells; delta_col <= max_radius_cells; ++delta_col) {
                if (delta_row == 0 && delta_col == 0) continue;
                const float offset_distance = std::hypot(static_cast<float>(delta_row), static_cast<float>(delta_col));
                if (offset_distance > static_cast<float>(max_radius_cells) + 0.25f || offset_distance < 1.0f) continue;
                const int row = agent_row + delta_row;
                const int col = agent_col + delta_col;
                if (row < 0 || row >= rows || col < 0 || col >= cols) continue;
                if (traversible.at<uint8_t>(row, col) == 0) continue;
                const float candidate_distance = dist_ptr[row * cols + col];
                if (!std::isfinite(candidate_distance)) continue;
                const float distance_drop = current_distance - candidate_distance;
                if (distance_drop <= 1e-4f) continue;
                if (!LineIsTraversible(traversible, agent_row, agent_col, row, col)) continue;
                const Eigen::Vector3f waypoint = mGraphSlam->GetNavigationMap()->CellToWorld(row, col, agent_position.y());
                const Eigen::Vector2f target_vector(waypoint.x() - agent_position.x(), waypoint.z() - agent_position.z());
                const float target_norm = target_vector.norm();
                float heading_error_deg = 0.0f;
                if (target_norm > 1e-6f) {
                    const Eigen::Vector2f target = target_vector / target_norm;
                    const float cross = forward.x() * target.y() - forward.y() * target.x();
                    const float dot = std::clamp(forward.dot(target), -1.0f, 1.0f);
                    heading_error_deg = std::atan2(cross, dot) * 180.0f / 3.14159265358979323846f;
                }
                const float step_size_penalty = std::abs(offset_distance - static_cast<float>(safe_step_cells)) * 0.03f;
                const float heading_penalty = std::abs(heading_error_deg) / 180.0f * std::max(1.0f, static_cast<float>(safe_step_cells) * 0.25f);
                const float progress_rate = distance_drop / std::max(1.0f, offset_distance);
                const float score = distance_drop + 0.25f * progress_rate - heading_penalty - step_size_penalty;
                const float neg_abs_heading = -std::abs(heading_error_deg);
                const bool better = !have_best
                    || score > best_score
                    || (score == best_score && distance_drop > best_drop)
                    || (score == best_score && distance_drop == best_drop && neg_abs_heading > best_neg_abs_heading);
                if (!better) continue;
                have_best = true;
                best_score = score;
                best_drop = distance_drop;
                best_neg_abs_heading = neg_abs_heading;
                best_heading = heading_error_deg;
                best_waypoint = waypoint;
            }
        }
        if (!have_best) {
            return py::none();
        }
        py::dict result;
        result["heading_error_deg"] = best_heading;
        result["waypoint"] = Vec3MetersToTuple(best_waypoint);
        result["score"] = best_score;
        result["distance_drop"] = best_drop;
        return result;
    }

    py::dict GetRuntimeStats() const {
        py::dict stats;
        if (!mGraphSlam) {
            stats["inseg_ms"] = 0.0;
            stats["connectivity_ms"] = 0.0;
            stats["yolo_ms"] = 0.0;
            stats["navigation_free_ms"] = 0.0;
            stats["navigation_occupied_ms"] = 0.0;
            return stats;
        }

        const auto &runtime_stats = mGraphSlam->GetLastRuntimeStats();
        stats["inseg_ms"] = runtime_stats.inseg_ms;
        stats["connectivity_ms"] = runtime_stats.connectivity_ms;
        stats["yolo_ms"] = runtime_stats.yolo_ms;
        stats["navigation_free_ms"] = runtime_stats.navigation_free_ms;
        stats["navigation_occupied_ms"] = runtime_stats.navigation_occupied_ms;
        return stats;
    }

    void ResetNavigationMap() {
        if (!mGraphSlam) {
            throw std::runtime_error("GraphSLAM is not initialized");
        }
        mGraphSlam->ResetNavigationMap();
    }

    void BeginProvisionalNavigationMap() {
        if (!mGraphSlam) {
            throw std::runtime_error("GraphSLAM is not initialized");
        }
        mGraphSlam->BeginProvisionalNavigationMap();
    }

    bool HasProvisionalNavigationMap() const {
        return mGraphSlam && mGraphSlam->HasProvisionalNavigationMap();
    }

    bool DropProvisionalNavigationMap() {
        if (!mGraphSlam) {
            throw std::runtime_error("GraphSLAM is not initialized");
        }
        return mGraphSlam->DropProvisionalNavigationMap();
    }

    bool CommitProvisionalNavigationMap() {
        if (!mGraphSlam) {
            throw std::runtime_error("GraphSLAM is not initialized");
        }
        return mGraphSlam->CommitProvisionalNavigationMap();
    }

    bool NavigationGraphOccupancyEnabled() const {
        return mGraphSlam && mGraphSlam->NavigationGraphOccupancyEnabled();
    }

    py::dict NavigationGraphOccupancyStats() const {
        PSLAM::NavigationGraphOccupancyStats stats;
        if (mGraphSlam) stats = mGraphSlam->GetNavigationMap()->GraphOccupancyStats();
        py::dict result;
        result["surfel_votes"] = stats.surfel_votes;
        result["removed_free_cells"] = stats.removed_free_cells;
        result["added_occupied_cells"] = stats.added_occupied_cells;
        return result;
    }

    void SetNavigationFloorSupport(bool enabled) {
        mNavigationConfig.floor_support_enabled = enabled;
        if (mGraphSlam) {
            // GraphSLAM applies this config to both the active and provisional maps
            mGraphSlam->SetNavigationMapConfig(mNavigationConfig);
        }
    }

    bool NavigationFloorSupportEnabled() const {
        return mNavigationConfig.floor_support_enabled;
    }

    py::dict NavigationFloorSupportStats() const {
        PSLAM::NavigationFloorSupportStats stats;
        stats.enabled = mNavigationConfig.floor_support_enabled;
        if (mGraphSlam) {
            stats = mGraphSlam->GetNavigationMap()->FloorSupportStats();
        }
        py::dict result;
        result["enabled"] = stats.enabled;
        result["nonfloor_cells"] = stats.nonfloor_cells;
        result["vetoed_free_cells"] = stats.vetoed_free_cells;
        result["potential_vetoed_free_cells"] = stats.potential_vetoed_free_cells;
        return result;
    }

    py::tuple NavigationShape() const {
        if (!mGraphSlam) return py::make_tuple(0, 0);
        const auto shape = mGraphSlam->GetNavigationMap()->Shape();
        return py::make_tuple(shape.first, shape.second);
    }

    float NavigationResolutionMeters() const {
        if (!mGraphSlam) return 0.0f;
        return mGraphSlam->GetNavigationMap()->ResolutionMeters();
    }

    py::tuple NavigationOriginXZ() const {
        if (!mGraphSlam) return py::make_tuple(0.0f, 0.0f);
        const auto origin = mGraphSlam->GetNavigationMap()->OriginXZ();
        return py::make_tuple(origin.x(), origin.y());
    }

    py::tuple NavigationWorldToCell(const py::array_t<float, py::array::c_style | py::array::forcecast> &xyz_m) const {
        if (!mGraphSlam) {
            throw std::runtime_error("GraphSLAM is not initialized");
        }
        const auto cell = mGraphSlam->GetNavigationMap()->WorldToCell(ArrayToVec3Meters(xyz_m));
        return py::make_tuple(cell.first, cell.second);
    }

    py::tuple NavigationCellToWorld(int row, int col, float y_m) const {
        if (!mGraphSlam) {
            throw std::runtime_error("GraphSLAM is not initialized");
        }
        return Vec3MetersToTuple(mGraphSlam->GetNavigationMap()->CellToWorld(row, col, y_m));
    }

    py::array_t<uint8_t> NavigationObservedMask() const {
        if (!mGraphSlam) return py::array_t<uint8_t>(std::vector<py::ssize_t>{0, 0});
        return MatToUint8Array(mGraphSlam->GetNavigationMap()->ObservedMask());
    }

    py::array_t<uint8_t> NavigationFreeMask() const {
        if (!mGraphSlam) return py::array_t<uint8_t>(std::vector<py::ssize_t>{0, 0});
        return MatToUint8Array(mGraphSlam->GetNavigationMap()->FreeMask());
    }

    py::array_t<uint8_t> NavigationOccupiedMask() const {
        if (!mGraphSlam) return py::array_t<uint8_t>(std::vector<py::ssize_t>{0, 0});
        return MatToUint8Array(mGraphSlam->GetNavigationMap()->OccupiedMask());
    }

    py::array_t<uint8_t> NavigationFrontierCells() const {
        if (!mGraphSlam) return py::array_t<uint8_t>(std::vector<py::ssize_t>{0, 0});
        return MatToUint8Array(mGraphSlam->GetNavigationMap()->FrontierCells());
    }

    py::array_t<uint8_t> RenderNavigationMap() const {
        if (!mGraphSlam) return py::array_t<uint8_t>(std::vector<py::ssize_t>{0, 0, 3});
        return MatToUint8Array(mGraphSlam->GetNavigationMap()->Render());
    }

    py::object SnapToFree(
        const py::array_t<float, py::array::c_style | py::array::forcecast> &xyz_m,
        float max_radius_m = 2.0f
    ) const {
        if (!mGraphSlam) {
            throw std::runtime_error("GraphSLAM is not initialized");
        }
        return OptionalVec3MetersToPython(mGraphSlam->GetNavigationMap()->SnapToFree(ArrayToVec3Meters(xyz_m), max_radius_m));
    }

    float NavigationGeodesicDistance(
        const py::array_t<float, py::array::c_style | py::array::forcecast> &start_m,
        const py::array_t<float, py::array::c_style | py::array::forcecast> &goal_m
    ) const {
        if (!mGraphSlam) {
            throw std::runtime_error("GraphSLAM is not initialized");
        }
        return mGraphSlam->GetNavigationMap()->GeodesicDistance(ArrayToVec3Meters(start_m), ArrayToVec3Meters(goal_m));
    }

    py::object NavigationFrontierGoal(
        const py::array_t<float, py::array::c_style | py::array::forcecast> &position_m,
        float min_goal_distance_m = 1.5f,
        int min_frontier_size = 20,
        bool prefer_far = false
    ) const {
        if (!mGraphSlam) {
            throw std::runtime_error("GraphSLAM is not initialized");
        }
        return OptionalVec3MetersToPython(
            mGraphSlam->GetNavigationMap()->FrontierGoal(
                ArrayToVec3Meters(position_m),
                min_goal_distance_m,
                min_frontier_size,
                prefer_far
            )
        );
    }

private:
    std::vector<SemanticNodeSnapshot> CollectSemanticNodes(bool include_unknown, int min_surfels) const {
        std::vector<SemanticNodeSnapshot> snapshots;
        if (!mGraphSlam) return snapshots;

        auto *graph = mGraphSlam->GetGraph();
        std::unique_lock<std::mutex> graph_lock(graph->mMutNode);
        snapshots.reserve(graph->nodes.size());
        for (const auto &pair : graph->nodes) {
            if (pair.first == 0 || !pair.second) continue;
            const auto &node = pair.second;
            std::string label = node->GetLabel();
            if (!include_unknown && label == PSLAM::Node::Unknown()) continue;

            int surfel_count = 0;
            {
                std::unique_lock<std::mutex> surfel_lock(node->mMutSurfel);
                surfel_count = static_cast<int>(node->surfels.size());
            }
            if (surfel_count < min_surfels) continue;

            SemanticNodeSnapshot snapshot;
            snapshot.id = pair.first;
            snapshot.label = std::move(label);
            snapshot.centroid_m = ReadVec3Locked(node, &PSLAM::Node::centroid) * 0.001f;
            snapshot.bbox_min_m = ReadVec3Locked(node, &PSLAM::Node::bbox_min) * 0.001f;
            snapshot.bbox_max_m = ReadVec3Locked(node, &PSLAM::Node::bbox_max) * 0.001f;
            snapshot.surfel_count = surfel_count;
            snapshot.semantic_confidence = ReadMaxSemanticProbability(node);
            {
                std::unique_lock<std::mutex> pred_lock(node->mMutPred);
                snapshot.class_probs = node->mClsProb;
            }
            snapshots.push_back(snapshot);
        }
        return snapshots;
    }

    static float ComputeSparseGoalLikelihood(
        const PSLAM::NavigationMap *navigation_map,
        const std::vector<SemanticNodeSnapshot> &nodes,
        const Eigen::Vector3f &viewpoint,
        const Eigen::Vector2f &direction_xz,
        float hfov_deg,
        float max_range_m,
        const std::vector<std::string> &goal_phrases,
        bool query_conditional_labels
    ) {
        if (!navigation_map || nodes.empty() || goal_phrases.empty()) return 0.0f;
        const Eigen::Vector2f viewpoint_xz(viewpoint.x(), viewpoint.z());
        const auto shape = navigation_map->Shape();
        float score = 0.0f;
        for (const auto &node : nodes) {
            if (!IsFiniteVec3(node.centroid_m)) continue;
            const auto [row, col] = navigation_map->WorldToCell(node.centroid_m);
            if (row < 0 || row >= shape.first || col < 0 || col >= shape.second) continue;
            float cue_probability = 0.0f;
            if (query_conditional_labels) {
                cue_probability = QueryProbability(node, goal_phrases);
                if (cue_probability <= 0.0f) continue;
            } else {
                if (!LabelInQuerySet(node.label, goal_phrases)) continue;
                cue_probability = MaxSemanticProbability(node);
            }
            const Eigen::Vector2f node_xz(node.centroid_m.x(), node.centroid_m.z());
            if (!InViewCone(viewpoint_xz, direction_xz, node_xz, hfov_deg, max_range_m)) continue;
            score += cue_probability;
        }
        return score;
    }

    static float ComputeRevisitPenalty(
        const PSLAM::NavigationMap *navigation_map,
        const cv::Mat &visited,
        const Eigen::Vector3f &goal,
        const std::vector<Eigen::Vector3f> &failed_goals,
        const std::vector<float> &failed_goal_penalty_scales,
        int visited_radius_cells,
        float failed_goal_radius_m
    ) {
        float penalty = 0.0f;
        if (navigation_map && !visited.empty()) {
            const auto [row, col] = navigation_map->WorldToCell(goal);
            const int radius = std::max(1, visited_radius_cells);
            const int row_min = std::max(0, row - radius);
            const int row_max = std::min(visited.rows, row + radius + 1);
            const int col_min = std::max(0, col - radius);
            const int col_max = std::min(visited.cols, col + radius + 1);
            int visited_count = 0;
            int total_count = 0;
            for (int r = row_min; r < row_max; ++r) {
                const auto *ptr = visited.ptr<uint8_t>(r);
                for (int c = col_min; c < col_max; ++c) {
                    ++total_count;
                    if (ptr[c] != 0) ++visited_count;
                }
            }
            if (total_count > 0) {
                penalty += static_cast<float>(visited_count) / static_cast<float>(total_count);
            }
        }
        penalty += ComputeFailedGoalPenalty(
            goal,
            failed_goals,
            failed_goal_penalty_scales,
            failed_goal_radius_m
        );
        return penalty;
    }

    static bool LineIsTraversible(
        const cv::Mat &traversible,
        int start_row,
        int start_col,
        int end_row,
        int end_col
    ) {
        cv::LineIterator iterator(
            traversible,
            cv::Point(start_col, start_row),
            cv::Point(end_col, end_row),
            8
        );
        for (int index = 0; index < iterator.count; ++index, ++iterator) {
            const cv::Point point = iterator.pos();
            if (point.y < 0 || point.y >= traversible.rows || point.x < 0 || point.x >= traversible.cols) {
                return false;
            }
            if (traversible.at<uint8_t>(point.y, point.x) == 0) {
                return false;
            }
        }
        return true;
    }

    PSLAM::CameraParameters mCamera;
    std::unique_ptr<PSLAM::ConfigPSLAM> mConfig;
    std::unique_ptr<PSLAM::GraphSLAM> mGraphSlam;
    bool mQueryConditionalLabels = false;
    bool mEuclideanTargetAssociation = false;
    bool mRequireReachableTarget = true;
    PSLAM::NavigationMapConfig mNavigationConfig;
    int mFrameCounter = 0;
};

PYBIND11_MODULE(_sgf, module) {
    module.doc() = "Minimal online SGF bindings for Habitat text-goal navigation";

    module.def(
        "_failed_goal_policy_metrics",
        &EvaluateFailedGoalPolicy,
        py::arg("goal_m"),
        py::arg("failed_goals_m"),
        py::arg("failed_goal_counts"),
        py::arg("failed_goal_penalty_scales"),
        py::arg("failed_goal_radius_m"),
        py::arg("failed_goal_repeat_penalty_count")
    );

    py::class_<OnlineSGF>(module, "OnlineSGF")
        .def(
            py::init<
                int,
                int,
                float,
                float,
                float,
                float,
                bool,
                const std::string &,
                const std::string &,
                const std::vector<std::string> &,
                const std::string &,
                int,
                int,
                float,
                float,
                float,
                float,
                float,
                float,
                float,
                int,
                float,
                float,
                float,
                int,
                int,
                int
            >(),
            py::arg("width"),
            py::arg("height"),
            py::arg("fx"),
            py::arg("fy"),
            py::arg("cx"),
            py::arg("cy"),
            py::arg("enable_semantics") = true,
            py::arg("yolo_model") = "",
            py::arg("python_executable") = "python3",
            py::arg("yolo_classes") = std::vector<std::string>{},
            py::arg("yolo_match_mode") = "fast_bbox",
            py::arg("min_pyr_level") = 2,
            py::arg("segment_filter") = 512,
            py::arg("depth_edge_threshold") = -1.0f,
            py::arg("yolo_conf_threshold") = 0.25f,
            py::arg("yolo_iou_threshold") = 0.5f,
            py::arg("yolo_match_iou_threshold") = 0.15f,
            py::arg("ema_alpha_min") = 0.05f,
            py::arg("ema_alpha_max") = 0.60f,
            py::arg("navigation_resolution_m") = 0.05f,
            py::arg("navigation_depth_stride") = 4,
            py::arg("navigation_max_range_m") = 5.0f,
            py::arg("navigation_camera_height_m") = 1.5f,
            py::arg("navigation_surfel_min_confidence") = 0.30f,
            py::arg("navigation_obstacle_confirm_hits") = 2,
            py::arg("navigation_free_confirm_hits") = 1,
            py::arg("navigation_obstacle_free_margin") = 1
        )
        .def("reset", &OnlineSGF::Reset)
        .def("reset_navigation_map", &OnlineSGF::ResetNavigationMap)
        .def("begin_provisional_navigation_map", &OnlineSGF::BeginProvisionalNavigationMap)
        .def("has_provisional_navigation_map", &OnlineSGF::HasProvisionalNavigationMap)
        .def("drop_provisional_navigation_map", &OnlineSGF::DropProvisionalNavigationMap)
        .def("commit_provisional_navigation_map", &OnlineSGF::CommitProvisionalNavigationMap)
        .def("navigation_graph_occupancy_enabled", &OnlineSGF::NavigationGraphOccupancyEnabled)
        .def("navigation_graph_occupancy_stats", &OnlineSGF::NavigationGraphOccupancyStats)
        .def("set_navigation_floor_support", &OnlineSGF::SetNavigationFloorSupport, py::arg("enabled"))
        .def("navigation_floor_support_enabled", &OnlineSGF::NavigationFloorSupportEnabled)
        .def("navigation_floor_support_stats", &OnlineSGF::NavigationFloorSupportStats)
        .def(
            "process_frame",
            &OnlineSGF::ProcessFrame,
            py::arg("color_bgr"),
            py::arg("depth"),
            py::arg("world_from_camera_cv"),
            py::arg("frame_index") = -1
        )
        .def(
            "get_nodes",
            &OnlineSGF::GetNodes,
            py::arg("include_unknown") = false,
            py::arg("min_surfels") = 0
        )
        .def(
            "get_surfels",
            &OnlineSGF::GetSurfels,
            py::arg("min_confidence") = 0.0f,
            py::arg("stable_only") = true
        )
        .def(
            "get_semantic_label_summary",
            &OnlineSGF::GetSemanticLabelSummary,
            py::arg("goal_phrases")
        )
        .def("get_last_detections", &OnlineSGF::GetLastDetections)
        .def(
            "set_query_conditional_labels",
            &OnlineSGF::SetQueryConditionalLabels,
            py::arg("enabled")
        )
        .def(
            "set_target_association",
            &OnlineSGF::SetTargetAssociation,
            py::arg("mode")
        )
        .def("target_association", &OnlineSGF::TargetAssociation)
        .def(
            "select_target_node",
            &OnlineSGF::SelectTargetNode,
            py::arg("agent_position_m"),
            py::arg("goal_phrases"),
            py::arg("min_surfels"),
            py::arg("min_confidence"),
            py::arg("excluded_node_ids"),
            py::arg("excluded_positions_m"),
            py::arg("exclusion_radius_m")
        )
        .def(
            "select_hint_node",
            &OnlineSGF::SelectHintNode,
            py::arg("agent_position_m"),
            py::arg("goal_phrases"),
            py::arg("min_surfels"),
            py::arg("min_confidence"),
            py::arg("excluded_node_ids"),
            py::arg("excluded_positions_m"),
            py::arg("exclusion_radius_m")
        )
        .def("get_runtime_stats", &OnlineSGF::GetRuntimeStats)
        .def("navigation_shape", &OnlineSGF::NavigationShape)
        .def("navigation_resolution_m", &OnlineSGF::NavigationResolutionMeters)
        .def("navigation_origin_xz", &OnlineSGF::NavigationOriginXZ)
        .def("navigation_world_to_cell", &OnlineSGF::NavigationWorldToCell, py::arg("xyz_m"))
        .def("navigation_cell_to_world", &OnlineSGF::NavigationCellToWorld, py::arg("row"), py::arg("col"), py::arg("y_m"))
        .def("navigation_observed_mask", &OnlineSGF::NavigationObservedMask)
        .def("navigation_free_mask", &OnlineSGF::NavigationFreeMask)
        .def("navigation_occupied_mask", &OnlineSGF::NavigationOccupiedMask)
        .def("navigation_frontier_cells", &OnlineSGF::NavigationFrontierCells)
        .def("render_navigation_map", &OnlineSGF::RenderNavigationMap)
        .def("snap_to_free", &OnlineSGF::SnapToFree, py::arg("xyz_m"), py::arg("max_radius_m") = 2.0f)
        .def("snap_to_free_many", &OnlineSGF::SnapToFreeMany, py::arg("xyzs_m"), py::arg("max_radius_m") = 2.0f)
        .def(
            "navigation_geodesic_distance",
            &OnlineSGF::NavigationGeodesicDistance,
            py::arg("start_m"),
            py::arg("goal_m")
        )
        .def(
            "navigation_geodesic_distances",
            &OnlineSGF::NavigationGeodesicDistances,
            py::arg("start_m"),
            py::arg("goals_m")
        )
        .def(
            "navigation_utility_frontier_goal",
            &OnlineSGF::NavigationUtilityFrontierGoal,
            py::arg("position_m"),
            py::arg("visited_mask"),
            py::arg("failed_goals_m"),
            py::arg("failed_goal_counts"),
            py::arg("failed_goal_penalty_scales"),
            py::arg("goal_phrases"),
            py::arg("visited_radius_cells"),
            py::arg("min_goal_distance_m"),
            py::arg("min_frontier_size"),
            py::arg("failed_goal_radius_m"),
            py::arg("failed_goal_repeat_penalty_count"),
            py::arg("failed_goal_repeat_penalty_scale"),
            py::arg("hfov_deg"),
            py::arg("max_range_m"),
            py::arg("info_gain_weight"),
            py::arg("goal_likelihood_weight"),
            py::arg("cost_weight"),
            py::arg("revisit_weight"),
            py::arg("excluded_node_ids") = std::vector<int>{}
        )
        .def(
            "select_short_term_goal",
            &OnlineSGF::SelectShortTermGoal,
            py::arg("agent_position_m"),
            py::arg("forward_xz"),
            py::arg("agent_row"),
            py::arg("agent_col"),
            py::arg("fmm_dist"),
            py::arg("traversible_mask"),
            py::arg("step_cells")
        )
        .def(
            "navigation_frontier_goal",
            &OnlineSGF::NavigationFrontierGoal,
            py::arg("position_m"),
            py::arg("min_goal_distance_m") = 1.5f,
            py::arg("min_frontier_size") = 20,
            py::arg("prefer_far") = false
        );

    sgf_pybind::RegisterHotPathBindings(module);
}
