#include "inseg.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <future>
#include <immintrin.h>
#include <limits>

#include <Eigen/Dense>
#include <opencv2/core/core.hpp>

namespace inseg_lib {

namespace {

#ifndef SGF_ENABLE_ASYNC_COMPACT
#define SGF_ENABLE_ASYNC_COMPACT 1
#endif

#ifndef SGF_ENABLE_ASYNC_INDEX_MAP
#define SGF_ENABLE_ASYNC_INDEX_MAP 1
#endif

inline int RoundPixelCoordinate(const float value) {
    return static_cast<int>(value + 0.5f);
}

inline void ResetMatBytes(cv::Mat* mat, const int byte_value) {
    if (mat->empty()) return;
    std::memset(mat->data, byte_value, mat->rows * static_cast<size_t>(mat->step));
}

inline float TransformPointComponent(const float c0,
                                     const float c1,
                                     const float c2,
                                     const float x,
                                     const float y,
                                     const float z,
                                     const float offset) {
    const float yz = (c1 * y) + (c2 * z);
    const float xyz = (c0 * x) + yz;
    return xyz + offset;
}

inline float RotateVectorComponent(const float c0,
                                   const float c1,
                                   const float c2,
                                   const float x,
                                   const float y,
                                   const float z) {
    return (c0 * x) + ((c1 * y) + (c2 * z));
}

inline float DotProduct3(const float ax,
                         const float ay,
                         const float az,
                         const float bx,
                         const float by,
                         const float bz) {
    return (ax * bx) + ((ay * by) + (az * bz));
}

// Accumulate the Y/Z terms before adding the X term.
inline float DotProduct3XLast(const float ax,
                              const float ay,
                              const float az,
                              const float bx,
                              const float by,
                              const float bz) {
    return ((ay * by) + (az * bz)) + (ax * bx);
}

// Keep normalization operations in single precision.
inline float ScalarSqrt(const float value) {
    const __m128 lane = _mm_set_ss(value);
    return _mm_cvtss_f32(_mm_sqrt_ss(lane));
}

inline float ScalarDivide(const float numerator, const float denominator) {
    const __m128 num_lane = _mm_set_ss(numerator);
    const __m128 den_lane = _mm_set_ss(denominator);
    return _mm_cvtss_f32(_mm_div_ss(num_lane, den_lane));
}

inline bool IsFiniteVec3(const cv::Vec3f& value) {
    return std::isfinite(value[0]) && std::isfinite(value[1]) && std::isfinite(value[2]);
}

inline int ReadLabel(const cv::Mat& label_map, const int y, const int x) {
    if (label_map.empty()) return 0;
    switch (label_map.type()) {
        case CV_16UC1:
            return static_cast<int>(label_map.ptr<uint16_t>(y)[x]);
        case CV_16SC1:
            return static_cast<int>(label_map.ptr<int16_t>(y)[x]);
        case CV_32SC1:
            return label_map.ptr<int>(y)[x];
        default:
            return 0;
    }
}

inline std::shared_ptr<Surfel> CreateSurfel(const float pos_x,
                                            const float pos_y,
                                            const float pos_z,
                                            const float normal_x,
                                            const float normal_y,
                                            const float normal_z,
                                            const float radius,
                                            const float confidence,
                                            const int label) {
    auto surfel = std::make_shared<Surfel>();
    surfel->confidence = confidence;
    surfel->pos.x() = pos_x;
    surfel->pos.y() = pos_y;
    surfel->pos.z() = pos_z;
    surfel->normal.x() = normal_x;
    surfel->normal.y() = normal_y;
    surfel->normal.z() = normal_z;
    surfel->radius = radius;
    surfel->label = label;
    return surfel;
}

inline std::vector<float>& MakeIndexDepthBuffer() {
    thread_local std::vector<float> buffer;
    return buffer;
}

inline void AppendSurfel(Map* map, std::shared_ptr<Surfel>&& surfel) {
    map->surfels.emplace_back(std::move(surfel));
    auto& stored_surfel = map->surfels.back();
    stored_surfel->index_in_graph = map->segments_->Add(stored_surfel);
}

inline void EnsureSurfelCapacityForFrame(std::vector<std::shared_ptr<Surfel>>* surfels,
                                         const int frame_rows,
                                         const int frame_cols) {
    if (!surfels || frame_rows <= 0 || frame_cols <= 0) {
        return;
    }
    const size_t frame_pixels = static_cast<size_t>(frame_rows) * static_cast<size_t>(frame_cols);
    const size_t growth_hint = std::max<size_t>(frame_pixels / 4, 16384);
    const size_t required = surfels->size() + growth_hint;
    if (required <= surfels->capacity()) {
        return;
    }

    size_t new_capacity = surfels->capacity();
    if (new_capacity == 0) {
        new_capacity = required;
    }
    while (new_capacity < required) {
        new_capacity += std::max<size_t>(new_capacity / 2, 65536);
    }
    surfels->reserve(new_capacity);
}

inline int ReadLabelFromRow(const uint16_t* const row_u16,
                            const int16_t* const row_s16,
                            const int* const row_s32,
                            const int x) {
    if (row_s32) {
        return row_s32[x];
    }
    if (row_u16) {
        return static_cast<int>(row_u16[x]);
    }
    if (row_s16) {
        return static_cast<int>(row_s16[x]);
    }
    return 0;
}

inline float PointToPlaneDistance(const cv::Vec3f& neighbor,
                                  const float point_x,
                                  const float point_y,
                                  const float point_z,
                                  const float normal_x,
                                  const float normal_y,
                                  const float normal_z) {
    if (!IsFiniteVec3(neighbor)) {
        return std::numeric_limits<float>::infinity();
    }
    const float diff_x = neighbor[0] - point_x;
    const float diff_y = neighbor[1] - point_y;
    const float diff_z = neighbor[2] - point_z;
    return std::fabs(
        DotProduct3(diff_x, diff_y, diff_z, normal_x, normal_y, normal_z));
}

inline void WriteLabel(cv::Mat* label_map,
                       const int y,
                       const int x,
                       const int value) {
    if (!label_map || label_map->empty()) return;
    switch (label_map->type()) {
        case CV_16UC1:
            label_map->ptr<uint16_t>(y)[x] = static_cast<uint16_t>(value);
            break;
        case CV_16SC1:
            label_map->ptr<int16_t>(y)[x] = static_cast<int16_t>(value);
            break;
        case CV_32SC1:
            label_map->ptr<int>(y)[x] = value;
            break;
        default:
            break;
    }
}

struct InversePose {
    float inv00;
    float inv01;
    float inv02;
    float itx;
    float inv10;
    float inv11;
    float inv12;
    float ity;
    float inv20;
    float inv21;
    float inv22;
    float itz;
};

inline InversePose ComputeInversePose(const Eigen::Matrix4f& pose) {
    const Eigen::Matrix4f inverse = pose.inverse();
    return {
        inverse(0, 0), inverse(0, 1), inverse(0, 2), inverse(0, 3),
        inverse(1, 0), inverse(1, 1), inverse(1, 2), inverse(1, 3),
        inverse(2, 0), inverse(2, 1), inverse(2, 2), inverse(2, 3),
    };
}

}  // namespace

void InSegReconstruction::SetPropagatedLabelMap() {
    label_map_propagated_frame_.copyTo(image_pyramid_.GetFrameLabelMapNotConst());
}

void InSegReconstruction::SetLabelMap() {
    label_map_frame_.copyTo(image_pyramid_.GetFrameLabelMapNotConst());
}

void InSegReconstruction::Label() {
    const int level = main_config.min_pyr_level;
    const cv::Mat& frame_point_cloud = image_pyramid_.GetFramePointCloud(level);
    const cv::Mat& frame_normal_map = image_pyramid_.GetFrameNormalMap(level);
    label_num_ = segmentation_->Segment(frame_point_cloud, frame_normal_map, &label_map_frame_);
    SetLabelMap();
}

void InSegReconstruction::CalculateLabelSizeList(const cv::Mat* label_map,
                                                 const int label_num,
                                                 std::vector<int16_t>* label_size_list) {
    if (!label_size_list || label_num <= 0) {
        return;
    }
    label_size_list->assign(label_num, 0);
    if (!label_map || label_map->empty()) {
        return;
    }
    for (int y = 0; y < label_map->rows; ++y) {
        for (int x = 0; x < label_map->cols; ++x) {
            const int label = ReadLabel(*label_map, y, x);
            if (label >= 0 && label < label_num) {
                ++(*label_size_list)[label];
            }
        }
    }
}

void InSegReconstruction::PropagateLabels() {
    if (map_.surfels.empty()) {
        return;
    }

    const int level = main_config.min_pyr_level;
    cv::Mat& frame_label_map = image_pyramid_.GetFrameLabelMapNotConst();
    const cv::Mat& frame_point_cloud = image_pyramid_.GetFramePointCloud(level);
    const cv::Mat& frame_normal_map = image_pyramid_.GetFrameNormalMap(level);
    const cv::Mat& model_point_cloud = image_pyramid_.GetModelPointCloud(level);
    const cv::Mat& model_normal_map = image_pyramid_.GetModelNormalMap(level);
    if (frame_label_map.empty() || frame_point_cloud.empty() || frame_normal_map.empty() ||
        model_point_cloud.empty() || model_normal_map.empty()) {
        return;
    }

    const float depth_uncertainty =
        static_cast<float>(main_config.uncertainty_coefficient_depth);
    constexpr float kPi = 3.1415927f;
    const float observation_dot_threshold =
        std::cos((config.update_point_angle_threshold / 180.0f) * kPi);
    const float normal_match_threshold =
        std::cos((config.label_point_dot_product_threshold / 180.0f) * kPi);

    const int old_world_label_num = world_label_num_;
    std::vector<int> overlap_counts;
    if (old_world_label_num > 0 && label_num_ > 0) {
        overlap_counts.assign(static_cast<size_t>(old_world_label_num) *
                                  static_cast<size_t>(label_num_),
                              0);
    }

    label_size_list_.assign(label_num_, 0);

    for (int y = 0; y < frame_point_cloud.rows; ++y) {
        const cv::Vec3f* const frame_point_row = frame_point_cloud.ptr<cv::Vec3f>(y);
        const cv::Vec3f* const frame_normal_row = frame_normal_map.ptr<cv::Vec3f>(y);
        const cv::Vec3f* const model_point_row = model_point_cloud.ptr<cv::Vec3f>(y);
        const cv::Vec3f* const model_normal_row = model_normal_map.ptr<cv::Vec3f>(y);
        uint16_t* const world_label_row = world_label_map_.ptr<uint16_t>(y);

        for (int x = 0; x < frame_point_cloud.cols; ++x) {
            const cv::Vec3f& frame_point = frame_point_row[x];
            const float point_x = frame_point[0];
            const float point_y = frame_point[1];
            const float point_z = frame_point[2];
            if (point_z <= 0.0f) {
                world_label_row[x] = 0;
                continue;
            }

            const cv::Vec3f& frame_normal = frame_normal_row[x];
            const float normal_x = frame_normal[0];
            const float normal_y = frame_normal[1];
            const float normal_z = frame_normal[2];
            if (normal_z == 0.0f) {
                world_label_row[x] = 0;
                continue;
            }

            const float ray_norm_sq =
                (point_x * point_x) + ((point_y * point_y) + (point_z * point_z));
            float neg_x = -point_x;
            float neg_y = -point_y;
            float neg_z = -point_z;
            if (ray_norm_sq > 0.0f) {
                const float ray_norm = ScalarSqrt(ray_norm_sq);
                neg_x = ScalarDivide(neg_x, ray_norm);
                neg_y = ScalarDivide(neg_y, ray_norm);
                neg_z = ScalarDivide(neg_z, ray_norm);
            }

            const float observation_dot =
                ((normal_z * neg_z) + (normal_y * neg_y)) + (normal_x * neg_x);
            if (observation_dot > observation_dot_threshold) {
                const int frame_label = ReadLabel(frame_label_map, y, x);
                if (frame_label != 0 && point_z > 0.0f) {
                    const cv::Vec3f& model_normal = model_normal_row[x];
                    if (model_normal[2] != 0.0f) {
                        const cv::Vec3f& model_point = model_point_row[x];
                        const float dx = model_point[0] - point_x;
                        const float dy = model_point[1] - point_y;
                        const float dz = model_point[2] - point_z;
                        const float distance_along_ray = std::fabs(
                            DotProduct3XLast(dx, dy, dz, neg_x, neg_y, neg_z));
                        if (distance_along_ray <=
                            (point_z * (depth_uncertainty * point_z))) {
                            const float normal_dot =
                                ((frame_normal[1] * model_normal[1]) +
                                 (model_normal[2] * frame_normal[2])) +
                                (frame_normal[0] * model_normal[0]);
                            if (normal_dot >= normal_match_threshold) {
                                if (frame_label >= 0 &&
                                    frame_label < static_cast<int>(label_size_list_.size())) {
                                    ++label_size_list_[frame_label];
                                    const int world_label = static_cast<int>(world_label_row[x]);
                                    if (world_label > 0 && world_label < old_world_label_num &&
                                        !overlap_counts.empty()) {
                                        ++overlap_counts[world_label * label_num_ + frame_label];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if (label_num_ <= 0) {
        return;
    }

    std::vector<int> label_mapping(label_num_, 0);
    int new_label_count = 0;
    for (int frame_label = 1; frame_label < label_num_; ++frame_label) {
        if (label_size_list_[frame_label] <= 0) {
            label_mapping[frame_label] = 0;
            continue;
        }

        std::vector<int> candidate_world_labels;
        float best_overlap_ratio = 0.0f;
        int best_world_label = 0;
        for (int world_label = 1; world_label < old_world_label_num; ++world_label) {
            if (overlap_counts.empty()) {
                break;
            }
            const int count = overlap_counts[world_label * label_num_ + frame_label];
            if (count == 0) {
                continue;
            }

            const float overlap_ratio =
                static_cast<float>(count) /
                static_cast<float>(label_size_list_[frame_label]);
            if (overlap_ratio > best_overlap_ratio) {
                best_overlap_ratio = overlap_ratio;
                best_world_label = world_label;
            }

            if (overlap_ratio > config.label_region_ratio &&
                label_size_list_[frame_label] > config.label_size_threshold) {
                candidate_world_labels.push_back(world_label);
            }
        }

        if (candidate_world_labels.size() > 1) {
            for (size_t i = 0; i < candidate_world_labels.size(); ++i) {
                auto& connections = label_connect_confidence_list_[candidate_world_labels[i]];
                for (size_t j = i + 1; j < candidate_world_labels.size(); ++j) {
                    const auto [it, inserted] =
                        connections.try_emplace(candidate_world_labels[j], 0);
                    if (!inserted) {
                        it->second += 2;
                    }
                }
            }
        }

        if (best_overlap_ratio <= config.label_merge_overlap_ratio) {
            if (label_size_list_[frame_label] <= config.label_min_size) {
                label_mapping[frame_label] = 0;
            } else {
                label_mapping[frame_label] = old_world_label_num + new_label_count;
                ++new_label_count;
            }
        } else {
            label_mapping[frame_label] = best_world_label;
        }
    }

    world_label_num_ += new_label_count + 1;

    for (int y = 0; y < frame_label_map.rows; ++y) {
        for (int x = 0; x < frame_label_map.cols; ++x) {
            const int label = ReadLabel(frame_label_map, y, x);
            const int mapped =
                (label >= 0 && label < static_cast<int>(label_mapping.size()))
                    ? label_mapping[label]
                    : 0;
            WriteLabel(&frame_label_map, y, x, mapped);
        }
    }

    const int decay_threshold = config.label_merge_confidence_threshold + 1;
    for (auto& [label, connections] : label_connect_confidence_list_) {
        (void)label;
        for (auto& [other_label, confidence] : connections) {
            (void)other_label;
            if (confidence <= decay_threshold) {
                confidence = std::max(confidence - 1, 0);
            }
        }
    }
}

void InSegReconstruction::InitializeMap(const Eigen::Matrix4f& pose) {
    pose_ = pose;
    map_.surfels.clear();

    Label();

    const int level = main_config.min_pyr_level;
    const cv::Mat& frame_point_cloud = image_pyramid_.GetFramePointCloud(level);
    const cv::Mat& frame_normal_map = image_pyramid_.GetFrameNormalMap(level);
    const cv::Mat& frame_label_map = image_pyramid_.GetFrameLabelMap();
    const cv::Mat& rgb_image = image_pyramid_.GetRGBImage();
    if (frame_point_cloud.empty() || frame_normal_map.empty()) {
        world_label_num_ = label_num_;
        MakeIndexMap();
        return;
    }

    const InversePose pose_inverse =
        ComputeInversePose(pose_);
    const float inv00 = pose_inverse.inv00;
    const float inv01 = pose_inverse.inv01;
    const float inv02 = pose_inverse.inv02;
    const float inv10 = pose_inverse.inv10;
    const float inv11 = pose_inverse.inv11;
    const float inv12 = pose_inverse.inv12;
    const float inv20 = pose_inverse.inv20;
    const float inv21 = pose_inverse.inv21;
    const float inv22 = pose_inverse.inv22;
    const float itx = pose_inverse.itx;
    const float ity = pose_inverse.ity;
    const float itz = pose_inverse.itz;

    constexpr float kPi = 3.1415927f;
    constexpr float kRadiusScale = 0.70710677f;
    constexpr float kConfidenceFalloff = 1.3888888f;

    const CamParams& cam_params = camera_.GetCamParams(level);
    const float inv_fx = 1.0f / cam_params.fx;
    const float corner_x = -cam_params.cx * inv_fx;
    const float corner_y = -cam_params.cy / cam_params.fy;
    const float inv_corner_norm =
        1.0f / std::sqrt((corner_x * corner_x) + (corner_y * corner_y));
    const float observation_dot_threshold =
        std::cos((config.update_point_angle_threshold / 180.0f) * kPi);
    const bool use_rgb =
        rgb_image.type() == CV_8UC3 && rgb_image.size() == frame_point_cloud.size();
    const int frame_label_type = frame_label_map.type();

    for (int y = 1; y < frame_point_cloud.rows - 1; ++y) {
        const cv::Vec3f* const point_row = frame_point_cloud.ptr<cv::Vec3f>(y);
        const cv::Vec3f* const normal_row = frame_normal_map.ptr<cv::Vec3f>(y);
        const cv::Vec3b* const rgb_row = use_rgb ? rgb_image.ptr<cv::Vec3b>(y) : nullptr;
        const uint16_t* const frame_label_row_u16 =
            frame_label_type == CV_16UC1 ? frame_label_map.ptr<uint16_t>(y) : nullptr;
        const int16_t* const frame_label_row_s16 =
            frame_label_type == CV_16SC1 ? frame_label_map.ptr<int16_t>(y) : nullptr;
        const int* const frame_label_row_s32 =
            frame_label_type == CV_32SC1 ? frame_label_map.ptr<int>(y) : nullptr;

        for (int x = 1; x < frame_point_cloud.cols - 1; ++x) {
            const cv::Vec3f& point_camera = point_row[x];
            const float point_camera_x = point_camera[0];
            const float point_camera_y = point_camera[1];
            const float point_camera_z = point_camera[2];
            if (point_camera_z <= 0.0f) {
                continue;
            }

            const cv::Vec3f& normal_camera = normal_row[x];
            const float normal_camera_x = normal_camera[0];
            const float normal_camera_y = normal_camera[1];
            const float normal_camera_z = normal_camera[2];
            if (normal_camera_z == 0.0f) {
                continue;
            }

            const int frame_label = ReadLabelFromRow(
                frame_label_row_u16, frame_label_row_s16, frame_label_row_s32, x);

            const float ray_norm_sq =
                ((point_camera_y * point_camera_y) +
                 (point_camera_z * point_camera_z)) +
                (point_camera_x * point_camera_x);
            float negative_ray_x = -point_camera_x;
            float negative_ray_y = -point_camera_y;
            float negative_ray_z = -point_camera_z;
            if (ray_norm_sq > 0.0f) {
                const float ray_norm = std::sqrt(ray_norm_sq);
                negative_ray_x /= ray_norm;
                negative_ray_y /= ray_norm;
                negative_ray_z /= ray_norm;
            }

            const float normal_alignment =
                (normal_camera_x * negative_ray_x) +
                ((normal_camera_z * negative_ray_z) +
                 (normal_camera_y * negative_ray_y));
            if (normal_alignment < observation_dot_threshold) {
                continue;
            }

            const float point_world_x =
                TransformPointComponent(inv00, inv01, inv02,
                                        point_camera_x, point_camera_y, point_camera_z, itx);
            const float point_world_y =
                TransformPointComponent(inv10, inv11, inv12,
                                        point_camera_x, point_camera_y, point_camera_z, ity);
            const float point_world_z =
                TransformPointComponent(inv20, inv21, inv22,
                                        point_camera_x, point_camera_y, point_camera_z, itz);
            const float normal_world_x =
                RotateVectorComponent(inv00, inv01, inv02,
                                      normal_camera_x, normal_camera_y, normal_camera_z);
            const float normal_world_y =
                RotateVectorComponent(inv10, inv11, inv12,
                                      normal_camera_x, normal_camera_y, normal_camera_z);
            const float normal_world_z =
                RotateVectorComponent(inv20, inv21, inv22,
                                      normal_camera_x, normal_camera_y, normal_camera_z);

            const float surfel_radius =
                (kRadiusScale * inv_fx * point_camera_z) / normal_alignment;
            const float radial =
                std::sqrt((point_camera_x / point_camera_z) * (point_camera_x / point_camera_z) +
                          (point_camera_y / point_camera_z) * (point_camera_y / point_camera_z)) *
                inv_corner_norm;
            const double observation_confidence_arg =
                static_cast<double>(-((radial * radial) * kConfidenceFalloff));
            const float observation_confidence =
                static_cast<float>(std::exp(observation_confidence_arg));

            auto surfel = CreateSurfel(point_world_x,
                                       point_world_y,
                                       point_world_z,
                                       normal_world_x,
                                       normal_world_y,
                                       normal_world_z,
                                       surfel_radius,
                                       observation_confidence,
                                       frame_label);
            surfel->is_valid = true;
            surfel->is_stable = true;
            if (use_rgb) {
                const cv::Vec3b& color = rgb_row[x];
                surfel->color[0] = color[0];
                surfel->color[1] = color[1];
                surfel->color[2] = color[2];
            }
            AppendSurfel(&map_, std::move(surfel));
        }
    }

    world_label_num_ = label_num_;
    MakeIndexMap();
}

void InSegReconstruction::MakeIndexMap() {
    if (map_.surfels.empty()) {
        return;
    }

    const int level = main_config.min_pyr_level;
    const CamParams& cam_params = camera_.GetCamParams(level);
    cv::Mat& model_point_cloud = image_pyramid_.GetModelPointCloud();
    if (model_point_cloud.rows != cam_params.img_height || model_point_cloud.cols != cam_params.img_width) {
        model_point_cloud.create(cam_params.img_height, cam_params.img_width, CV_32FC3);
    }
    ResetMatBytes(&model_point_cloud, 0);

    cv::Mat& model_normal_map = image_pyramid_.GetModelNormalMapNotConst(level);
    if (model_normal_map.rows != cam_params.img_height || model_normal_map.cols != cam_params.img_width) {
        model_normal_map.create(cam_params.img_height, cam_params.img_width, CV_32FC3);
    }
    ResetMatBytes(&model_normal_map, 0);

    const CamParams& index_cam_params = camera_.GetIndexCamParams();
    if (index_map_.rows != index_cam_params.img_height || index_map_.cols != index_cam_params.img_width) {
        index_map_.create(index_cam_params.img_height, index_cam_params.img_width, CV_32SC1);
    }
    ResetMatBytes(&index_map_, 0xFF);

    if (world_label_map_.rows != cam_params.img_height || world_label_map_.cols != cam_params.img_width) {
        world_label_map_.create(cam_params.img_height, cam_params.img_width, CV_16UC1);
    }
    ResetMatBytes(&world_label_map_, 0);
    image_pyramid_.SetModelLabelMap(&world_label_map_);

    world_label_match_list_.Resize(world_label_num_);
    for (const auto& [label, connections] : label_connect_confidence_list_) {
        for (const auto& [other_label, confidence] : connections) {
            if (confidence <= config.label_merge_confidence_threshold) {
                continue;
            }
            world_label_match_list_.UnionSet(label, other_label);
            map_.Merge(label, other_label);
        }
    }

    const auto& surfels = map_.surfels;
    const float m00 = pose_(0, 0);
    const float m01 = pose_(0, 1);
    const float m02 = pose_(0, 2);
    const float m10 = pose_(1, 0);
    const float m11 = pose_(1, 1);
    const float m12 = pose_(1, 2);
    const float m20 = pose_(2, 0);
    const float m21 = pose_(2, 1);
    const float m22 = pose_(2, 2);
    const float tx = pose_(0, 3);
    const float ty = pose_(1, 3);
    const float tz = pose_(2, 3);

    const float cam_fx = cam_params.fx;
    const float cam_fy = cam_params.fy;
    const float cam_cx = cam_params.cx;
    const float cam_cy = cam_params.cy;
    const float index_fx = index_cam_params.fx;
    const float index_fy = index_cam_params.fy;
    const float index_cx = index_cam_params.cx;
    const float index_cy = index_cam_params.cy;

    const size_t index_step = index_map_.step / sizeof(int);
    auto* const index_data = reinterpret_cast<int*>(index_map_.data);
    auto& index_depth_cache = MakeIndexDepthBuffer();
    index_depth_cache.assign(
        static_cast<size_t>(index_map_.rows) * index_step,
        std::numeric_limits<float>::infinity());
    auto* const index_depth_data = index_depth_cache.data();
    const size_t point_step = model_point_cloud.step / sizeof(float);
    auto* const point_data = reinterpret_cast<float*>(model_point_cloud.data);
    const size_t normal_step = model_normal_map.step / sizeof(float);
    auto* const normal_data = reinterpret_cast<float*>(model_normal_map.data);
    const size_t label_step = world_label_map_.step / sizeof(uint16_t);
    auto* const label_data = reinterpret_cast<uint16_t*>(world_label_map_.data);
    const auto* const surfel_refs = surfels.data();

    for (int surfel_index = 0; surfel_index < static_cast<int>(surfels.size()); ++surfel_index) {
        const auto& surfel_ref = surfel_refs[surfel_index];
        Surfel* const surfel = surfel_ref.get();
        if (!surfel || !surfel->is_valid) {
            continue;
        }

        ++surfel->time_invalid;
        if (!surfel->is_stable && surfel->time_invalid > config.surfel_invalid_frame_threshold) {
            surfel->is_valid = false;
            continue;
        }

        surfel->label = world_label_match_list_.Find(surfel->label);

        const float normal_x = surfel->normal.x();
        const float normal_y = surfel->normal.y();
        const float normal_z = surfel->normal.z();
        const float normal_camera_x =
            RotateVectorComponent(m00, m01, m02, normal_x, normal_y, normal_z);
        const float normal_camera_y =
            RotateVectorComponent(m10, m11, m12, normal_x, normal_y, normal_z);
        const float normal_camera_z =
            RotateVectorComponent(m20, m21, m22, normal_x, normal_y, normal_z);
        if (normal_camera_z > 0.0f) {
            continue;
        }

        const float pos_x = surfel->pos.x();
        const float pos_y = surfel->pos.y();
        const float pos_z = surfel->pos.z();
        const float pos_camera_x =
            TransformPointComponent(m00, m01, m02, pos_x, pos_y, pos_z, tx);
        const float pos_camera_y =
            TransformPointComponent(m10, m11, m12, pos_x, pos_y, pos_z, ty);
        const float pos_camera_z =
            TransformPointComponent(m20, m21, m22, pos_x, pos_y, pos_z, tz);
        if (pos_camera_z == 0.0f) {
            continue;
        }

        const float u_normalized = pos_camera_x / pos_camera_z;
        const float v_normalized = pos_camera_y / pos_camera_z;

        const float x_index_f = (u_normalized * index_fx) + index_cx;
        const float y_index_f = (v_normalized * index_fy) + index_cy;
        const int x_index = RoundPixelCoordinate(x_index_f);
        const int y_index = RoundPixelCoordinate(y_index_f);
        if (x_index < 0 || x_index >= index_cam_params.img_width || y_index < 0 || y_index >= index_cam_params.img_height) {
            continue;
        }

        int* const index_cell = index_data + y_index * index_step + x_index;
        float* const index_depth_cell = index_depth_data + y_index * index_step + x_index;
        if (*index_cell < 0) {
            *index_cell = surfel_index;
            *index_depth_cell = pos_camera_z;
        } else {
            if (*index_depth_cell > pos_camera_z) {
                *index_cell = surfel_index;
                *index_depth_cell = pos_camera_z;
            }
        }

        const float x_model_f = (u_normalized * cam_fx) + cam_cx;
        const float y_model_f = (v_normalized * cam_fy) + cam_cy;
        const int x_model = RoundPixelCoordinate(x_model_f);
        const int y_model = RoundPixelCoordinate(y_model_f);
        if (x_model < 0 || x_model >= model_point_cloud.cols || y_model < 0 || y_model >= model_point_cloud.rows) {
            continue;
        }

        float* const rendered_point = point_data + y_model * point_step + 3 * x_model;
        if ((rendered_point[2] == 0.0f || rendered_point[2] >= pos_camera_z) && normal_camera_z < 0.0f) {
            if (surfel->is_stable) {
                rendered_point[0] = pos_camera_x;
                rendered_point[1] = pos_camera_y;
                rendered_point[2] = pos_camera_z;
                float* const rendered_normal = normal_data + y_model * normal_step + 3 * x_model;
                rendered_normal[0] = normal_camera_x;
                rendered_normal[1] = normal_camera_y;
                rendered_normal[2] = normal_camera_z;
            }
            label_data[y_model * label_step + x_model] = static_cast<uint16_t>(surfel->label);
        }
    }

    image_pyramid_.PyrModelDown(camera_.GetMaxLevel());
}

void InSegReconstruction::UpdateMap() {
    if (config.remove_dynamic_point) {
        return;
    }
    if (map_.surfels.empty()) {
        return;
    }

    const int level = main_config.min_pyr_level;
    const cv::Mat& frame_point_cloud = image_pyramid_.GetFramePointCloud(level);
    const cv::Mat& frame_normal_map = image_pyramid_.GetFrameNormalMap(level);
    const cv::Mat& frame_label_map = image_pyramid_.GetFrameLabelMap();
    const cv::Mat& rgb_image = image_pyramid_.GetRGBImage();
    if (frame_point_cloud.empty() || frame_normal_map.empty() || index_map_.empty()) {
        return;
    }

    EnsureSurfelCapacityForFrame(&map_.surfels, frame_point_cloud.rows, frame_point_cloud.cols);

    const InversePose pose_inverse =
        ComputeInversePose(pose_);
    const float inv00 = pose_inverse.inv00;
    const float inv01 = pose_inverse.inv01;
    const float inv02 = pose_inverse.inv02;
    const float inv10 = pose_inverse.inv10;
    const float inv11 = pose_inverse.inv11;
    const float inv12 = pose_inverse.inv12;
    const float inv20 = pose_inverse.inv20;
    const float inv21 = pose_inverse.inv21;
    const float inv22 = pose_inverse.inv22;
    const float itx = pose_inverse.itx;
    const float ity = pose_inverse.ity;
    const float itz = pose_inverse.itz;

    constexpr float kPi = 3.1415927f;
    constexpr float kRadiusScale = 0.70710677f;
    constexpr float kConfidenceFalloff = 1.3888888f;
    constexpr int kIndexBlockArea = 16;

    const CamParams& cam_params = camera_.GetCamParams(level);
    const float inv_fx = 1.0f / cam_params.fx;
    const float corner_x = -cam_params.cx * inv_fx;
    const float corner_y = -cam_params.cy / cam_params.fy;
    const float inv_corner_norm =
        1.0f / std::sqrt((corner_x * corner_x) + (corner_y * corner_y));
    const float observation_dot_threshold =
        std::cos((config.update_point_angle_threshold / 180.0f) * kPi);
    const float normal_match_threshold =
        std::cos((config.update_point_dot_product_threshold / 180.0f) * kPi);
    const float depth_uncertainty =
        static_cast<float>(main_config.uncertainty_coefficient_depth);
    const bool use_rgb =
        rgb_image.type() == CV_8UC3 && rgb_image.size() == frame_point_cloud.size();
    const int frame_label_type = frame_label_map.type();
    int surfel_count = static_cast<int>(map_.surfels.size());
    const std::shared_ptr<Surfel>* surfel_refs = map_.surfels.data();

    for (int y = 1; y < frame_point_cloud.rows - 1; ++y) {
        const cv::Vec3f* const point_row = frame_point_cloud.ptr<cv::Vec3f>(y);
        const cv::Vec3f* const point_row_above = frame_point_cloud.ptr<cv::Vec3f>(y - 1);
        const cv::Vec3f* const point_row_below = frame_point_cloud.ptr<cv::Vec3f>(y + 1);
        const cv::Vec3f* const normal_row = frame_normal_map.ptr<cv::Vec3f>(y);
        const cv::Vec3b* const rgb_row = use_rgb ? rgb_image.ptr<cv::Vec3b>(y) : nullptr;
        const uint16_t* const frame_label_row_u16 =
            frame_label_type == CV_16UC1 ? frame_label_map.ptr<uint16_t>(y) : nullptr;
        const int16_t* const frame_label_row_s16 =
            frame_label_type == CV_16SC1 ? frame_label_map.ptr<int16_t>(y) : nullptr;
        const int* const frame_label_row_s32 =
            frame_label_type == CV_32SC1 ? frame_label_map.ptr<int>(y) : nullptr;
        const int index_y = y * 4;
        const int* const index_rows[4] = {
            index_map_.ptr<int>(index_y),
            index_map_.ptr<int>(index_y + 1),
            index_map_.ptr<int>(index_y + 2),
            index_map_.ptr<int>(index_y + 3),
        };

        for (int x = 1; x < frame_point_cloud.cols - 1; ++x) {
            const cv::Vec3f& point_camera = point_row[x];
            const cv::Vec3f& normal_camera = normal_row[x];
            if (!IsFiniteVec3(point_camera) || point_camera[2] <= 0.0f ||
                !IsFiniteVec3(normal_camera) || normal_camera[2] == 0.0f) {
                continue;
            }

            const float point_camera_x = point_camera[0];
            const float point_camera_y = point_camera[1];
            const float point_camera_z = point_camera[2];
            const float normal_camera_x = normal_camera[0];
            const float normal_camera_y = normal_camera[1];
            const float normal_camera_z = normal_camera[2];

            const float local_threshold =
                0.5f * depth_uncertainty * point_camera_z * point_camera_z;
            if (PointToPlaneDistance(point_row_above[x - 1],
                                     point_camera_x, point_camera_y, point_camera_z,
                                     normal_camera_x, normal_camera_y, normal_camera_z) > local_threshold ||
                PointToPlaneDistance(point_row_above[x],
                                     point_camera_x, point_camera_y, point_camera_z,
                                     normal_camera_x, normal_camera_y, normal_camera_z) > local_threshold ||
                PointToPlaneDistance(point_row_above[x + 1],
                                     point_camera_x, point_camera_y, point_camera_z,
                                     normal_camera_x, normal_camera_y, normal_camera_z) > local_threshold ||
                PointToPlaneDistance(point_row[x - 1],
                                     point_camera_x, point_camera_y, point_camera_z,
                                     normal_camera_x, normal_camera_y, normal_camera_z) > local_threshold ||
                PointToPlaneDistance(point_row[x + 1],
                                     point_camera_x, point_camera_y, point_camera_z,
                                     normal_camera_x, normal_camera_y, normal_camera_z) > local_threshold ||
                PointToPlaneDistance(point_row_below[x - 1],
                                     point_camera_x, point_camera_y, point_camera_z,
                                     normal_camera_x, normal_camera_y, normal_camera_z) > local_threshold ||
                PointToPlaneDistance(point_row_below[x],
                                     point_camera_x, point_camera_y, point_camera_z,
                                     normal_camera_x, normal_camera_y, normal_camera_z) > local_threshold ||
                PointToPlaneDistance(point_row_below[x + 1],
                                     point_camera_x, point_camera_y, point_camera_z,
                                     normal_camera_x, normal_camera_y, normal_camera_z) > local_threshold) {
                continue;
            }

            const float ray_norm_sq =
                ((point_camera_y * point_camera_y) +
                 (point_camera_z * point_camera_z)) +
                (point_camera_x * point_camera_x);
            float negative_ray_x = -point_camera_x;
            float negative_ray_y = -point_camera_y;
            float negative_ray_z = -point_camera_z;
            if (ray_norm_sq > 0.0f) {
                const float ray_norm = std::sqrt(ray_norm_sq);
                negative_ray_x /= ray_norm;
                negative_ray_y /= ray_norm;
                negative_ray_z /= ray_norm;
            }
            const float normal_alignment =
                (normal_camera_x * negative_ray_x) +
                ((normal_camera_z * negative_ray_z) +
                 (normal_camera_y * negative_ray_y));
            if (normal_alignment < observation_dot_threshold) {
                continue;
            }

            const float surfel_radius =
                (kRadiusScale * inv_fx * point_camera_z) / normal_alignment;
            const float radial =
                std::sqrt((point_camera_x / point_camera_z) * (point_camera_x / point_camera_z) +
                          (point_camera_y / point_camera_z) * (point_camera_y / point_camera_z)) *
                inv_corner_norm;
            const double observation_confidence_arg =
                static_cast<double>(-((radial * radial) * kConfidenceFalloff));
            const float observation_confidence =
                static_cast<float>(std::exp(observation_confidence_arg));

            const float point_world_x =
                TransformPointComponent(inv00, inv01, inv02, point_camera_x, point_camera_y, point_camera_z, itx);
            const float point_world_y =
                TransformPointComponent(inv10, inv11, inv12, point_camera_x, point_camera_y, point_camera_z, ity);
            const float point_world_z =
                TransformPointComponent(inv20, inv21, inv22, point_camera_x, point_camera_y, point_camera_z, itz);
            const float normal_world_x =
                RotateVectorComponent(inv00, inv01, inv02, normal_camera_x, normal_camera_y, normal_camera_z);
            const float normal_world_y =
                RotateVectorComponent(inv10, inv11, inv12, normal_camera_x, normal_camera_y, normal_camera_z);
            const float normal_world_z =
                RotateVectorComponent(inv20, inv21, inv22, normal_camera_x, normal_camera_y, normal_camera_z);

            const float view_dir_world_x =
                RotateVectorComponent(inv00, inv01, inv02, negative_ray_x, negative_ray_y, negative_ray_z);
            const float view_dir_world_y =
                RotateVectorComponent(inv10, inv11, inv12, negative_ray_x, negative_ray_y, negative_ray_z);
            const float view_dir_world_z =
                RotateVectorComponent(inv20, inv21, inv22, negative_ray_x, negative_ray_y, negative_ray_z);

            const int index_x = x * 4;
            const float match_threshold = depth_uncertainty * point_camera_z * point_camera_z;

            int candidate_indices[kIndexBlockArea];
            int candidate_count = 0;
            for (int yy = 0; yy < 4; ++yy) {
                const int* const index_row = index_rows[yy];
                for (int xx = 0; xx < 4; ++xx) {
                    const int candidate_index = index_row[index_x + xx];
                    if (candidate_index < 0 || candidate_index >= surfel_count) {
                        continue;
                    }
                    bool already_seen = false;
                    for (int i = 0; i < candidate_count; ++i) {
                        if (candidate_indices[i] == candidate_index) {
                            already_seen = true;
                            break;
                        }
                    }
                    if (!already_seen) {
                        candidate_indices[candidate_count++] = candidate_index;
                    }
                }
            }

            int best_surfel_index = -1;
            float best_position_error = std::numeric_limits<float>::max();
            for (int candidate_offset = 0; candidate_offset < candidate_count; ++candidate_offset) {
                const int candidate_index = candidate_indices[candidate_offset];
                const auto& surfel_ref = surfel_refs[candidate_index];
                const Surfel* const surfel = surfel_ref.get();
                if (!surfel) {
                    continue;
                }

                const float surfel_pos_x = surfel->pos.x();
                const float surfel_pos_y = surfel->pos.y();
                const float surfel_pos_z = surfel->pos.z();
                const float position_error = std::fabs(
                    ((surfel_pos_x - point_world_x) * view_dir_world_x) +
                    (((surfel_pos_y - point_world_y) * view_dir_world_y) +
                     ((surfel_pos_z - point_world_z) * view_dir_world_z)));
                if (position_error >= match_threshold) {
                    continue;
                }
                const float normal_dot =
                    (surfel->normal.x() * normal_world_x) +
                    ((surfel->normal.y() * normal_world_y) +
                     (surfel->normal.z() * normal_world_z));
                if (normal_dot <= normal_match_threshold) {
                    continue;
                }
                if (surfel_radius > (1.5f * surfel->radius)) {
                    continue;
                }
                if (position_error < best_position_error) {
                    best_position_error = position_error;
                    best_surfel_index = candidate_index;
                }
            }

            if (best_surfel_index > 0) {
                auto& surfel_ref = map_.surfels[best_surfel_index];
                Surfel* const surfel = surfel_ref.get();
                if (!surfel) {
                    continue;
                }
                const int frame_label = ReadLabelFromRow(
                    frame_label_row_u16, frame_label_row_s16, frame_label_row_s32, x);

                const float old_confidence = surfel->confidence;
                const float merged_confidence = observation_confidence + old_confidence;
                const float inv_confidence = 1.0f / merged_confidence;

                const float merged_position_x =
                    ((point_world_x * observation_confidence) +
                     (surfel->pos.x() * old_confidence)) * inv_confidence;
                const float merged_position_y =
                    ((point_world_y * observation_confidence) +
                     (surfel->pos.y() * old_confidence)) * inv_confidence;
                const float merged_position_z =
                    ((point_world_z * observation_confidence) +
                     (surfel->pos.z() * old_confidence)) * inv_confidence;

                float merged_normal_x =
                    ((normal_world_x * observation_confidence) +
                     (surfel->normal.x() * old_confidence)) * inv_confidence;
                float merged_normal_y =
                    ((normal_world_y * observation_confidence) +
                     (surfel->normal.y() * old_confidence)) * inv_confidence;
                float merged_normal_z =
                    ((normal_world_z * observation_confidence) +
                     (surfel->normal.z() * old_confidence)) * inv_confidence;
                const float merged_normal_norm = ScalarSqrt(
                    ((merged_normal_y * merged_normal_y) +
                     (merged_normal_z * merged_normal_z)) +
                    (merged_normal_x * merged_normal_x));
                if (merged_normal_norm > 0.0f) {
                    merged_normal_x = ScalarDivide(merged_normal_x, merged_normal_norm);
                    merged_normal_y = ScalarDivide(merged_normal_y, merged_normal_norm);
                    merged_normal_z = ScalarDivide(merged_normal_z, merged_normal_norm);
                }
                const float merged_radius =
                    ((surfel->radius * old_confidence) +
                     (surfel_radius * observation_confidence)) * inv_confidence;

                if (use_rgb) {
                    const cv::Vec3b& color = rgb_row[x];
                    surfel->color[0] = static_cast<unsigned char>(
                        ((static_cast<float>(color[0]) * observation_confidence) +
                         (static_cast<float>(surfel->color[0]) * old_confidence)) *
                        inv_confidence);
                    surfel->color[1] = static_cast<unsigned char>(
                        ((static_cast<float>(color[1]) * observation_confidence) +
                         (static_cast<float>(surfel->color[1]) * old_confidence)) *
                        inv_confidence);
                    surfel->color[2] = static_cast<unsigned char>(
                        ((static_cast<float>(color[2]) * observation_confidence) +
                         (static_cast<float>(surfel->color[2]) * old_confidence)) *
                        inv_confidence);
                }

                surfel->confidence = merged_confidence;
                surfel->radius = merged_radius;
                surfel->pos.x() = merged_position_x;
                surfel->pos.y() = merged_position_y;
                surfel->pos.z() = merged_position_z;
                surfel->normal.x() = merged_normal_x;
                surfel->normal.y() = merged_normal_y;
                surfel->normal.z() = merged_normal_z;

                if (frame_label != 0) {
                    surfel->label_confidence +=
                        (surfel->label == frame_label) ? 1 : -1;
                    if (surfel->label_confidence < 0) {
                        map_.SetLabel(best_surfel_index, frame_label);
                        surfel->label_confidence = 0;
                    }
                }
                if (merged_confidence > static_cast<float>(config.surfel_stable_frame_threshold)) {
                    surfel->is_stable = true;
                }
                continue;
            }

            if (best_surfel_index != -1) {
                continue;
            }

            const int frame_label = ReadLabelFromRow(
                frame_label_row_u16, frame_label_row_s16, frame_label_row_s32, x);
            auto surfel = CreateSurfel(point_world_x,
                                       point_world_y,
                                       point_world_z,
                                       normal_world_x,
                                       normal_world_y,
                                       normal_world_z,
                                       surfel_radius,
                                       observation_confidence,
                                       frame_label);
            surfel->is_valid = true;
            surfel->is_stable = false;
            if (use_rgb) {
                const cv::Vec3b& color = rgb_row[x];
                surfel->color[0] = color[0];
                surfel->color[1] = color[1];
                surfel->color[2] = color[2];
            }
            AppendSurfel(&map_, std::move(surfel));
            ++surfel_count;
            surfel_refs = map_.surfels.data();
        }
    }
}

void InSegReconstruction::ProcessFrame(const cv::Mat& depth_map, const cv::Mat& color_image) {
    if (depth_map.empty() || color_image.empty()) return;

    // Use async thread to remove invalid surfels, avoid blocking main thread
    static int frame_counter = 0;
    bool do_compact = SGF_ENABLE_ASYNC_COMPACT && (frame_counter++ % 20 == 0);

    if (SGF_ENABLE_ASYNC_INDEX_MAP) {
        auto background_future = std::async(std::launch::async, [this, do_compact]() {
            if (do_compact) {
                auto& surfels = map_.surfels;
                surfels.erase(std::remove_if(surfels.begin(), surfels.end(),
                                             [](const auto& s) { return !s->is_valid; }),
                              surfels.end());
            }
            MakeIndexMap();
        });
        image_pyramid_.CreateDepthPyramid(depth_map, color_image, map_config.normalize_depth, camera_,
                                          main_config.uncertainty_coefficient_depth);
        Label();
        background_future.wait();
    } else {
        image_pyramid_.CreateDepthPyramid(depth_map, color_image, map_config.normalize_depth, camera_,
                                          main_config.uncertainty_coefficient_depth);
        Label();
        if (do_compact) {
            auto& surfels = map_.surfels;
            surfels.erase(std::remove_if(surfels.begin(), surfels.end(),
                                         [](const auto& s) { return !s->is_valid; }),
                          surfels.end());
        }
        MakeIndexMap();
    }

    PropagateLabels();
    UpdateMap();

    image_pyramid_.GetFrameLabelMapNotConst().copyTo(label_map_propagated_frame_);
}

} // namespace inseg_lib
