#include "image_pyramid.h"

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc.hpp>

namespace inseg_lib {

namespace {

constexpr int kVec3Type = CV_32FC3;

inline bool IsValidPoint(const cv::Vec3f& value) {
    return value[2] != 0.0f;
}

inline bool IsFiniteDepth(const float depth) {
    return std::isfinite(depth) && depth > 0.0f;
}

}  // namespace

ImagePyramid::ImagePyramid(const MapConfig& config, const int max_level, const int min_level)
    : min_level_(min_level), config_(config) {
    frame_pyr_depth_img_.resize(max_level);
    frame_pyr_point_cloud_.resize(max_level);
    model_pyr_point_cloud_.resize(max_level);
    frame_pyr_rgb_img_.resize(max_level);
    model_pyr_normal_map_.resize(max_level);
    frame_pyr_normal_map_.resize(max_level);
    frame_pyr_smooth_depth_img_.resize(max_level);
    frame_pyr_smooth_point_cloud_.resize(max_level);
}

cv::Mat& ImagePyramid::GetFramePointCloud() {
    return frame_pyr_point_cloud_[min_level_];
}

const cv::Mat& ImagePyramid::GetFramePointCloud(int level) const {
    return frame_pyr_point_cloud_[level];
}

const cv::Mat& ImagePyramid::GetFrameNormalMap(int level) const {
    return frame_pyr_normal_map_[level];
}

const cv::Mat& ImagePyramid::GetRGBImage() const {
    return frame_pyr_rgb_img_[min_level_];
}

const cv::Mat& ImagePyramid::GetRGBImageHD() const {
    return frame_rgb_img_;
}

cv::Mat& ImagePyramid::GetFrameLabelMapNotConst() {
    return frame_label_map_;
}

cv::Mat& ImagePyramid::GetModelLabelMapNotConst() {
    return model_label_map_;
}

const cv::Mat& ImagePyramid::GetFrameLabelMap() const {
    return frame_label_map_;
}

void ImagePyramid::PyrDownMap(const cv::Mat& upper, cv::Mat* cur) {
    if (!cur || upper.empty()) {
        return;
    }

    const int rows = upper.rows / 2;
    const int cols = upper.cols / 2;
    if (rows <= 0 || cols <= 0) {
        cur->release();
        return;
    }

    cur->create(rows, cols, upper.type());
    cur->setTo(cv::Scalar::all(0));

    for (int y = 0; y < rows; ++y) {
        cv::Vec3f* dst_row = cur->ptr<cv::Vec3f>(y);
        const cv::Vec3f* src_row0 = upper.ptr<cv::Vec3f>(2 * y);
        const cv::Vec3f* src_row1 = upper.ptr<cv::Vec3f>(2 * y + 1);
        for (int x = 0; x < cols; ++x) {
            cv::Vec3f sum(0.0f, 0.0f, 0.0f);
            int count = 0;

            const cv::Vec3f& v00 = src_row0[2 * x];
            if (IsValidPoint(v00)) {
                sum += v00;
                ++count;
            }
            const cv::Vec3f& v01 = src_row0[2 * x + 1];
            if (IsValidPoint(v01)) {
                sum += v01;
                ++count;
            }
            const cv::Vec3f& v10 = src_row1[2 * x];
            if (IsValidPoint(v10)) {
                sum += v10;
                ++count;
            }
            const cv::Vec3f& v11 = src_row1[2 * x + 1];
            if (IsValidPoint(v11)) {
                sum += v11;
                ++count;
            }

            if (count > 0) {
                dst_row[x] = sum * (1.0f / static_cast<float>(count));
            }
        }
    }
}

void ImagePyramid::PyrModelDown(const int max_level) {
    for (int level = min_level_ + 1; level < max_level; ++level) {
        PyrDownMap(model_pyr_point_cloud_[level - 1], &model_pyr_point_cloud_[level]);
        PyrDownMap(model_pyr_normal_map_[level - 1], &model_pyr_normal_map_[level]);
    }
}

void ImagePyramid::PyrFrameDown(const int max_level) {
    for (int level = min_level_ + 1; level < max_level; ++level) {
        PyrDownMap(frame_pyr_point_cloud_[level - 1], &frame_pyr_point_cloud_[level]);
        PyrDownMap(frame_pyr_normal_map_[level - 1], &frame_pyr_normal_map_[level]);
    }
}

cv::Mat& ImagePyramid::GetModelPointCloud() {
    return model_pyr_point_cloud_[min_level_];
}

const cv::Mat& ImagePyramid::GetModelPointCloud(int level) const {
    return model_pyr_point_cloud_[level];
}

cv::Mat& ImagePyramid::GetModelNormalMapNotConst(int level) {
    return model_pyr_normal_map_[level];
}

cv::Mat& ImagePyramid::GetFrameNormalMapNotConst(int level) {
    return frame_pyr_normal_map_[level];
}

const cv::Mat& ImagePyramid::GetModelNormalMap(int level) const {
    return model_pyr_normal_map_[level];
}

void ImagePyramid::SetModelLabelMap(cv::Mat* label) {
    if (label) {
        model_label_map_ = *label;
    } else {
        model_label_map_.release();
    }
}

void ImagePyramid::SetFrameLabelMap(cv::Mat* label) {
    if (label) {
        frame_label_map_ = *label;
    } else {
        frame_label_map_.release();
    }
}

const cv::Mat& ImagePyramid::GetModelLabelMap() const {
    return model_label_map_;
}

float AddBilateralFilterElem(const float depth,
                             const int16_t coefficient,
                             const float reference_depth,
                             const float depth_variance) {
    if (depth <= 0.0f) {
        return 0.0f;
    }
    const float denom = std::max(depth_variance + depth_variance, 1e-12f);
    const float diff = reference_depth - depth;
    return static_cast<float>(coefficient) * std::exp(-(diff * diff) / denom);
}

void FilterBilateral3x3(const cv::Mat& source,
                        const float uncertainty_coef_depth,
                        cv::Mat* destination) {
    if (!destination || source.empty()) {
        return;
    }

    destination->create(source.rows, source.cols, CV_32FC1);
    destination->setTo(0.0f);

    for (int y = 1; y < source.rows - 1; ++y) {
        const float* prev = source.ptr<float>(y - 1);
        const float* cur = source.ptr<float>(y);
        const float* next = source.ptr<float>(y + 1);
        float* dst = destination->ptr<float>(y);

        for (int x = 1; x < source.cols - 1; ++x) {
            const float center = cur[x];
            if (center <= 0.0f) {
                continue;
            }

            const float depth_scale = ((uncertainty_coef_depth * center) * center) * 0.5f;
            const float depth_variance = depth_scale * depth_scale;

            float weight_sum = 0.0f;
            float value_sum = 0.0f;

            const float samples[9] = {
                prev[x - 1], prev[x], prev[x + 1],
                cur[x - 1], cur[x], cur[x + 1],
                next[x - 1], next[x], next[x + 1],
            };
            const int16_t coeffs[9] = {1, 2, 1, 2, 4, 2, 1, 2, 1};

            for (int i = 0; i < 9; ++i) {
                const float weight = AddBilateralFilterElem(samples[i], coeffs[i], center, depth_variance);
                weight_sum += weight;
                value_sum += weight * samples[i];
            }

            if (weight_sum > 0.0f) {
                dst[x] = value_sum / weight_sum;
            }
        }
    }
}

void CalculateNormalMap(const cv::Mat& point_cloud, cv::Mat* normal_image) {
    if (!normal_image || point_cloud.empty()) {
        return;
    }

    normal_image->create(point_cloud.rows, point_cloud.cols, kVec3Type);
    normal_image->setTo(cv::Scalar::all(0));

    for (int y = 1; y < point_cloud.rows - 1; ++y) {
        const cv::Vec3f* prev = point_cloud.ptr<cv::Vec3f>(y - 1);
        const cv::Vec3f* cur = point_cloud.ptr<cv::Vec3f>(y);
        const cv::Vec3f* next = point_cloud.ptr<cv::Vec3f>(y + 1);
        cv::Vec3f* dst = normal_image->ptr<cv::Vec3f>(y);

        for (int x = 1; x < point_cloud.cols - 1; ++x) {
            const cv::Vec3f& center = cur[x];
            const cv::Vec3f& left = cur[x - 1];
            const cv::Vec3f& right = cur[x + 1];
            const cv::Vec3f& up = prev[x];
            const cv::Vec3f& down = next[x];

            if (center[2] <= 0.0f || left[2] <= 0.0f || right[2] <= 0.0f ||
                up[2] <= 0.0f || down[2] <= 0.0f) {
                continue;
            }

            const float vx = up[0] - down[0];
            const float vy = up[1] - down[1];
            const float vz = up[2] - down[2];
            const float hx = left[0] - right[0];
            const float hy = left[1] - right[1];
            const float hz = left[2] - right[2];

            float nx = vz * hy - vy * hz;
            float ny = vx * hz - hx * vz;
            float nz = vy * hx - vx * hy;

            const float norm_sq = (ny * ny + nz * nz) + nx * nx;
            if (norm_sq > 0.0f) {
                const float norm = std::sqrt(norm_sq);
                nx /= norm;
                ny /= norm;
                nz /= norm;
            }
            if (nz > 0.0f) {
                nx = -nx;
                ny = -ny;
                nz = -nz;
            }
            dst[x] = cv::Vec3f(nx, ny, nz);
        }
    }
}

void CalculatePointCloud(const cv::Mat& depth_image,
                         const CamParams& cam_params,
                         const bool normalize_depth,
                         cv::Mat* point_cloud) {
    if (!point_cloud || depth_image.empty()) {
        return;
    }

    point_cloud->create(depth_image.rows, depth_image.cols, kVec3Type);
    point_cloud->setTo(cv::Scalar::all(0));

    std::vector<float> normalized_x(depth_image.cols);
    std::vector<float> normalized_y(depth_image.rows);
    for (int x = 0; x < depth_image.cols; ++x) {
        normalized_x[x] = (static_cast<float>(x) - cam_params.cx) / cam_params.fx;
    }
    for (int y = 0; y < depth_image.rows; ++y) {
        normalized_y[y] = (static_cast<float>(y) - cam_params.cy) / cam_params.fy;
    }

    for (int y = 0; y < depth_image.rows; ++y) {
        const float* depth_row = depth_image.ptr<float>(y);
        cv::Vec3f* out_row = point_cloud->ptr<cv::Vec3f>(y);
        for (int x = 0; x < depth_image.cols; ++x) {
            const float depth = depth_row[x];
            if (depth == 0.0f) {
                continue;
            }

            const float nx = normalized_x[x];
            const float ny = normalized_y[y];
            if (normalize_depth) {
                float px = nx;
                float py = ny;
                float pz = 1.0f;
                const float norm_sq = (py * py + 1.0f) + px * px;
                if (norm_sq > 0.0f) {
                    const float norm = std::sqrt(norm_sq);
                    px /= norm;
                    py /= norm;
                    pz = 1.0f / norm;
                }
                out_row[x] = cv::Vec3f(px * depth, py * depth, pz * depth);
            } else {
                out_row[x] = cv::Vec3f(nx * depth, ny * depth, depth);
            }
        }
    }
}

void ImagePyramid::CreateDepthPyramid(const cv::Mat& depth_image,
                                      const cv::Mat& color_image,
                                      const bool normalize_depth,
                                      const InSegCamera& camera,
                                      const float uncertainty_coefficient_depth) {
    const int max_level = camera.GetMaxLevel();
    if (max_level <= 0) {
        return;
    }

    depth_image.convertTo(frame_pyr_depth_img_[0], CV_32FC1);
    color_image.copyTo(frame_rgb_img_);
    color_image.copyTo(frame_pyr_rgb_img_[0]);

    for (int y = 0; y < frame_pyr_depth_img_[0].rows; ++y) {
        float* row = frame_pyr_depth_img_[0].ptr<float>(y);
        for (int x = 0; x < frame_pyr_depth_img_[0].cols; ++x) {
            const float depth = row[x];
            if (!IsFiniteDepth(depth) ||
                depth < static_cast<float>(config_.near_depth_threshold) ||
                depth > static_cast<float>(config_.far_depth_threshold)) {
                row[x] = 0.0f;
            }
        }
    }

    for (int level = 1; level < max_level; ++level) {
        const cv::Size size(frame_pyr_depth_img_[level - 1].cols / 2,
                            frame_pyr_depth_img_[level - 1].rows / 2);
        cv::resize(frame_pyr_depth_img_[level - 1], frame_pyr_depth_img_[level], size, 0.0, 0.0,
                   cv::INTER_NEAREST);
        cv::resize(frame_pyr_rgb_img_[level - 1], frame_pyr_rgb_img_[level], size, 0.0, 0.0,
                   cv::INTER_LINEAR);
    }

    for (int level = 0; level < max_level; ++level) {
        if (config_.num_iterations_bilateral_filtering <= 0) {
            frame_pyr_depth_img_[level].copyTo(frame_pyr_smooth_depth_img_[level]);
        } else {
            frame_pyr_depth_img_[level].copyTo(frame_pyr_smooth_depth_img_[level]);
            cv::Mat tmp;
            for (int iter = 0; iter < config_.num_iterations_bilateral_filtering; ++iter) {
                FilterBilateral3x3(frame_pyr_smooth_depth_img_[level],
                                  uncertainty_coefficient_depth,
                                  &tmp);
                tmp.copyTo(frame_pyr_smooth_depth_img_[level]);
            }
        }

        const CamParams& cam_params = camera.GetCamParams(level);
        CalculatePointCloud(frame_pyr_depth_img_[level], cam_params, normalize_depth,
                            &frame_pyr_point_cloud_[level]);
        if (config_.num_iterations_bilateral_filtering <= 0) {
            frame_pyr_point_cloud_[level].copyTo(frame_pyr_smooth_point_cloud_[level]);
        } else {
            CalculatePointCloud(frame_pyr_smooth_depth_img_[level], cam_params, normalize_depth,
                                &frame_pyr_smooth_point_cloud_[level]);
        }
        CalculateNormalMap(frame_pyr_smooth_point_cloud_[level], &frame_pyr_normal_map_[level]);
    }
}

}  // namespace inseg_lib
