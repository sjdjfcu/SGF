#include "camera.h"

namespace inseg_lib {

CamParams CamParams::CreateScaledCamParams(const float scale) const {
    CamParams scaled = *this;
    scaled.fx *= scale;
    scaled.fy *= scale;
    scaled.cx = ((scaled.cx + 0.5f) * scale) - 0.5f;
    scaled.cy = ((scaled.cy + 0.5f) * scale) - 0.5f;
    scaled.img_width = static_cast<int>(static_cast<float>(scaled.img_width) * scale);
    scaled.img_height = static_cast<int>(static_cast<float>(scaled.img_height) * scale);
    return scaled;
}

Eigen::Vector2f CamParams::Denormalize(const Eigen::Vector2f& xy) const {
    return Eigen::Vector2f(xy.x() * fx + cx, xy.y() * fy + cy);
}

Eigen::Vector2f CamParams::Normalize(const Eigen::Vector2f& uv) const {
    return Eigen::Vector2f((uv.x() - cx) / fx, (uv.y() - cy) / fy);
}

void InSegCamera::Init(const CamParams& cparam, const int max_levels, const int min_levels) {
    camera_params_.clear();
    camera_params_.reserve(max_levels);
    if (max_levels <= 0) {
        camera_param_index_ = CamParams();
        return;
    }

    camera_params_.push_back(cparam);
    for (int level = 1; level < max_levels; ++level) {
        camera_params_.push_back(camera_params_.back().CreateScaledCamParams(0.5f));
    }

    camera_param_index_ = camera_params_[min_levels].CreateScaledCamParams(4.0f);
}

const CamParams& InSegCamera::GetCamParams(const int level) const {
    return camera_params_[level];
}

const CamParams& InSegCamera::GetIndexCamParams() const {
    return camera_param_index_;
}

int InSegCamera::GetMaxLevel() const {
    return static_cast<int>(camera_params_.size());
}

}  // namespace inseg_lib
