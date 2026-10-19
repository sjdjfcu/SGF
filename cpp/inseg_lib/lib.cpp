#include "lib.h"

namespace inseg_lib {

InSegLib::InSegLib(std::shared_ptr<inseg_lib::SegmentsInterface> segments,
                   InSegConfig& config,
                   MainConfig& main_config,
                   MapConfig& map_config,
                   const SegmentationConfig& segmentation_config)
    : inseg_(std::move(segments), config, main_config, map_config, segmentation_config) {}

void InSegLib::ProcessFrame(const cv::Mat& depth_map, const cv::Mat& color_map) {
    inseg_.ProcessFrame(depth_map, color_map);
}

void InSegLib::InitializeMap(const Eigen::Matrix4f& pose) {
    initialized = true;
    inseg_.InitializeMap(pose);
}

void InSegLib::InitializeCamera(const Eigen::Matrix3d& intrinsics,
                                const int image_width,
                                const int image_height) {
    CamParams params;
    params.fx = static_cast<float>(intrinsics(0, 0));
    params.fy = static_cast<float>(intrinsics(1, 1));
    params.cx = static_cast<float>(intrinsics(0, 2));
    params.cy = static_cast<float>(intrinsics(1, 2));
    params.img_width = image_width;
    params.img_height = image_height;
    inseg_.InitializeCamera(params);
}

void InSegLib::Reset() {
    initialized = false;
    inseg_.Reset();
}

const Eigen::Matrix4f& InSegLib::pose() const {
    return inseg_.pose();
}

void InSegLib::SetLabelMap() {
    inseg_.SetLabelMap();
}

void InSegLib::SetPropagatedLabelMap() {
    inseg_.SetPropagatedLabelMap();
}

void InSegLib::set_pose(const Eigen::Matrix4f& pose) {
    inseg_.set_pose(pose);
}

Map& InSegLib::map() {
    return inseg_.map();
}

ImagePyramid& InSegLib::image_pyramid() {
    return inseg_.image_pyramid();
}

const cv::Mat& InSegLib::GetSegmentedLabelMap() {
    const cv::Mat& frame_label_map = inseg_.GetFrameLabelMap();
    if (frame_label_map.empty()) {
        segmented_labels_.release();
        return segmented_labels_;
    }

    if (segmented_labels_.rows != frame_label_map.rows ||
        segmented_labels_.cols != frame_label_map.cols ||
        segmented_labels_.type() != CV_8UC3) {
        segmented_labels_.create(frame_label_map.rows, frame_label_map.cols, CV_8UC3);
    }

    segmented_labels_.setTo(cv::Scalar::all(0));
    for (int y = 0; y < frame_label_map.rows; ++y) {
        const uint16_t* label_row = frame_label_map.ptr<uint16_t>(y);
        cv::Vec3b* out_row = segmented_labels_.ptr<cv::Vec3b>(y);
        for (int x = 0; x < frame_label_map.cols; ++x) {
            out_row[x] = CalculateLabelColor(static_cast<int16_t>(label_row[x]));
        }
    }
    return segmented_labels_;
}

const cv::Mat& InSegLib::GetFrameLabelMap() {
    return inseg_.GetFrameLabelMap();
}

int InSegLib::GetLabelSize() {
    return inseg_.GetLabelSize();
}

}  // namespace inseg_lib
