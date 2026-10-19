#include "inseg.h"

namespace inseg_lib {

InSegReconstruction::InSegReconstruction(
    std::shared_ptr<inseg_lib::SegmentsInterface> segments,
    InSegConfig& config_in,
    MainConfig& main_config_in,
    MapConfig& map_config_in,
    const SegmentationConfig& segmentation_config)
    : map_config(map_config_in),
      config(config_in),
      main_config(main_config_in),
      segmentation_(std::make_unique<InSegSegmentation>(
          config.depth_edge_threshold,
          static_cast<float>(main_config.uncertainty_coefficient_depth),
          segmentation_config.min_region_size)),
      image_pyramid_(map_config, main_config.max_pyr_level, main_config.min_pyr_level),
      map_(std::move(segments), map_config) {
    pose_.setIdentity();
}

void InSegReconstruction::InitializeCamera(const CamParams& cparam) {
    camera_.Init(cparam, main_config.max_pyr_level, main_config.min_pyr_level);
}

void InSegReconstruction::Reset() {
    map_.Clear();
    index_map_.release();
    label_map_frame_.release();
    label_map_propagated_frame_.release();
    world_label_map_.release();
    label_num_ = 0;
    label_num_frame_ = 0;
    world_label_num_ = 0;
    label_size_list_.clear();
    instance_labels.clear();
    label_connect_confidence_list_.clear();
    world_label_match_list_ = DisjointSet();
}

const cv::Mat& InSegReconstruction::GetFrameLabelMap() {
    return image_pyramid_.GetFrameLabelMap();
}

int InSegReconstruction::GetLabelSize() const {
    return world_label_num_;
}

void InSegReconstruction::set_pose(const Eigen::Matrix4f& pose) {
    pose_ = pose;
}

}  // namespace inseg_lib
