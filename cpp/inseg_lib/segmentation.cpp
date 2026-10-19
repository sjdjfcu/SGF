#include "segmentation.h"

#include <cmath>
#include <map>

#include "main_config.h"

namespace inseg_lib {

namespace {

inline const cv::Vec3f& ReadVec3(const cv::Mat& mat, int y, int x) {
    return mat.ptr<cv::Vec3f>(y)[x];
}

inline uint16_t& ReadLabelRef(cv::Mat* mat, int y, int x) {
    return mat->ptr<uint16_t>(y)[x];
}

inline const uint16_t& ReadLabel(const cv::Mat& mat, int y, int x) {
    return mat.ptr<uint16_t>(y)[x];
}

inline float AbsProjectedDistance(const cv::Vec3f& point,
                                  const cv::Vec3f& normal,
                                  const cv::Vec3f& other) {
    return std::fabs((other[2] - point[2]) * normal[2] +
                     (other[1] - point[1]) * normal[1] +
                     (other[0] - point[0]) * normal[0]);
}

}  // namespace

float InSegSegmentation::CalculateAngleMin(const Eigen::Vector3f& point,
                                           const Eigen::Vector3f& normal,
                                           const Eigen::Vector3f& other_vert,
                                           const Eigen::Vector3f& other_normal,
                                           float min_dot) {
    if (point.z() <= 0.0f || normal.z() == 0.0f ||
        other_vert.z() <= 0.0f || other_normal.z() == 0.0f) {
        return 0.0f;
    }

    float candidate = 1.0f;
    if ((point - other_vert).dot(normal) < 0.1f) {
        candidate = normal.dot(other_normal);
    }
    return std::min(candidate, min_dot);
}

void InSegSegmentation::CalculateGeometricalEdge(const cv::Mat& point_cloud,
                                                 const cv::Mat& normal_map) {
    normal_edge_.create(normal_map.rows, normal_map.cols, CV_8UC1);
    normal_edge_.setTo(0);

    for (int y = 1; y < normal_map.rows - 1; ++y) {
        uint8_t* edge_row = normal_edge_.ptr<uint8_t>(y);
        for (int x = 1; x < normal_map.cols - 1; ++x) {
            const cv::Vec3f& point_cv = ReadVec3(point_cloud, y, x);
            const cv::Vec3f& normal_cv = ReadVec3(normal_map, y, x);
            if (point_cv[2] <= 0.0f || normal_cv[2] == 0.0f) {
                continue;
            }

            const Eigen::Vector3f point(point_cv[0], point_cv[1], point_cv[2]);
            const Eigen::Vector3f normal(normal_cv[0], normal_cv[1], normal_cv[2]);

            float min_dot = 1.0f;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) {
                        continue;
                    }
                    const cv::Vec3f& other_point_cv = ReadVec3(point_cloud, y + dy, x + dx);
                    const cv::Vec3f& other_normal_cv = ReadVec3(normal_map, y + dy, x + dx);
                    min_dot = CalculateAngleMin(
                        point,
                        normal,
                        Eigen::Vector3f(other_point_cv[0], other_point_cv[1], other_point_cv[2]),
                        Eigen::Vector3f(other_normal_cv[0], other_normal_cv[1], other_normal_cv[2]),
                        min_dot);
                }
            }

            const float threshold =
                0.5f * uncertainty_coefficient_depth_ * point_cv[2] * point_cv[2];
            bool geometrically_close = true;
            for (int dy = -1; dy <= 1 && geometrically_close; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) {
                        continue;
                    }
                    const cv::Vec3f& other = ReadVec3(point_cloud, y + dy, x + dx);
                    if (AbsProjectedDistance(point_cv, normal_cv, other) > threshold) {
                        geometrically_close = false;
                        break;
                    }
                }
            }

            if (geometrically_close && min_dot > depth_edge_threshold_) {
                edge_row[x] = static_cast<uint8_t>(kNoEdge);
            }
        }
    }
}

void InSegSegmentation::Segment(cv::Mat* label_map) {
    label_map->create(normal_edge_.rows, normal_edge_.cols, CV_16UC1);
    label_map->setTo(0);

    const int rows = normal_edge_.rows;
    const int cols = normal_edge_.cols;
    labels_.Resize(rows * cols);

    int flat_base = 0;
    for (int y = 0; y < rows; ++y) {
        const uint8_t* edge_row = normal_edge_.ptr<uint8_t>(y);
        for (int x = 0; x < cols; ++x) {
            const int flat = flat_base + x;
            if (edge_row[x] == 0) {
                labels_.UnionSet(flat, 0);
                continue;
            }
            if (x > 0 && edge_row[x - 1] != 0) {
                labels_.UnionSet(flat, flat - 1);
            }
            if (y > 0 && normal_edge_.ptr<uint8_t>(y - 1)[x] != 0) {
                labels_.UnionSet(flat, flat - cols);
            }
        }
        flat_base += cols;
    }
}

void InSegSegmentation::RegionGrowing(const cv::Mat& point_cloud,
                                      const cv::Mat& normal_map,
                                      cv::Mat* label_map) {
    cv::Mat label_map_copy;
    label_map->copyTo(label_map_copy);

    for (int y = 1; y < label_map->rows - 1; ++y) {
        for (int x = 1; x < label_map->cols - 1; ++x) {
            const cv::Vec3f& point = ReadVec3(point_cloud, y, x);
            const cv::Vec3f& normal = ReadVec3(normal_map, y, x);
            if (point[2] <= 0.0f || normal[2] == 0.0f || ReadLabel(label_map_copy, y, x) != 0) {
                continue;
            }

            const float threshold =
                0.25f * uncertainty_coefficient_depth_ * point[2] * point[2];

            float best_dot = 0.0f;
            uint16_t best_label = 0;

            const int offsets[4][2] = {{0, -1}, {0, 1}, {-1, 0}, {1, 0}};
            for (const auto& offset : offsets) {
                const int ny = y + offset[0];
                const int nx = x + offset[1];
                const uint16_t candidate_label = ReadLabel(label_map_copy, ny, nx);
                if (candidate_label == 0) {
                    continue;
                }

                const cv::Vec3f& other_point = ReadVec3(point_cloud, ny, nx);
                const cv::Vec3f& other_normal = ReadVec3(normal_map, ny, nx);
                if (other_point[2] <= 0.0f || other_normal[2] >= 0.0f) {
                    continue;
                }
                if (AbsProjectedDistance(point, normal, other_point) > threshold) {
                    continue;
                }

                const float dot = normal[0] * other_normal[0] +
                                  normal[1] * other_normal[1] +
                                  normal[2] * other_normal[2];
                if (dot > best_dot) {
                    best_dot = dot;
                    best_label = candidate_label;
                }
            }

            if (best_dot >= 0.1f) {
                ReadLabelRef(label_map, y, x) = best_label;
            } else {
                ReadLabelRef(label_map, y, x) = 0;
            }
        }
    }
}

void InSegSegmentation::CreateEdges(const cv::Mat& normal_edge,
                                    const cv::Mat& point_cloud,
                                    const cv::Mat& normal_map) {
    (void)normal_edge;
    (void)point_cloud;
    (void)normal_map;
}

float InSegSegmentation::CalculateEdgeWeight(const Eigen::Vector3f& point,
                                             const Eigen::Vector3f& normal,
                                             const Eigen::Vector3f& other_vert,
                                             const Eigen::Vector3f& other_normal) {
    float weight = 1.0f - normal.dot(other_normal);
    if ((point - other_vert).dot(normal) < 0.0f) {
        weight *= weight;
    }
    return weight;
}

int InSegSegmentation::Relabel(cv::Mat* label_map) {
    std::map<int, int> root_to_label;
    int next_label = 0;

    for (int y = 0; y < label_map->rows; ++y) {
        uint16_t* row = label_map->ptr<uint16_t>(y);
        for (int x = 0; x < label_map->cols; ++x) {
            const int index = y * label_map->cols + x;
            const int root = labels_.Find(index);
            if (labels_.GetSize(index) < min_region_size_) {
                labels_.UnionSet(root, 0);
                row[x] = 0;
                continue;
            }

            auto it = root_to_label.find(root);
            if (it == root_to_label.end()) {
                it = root_to_label.emplace(root, next_label++).first;
            }
            row[x] = static_cast<uint16_t>(it->second);
        }
    }

    return labels_.GetTotalSize();
}

int InSegSegmentation::Segment(const cv::Mat& point_cloud,
                               const cv::Mat& normal_map,
                               cv::Mat* label_map) {
    CalculateGeometricalEdge(point_cloud, normal_map);
    Segment(label_map);
    const int label_count = Relabel(label_map);
    RegionGrowing(point_cloud, normal_map, label_map);
    return label_count;
}

}  // namespace inseg_lib
