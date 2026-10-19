#include "hot_path_bindings.h"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <Eigen/Core>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace {

cv::Mat U8MaskFromArray(const py::array &array) {
    const auto info = array.request();
    if (info.ndim != 2) {
        throw std::runtime_error("mask must be a 2D array");
    }
    const int rows = static_cast<int>(info.shape[0]);
    const int cols = static_cast<int>(info.shape[1]);
    cv::Mat mask(rows, cols, CV_8UC1, cv::Scalar(0));
    if (info.format == py::format_descriptor<bool>::format()) {
        const auto *src = static_cast<const bool *>(info.ptr);
        for (int index = 0; index < rows * cols; ++index) {
            mask.data[index] = src[index] ? 1 : 0;
        }
        return mask;
    }
    const auto casted = py::array_t<float, py::array::c_style | py::array::forcecast>(array);
    const auto casted_info = casted.request();
    const auto *src = static_cast<const float *>(casted_info.ptr);
    for (int index = 0; index < rows * cols; ++index) {
        mask.data[index] = src[index] != 0.0f ? 1 : 0;
    }
    return mask;
}

cv::Mat FloatMatFromArray(const py::array &array, int expected_rows, int expected_cols) {
    const auto casted = py::array_t<float, py::array::c_style | py::array::forcecast>(array);
    const auto info = casted.request();
    if (info.ndim != 2 || info.shape[0] != expected_rows || info.shape[1] != expected_cols) {
        throw std::runtime_error("array shape mismatch");
    }
    cv::Mat mat(expected_rows, expected_cols, CV_32FC1, info.ptr);
    return mat.clone();
}

py::array_t<float> FloatArrayFromMat(const cv::Mat &mat) {
    py::array_t<float> array({static_cast<py::ssize_t>(mat.rows), static_cast<py::ssize_t>(mat.cols)});
    auto dst = array.mutable_unchecked<2>();
    for (int row = 0; row < mat.rows; ++row) {
        const auto *src = mat.ptr<float>(row);
        for (int col = 0; col < mat.cols; ++col) {
            dst(row, col) = src[col];
        }
    }
    return array;
}

py::array_t<uint8_t> U8ArrayFromMat(const cv::Mat &mat) {
    py::array_t<uint8_t> array({static_cast<py::ssize_t>(mat.rows), static_cast<py::ssize_t>(mat.cols)});
    auto dst = array.mutable_unchecked<2>();
    for (int row = 0; row < mat.rows; ++row) {
        const auto *src = mat.ptr<uint8_t>(row);
        for (int col = 0; col < mat.cols; ++col) {
            dst(row, col) = src[col];
        }
    }
    return array;
}

py::array_t<float> FloatVectorArray(const std::vector<float> &values) {
    py::array_t<float> array(static_cast<py::ssize_t>(values.size()));
    if (!values.empty()) {
        std::memcpy(array.mutable_data(), values.data(), values.size() * sizeof(float));
    }
    return array;
}

py::array_t<float> XYArray(const std::vector<float> &values) {
    const py::ssize_t rows = static_cast<py::ssize_t>(values.size() / 2);
    py::array_t<float> array({rows, static_cast<py::ssize_t>(2)});
    if (!values.empty()) {
        std::memcpy(array.mutable_data(), values.data(), values.size() * sizeof(float));
    }
    return array;
}

py::array_t<float> XYZArray(const std::vector<Eigen::Vector3f> &values) {
    py::array_t<float> array({static_cast<py::ssize_t>(values.size()), static_cast<py::ssize_t>(3)});
    auto out = array.mutable_unchecked<2>();
    for (py::ssize_t index = 0; index < static_cast<py::ssize_t>(values.size()); ++index) {
        out(index, 0) = values[static_cast<size_t>(index)].x();
        out(index, 1) = values[static_cast<size_t>(index)].y();
        out(index, 2) = values[static_cast<size_t>(index)].z();
    }
    return array;
}

std::vector<Eigen::Vector3f> Vec3ListFromArray(const py::array_t<float, py::array::c_style | py::array::forcecast> &array) {
    const auto info = array.request();
    if (info.size == 0) return {};
    if (info.ndim != 2 || info.shape[1] != 3) {
        throw std::runtime_error("points_world must be an Nx3 float array");
    }
    const auto *ptr = static_cast<const float *>(info.ptr);
    std::vector<Eigen::Vector3f> points;
    points.reserve(static_cast<size_t>(info.shape[0]));
    for (py::ssize_t index = 0; index < info.shape[0]; ++index) {
        points.emplace_back(ptr[index * 3 + 0], ptr[index * 3 + 1], ptr[index * 3 + 2]);
    }
    return points;
}

std::vector<float> ScoreVectorFromArray(
    const py::array_t<float, py::array::c_style | py::array::forcecast> &array,
    size_t limit
) {
    const auto info = array.request();
    const size_t count = std::min(static_cast<size_t>(info.size), limit);
    const auto *ptr = static_cast<const float *>(info.ptr);
    return std::vector<float>(ptr, ptr + count);
}

py::object OptionalCellObject(int row, int col) {
    return py::make_tuple(row, col);
}

bool ParseCell(const py::object &cell, int *row, int *col) {
    if (cell.is_none()) return false;
    const auto tuple = py::cast<py::tuple>(cell);
    if (tuple.size() < 2) return false;
    *row = py::cast<int>(tuple[0]);
    *col = py::cast<int>(tuple[1]);
    return true;
}

void SetPatchFloat(cv::Mat &mat, int row, int col, int radius_cells, float value) {
    const int radius = std::max(0, radius_cells);
    const int row_min = std::max(0, row - radius);
    const int row_max = std::min(mat.rows, row + radius + 1);
    const int col_min = std::max(0, col - radius);
    const int col_max = std::min(mat.cols, col + radius + 1);
    for (int r = row_min; r < row_max; ++r) {
        auto *ptr = mat.ptr<float>(r);
        for (int c = col_min; c < col_max; ++c) {
            ptr[c] = value;
        }
    }
}

py::dict BuildCostMap(
    const py::array &free_mask,
    const py::array &occupied_mask,
    float resolution_m,
    float agent_radius_m,
    float clearance_margin_m,
    float clearance_penalty,
    float collision_penalty,
    float max_cost,
    float min_speed,
    float goal_relaxation_radius_m,
    const py::object &collision_mask,
    const py::object &agent_cell,
    const py::object &goal_cell,
    const py::object &goal_clearance_margin_m
) {
    const cv::Mat free = U8MaskFromArray(free_mask);
    const cv::Mat occupied = U8MaskFromArray(occupied_mask);
    if (free.size() != occupied.size()) {
        throw std::runtime_error("free_mask and occupied_mask must have the same shape");
    }

    const int rows = free.rows;
    const int cols = free.cols;
    const float resolution = std::max(1e-6f, resolution_m);
    cv::Mat blocked(rows, cols, CV_8UC1, cv::Scalar(0));
    cv::Mat traversible(rows, cols, CV_8UC1, cv::Scalar(0));
    for (int row = 0; row < rows; ++row) {
        const auto *free_ptr = free.ptr<uint8_t>(row);
        const auto *occ_ptr = occupied.ptr<uint8_t>(row);
        auto *blocked_ptr = blocked.ptr<uint8_t>(row);
        auto *trav_ptr = traversible.ptr<uint8_t>(row);
        for (int col = 0; col < cols; ++col) {
            const bool is_blocked = occ_ptr[col] != 0 || free_ptr[col] == 0;
            blocked_ptr[col] = is_blocked ? 1 : 0;
            trav_ptr[col] = is_blocked ? 0 : 1;
        }
    }

    cv::Mat clearance_m(rows, cols, CV_32FC1, cv::Scalar(0.0f));
    if (!blocked.empty() && cv::countNonZero(blocked) == 0) {
        clearance_m.setTo(static_cast<float>(std::hypot(rows, cols)) * resolution);
    } else if (!blocked.empty()) {
        cv::distanceTransform(traversible, clearance_m, cv::DIST_L2, 5);
        clearance_m *= resolution;
    }

    const float base_margin = std::max(0.0f, clearance_margin_m);
    const float base_required = std::max(resolution, agent_radius_m + base_margin);
    cv::Mat required(rows, cols, CV_32FC1, cv::Scalar(base_required));
    int goal_row = 0;
    int goal_col = 0;
    float goal_relaxed_margin = 0.0f;
    const bool has_goal = ParseCell(goal_cell, &goal_row, &goal_col)
        && !goal_clearance_margin_m.is_none()
        && goal_row >= 0 && goal_row < rows && goal_col >= 0 && goal_col < cols;
    if (has_goal) {
        goal_relaxed_margin = std::max(0.0f, py::cast<float>(goal_clearance_margin_m));
        if (goal_relaxed_margin < base_margin) {
            const float relaxed_required = std::max(resolution, agent_radius_m + goal_relaxed_margin);
            const int radius_cells = std::max(1, static_cast<int>(std::ceil(goal_relaxation_radius_m / resolution)));
            const int row_min = std::max(0, goal_row - radius_cells);
            const int row_max = std::min(rows, goal_row + radius_cells + 1);
            const int col_min = std::max(0, goal_col - radius_cells);
            const int col_max = std::min(cols, goal_col + radius_cells + 1);
            for (int row = row_min; row < row_max; ++row) {
                auto *ptr = required.ptr<float>(row);
                for (int col = col_min; col < col_max; ++col) {
                    const float distance = std::hypot(static_cast<float>(row - goal_row), static_cast<float>(col - goal_col));
                    const float weight = std::clamp(1.0f - distance / std::max(1.0f, static_cast<float>(radius_cells)), 0.0f, 1.0f);
                    ptr[col] = ptr[col] * (1.0f - weight) + relaxed_required * weight;
                }
            }
        }
    }

    cv::Mat cost(rows, cols, CV_32FC1, cv::Scalar(1.0f));
    for (int row = 0; row < rows; ++row) {
        const auto *clearance_ptr = clearance_m.ptr<float>(row);
        const auto *required_ptr = required.ptr<float>(row);
        auto *cost_ptr = cost.ptr<float>(row);
        for (int col = 0; col < cols; ++col) {
            const float denominator = std::max(required_ptr[col], resolution);
            const float deficiency = std::clamp((required_ptr[col] - clearance_ptr[col]) / denominator, 0.0f, 1.0f);
            cost_ptr[col] += clearance_penalty * deficiency * deficiency;
        }
    }

    int collision_cells = 0;
    if (!collision_mask.is_none()) {
        const cv::Mat collision = FloatMatFromArray(py::cast<py::array>(collision_mask), rows, cols);
        for (int row = 0; row < rows; ++row) {
            const auto *collision_ptr = collision.ptr<float>(row);
            auto *cost_ptr = cost.ptr<float>(row);
            for (int col = 0; col < cols; ++col) {
                const float value = std::clamp(collision_ptr[col], 0.0f, 1.0f);
                if (value > 0.0f) ++collision_cells;
                cost_ptr[col] += collision_penalty * value;
            }
        }
    }

    int agent_row = 0;
    int agent_col = 0;
    if (ParseCell(agent_cell, &agent_row, &agent_col)) {
        SetPatchFloat(cost, agent_row, agent_col, 1, 1.0f);
    }

    const float clipped_max_cost = std::max(1.0f, max_cost);
    cv::Mat speed(rows, cols, CV_32FC1, cv::Scalar(0.0f));
    int traversible_cells = 0;
    float min_cost_value = std::numeric_limits<float>::infinity();
    float max_cost_value = -std::numeric_limits<float>::infinity();
    for (int row = 0; row < rows; ++row) {
        const auto *trav_ptr = traversible.ptr<uint8_t>(row);
        auto *cost_ptr = cost.ptr<float>(row);
        auto *speed_ptr = speed.ptr<float>(row);
        for (int col = 0; col < cols; ++col) {
            cost_ptr[col] = std::clamp(cost_ptr[col], 1.0f, clipped_max_cost);
            if (trav_ptr[col] == 0) continue;
            ++traversible_cells;
            min_cost_value = std::min(min_cost_value, cost_ptr[col]);
            max_cost_value = std::max(max_cost_value, cost_ptr[col]);
            speed_ptr[col] = std::max(min_speed, 1.0f / std::max(cost_ptr[col], 1e-6f));
        }
    }
    const int blocked_cells = rows * cols - traversible_cells;

    py::dict diagnostics;
    diagnostics["traversible_cells"] = traversible_cells;
    diagnostics["blocked_cells"] = blocked_cells;
    diagnostics["collision_penalty_cells"] = collision_cells;
    diagnostics["min_cost"] = traversible_cells > 0 ? py::cast(min_cost_value) : py::cast(std::numeric_limits<float>::infinity());
    diagnostics["max_cost"] = traversible_cells > 0 ? py::cast(max_cost_value) : py::cast(std::numeric_limits<float>::infinity());
    if (has_goal && !goal_clearance_margin_m.is_none()) {
        diagnostics["goal_clearance_margin_m"] = goal_relaxed_margin;
    } else {
        diagnostics["goal_clearance_margin_m"] = py::none();
    }

    py::dict result;
    result["traversible"] = U8ArrayFromMat(traversible);
    result["cost"] = FloatArrayFromMat(cost);
    result["speed"] = FloatArrayFromMat(speed);
    result["clearance_m"] = FloatArrayFromMat(clearance_m);
    result["diagnostics"] = diagnostics;
    return result;
}

py::tuple ProjectPixelsToWorld(
    const py::array_t<float, py::array::c_style | py::array::forcecast> &xy,
    const py::array_t<float, py::array::c_style | py::array::forcecast> &depth,
    float fx,
    float fy,
    float cx,
    float cy,
    const py::array_t<float, py::array::c_style | py::array::forcecast> &world_from_camera_cv,
    float min_depth_m,
    float max_depth_m
) {
    const auto xy_info = xy.request();
    if (xy_info.size == 0) {
        return py::make_tuple(
            py::array_t<float>(std::vector<py::ssize_t>{0, 3}),
            py::array_t<bool>(std::vector<py::ssize_t>{0})
        );
    }
    if (xy_info.ndim != 2 || xy_info.shape[1] != 2) {
        throw std::runtime_error("xy must be an Nx2 float array");
    }
    const auto depth_info = depth.request();
    if (depth_info.ndim != 2) {
        throw std::runtime_error("depth must be a 2D float array");
    }
    const auto transform_info = world_from_camera_cv.request();
    if (transform_info.ndim != 2 || transform_info.shape[0] != 4 || transform_info.shape[1] != 4) {
        throw std::runtime_error("world_from_camera_cv must be a 4x4 float array");
    }

    const auto *xy_ptr = static_cast<const float *>(xy_info.ptr);
    const auto *depth_ptr = static_cast<const float *>(depth_info.ptr);
    const auto *transform_ptr = static_cast<const float *>(transform_info.ptr);
    Eigen::Matrix<float, 4, 4, Eigen::RowMajor> transform =
        Eigen::Map<const Eigen::Matrix<float, 4, 4, Eigen::RowMajor>>(transform_ptr);
    const int height = static_cast<int>(depth_info.shape[0]);
    const int width = static_cast<int>(depth_info.shape[1]);
    const py::ssize_t count = xy_info.shape[0];
    py::array_t<bool> valid_array(count);
    auto valid = valid_array.mutable_unchecked<1>();
    std::vector<Eigen::Vector3f> points;
    points.reserve(static_cast<size_t>(count));
    const bool valid_intrinsics = fx > 0.0f && fy > 0.0f;

    for (py::ssize_t index = 0; index < count; ++index) {
        valid(index) = false;
        const float x_px = xy_ptr[index * 2 + 0];
        const float y_px = xy_ptr[index * 2 + 1];
        if (!std::isfinite(x_px) || !std::isfinite(y_px) || !valid_intrinsics) continue;
        const int u = static_cast<int>(std::lround(x_px));
        const int v = static_cast<int>(std::lround(y_px));
        if (u < 0 || u >= width || v < 0 || v >= height) continue;
        const float z = depth_ptr[v * width + u];
        if (!std::isfinite(z) || z <= min_depth_m || z > max_depth_m) continue;
        const float x = (x_px - cx) / fx * z;
        const float y = (y_px - cy) / fy * z;
        const Eigen::Vector4f camera_point(x, y, z, 1.0f);
        const Eigen::Vector3f world_point = (transform * camera_point).head<3>();
        if (!std::isfinite(world_point.x()) || !std::isfinite(world_point.y()) || !std::isfinite(world_point.z())) continue;
        valid(index) = true;
        points.push_back(world_point);
    }
    return py::make_tuple(XYZArray(points), valid_array);
}

std::vector<int32_t> DbscanLabelsVector(const std::vector<Eigen::Vector3f> &points, float eps_m, int min_samples) {
    const size_t count = points.size();
    std::vector<int32_t> labels(count, -1);
    if (count == 0) return labels;

    const float eps = std::max(0.0f, eps_m);
    const float eps_sq = eps * eps;
    std::vector<std::vector<int>> neighbors(count);
    for (size_t i = 0; i < count; ++i) {
        for (size_t j = 0; j < count; ++j) {
            const float dx = points[i].x() - points[j].x();
            const float dz = points[i].z() - points[j].z();
            if (dx * dx + dz * dz <= eps_sq) {
                neighbors[i].push_back(static_cast<int>(j));
            }
        }
    }

    std::vector<uint8_t> visited(count, 0);
    int cluster_id = 0;
    const int min_pts = std::max(1, min_samples);
    for (size_t index = 0; index < count; ++index) {
        if (visited[index]) continue;
        visited[index] = 1;
        if (static_cast<int>(neighbors[index].size()) < min_pts) {
            continue;
        }

        labels[index] = cluster_id;
        std::vector<int> seeds = neighbors[index];
        std::vector<uint8_t> queued(count, 0);
        for (int seed : seeds) {
            if (seed >= 0 && static_cast<size_t>(seed) < count) queued[static_cast<size_t>(seed)] = 1;
        }
        size_t cursor = 0;
        while (cursor < seeds.size()) {
            const int candidate = seeds[cursor++];
            if (candidate < 0 || static_cast<size_t>(candidate) >= count) continue;
            const size_t candidate_index = static_cast<size_t>(candidate);
            if (!visited[candidate_index]) {
                visited[candidate_index] = 1;
                if (static_cast<int>(neighbors[candidate_index].size()) >= min_pts) {
                    for (int neighbor : neighbors[candidate_index]) {
                        if (neighbor < 0 || static_cast<size_t>(neighbor) >= count) continue;
                        if (queued[static_cast<size_t>(neighbor)]) continue;
                        queued[static_cast<size_t>(neighbor)] = 1;
                        seeds.push_back(neighbor);
                    }
                }
            }
            if (labels[candidate_index] < 0) {
                labels[candidate_index] = cluster_id;
            }
        }
        cluster_id += 1;
    }
    return labels;
}

py::array_t<int32_t> DbscanLabelsXz(
    const py::array_t<float, py::array::c_style | py::array::forcecast> &points_world,
    float eps_m,
    int min_samples
) {
    const auto points = Vec3ListFromArray(points_world);
    const auto labels = DbscanLabelsVector(points, eps_m, min_samples);
    py::array_t<int32_t> array(static_cast<py::ssize_t>(labels.size()));
    if (!labels.empty()) {
        std::memcpy(array.mutable_data(), labels.data(), labels.size() * sizeof(int32_t));
    }
    return array;
}

py::dict EmptyMatches(int raw_count, int coarse_count) {
    py::dict result;
    result["goal_xy"] = py::array_t<float>(std::vector<py::ssize_t>{0, 2});
    result["current_xy"] = py::array_t<float>(std::vector<py::ssize_t>{0, 2});
    result["fine_conf"] = py::array_t<float>(static_cast<py::ssize_t>(0));
    result["coarse_sim"] = py::array_t<float>(static_cast<py::ssize_t>(0));
    result["raw_match_count"] = raw_count;
    result["coarse_filtered_count"] = coarse_count;
    result["fine_conf_filtered_count"] = 0;
    return result;
}

py::dict FilterMatchScores(
    const py::array_t<float, py::array::c_style | py::array::forcecast> &goal_xy,
    const py::array_t<float, py::array::c_style | py::array::forcecast> &current_xy,
    const py::array_t<float, py::array::c_style | py::array::forcecast> &fine_conf,
    const py::array_t<float, py::array::c_style | py::array::forcecast> &coarse_sim,
    float fine_conf_threshold,
    float min_coarse_sim
) {
    const auto goal_info = goal_xy.request();
    const auto current_info = current_xy.request();
    const auto fine_info = fine_conf.request();
    const auto coarse_info = coarse_sim.request();
    const size_t goal_count = goal_info.size / 2;
    const size_t current_count = current_info.size / 2;
    const size_t count = std::min({goal_count, current_count, static_cast<size_t>(fine_info.size), static_cast<size_t>(coarse_info.size)});
    const auto *goal_ptr = static_cast<const float *>(goal_info.ptr);
    const auto *current_ptr = static_cast<const float *>(current_info.ptr);
    const auto *fine_ptr = static_cast<const float *>(fine_info.ptr);
    const auto *coarse_ptr = static_cast<const float *>(coarse_info.ptr);

    std::vector<float> out_goal;
    std::vector<float> out_current;
    std::vector<float> out_fine;
    std::vector<float> out_coarse;
    out_goal.reserve(count * 2);
    out_current.reserve(count * 2);
    out_fine.reserve(count);
    out_coarse.reserve(count);
    int coarse_filtered = 0;
    for (size_t index = 0; index < count; ++index) {
        const float gx = goal_ptr[index * 2 + 0];
        const float gy = goal_ptr[index * 2 + 1];
        const float cx = current_ptr[index * 2 + 0];
        const float cy = current_ptr[index * 2 + 1];
        const float fine = fine_ptr[index];
        const float coarse = coarse_ptr[index];
        const bool finite = std::isfinite(gx) && std::isfinite(gy) && std::isfinite(cx) && std::isfinite(cy)
            && std::isfinite(fine) && std::isfinite(coarse);
        if (!finite || coarse < min_coarse_sim) continue;
        ++coarse_filtered;
        if (fine < fine_conf_threshold) continue;
        out_goal.push_back(gx);
        out_goal.push_back(gy);
        out_current.push_back(cx);
        out_current.push_back(cy);
        out_fine.push_back(fine);
        out_coarse.push_back(coarse);
    }

    py::dict result;
    result["goal_xy"] = XYArray(out_goal);
    result["current_xy"] = XYArray(out_current);
    result["fine_conf"] = FloatVectorArray(out_fine);
    result["coarse_sim"] = FloatVectorArray(out_coarse);
    result["raw_match_count"] = static_cast<int>(count);
    result["coarse_filtered_count"] = coarse_filtered;
    result["fine_conf_filtered_count"] = static_cast<int>(out_fine.size());
    return result;
}

py::tuple FilterGoalImageRegion(
    const py::array_t<float, py::array::c_style | py::array::forcecast> &goal_xy,
    const py::array_t<float, py::array::c_style | py::array::forcecast> &current_xy,
    const py::array_t<float, py::array::c_style | py::array::forcecast> &fine_conf,
    const py::array_t<float, py::array::c_style | py::array::forcecast> &coarse_sim,
    int raw_match_count,
    int coarse_filtered_count,
    int goal_height,
    int goal_width,
    float min_x_frac,
    float max_x_frac,
    float min_y_frac,
    float max_y_frac
) {
    const auto goal_info = goal_xy.request();
    const auto current_info = current_xy.request();
    const auto fine_info = fine_conf.request();
    const auto coarse_info = coarse_sim.request();
    const size_t count = std::min({
        static_cast<size_t>(goal_info.size / 2),
        static_cast<size_t>(current_info.size / 2),
        static_cast<size_t>(fine_info.size),
        static_cast<size_t>(coarse_info.size),
    });
    if (goal_width <= 0 || goal_height <= 0 || count == 0) {
        py::dict result = FilterMatchScores(goal_xy, current_xy, fine_conf, coarse_sim, -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity());
        result["raw_match_count"] = raw_match_count;
        result["coarse_filtered_count"] = coarse_filtered_count;
        result["fine_conf_filtered_count"] = static_cast<int>(count);
        return py::make_tuple(result, 0);
    }

    const float min_x = std::clamp(min_x_frac, 0.0f, 1.0f) * static_cast<float>(goal_width);
    const float max_x = std::clamp(max_x_frac, 0.0f, 1.0f) * static_cast<float>(goal_width);
    const float min_y = std::clamp(min_y_frac, 0.0f, 1.0f) * static_cast<float>(goal_height);
    const float max_y = std::clamp(max_y_frac, 0.0f, 1.0f) * static_cast<float>(goal_height);
    if (min_x <= 0.0f && max_x >= static_cast<float>(goal_width) && min_y <= 0.0f && max_y >= static_cast<float>(goal_height)) {
        py::dict result = FilterMatchScores(goal_xy, current_xy, fine_conf, coarse_sim, -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity());
        result["raw_match_count"] = raw_match_count;
        result["coarse_filtered_count"] = coarse_filtered_count;
        result["fine_conf_filtered_count"] = static_cast<int>(count);
        return py::make_tuple(result, 0);
    }
    if (max_x <= min_x || max_y <= min_y) {
        return py::make_tuple(EmptyMatches(raw_match_count, coarse_filtered_count), static_cast<int>(count));
    }

    const auto *goal_ptr = static_cast<const float *>(goal_info.ptr);
    const auto *current_ptr = static_cast<const float *>(current_info.ptr);
    const auto *fine_ptr = static_cast<const float *>(fine_info.ptr);
    const auto *coarse_ptr = static_cast<const float *>(coarse_info.ptr);
    std::vector<float> out_goal;
    std::vector<float> out_current;
    std::vector<float> out_fine;
    std::vector<float> out_coarse;
    int keep_count = 0;
    for (size_t index = 0; index < count; ++index) {
        const float gx = goal_ptr[index * 2 + 0];
        const float gy = goal_ptr[index * 2 + 1];
        const bool keep = gx >= min_x && gx <= max_x && gy >= min_y && gy <= max_y;
        if (!keep) continue;
        ++keep_count;
        out_goal.push_back(gx);
        out_goal.push_back(gy);
        out_current.push_back(current_ptr[index * 2 + 0]);
        out_current.push_back(current_ptr[index * 2 + 1]);
        out_fine.push_back(fine_ptr[index]);
        out_coarse.push_back(coarse_ptr[index]);
    }

    py::dict result;
    result["goal_xy"] = XYArray(out_goal);
    result["current_xy"] = XYArray(out_current);
    result["fine_conf"] = FloatVectorArray(out_fine);
    result["coarse_sim"] = FloatVectorArray(out_coarse);
    result["raw_match_count"] = raw_match_count;
    result["coarse_filtered_count"] = coarse_filtered_count;
    result["fine_conf_filtered_count"] = keep_count;
    return py::make_tuple(result, static_cast<int>(count) - keep_count);
}

py::object SelectConfidenceCluster(
    const py::array_t<float, py::array::c_style | py::array::forcecast> &points_world,
    const py::array_t<float, py::array::c_style | py::array::forcecast> &fine_conf,
    const py::array_t<float, py::array::c_style | py::array::forcecast> &coarse_sim,
    float eps_m,
    int min_samples
) {
    const auto points = Vec3ListFromArray(points_world);
    const size_t count = points.size();
    const auto fine = ScoreVectorFromArray(fine_conf, count);
    const auto coarse = ScoreVectorFromArray(coarse_sim, count);
    if (count == 0 || fine.size() != count || coarse.size() != count) {
        return py::none();
    }
    const auto labels = DbscanLabelsVector(points, eps_m, std::max(1, min_samples));
    int best_label = -1;
    float best_confidence_sum = -std::numeric_limits<float>::infinity();
    int best_size = -1;
    for (int32_t label : labels) {
        if (label < 0) continue;
        float confidence_sum = 0.0f;
        int size = 0;
        for (size_t index = 0; index < count; ++index) {
            if (labels[index] != label) continue;
            confidence_sum += fine[index];
            ++size;
        }
        if (confidence_sum > best_confidence_sum || (confidence_sum == best_confidence_sum && size > best_size)) {
            best_confidence_sum = confidence_sum;
            best_size = size;
            best_label = label;
        }
    }
    if (best_label < 0 || best_size <= 0) {
        return py::none();
    }

    py::array_t<uint8_t> mask_array(static_cast<py::ssize_t>(count));
    auto mask = mask_array.mutable_unchecked<1>();
    float confidence_sum = 0.0f;
    float coarse_sum = 0.0f;
    int size = 0;
    for (py::ssize_t index = 0; index < static_cast<py::ssize_t>(count); ++index) {
        const bool selected = labels[static_cast<size_t>(index)] == best_label;
        mask(index) = selected ? 1 : 0;
        if (!selected) continue;
        confidence_sum += fine[static_cast<size_t>(index)];
        coarse_sum += coarse[static_cast<size_t>(index)];
        ++size;
    }

    py::dict result;
    result["mask"] = mask_array;
    result["confidence_sum"] = confidence_sum;
    result["mean_confidence"] = confidence_sum / std::max(1, size);
    result["mean_coarse_sim"] = coarse_sum / std::max(1, size);
    result["size"] = size;
    return result;
}

}  // namespace

namespace sgf_pybind {

void RegisterHotPathBindings(py::module_ &module) {
    module.def(
        "build_cost_map",
        &BuildCostMap,
        py::arg("free_mask"),
        py::arg("occupied_mask"),
        py::arg("resolution_m"),
        py::arg("agent_radius_m"),
        py::arg("clearance_margin_m") = 0.05f,
        py::arg("clearance_penalty") = 9.0f,
        py::arg("collision_penalty") = 12.0f,
        py::arg("max_cost") = 64.0f,
        py::arg("min_speed") = 1e-3f,
        py::arg("goal_relaxation_radius_m") = 0.55f,
        py::arg("collision_mask") = py::none(),
        py::arg("agent_cell") = py::none(),
        py::arg("goal_cell") = py::none(),
        py::arg("goal_clearance_margin_m") = py::none()
    );
    module.def(
        "project_pixels_to_world",
        &ProjectPixelsToWorld,
        py::arg("xy"),
        py::arg("depth"),
        py::arg("fx"),
        py::arg("fy"),
        py::arg("cx"),
        py::arg("cy"),
        py::arg("world_from_camera_cv"),
        py::arg("min_depth_m") = 0.0f,
        py::arg("max_depth_m") = 10.0f
    );
    module.def(
        "dbscan_labels_xz",
        &DbscanLabelsXz,
        py::arg("points_world"),
        py::arg("eps_m"),
        py::arg("min_samples")
    );
    module.def(
        "filter_match_scores",
        &FilterMatchScores,
        py::arg("goal_xy"),
        py::arg("current_xy"),
        py::arg("fine_conf"),
        py::arg("coarse_sim"),
        py::arg("fine_conf_threshold"),
        py::arg("min_coarse_sim")
    );
    module.def(
        "filter_goal_image_region",
        &FilterGoalImageRegion,
        py::arg("goal_xy"),
        py::arg("current_xy"),
        py::arg("fine_conf"),
        py::arg("coarse_sim"),
        py::arg("raw_match_count"),
        py::arg("coarse_filtered_count"),
        py::arg("goal_height"),
        py::arg("goal_width"),
        py::arg("min_x_frac"),
        py::arg("max_x_frac"),
        py::arg("min_y_frac"),
        py::arg("max_y_frac")
    );
    module.def(
        "select_confidence_cluster",
        &SelectConfidenceCluster,
        py::arg("points_world"),
        py::arg("fine_conf"),
        py::arg("coarse_sim"),
        py::arg("eps_m"),
        py::arg("min_samples")
    );
}

}  // namespace sgf_pybind
