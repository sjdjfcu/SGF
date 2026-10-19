#include "navigation_map.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <queue>
#include <tuple>

#include <opencv2/imgproc.hpp>

namespace PSLAM {

namespace {

constexpr uint16_t kMaxUint16 = 65000;

void IncrementU16(cv::Mat &mat, int row, int col, int weight = 1) {
    auto &value = mat.at<uint16_t>(row, col);
    const int bumped = static_cast<int>(value) + weight;
    value = static_cast<uint16_t>(std::min<int>(bumped, kMaxUint16));
}

Eigen::Matrix4f WorldFromCameraMeters(const Eigen::Matrix4f &camera_from_world_mm) {
    Eigen::Matrix4f world_from_camera = camera_from_world_mm.inverse();
    world_from_camera.block<3, 1>(0, 3) *= 0.001f;
    return world_from_camera;
}

float DepthMetersAt(const cv::Mat &depth_mm, int row, int col) {
    if (depth_mm.type() == CV_16UC1) {
        return static_cast<float>(depth_mm.at<uint16_t>(row, col)) * 0.001f;
    }
    if (depth_mm.type() == CV_32FC1) {
        return depth_mm.at<float>(row, col);
    }
    if (depth_mm.type() == CV_64FC1) {
        return static_cast<float>(depth_mm.at<double>(row, col));
    }
    return 0.0f;
}

}  // namespace

NavigationMap::NavigationMap(NavigationMapConfig config) : mConfig(config) {
    Reset();
}

void NavigationMap::Reset() {
    mGraphOccupancyStats = {};
    mFree = cv::Mat::zeros(1, 1, CV_16UC1);
    mOccupied = cv::Mat::zeros(1, 1, CV_16UC1);
    mObserved = cv::Mat::zeros(1, 1, CV_8UC1);
    mFloorHits = cv::Mat::zeros(1, 1, CV_16UC1);
    mNonFloorHits = cv::Mat::zeros(1, 1, CV_16UC1);
    mOriginXZ.setZero();
    mFloorY = 0.0f;
    mHasFloor = false;
    mHasLastCameraCell = false;
    mLastCameraRow = 0;
    mLastCameraCol = 0;
    mFloorHistogram.clear();
    mFloorEstimate = 0.0f;
    mHasFloorEstimate = false;
}

namespace {
constexpr float kFloorHistogramBinM = 0.01f;
}

void NavigationMap::UpdateFloorEstimate(const std::vector<float> &heights_m, int min_support) {
    const float lo = mFloorY - 0.5f * mConfig.camera_height_m;
    const float hi = mFloorY + mConfig.height_min_m;
    const int bins = std::max(1, static_cast<int>(std::ceil((hi - lo) / kFloorHistogramBinM)));
    if (static_cast<int>(mFloorHistogram.size()) != bins) {
        mFloorHistogram.assign(static_cast<size_t>(bins), 0u);
    }
    for (const float y : heights_m) {
        if (!(y >= lo && y < hi)) continue;
        const int bin = std::min(bins - 1, static_cast<int>((y - lo) / kFloorHistogramBinM));
        mFloorHistogram[static_cast<size_t>(bin)] += 1u;
    }
    int peak = 0;
    for (int bin = 1; bin < bins; ++bin) {
        if (mFloorHistogram[static_cast<size_t>(bin)] > mFloorHistogram[static_cast<size_t>(peak)]) peak = bin;
    }
    if (static_cast<int>(mFloorHistogram[static_cast<size_t>(peak)]) >= std::max(1, min_support)) {
        mFloorEstimate = lo + (static_cast<float>(peak) + 0.5f) * kFloorHistogramBinM;
        mHasFloorEstimate = true;
    }
}

void NavigationMap::SetConfig(const NavigationMapConfig &config) {
    mConfig = config;
    if (mConfig.resolution_m <= 0.0f) mConfig.resolution_m = 0.05f;
    if (mConfig.depth_stride <= 0) mConfig.depth_stride = 1;
    if (mConfig.obstacle_confirm_hits <= 0) mConfig.obstacle_confirm_hits = 1;
    if (mConfig.free_confirm_hits <= 0) mConfig.free_confirm_hits = 1;
    if (mConfig.obstacle_free_margin < 0) mConfig.obstacle_free_margin = 0;
}

std::pair<int, int> NavigationMap::Shape() const {
    return {mFree.rows, mFree.cols};
}

std::pair<int, int> NavigationMap::WorldToCell(const Eigen::Vector3f &xyz_m) const {
    const int row = static_cast<int>(std::floor((xyz_m.z() - mOriginXZ.y()) / mConfig.resolution_m));
    const int col = static_cast<int>(std::floor((xyz_m.x() - mOriginXZ.x()) / mConfig.resolution_m));
    return {row, col};
}

Eigen::Vector3f NavigationMap::CellToWorld(int row, int col, float y_m) const {
    const float x = mOriginXZ.x() + (static_cast<float>(col) + 0.5f) * mConfig.resolution_m;
    const float z = mOriginXZ.y() + (static_cast<float>(row) + 0.5f) * mConfig.resolution_m;
    return Eigen::Vector3f(x, y_m, z);
}

bool NavigationMap::InBounds(int row, int col) const {
    return row >= 0 && row < mFree.rows && col >= 0 && col < mFree.cols;
}

void NavigationMap::EnsureBounds(const Eigen::Vector2f &xz_min, const Eigen::Vector2f &xz_max, float margin_m) {
    const float res = mConfig.resolution_m;
    const float need_min_x = xz_min.x() - margin_m;
    const float need_min_z = xz_min.y() - margin_m;
    const float need_max_x = xz_max.x() + margin_m;
    const float need_max_z = xz_max.y() + margin_m;

    const float origin_x = mOriginXZ.x();
    const float origin_z = mOriginXZ.y();
    const float far_x = origin_x + static_cast<float>(mFree.cols) * res;
    const float far_z = origin_z + static_cast<float>(mFree.rows) * res;

    const float grow_left = std::max(0.0f, origin_x - need_min_x);
    const float grow_right = std::max(0.0f, need_max_x - far_x);
    const float grow_top = std::max(0.0f, origin_z - need_min_z);
    const float grow_bottom = std::max(0.0f, need_max_z - far_z);
    if (grow_left == 0.0f && grow_right == 0.0f && grow_top == 0.0f && grow_bottom == 0.0f) {
        return;
    }

    const int cells_left = static_cast<int>(std::ceil(grow_left / res));
    const int cells_right = static_cast<int>(std::ceil(grow_right / res));
    const int cells_top = static_cast<int>(std::ceil(grow_top / res));
    const int cells_bottom = static_cast<int>(std::ceil(grow_bottom / res));
    const int new_rows = mFree.rows + cells_top + cells_bottom;
    const int new_cols = mFree.cols + cells_left + cells_right;

    auto grow = [&](const cv::Mat &src, int type) {
        cv::Mat dst = cv::Mat::zeros(new_rows, new_cols, type);
        src.copyTo(dst(cv::Rect(cells_left, cells_top, src.cols, src.rows)));
        return dst;
    };

    mFree = grow(mFree, CV_16UC1);
    mOccupied = grow(mOccupied, CV_16UC1);
    mObserved = grow(mObserved, CV_8UC1);
    mFloorHits = grow(mFloorHits, CV_16UC1);
    mNonFloorHits = grow(mNonFloorHits, CV_16UC1);
    mOriginXZ.x() = origin_x - static_cast<float>(cells_left) * res;
    mOriginXZ.y() = origin_z - static_cast<float>(cells_top) * res;
    mLastCameraRow += cells_top;
    mLastCameraCol += cells_left;
}

float NavigationMap::UpdateFloor(float camera_y_m) {
    if (!mHasFloor) {
        mFloorY = camera_y_m - mConfig.camera_height_m;
        mHasFloor = true;
    }
    return mFloorY;
}

std::pair<float, float> NavigationMap::HeightBand(float camera_y_m) {
    const float floor_y = UpdateFloor(camera_y_m);
    return {floor_y + mConfig.height_min_m, floor_y + mConfig.height_max_m};
}

void NavigationMap::MarkFree(int row, int col) {
    if (!InBounds(row, col)) return;
    IncrementU16(mFree, row, col, 1);
    MarkObserved(row, col);
}

void NavigationMap::MarkOccupied(int row, int col, int weight) {
    if (!InBounds(row, col)) return;
    IncrementU16(mOccupied, row, col, weight);
    MarkObserved(row, col);
}

void NavigationMap::MarkObserved(int row, int col) {
    if (!InBounds(row, col)) return;
    auto &value = mObserved.at<uint8_t>(row, col);
    value = static_cast<uint8_t>(std::min<int>(static_cast<int>(value) + 1, 255));
}

void NavigationMap::UpdateFreeSpace(
    const cv::Mat &depth_mm,
    const Eigen::Matrix4f &camera_from_world_mm,
    const CameraParameters &camera
) {
    if (depth_mm.empty() || depth_mm.channels() != 1) return;
    if (camera.fx == 0.0f || camera.fy == 0.0f) return;

    const Eigen::Matrix4f world_from_camera = WorldFromCameraMeters(camera_from_world_mm);
    const Eigen::Vector3f camera_origin = world_from_camera.block<3, 1>(0, 3);
    const auto [height_min, height_max] = HeightBand(camera_origin.y());

    struct Endpoint {
        Eigen::Vector3f world_m;
        bool height_ok = false;
        int surface = 0;  // 1 = floor, -1 = off the floor, 0 = above the band
    };
    std::vector<Endpoint> endpoints;
    endpoints.reserve((depth_mm.rows / std::max(1, mConfig.depth_stride) + 1) *
                      (depth_mm.cols / std::max(1, mConfig.depth_stride) + 1));

    Eigen::Vector2f xz_min(camera_origin.x(), camera_origin.z());
    Eigen::Vector2f xz_max(camera_origin.x(), camera_origin.z());
    const int stride = std::max(1, mConfig.depth_stride);
    for (int row = 0; row < depth_mm.rows; row += stride) {
        for (int col = 0; col < depth_mm.cols; col += stride) {
            const float z_m = DepthMetersAt(depth_mm, row, col);
            if (!std::isfinite(z_m) || z_m <= mConfig.min_range_m || z_m >= mConfig.max_range_m) {
                continue;
            }

            const float x_m = (static_cast<float>(col) - camera.cx) * z_m / camera.fx;
            const float y_m = (static_cast<float>(row) - camera.cy) * z_m / camera.fy;
            const Eigen::Vector4f camera_point(x_m, y_m, z_m, 1.0f);
            const Eigen::Vector3f world_point = (world_from_camera * camera_point).head<3>();
            const bool height_ok = world_point.y() >= height_min && world_point.y() <= height_max;
            endpoints.push_back({world_point, height_ok, 0});
            xz_min.x() = std::min(xz_min.x(), world_point.x());
            xz_min.y() = std::min(xz_min.y(), world_point.z());
            xz_max.x() = std::max(xz_max.x(), world_point.x());
            xz_max.y() = std::max(xz_max.y(), world_point.z());
        }
    }
    if (endpoints.empty()) return;

    {
        std::vector<float> heights;
        heights.reserve(endpoints.size());
        for (const auto &endpoint : endpoints) heights.push_back(endpoint.world_m.y());
        // Minimum support: one image row of samples
        UpdateFloorEstimate(heights, depth_mm.cols / stride);
    }
    if (mHasFloorEstimate) {
        const float step_m = mConfig.height_min_m;
        for (auto &endpoint : endpoints) {
            const float y = endpoint.world_m.y();
            if (std::abs(y - mFloorEstimate) < step_m) {
                endpoint.surface = 1;
            } else if (y <= height_max) {
                endpoint.surface = -1;
            }
        }
    }

    EnsureBounds(xz_min, xz_max);
    const auto [cam_row, cam_col] = WorldToCell(camera_origin);
    // Cells crossed by the agent since the last update are floor
    {
        const int from_row = mHasLastCameraCell ? mLastCameraRow : cam_row;
        const int from_col = mHasLastCameraCell ? mLastCameraCol : cam_col;
        const int dr = cam_row - from_row;
        const int dc = cam_col - from_col;
        const int steps = std::max(std::abs(dr), std::abs(dc));
        for (int step = 0; step <= steps; ++step) {
            const float t = steps == 0 ? 0.0f : static_cast<float>(step) / static_cast<float>(steps);
            const int row = static_cast<int>(std::lround(static_cast<float>(from_row) + static_cast<float>(dr) * t));
            const int col = static_cast<int>(std::lround(static_cast<float>(from_col) + static_cast<float>(dc) * t));
            for (int nr = row - 1; nr <= row + 1; ++nr) {
                for (int nc = col - 1; nc <= col + 1; ++nc) {
                    if (InBounds(nr, nc)) IncrementU16(mFloorHits, nr, nc, 1);
                }
            }
        }
        mHasLastCameraCell = true;
        mLastCameraRow = cam_row;
        mLastCameraCol = cam_col;
    }
    const float floor_plane = mHasFloorEstimate ? mFloorEstimate : mFloorY;
    const float drop_level = floor_plane - mConfig.height_min_m;
    for (const auto &endpoint : endpoints) {
        const auto [end_row, end_col] = WorldToCell(endpoint.world_m);
        const int dr = end_row - cam_row;
        const int dc = end_col - cam_col;
        const int steps = std::max(std::abs(dr), std::abs(dc));
        if (steps == 0) continue;
        for (int step = 0; step < steps; ++step) {
            const float t = static_cast<float>(step) / static_cast<float>(steps);
            const int row = static_cast<int>(std::lround(static_cast<float>(cam_row) + static_cast<float>(dr) * t));
            const int col = static_cast<int>(std::lround(static_cast<float>(cam_col) + static_cast<float>(dc) * t));
            MarkFree(row, col);
        }
        if (InBounds(end_row, end_col)) {
            if (endpoint.height_ok) {
                MarkOccupied(end_row, end_col, 1);
            } else {
                MarkObserved(end_row, end_col);
            }
            if (endpoint.surface > 0) {
                IncrementU16(mFloorHits, end_row, end_col, 1);
            } else if (endpoint.surface < 0) {
                IncrementU16(mNonFloorHits, end_row, end_col, 1);
            }
        }
        // Ray ending below the floor: no floor between plane crossing and endpoint
        if (mHasFloorEstimate && endpoint.world_m.y() < drop_level) {
            const float cam_y = camera_origin.y();
            const float denom = cam_y - endpoint.world_m.y();
            if (denom > 1e-4f && cam_y > floor_plane) {
                const float s = std::clamp((cam_y - floor_plane) / denom, 0.0f, 1.0f);
                const Eigen::Vector3f crossing = camera_origin + s * (endpoint.world_m - camera_origin);
                const auto [cross_row, cross_col] = WorldToCell(crossing);
                const int sr = end_row - cross_row;
                const int sc = end_col - cross_col;
                const int seg = std::max(std::abs(sr), std::abs(sc));
                for (int step = 0; step <= seg; ++step) {
                    const float t = seg == 0 ? 0.0f : static_cast<float>(step) / static_cast<float>(seg);
                    const int row = static_cast<int>(std::lround(static_cast<float>(cross_row) + static_cast<float>(sr) * t));
                    const int col = static_cast<int>(std::lround(static_cast<float>(cross_col) + static_cast<float>(sc) * t));
                    if (InBounds(row, col)) IncrementU16(mNonFloorHits, row, col, 1);
                }
            }
        }
    }
}

void NavigationMap::UpdateOccupiedFromGraph(Graph *graph) {
    mGraphOccupancyStats = {};
    if (!graph) return;

    std::vector<Eigen::Vector3f> points;
    {
        std::unique_lock<std::mutex> graph_lock(graph->mMutNode);
        for (const auto &pair : graph->nodes) {
            if (pair.first == 0 || !pair.second) continue;
            const auto &node = pair.second;
            std::unique_lock<std::mutex> surfel_lock(node->mMutSurfel);
            for (const auto &surfel_pair : node->surfels) {
                const auto &surfel = surfel_pair.second;
                if (!surfel || !surfel->is_valid || !surfel->is_stable) continue;
                if (surfel->confidence < mConfig.surfel_min_confidence) continue;
                points.emplace_back(surfel->pos * 0.001f);
            }
        }
    }
    if (points.empty()) return;

    const float camera_y_guess = mHasFloor ? (mFloorY + mConfig.camera_height_m) : 0.0f;
    const auto [height_min, height_max] = HeightBand(camera_y_guess);
    Eigen::Vector2f xz_min(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    Eigen::Vector2f xz_max(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest());
    std::vector<Eigen::Vector3f> kept;
    kept.reserve(points.size());
    for (const auto &point : points) {
        if (point.y() < height_min || point.y() > height_max) continue;
        kept.push_back(point);
        xz_min.x() = std::min(xz_min.x(), point.x());
        xz_min.y() = std::min(xz_min.y(), point.z());
        xz_max.x() = std::max(xz_max.x(), point.x());
        xz_max.y() = std::max(xz_max.y(), point.z());
    }
    if (kept.empty()) return;

    EnsureBounds(xz_min, xz_max);
    for (const auto &point : kept) {
        const auto [row, col] = WorldToCell(point);
        if (!InBounds(row, col)) continue;
        const int free = mFree.at<uint16_t>(row, col);
        const int before = mOccupied.at<uint16_t>(row, col);
        const bool supported = !(mConfig.floor_support_enabled && IsNonFloorCell(row, col));
        const bool was_free = supported && free >= mConfig.free_confirm_hits
            && free > before && mObserved.at<uint8_t>(row, col) > 0;
        const int margin = std::max(0, mConfig.obstacle_free_margin);
        const bool was_occupied = !supported
            || (before >= mConfig.obstacle_confirm_hits && before >= free + margin);
        MarkOccupied(row, col, 2);
        const int after = mOccupied.at<uint16_t>(row, col);
        ++mGraphOccupancyStats.surfel_votes;
        if (was_free && free <= after) ++mGraphOccupancyStats.removed_free_cells;
        if (!was_occupied && after >= mConfig.obstacle_confirm_hits && after >= free + margin) {
            ++mGraphOccupancyStats.added_occupied_cells;
        }
    }
}

cv::Mat NavigationMap::ObservedMask() const {
    return mObserved > 0;
}

cv::Mat NavigationMap::OccupiedMask() const {
    cv::Mat mask = cv::Mat::zeros(mOccupied.rows, mOccupied.cols, CV_8UC1);
    for (int row = 0; row < mOccupied.rows; ++row) {
        const auto *occ = mOccupied.ptr<uint16_t>(row);
        const auto *free = mFree.ptr<uint16_t>(row);
        auto *dst = mask.ptr<uint8_t>(row);
        for (int col = 0; col < mOccupied.cols; ++col) {
            const int margin = std::max(0, mConfig.obstacle_free_margin);
            if (occ[col] >= mConfig.obstacle_confirm_hits && occ[col] >= free[col] + margin) {
                dst[col] = 255;
            } else if (mConfig.floor_support_enabled && IsNonFloorCell(row, col)) {
                dst[col] = 255;
            }
        }
    }
    return mask;
}

cv::Mat NavigationMap::FreeMask() const {
    cv::Mat mask = cv::Mat::zeros(mFree.rows, mFree.cols, CV_8UC1);
    for (int row = 0; row < mFree.rows; ++row) {
        const auto *free = mFree.ptr<uint16_t>(row);
        const auto *occ = mOccupied.ptr<uint16_t>(row);
        const auto *obs = mObserved.ptr<uint8_t>(row);
        auto *dst = mask.ptr<uint8_t>(row);
        for (int col = 0; col < mFree.cols; ++col) {
            if (mConfig.floor_support_enabled && IsNonFloorCell(row, col)) continue;
            if (free[col] >= mConfig.free_confirm_hits && free[col] > occ[col] && obs[col] > 0) {
                dst[col] = 255;
            }
        }
    }
    return mask;
}

NavigationFloorSupportStats NavigationMap::FloorSupportStats() const {
    NavigationFloorSupportStats stats;
    stats.enabled = mConfig.floor_support_enabled;
    for (int row = 0; row < mFree.rows; ++row) {
        const auto *free = mFree.ptr<uint16_t>(row);
        const auto *occ = mOccupied.ptr<uint16_t>(row);
        const auto *obs = mObserved.ptr<uint8_t>(row);
        for (int col = 0; col < mFree.cols; ++col) {
            if (!IsNonFloorCell(row, col)) continue;
            ++stats.nonfloor_cells;
            if (free[col] >= mConfig.free_confirm_hits && free[col] > occ[col] && obs[col] > 0) {
                ++stats.potential_vetoed_free_cells;
            }
        }
    }
    stats.vetoed_free_cells = stats.enabled ? stats.potential_vetoed_free_cells : 0;
    return stats;
}

cv::Mat NavigationMap::UnknownMask() const {
    return mObserved == 0;
}

cv::Mat NavigationMap::FrontierCells() const {
    const cv::Mat free = FreeMask();
    const cv::Mat unknown = UnknownMask();
    if (cv::countNonZero(free) == 0 || cv::countNonZero(unknown) == 0) {
        return cv::Mat::zeros(free.rows, free.cols, CV_8UC1);
    }
    cv::Mat unknown_dilated;
    cv::dilate(unknown, unknown_dilated, cv::Mat::ones(3, 3, CV_8UC1), cv::Point(-1, -1), 1);
    cv::Mat frontier;
    cv::bitwise_and(free, unknown_dilated, frontier);
    return frontier;
}

std::optional<Eigen::Vector3f> NavigationMap::SnapToFree(const Eigen::Vector3f &xyz_m, float max_radius_m) const {
    const cv::Mat free = FreeMask();
    if (cv::countNonZero(free) == 0) return std::nullopt;

    const auto [row, col] = WorldToCell(xyz_m);
    const int max_cells = std::max(1, static_cast<int>(std::ceil(max_radius_m / mConfig.resolution_m)));
    const int start_row = std::clamp(row, 0, free.rows - 1);
    const int start_col = std::clamp(col, 0, free.cols - 1);
    if (free.at<uint8_t>(start_row, start_col) > 0 &&
        std::abs(start_row - row) <= max_cells && std::abs(start_col - col) <= max_cells) {
        return CellToWorld(start_row, start_col, xyz_m.y());
    }

    cv::Mat visited = cv::Mat::zeros(free.rows, free.cols, CV_8UC1);
    std::queue<std::tuple<int, int, int>> queue;
    queue.emplace(start_row, start_col, 0);
    visited.at<uint8_t>(start_row, start_col) = 1;
    const int neighbors[8][2] = {
        {-1, 0}, {1, 0}, {0, -1}, {0, 1},
        {-1, -1}, {-1, 1}, {1, -1}, {1, 1},
    };
    while (!queue.empty()) {
        const auto [r, c, d] = queue.front();
        queue.pop();
        if (free.at<uint8_t>(r, c) > 0) {
            return CellToWorld(r, c, xyz_m.y());
        }
        if (d >= max_cells) continue;
        for (const auto &offset : neighbors) {
            const int nr = r + offset[0];
            const int nc = c + offset[1];
            if (nr < 0 || nr >= free.rows || nc < 0 || nc >= free.cols) continue;
            if (visited.at<uint8_t>(nr, nc) != 0) continue;
            visited.at<uint8_t>(nr, nc) = 1;
            queue.emplace(nr, nc, d + 1);
        }
    }
    return std::nullopt;
}

cv::Mat NavigationMap::DijkstraFrom(int start_row, int start_col) const {
    const cv::Mat free = FreeMask();
    cv::Mat dist(free.rows, free.cols, CV_32FC1, cv::Scalar(std::numeric_limits<float>::infinity()));
    if (start_row < 0 || start_row >= free.rows || start_col < 0 || start_col >= free.cols) return dist;

    using QueueItem = std::tuple<float, int, int>;
    struct Greater {
        bool operator()(const QueueItem &lhs, const QueueItem &rhs) const {
            return std::get<0>(lhs) > std::get<0>(rhs);
        }
    };
    std::priority_queue<QueueItem, std::vector<QueueItem>, Greater> queue;
    dist.at<float>(start_row, start_col) = 0.0f;
    queue.emplace(0.0f, start_row, start_col);
    const float sqrt2 = 1.41421356237f;
    const int neighbors[8][3] = {
        {-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0},
        {-1, -1, 1}, {-1, 1, 1}, {1, -1, 1}, {1, 1, 1},
    };
    while (!queue.empty()) {
        const auto [distance, row, col] = queue.top();
        queue.pop();
        if (distance > dist.at<float>(row, col)) continue;
        for (const auto &offset : neighbors) {
            const int nr = row + offset[0];
            const int nc = col + offset[1];
            if (nr < 0 || nr >= free.rows || nc < 0 || nc >= free.cols) continue;
            if (free.at<uint8_t>(nr, nc) == 0) continue;
            const float step = offset[2] == 0 ? 1.0f : sqrt2;
            const float nd = distance + step * mConfig.resolution_m;
            auto &slot = dist.at<float>(nr, nc);
            if (nd < slot) {
                slot = nd;
                queue.emplace(nd, nr, nc);
            }
        }
    }
    return dist;
}

cv::Mat NavigationMap::GeodesicDistanceMap(const Eigen::Vector3f &start_m) const {
    const auto [sr, sc] = WorldToCell(start_m);
    return DijkstraFrom(sr, sc);
}

float NavigationMap::GeodesicDistance(const Eigen::Vector3f &start_m, const Eigen::Vector3f &goal_m) const {
    const auto [gr, gc] = WorldToCell(goal_m);
    const cv::Mat dist = GeodesicDistanceMap(start_m);
    if (gr < 0 || gr >= dist.rows || gc < 0 || gc >= dist.cols) {
        return std::numeric_limits<float>::infinity();
    }
    const float value = dist.at<float>(gr, gc);
    return std::isfinite(value) ? value : std::numeric_limits<float>::infinity();
}

std::vector<float> NavigationMap::GeodesicDistancesFrom(
    const Eigen::Vector3f &start_m,
    const std::vector<Eigen::Vector3f> &goals_m
) const {
    const cv::Mat dist = GeodesicDistanceMap(start_m);
    std::vector<float> distances;
    distances.reserve(goals_m.size());
    for (const auto &goal_m : goals_m) {
        const auto [gr, gc] = WorldToCell(goal_m);
        if (gr < 0 || gr >= dist.rows || gc < 0 || gc >= dist.cols) {
            distances.push_back(std::numeric_limits<float>::infinity());
            continue;
        }
        const float value = dist.at<float>(gr, gc);
        distances.push_back(std::isfinite(value) ? value : std::numeric_limits<float>::infinity());
    }
    return distances;
}

std::optional<Eigen::Vector3f> NavigationMap::FrontierGoal(
    const Eigen::Vector3f &position_m,
    float min_goal_distance_m,
    int min_frontier_size,
    bool prefer_far
) const {
    const cv::Mat frontier = FrontierCells();
    if (cv::countNonZero(frontier) == 0) return std::nullopt;

    cv::Mat labels, stats, centroids;
    const int label_count = cv::connectedComponentsWithStats(frontier, labels, stats, centroids, 8, CV_32S);
    struct Candidate {
        Eigen::Vector3f point;
        float distance = std::numeric_limits<float>::infinity();
        int area = 0;
    };
    std::vector<Candidate> candidates;
    for (int label = 1; label < label_count; ++label) {
        const int area = stats.at<int>(label, cv::CC_STAT_AREA);
        if (area < min_frontier_size) continue;
        const double centroid_col = centroids.at<double>(label, 0);
        const double centroid_row = centroids.at<double>(label, 1);
        const Eigen::Vector3f candidate_world = CellToWorld(
            static_cast<int>(std::lround(centroid_row)),
            static_cast<int>(std::lround(centroid_col)),
            position_m.y()
        );
        const auto snapped = SnapToFree(candidate_world);
        if (!snapped.has_value()) continue;
        const float xz_distance = std::hypot(snapped->x() - position_m.x(), snapped->z() - position_m.z());
        if (xz_distance < min_goal_distance_m) continue;
        const float geo = GeodesicDistance(position_m, *snapped);
        if (!std::isfinite(geo)) continue;
        candidates.push_back({*snapped, geo, area});
    }
    if (candidates.empty()) return std::nullopt;
    std::sort(candidates.begin(), candidates.end(), [prefer_far](const Candidate &lhs, const Candidate &rhs) {
        if (prefer_far) return lhs.distance > rhs.distance;
        return lhs.distance < rhs.distance;
    });
    return candidates.front().point;
}

cv::Mat NavigationMap::Render() const {
    cv::Mat image(mFree.rows, mFree.cols, CV_8UC3, cv::Scalar(255, 255, 255));
    const cv::Mat free = FreeMask();
    const cv::Mat occupied = OccupiedMask();
    const cv::Mat frontier = FrontierCells();
    image.setTo(cv::Scalar(226, 231, 238), free);
    image.setTo(cv::Scalar(60, 60, 60), occupied);
    image.setTo(cv::Scalar(34, 140, 240), frontier);
    return image;
}

}  // namespace PSLAM
