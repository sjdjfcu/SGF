#pragma once

#include "../camera/CameraParameters.h"
#include "graph.h"

#include <Eigen/Core>
#include <opencv2/core.hpp>

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace PSLAM {

struct NavigationMapConfig {
    float resolution_m = 0.05f;
    float height_min_m = 0.15f;
    float height_max_m = 1.80f;
    int depth_stride = 4;
    float min_range_m = 0.30f;
    float max_range_m = 5.0f;
    float camera_height_m = 1.5f;
    float surfel_min_confidence = 0.30f;
    int obstacle_confirm_hits = 2;
    int free_confirm_hits = 1;
    int obstacle_free_margin = 1;
    // If false, floor evidence is still collected but does not veto free cells
    bool floor_support_enabled = true;
};

struct NavigationFloorSupportStats {
    bool enabled = true;
    uint64_t nonfloor_cells = 0;
    uint64_t vetoed_free_cells = 0;
    uint64_t potential_vetoed_free_cells = 0;
};

// Cell transitions caused by the most recent graph update
struct NavigationGraphOccupancyStats {
    uint64_t surfel_votes = 0;
    uint64_t removed_free_cells = 0;
    uint64_t added_occupied_cells = 0;
};

class NavigationMap {
public:
    explicit NavigationMap(NavigationMapConfig config = NavigationMapConfig{});

    void Reset();
    void SetConfig(const NavigationMapConfig &config);
    const NavigationMapConfig &Config() const { return mConfig; }

    void UpdateFreeSpace(
        const cv::Mat &depth_mm,
        const Eigen::Matrix4f &camera_from_world_mm,
        const CameraParameters &camera
    );
    void UpdateOccupiedFromGraph(Graph *graph);

    std::pair<int, int> Shape() const;
    float ResolutionMeters() const { return mConfig.resolution_m; }
    Eigen::Vector2f OriginXZ() const { return mOriginXZ; }
    std::pair<int, int> WorldToCell(const Eigen::Vector3f &xyz_m) const;
    Eigen::Vector3f CellToWorld(int row, int col, float y_m) const;

    cv::Mat ObservedMask() const;
    cv::Mat OccupiedMask() const;
    cv::Mat FreeMask() const;
    cv::Mat UnknownMask() const;
    cv::Mat FrontierCells() const;
    cv::Mat Render() const;
    NavigationFloorSupportStats FloorSupportStats() const;
    NavigationGraphOccupancyStats GraphOccupancyStats() const { return mGraphOccupancyStats; }

    std::optional<Eigen::Vector3f> SnapToFree(const Eigen::Vector3f &xyz_m, float max_radius_m = 2.0f) const;
    cv::Mat GeodesicDistanceMap(const Eigen::Vector3f &start_m) const;
    float GeodesicDistance(const Eigen::Vector3f &start_m, const Eigen::Vector3f &goal_m) const;
    std::vector<float> GeodesicDistancesFrom(
        const Eigen::Vector3f &start_m,
        const std::vector<Eigen::Vector3f> &goals_m
    ) const;
    std::optional<Eigen::Vector3f> FrontierGoal(
        const Eigen::Vector3f &position_m,
        float min_goal_distance_m = 1.5f,
        int min_frontier_size = 20,
        bool prefer_far = false
    ) const;

private:
    void EnsureBounds(const Eigen::Vector2f &xz_min, const Eigen::Vector2f &xz_max, float margin_m = 2.0f);
    float UpdateFloor(float camera_y_m);
    std::pair<float, float> HeightBand(float camera_y_m);
    void MarkFree(int row, int col);
    void MarkOccupied(int row, int col, int weight = 1);
    void MarkObserved(int row, int col);
    bool InBounds(int row, int col) const;
    // Surfaces observed but none on the floor plane, e.g. a stair or a drop
    bool IsNonFloorCell(int row, int col) const {
        return mNonFloorHits.at<uint16_t>(row, col) > 0 && mFloorHits.at<uint16_t>(row, col) == 0;
    }
    cv::Mat DijkstraFrom(int start_row, int start_col) const;

    NavigationMapConfig mConfig;
    NavigationGraphOccupancyStats mGraphOccupancyStats;
    cv::Mat mFree;
    cv::Mat mOccupied;
    cv::Mat mObserved;
    // Per-cell endpoint counts on and off the floor plane
    cv::Mat mFloorHits;
    cv::Mat mNonFloorHits;
    Eigen::Vector2f mOriginXZ = Eigen::Vector2f::Zero();
    float mFloorY = 0.0f;
    bool mHasFloor = false;
    // Measured floor height (mode of observed surface heights)
    void UpdateFloorEstimate(const std::vector<float> &heights_m, int min_support);
    std::vector<uint32_t> mFloorHistogram;
    float mFloorEstimate = 0.0f;
    bool mHasFloorEstimate = false;
    // Camera cell at the previous update, for marking the agent's trail
    bool mHasLastCameraCell = false;
    int mLastCameraRow = 0;
    int mLastCameraCol = 0;
};

}  // namespace PSLAM
