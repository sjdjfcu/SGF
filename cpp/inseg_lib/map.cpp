#include "map.h"

#include <algorithm>
#include <fstream>
#include <mutex>
#include <sstream>
#include <unordered_map>

namespace inseg_lib {

namespace {

cv::Vec3b SelectColor(const Surfel& surfel, const int color_type) {
    switch (color_type) {
        case 1:
            return cv::Vec3b(surfel.color[0], surfel.color[1], surfel.color[2]);
        case 2:
        default:
            return CalculateLabelColor(static_cast<int16_t>(surfel.label));
    }
}

Eigen::Vector3f TransformPoint(const Eigen::Matrix4f& pose, const Eigen::Vector3f& point) {
    const Eigen::Vector4f homo(point.x(), point.y(), point.z(), 1.0f);
    return (pose * homo).head<3>();
}

Eigen::Vector3f TransformNormal(const Eigen::Matrix4f& pose, const Eigen::Vector3f& normal) {
    Eigen::Vector3f transformed = pose.block<3, 3>(0, 0) * normal;
    const float norm = transformed.norm();
    if (norm > 0.0f) {
        transformed /= norm;
    }
    return transformed;
}

std::vector<cv::Vec3b>& LabelColors() {
    static std::vector<cv::Vec3b> colors;
    static std::once_flag once;
    std::call_once(once, []() {
        colors.emplace_back(255, 255, 255);

        std::ifstream file("colors.txt");
        std::string line;
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            int c0 = 0;
            int c1 = 0;
            int c2 = 0;
            if (ss >> c0 >> c1 >> c2) {
                colors.emplace_back(
                    static_cast<unsigned char>(c0),
                    static_cast<unsigned char>(c1),
                    static_cast<unsigned char>(c2));
            }
        }
    });
    return colors;
}

}  // namespace

Map::Map(std::shared_ptr<inseg_lib::SegmentsInterface> segments, MapConfig& config)
    : segments_(std::move(segments)) {
    (void)config;
    surfels.reserve(1000000);
}

Map::~Map() = default;

bool Map::SaveSegments(const std::string& file,
                       const std::string& file_list,
                       const int color_type) const {
    std::unordered_map<int, std::vector<std::shared_ptr<Surfel>>> grouped;
    for (const auto& surfel : surfels) {
        if (!surfel) {
            continue;
        }
        grouped[surfel->label].push_back(surfel);
    }

    std::vector<int> labels;
    labels.reserve(grouped.size());
    for (const auto& pair : grouped) {
        labels.push_back(pair.first);
    }
    std::sort(labels.begin(), labels.end());

    std::ofstream list_out;
    if (!file_list.empty()) {
        list_out.open(file_list);
        if (!list_out.is_open()) {
            return false;
        }
    }

    for (int label : labels) {
        const std::string segment_file = file + "_" + std::to_string(label) + ".ply";
        if (!SaveSurfels(segment_file, color_type, grouped[label])) {
            return false;
        }
        if (list_out.is_open()) {
            list_out << segment_file << "\n";
        }
    }
    return true;
}

bool Map::SaveModel(const std::string& filename,
                    const int color_type,
                    Eigen::Matrix4f pose) const {
    return SaveSurfels(filename, color_type, surfels, pose);
}

int Map::GetSize() const {
    return static_cast<int>(surfels.size());
}

const std::shared_ptr<Surfel> Map::GetSurfel(const int index) const {
    if (index < 0 || index >= static_cast<int>(surfels.size())) {
        return nullptr;
    }
    return surfels[index];
}

void Map::SetColor(const int index, const cv::Vec3b& col) {
    if (index <= 0 || index >= static_cast<int>(surfels.size()) || !surfels[index]) {
        return;
    }
    surfels[index]->color = {{col[0], col[1], col[2]}};
}

void Map::SetLabelConfidence(const int index, const float value) {
    if (index <= 0 || index >= static_cast<int>(surfels.size()) || !surfels[index]) {
        return;
    }
    surfels[index]->label_confidence = static_cast<int>(value);
}

void Map::SetStable(const int index, bool value) {
    if (index <= 0 || index >= static_cast<int>(surfels.size()) || !surfels[index]) {
        return;
    }
    surfels[index]->is_stable = value;
}

void Map::SetInvalid(const int index) {
    if (index <= 0 || index >= static_cast<int>(surfels.size()) || !surfels[index]) {
        return;
    }
    surfels[index]->is_valid = false;
}

void Map::Clear() {
    surfels.clear();
    if (segments_) {
        segments_->Clear();
    }
}

void Map::Add(const std::shared_ptr<Surfel> surfel) {
    if (!surfel) {
        return;
    }
    surfels.push_back(surfel);
    if (segments_) {
        surfel->index_in_graph = segments_->Add(surfel);
    }
}

void Map::UpdateSurfel(const int index,
                       const Eigen::Vector3f& pos,
                       const Eigen::Vector3f& normal,
                       const float confidence,
                       const float radius) {
    if (index <= 0 || index >= static_cast<int>(surfels.size()) || !surfels[index]) {
        return;
    }
    auto& surfel = surfels[index];
    surfel->pos = pos;
    surfel->normal = normal;
    surfel->confidence = confidence;
    surfel->radius = radius;
    if (segments_) {
        segments_->Update(surfel, pos, normal);
    }
}

void Map::Merge(const int index, const int index2) {
    if (segments_) {
        segments_->Merge(index, index2);
    }
}

void Map::SetLabel(const int index, const int label) {
    if (index <= 0 || index >= static_cast<int>(surfels.size()) || !surfels[index]) {
        return;
    }
    auto& surfel = surfels[index];
    if (surfel->label == label) {
        return;
    }
    if (segments_) {
        surfel->index_in_graph = segments_->UpdateLabel(surfel, label);
    }
    surfel->label = label;
}

void Map::WritePlyHeader(std::ofstream& fout, const int vertex) const {
    fout << "ply\n";
    fout << "format ascii 1.0\n";
    fout << "element vertex " << vertex << "\n";
    fout << "property float x\n";
    fout << "property float y\n";
    fout << "property float z\n";
    fout << "property int label\n";
    fout << "property float nx\n";
    fout << "property float ny\n";
    fout << "property float nz\n";
    fout << "property float curvature\n";
    fout << "property uchar red\n";
    fout << "property uchar green\n";
    fout << "property uchar blue\n";
    fout << "property uchar alpha\n";
    fout << "property float quality\n";
    fout << "property float radius\n";
    fout << "end_header\n";
}

bool Map::SaveSurfels(const std::string& filename,
                      const int color_type,
                      const std::vector<std::shared_ptr<Surfel>>& surfel_list,
                      Eigen::Matrix4f pose) const {
    std::ofstream fout(filename);
    if (!fout.is_open()) {
        return false;
    }

    int valid_count = 0;
    for (const auto& surfel : surfel_list) {
        if (surfel) {
            ++valid_count;
        }
    }

    WritePlyHeader(fout, valid_count);
    for (const auto& surfel : surfel_list) {
        if (!surfel) {
            continue;
        }

        const Eigen::Vector3f pos = TransformPoint(pose, surfel->pos);
        const Eigen::Vector3f normal = TransformNormal(pose, surfel->normal);
        const cv::Vec3b color = SelectColor(*surfel, color_type);

        fout << pos.x() << " " << pos.y() << " " << pos.z() << " "
             << surfel->label << " "
             << normal.x() << " " << normal.y() << " " << normal.z() << " "
             << 0.0f << " "
             << static_cast<int>(color[2]) << " "
             << static_cast<int>(color[1]) << " "
             << static_cast<int>(color[0]) << " "
             << 255 << " "
             << surfel->confidence << " "
             << surfel->radius << "\n";
    }
    return true;
}

const cv::Vec3b& CalculateLabelColor(const int16_t label) {
    auto& colors = LabelColors();
    static std::mutex mutex;
    static cv::RNG rng(1234567);
    std::lock_guard<std::mutex> lock(mutex);

    if (label < 0) {
        return colors[0];
    }
    while (label >= static_cast<int16_t>(colors.size())) {
        cv::Vec3b color;
        do {
            color = cv::Vec3b(
                static_cast<unsigned char>(rng.uniform(0, 256)),
                static_cast<unsigned char>(rng.uniform(0, 256)),
                static_cast<unsigned char>(rng.uniform(0, 256)));
        } while (static_cast<int>(color[0]) + static_cast<int>(color[1]) + static_cast<int>(color[2]) <= 179);
        colors.push_back(color);
    }
    return colors[label];
}

}  // namespace inseg_lib
