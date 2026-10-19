#ifndef INSEG_LIB_SURFEL_H_
#define INSEG_LIB_SURFEL_H_

#include <array>
#include <Eigen/Dense>
#include <vector>
#include <map>
#include <memory>

namespace inseg_lib {

class Surfel {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Surfel() = default;
    Surfel(const Eigen::Vector3f& pos, const Eigen::Vector3f& normal,
           const float radius, const float confidence, const int label);
    ~Surfel() = default;
    bool operator< (const Surfel &ref) const;

    void SetLabel(const int label);
    int GetLabel() const;
    
    int index_in_graph = -1;

    // Accumulated observation weight used for fusion and stability.
    float confidence = 0;
    // Age counter used to retire unstable surfels.
    int time_invalid = 0;

    // Support for the current label; negative values trigger reassignment.
    int label_confidence = 0;

    // Active surfels participate in projection and map updates.
    bool is_valid = false;
    // Stable surfels are retained regardless of age.
    bool is_stable = false;

    // Color used for visualization.
    std::array<unsigned char, 3> color = {{255, 255, 255}};

    Eigen::Vector3f pos;
    Eigen::Vector3f normal;
    float radius = 0;
    int label = 0;
};
}
typedef std::shared_ptr<inseg_lib::Surfel> SurfelPtr;

#endif  // INSEG_LIB_SURFEL_H_
