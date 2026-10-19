#include "surfel.h"

namespace inseg_lib {

Surfel::Surfel(const Eigen::Vector3f& pos_in,
               const Eigen::Vector3f& normal_in,
               const float radius_in,
               const float confidence_in,
               const int label_in)
    : index_in_graph(-1),
      confidence(confidence_in),
      time_invalid(0),
      label_confidence(0),
      is_valid(false),
      is_stable(false),
      color{{255, 255, 255}},
      pos(pos_in),
      normal(normal_in),
      radius(radius_in),
      label(label_in) {}

bool Surfel::operator<(const Surfel& ref) const {
    return is_valid && !ref.is_valid;
}

void Surfel::SetLabel(const int label_in) {
    label = label_in;
}

int Surfel::GetLabel() const {
    return label;
}

}  // namespace inseg_lib
