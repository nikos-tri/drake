#include "drake/automotive/gen/dropout_filter_parameters.h"

// GENERATED FILE DO NOT EDIT
// See drake/tools/lcm_vector_gen.py.

namespace drake {
namespace automotive {

const int DropoutFilterParametersIndices::kNumCoordinates;
const int DropoutFilterParametersIndices::kDropoutDutyCycle;

const std::vector<std::string>&
DropoutFilterParametersIndices::GetCoordinateNames() {
  static const never_destroyed<std::vector<std::string>> coordinates(
      std::vector<std::string>{
          "dropout_duty_cycle",
      });
  return coordinates.access();
}

}  // namespace automotive
}  // namespace drake
