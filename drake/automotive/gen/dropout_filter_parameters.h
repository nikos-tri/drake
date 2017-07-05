#pragma once

// GENERATED FILE DO NOT EDIT
// See drake/tools/lcm_vector_gen.py.

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "drake/common/never_destroyed.h"
#include "drake/systems/framework/basic_vector.h"

namespace drake {
namespace automotive {

/// Describes the row indices of a DropoutFilterParameters.
struct DropoutFilterParametersIndices {
  /// The total number of rows (coordinates).
  static const int kNumCoordinates = 1;

  // The index of each individual coordinate.
  static const int kDropoutDutyCycle = 0;

  /// Returns a vector containing the names of each coordinate within this
  /// class. The indices within the returned vector matches that of this class.
  /// In other words, `DropoutFilterParametersIndices::GetCoordinateNames()[i]`
  /// is the name for `BasicVector::GetAtIndex(i)`.
  static const std::vector<std::string>& GetCoordinateNames();
};

/// Specializes BasicVector with specific getters and setters.
template <typename T>
class DropoutFilterParameters : public systems::BasicVector<T> {
 public:
  /// An abbreviation for our row index constants.
  typedef DropoutFilterParametersIndices K;

  /// Default constructor.  Sets all rows to their default value:
  /// @arg @c dropout_duty_cycle defaults to 50.0 in units of percentage.
  DropoutFilterParameters() : systems::BasicVector<T>(K::kNumCoordinates) {
    this->set_dropout_duty_cycle(50.0);
  }

  DropoutFilterParameters<T>* DoClone() const override {
    return new DropoutFilterParameters;
  }

  /// @name Getters and Setters
  //@{
  /// percentage of time that filter will drop frames
  /// @note @c dropout_duty_cycle is expressed in units of percentage.
  /// @note @c dropout_duty_cycle has a limited domain of [0.0, 100.0].
  const T& dropout_duty_cycle() const {
    return this->GetAtIndex(K::kDropoutDutyCycle);
  }
  void set_dropout_duty_cycle(const T& dropout_duty_cycle) {
    this->SetAtIndex(K::kDropoutDutyCycle, dropout_duty_cycle);
  }
  //@}

  /// See DropoutFilterParametersIndices::GetCoordinateNames().
  static const std::vector<std::string>& GetCoordinateNames() {
    return DropoutFilterParametersIndices::GetCoordinateNames();
  }

  /// Returns whether the current values of this vector are well-formed.
  decltype(T() < T()) IsValid() const {
    using std::isnan;
    auto result = (T(0) == T(0));
    result = result && !isnan(dropout_duty_cycle());
    result = result && (dropout_duty_cycle() >= T(0.0));
    result = result && (dropout_duty_cycle() <= T(100.0));
    return result;
  }
};

}  // namespace automotive
}  // namespace drake
