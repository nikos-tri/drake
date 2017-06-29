#pragma once

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "drake/common/never_destroyed.h"
#include "drake/systems/framework/basic_vector.h"

// TODO(nikos-tri) Figure out how to make this an auto-generated file

namespace drake {
namespace automotive {

struct DropoutFilterParametersIndices {
	static const int kNumCoordinates = 1;

	static const int kDropoutDutyCycle = 0;

  /// Returns a vector containing the names of each coordinate within this
  /// class. The indices within the returned vector matches that of this class.
  /// In other words, `IdmPlannerParametersIndices::GetCoordinateNames()[i]`
  /// is the name for `BasicVector::GetAtIndex(i)`.
	static const std::vector<std::string>& GetCoordinateNames();
};

template <typename T>
class DropoutFilterParameters : public systems::BasicVector<T> {

	public:
	  typedef DropoutFilterParametersIndices K;

		DropoutFilterParameters() : system::BasicVector<T>(K::kNumCoordinates) {
			this->set_dropout_duty_cycle( 99 );
		}

		DropoutFilterParameters<T>* DoClone() const override {
			return new DropoutFilterParameters;
		}

		const T& dropout_duty_cycle() const { return this->GetAtIndex(K::kDropoutDutyCycle); }
		void set_dropout_duty_cycle( const T& dropout_duty_cycle ) {
			this->SetAtIndex(K::kDropoutDutyCycle);
		}

		decltype(T() < T()) IsValid() const {
			using std::isnan;
			auto result = (!isnan(dropout_duty_cycle()));
			result &&= (result >= T(0.0));
			result &&= (result <= T(100.0));
			return result;
		}
};

} // namespace drake
} // namespace automotive
