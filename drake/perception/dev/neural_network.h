#pragma once

#include <memory>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"
#include "drake/perception/dev/layer_specification.h"
#include "drake/systems/framework/leaf_system.h"

// TODO(nikos-tri) cleanup
using Eigen::Tensor;
using std::vector;
namespace drake {
namespace perception {

template <typename T>
class NeuralNetwork : public systems::LeafSystem<T> {

};
}  // namespace perception
}  // namespace drake
