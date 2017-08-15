#pragma once

#include <memory>
#include <Eigen/Dense>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include "drake/common/eigen_types.h"
#include "drake/systems/framework/leaf_system.h"

namespace drake {
namespace perception {

enum class LayerType { FullyConnected, Convolutional, Dropout };
// The 'None' type exists to accomodate architectures that have several
// convolutional layers with no nonlinearity in between
enum class NonlinearityType { Relu, Sigmoid, Atan, None };

// Store intermediate results between layers
// TODO(nikos-tri) change this to be a union
template <typename T>
struct LayerResult {
  VectorX<T> v;
  Eigen::Tensor<T, 3> t;
};

// TODO(nikos-tri) Consider subclassing this into two class of specifications,
// to avoid the clumsiness of some functions like evaluate()
template <typename T>
class LayerSpecification {
 public:
  LayerSpecification(
      const MatrixX<T>& weights_matrix, const VectorX<T>& bias_vector,
      NonlinearityType nonlinearity_type = NonlinearityType::None)
      : matrix_weights_{weights_matrix},
        vector_bias_{bias_vector},
        layer_type_{LayerType::FullyConnected},
        nonlinearity_type_{nonlinearity_type} {}

  LayerSpecification(
      const std::vector<Eigen::Tensor<T, 3>>& weights_tensors,
      const std::vector<MatrixX<T>>& bias_matrices,
      NonlinearityType nonlinearity_type = NonlinearityType::None)
      : filter_bank_{weights_tensors},
        matrix_bias_bank_{bias_matrices},
        layer_type_{LayerType::Convolutional},
        nonlinearity_type_{nonlinearity_type} {}

  LayerResult<T> Evaluate(const LayerResult<T>& input,
                          const systems::BasicVector<T>& weights_vector,
                          const systems::BasicVector<T>& bias_vector) const;

  Eigen::Tensor<T, 3> ConvolveAndBias(
      const Eigen::Tensor<T, 3>& input,
      const std::vector<Eigen::Tensor<T, 3>>& filters,
      const std::vector<MatrixX<T>>& biases) const;
  Eigen::Tensor<T, 3> AddBias(const Eigen::Tensor<T, 3>& input,
                              const MatrixX<T>& bias) const;
  void StackSlice(const Eigen::Tensor<T, 3>& slice, Eigen::Tensor<T, 3>* result,
                  int offset) const;

	
	VectorX<T> ReshapeTensorToVector( const Eigen::Tensor<T,3>& tensor ) const;
  LayerResult<T> Relu(const LayerResult<T>& in) const;

  // Functions for encoding/decoding parameters to/from a BasicVector, which is
  // the form in which they are stored in the Context

  // Encoding:
  // High-level
  std::unique_ptr<systems::BasicVector<T>> WeightsToBasicVector() const;
  std::unique_ptr<systems::BasicVector<T>> BiasToBasicVector() const;

  // Mid-level
  std::unique_ptr<systems::BasicVector<T>> FilterBankToBasicVector(
      const std::vector<Eigen::Tensor<T, 3>>& tensor) const;
  std::unique_ptr<systems::BasicVector<T>> MatrixBankToBasicVector(
      const std::vector<MatrixX<T>>& bias_bank) const;

  // Decoding
  std::unique_ptr<MatrixX<T>> WeightsMatrixFromBasicVector(
      const systems::BasicVector<T>& basic_vector) const;
  std::vector<std::unique_ptr<Eigen::Tensor<T, 3>>> FilterBankFromBasicVector(
      const systems::BasicVector<T>& basic_vector) const;

  std::unique_ptr<VectorX<T>> BiasVectorFromBasicVector(
      const systems::BasicVector<T>& basic_vector) const;
  std::vector<std::unique_ptr<MatrixX<T>>> BiasMatrixBankFromBasicVector(
      const systems::BasicVector<T>& basic_vector) const;

  // getters
  bool is_convolutional() const;
  bool is_fully_connected() const;
  bool is_relu() const;
  LayerType get_layer_type() const;
  NonlinearityType get_nonlinearity_type() const;
  int get_num_outputs() const;

 private:
  std::unique_ptr<systems::BasicVector<T>> VectorToBasicVector(
      const VectorX<T>& vector) const;
  std::unique_ptr<systems::BasicVector<T>> MatrixToBasicVector(
      const MatrixX<T>& matrix) const;

  // std::unique_ptr<MatrixX<T>> BasicVectorToMatrix(
  //    int rows, int cols, const systems::BasicVector<T>& basic_vector) const;
  // std::unique_ptr<MatrixX<T>> BasicVectorToMatrix(
  //    int rows, int cols, int depth,
  //    const systems::BasicVector<T>& basic_vector) const;

  std::unique_ptr<VectorX<T>> VectorFromBasicVector(
      const systems::BasicVector<T>& basic_vector) const;
  std::unique_ptr<MatrixX<T>> MatrixFromBasicVector(
      const systems::BasicVector<T>& basic_vector, int rows, int cols,
      int offset = 0) const;
  std::unique_ptr<Eigen::Tensor<T, 3>> Tensor3FromBasicVector(
      const systems::BasicVector<T>& basic_vector, int rows, int cols,
      int depth, int offset = 0) const;

  //	union {
  MatrixX<T> matrix_weights_;
  std::vector<Eigen::Tensor<T, 3>> filter_bank_;
  //	  int num_weights_; //useful for checking that BasicVectors have the
  //right size

  //	};
  //	union {
  VectorX<T> vector_bias_;
  std::vector<MatrixX<T>> matrix_bias_bank_;
  //	};

  // Weights<T> weights_;
  // Bias<T> bias_;
  LayerType layer_type_;
  NonlinearityType nonlinearity_type_;
  // the dimensions of the weights
  int weights_rows_, weights_columns_,
      weights_depth_;  // depth is one if weights is a matrix
  // dimensions of the bias
  int bias_rows_, bias_columns_;
};

}  // namespace perception
}  // namespace drake
