#include "drake/perception/dev/layer_specification.h"

namespace drake {
using systems::BasicVector;
using Eigen::Tensor;
namespace perception {

template <typename T>
LayerResult<T> LayerSpecification<T>::Evaluate( const LayerResult<T>& input, const BasicVector<T>& weights_vector, const BasicVector<T>& bias_vector ) const {
	// Can't call a convolutional layer with a vector input
	LayerResult<T> r;
	if ( is_fully_connected() ) {
		auto weights = *(WeightsMatrixFromBasicVector( weights_vector ));
		auto bias = *(BiasVectorFromBasicVector( bias_vector ));
		r.v = weights*input.v + bias;
	} else if ( is_convolutional() ) {
		//Convolve( input.tI//
		//TODO(nikos-tri) add bias
		DRAKE_DEMAND(false);
	} else {
		DRAKE_DEMAND(false);
	}

	if ( is_relu() ) {
		r = Relu( r );
	} else {
		// ...eventually handle the other nonlinearities
	}
	return r;
}

template <typename T>
LayerResult<T> LayerSpecification<T>::Relu(const LayerResult<T>& in) const {
	// TODO(nikos-tri) Relu only supports fully connected at this point
	DRAKE_DEMAND( is_fully_connected() );
	LayerResult<T> result;
	result.v = in.v;
  for (int i = 0; i < result.v.size(); i++) {
    if (result.v(i) < 0) {
      result.v(i) = 0;
    }
  }
  return result;
}

template <typename T>
Tensor<T, 3> LayerSpecification<T>::Convolve(Tensor<T, 3> input, std::vector<Tensor<T, 3>> filters) const {

	auto input_dimensions = input.dimensions();
  Tensor<T, 3> output_volume( input_dimensions[0], input_dimensions[1], filters.size() );
  Tensor<T, 3> output_slice;  // used as a temporary variable

  //// specify first and second dimensions for convolution
  //// (numbering of dimensions starts from zero, annoyingly)
  int count = 0;
  Eigen::array<ptrdiff_t, 1> dims({{2}});
  for (auto filter : filters) {
    output_slice = input.convolve(filter, dims);
    StackSlice(output_slice, &output_volume, count++ );
  }
  return output_volume;

}

template <typename T>
void LayerSpecification<T>::StackSlice(const Tensor<T, 3>& slice, Tensor<T, 3>* result, int offset ) const {
  auto dimensions_slice = slice.dimensions();
  auto dimensions_result = result->dimensions();

  DRAKE_THROW_UNLESS(dimensions_slice[0] == dimensions_result[0]);
  DRAKE_THROW_UNLESS(dimensions_slice[1] == dimensions_result[1]);

  for (int i = 0; i < dimensions_slice[0]; i++) {
    for (int j = 0; j < dimensions_slice[1]; j++) {
      (*result)(i, j, offset) = slice(i, j, 0);
    }
  }
}

template <typename T>
bool LayerSpecification<T>::is_relu() const {
	return (nonlinearity_type_ == NonlinearityType::Relu);
}

template <typename T>
std::unique_ptr<BasicVector<T>> LayerSpecification<T>::VectorToBasicVector(
    const VectorX<T>& vector) const {
  std::unique_ptr<BasicVector<T>> uptr(new BasicVector<T>(vector));
  return uptr;
}

template <typename T>
std::unique_ptr<BasicVector<T>> LayerSpecification<T>::MatrixToBasicVector(
    const MatrixX<T>& matrix) const {
  VectorX<T> data_vector(matrix.size());

  int vector_index = 0;
  for (int i = 0; i < matrix.rows(); i++) {
    for (int j = 0; j < matrix.cols(); j++) {
      data_vector(vector_index) = matrix(i, j);
      vector_index++;
    }
  }
  std::unique_ptr<BasicVector<T>> uptr(new BasicVector<T>(data_vector));
  return uptr;
}

template <typename T>
std::unique_ptr<BasicVector<T>> LayerSpecification<T>::FilterBankToBasicVector(
    const std::vector<Tensor<T, 3>>& filter_bank) const {

	int total_elements = 0;
  for ( decltype(filter_bank.size()) i = 0; i < filter_bank.size(); i++ ) {
  	total_elements += filter_bank[i].size();
	}

	VectorX<T> data_vector(total_elements);

  int vector_index = 0;
  for ( int m = 0; m < (int)(filter_bank.size()); m++ ) {
  auto tensor_dimensions = filter_bank[m].dimensions();
  for (int i = 0; i < tensor_dimensions[0]; i++) {
    for (int j = 0; j < tensor_dimensions[1]; j++) {
      for (int k = 0; k < tensor_dimensions[2]; k++) {
        data_vector(vector_index++) = filter_bank[m](i, j, k);
      }
    }
  }
	}

  std::unique_ptr<BasicVector<T>> uptr(new BasicVector<T>(data_vector));
  return uptr;
}

template <typename T>
std::unique_ptr<VectorX<T>> LayerSpecification<T>::VectorFromBasicVector(
    const BasicVector<T>& basic_vector ) const {

  std::unique_ptr<VectorX<T>> uptr(new VectorX<T>(basic_vector.get_value()));
  return uptr;
}

template <typename T>
std::unique_ptr<MatrixX<T>> LayerSpecification<T>::MatrixFromBasicVector(
    const BasicVector<T>& basic_vector, int rows, int cols) const {
  MatrixX<T>* weights = new MatrixX<T>(rows, cols);
  *weights = MatrixX<T>::Zero(rows, cols);

  VectorX<T> v = basic_vector.get_value();

  int vector_index = 0;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      (*weights)(i, j) = v(vector_index);
      vector_index++;
    }
  }

  std::unique_ptr<MatrixX<T>> uptr(weights);
  return uptr;
}

template <typename T>
std::unique_ptr<Tensor<T, 3>> LayerSpecification<T>::Tensor3FromBasicVector(
    const BasicVector<T>& basic_vector, int rows, int cols, int depth, int offset) const {
  Tensor<T, 3>* weights = new Tensor<T, 3>(rows, cols, depth);
  weights->setZero();

  VectorX<T> v = basic_vector.get_value();

  int vector_index = offset;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
    	for (int k = 0; k < depth; k++) {
      	(*weights)(i, j, k) = v(vector_index);
      	vector_index++;
			}
    }
  }

  std::unique_ptr<Tensor<T,3>> uptr(weights);
  return uptr;
}

// Functions for encoding/decoding parameters to/from a BasicVector, which is
// the form in which they are stored in the Context
template <typename T>
std::unique_ptr<systems::BasicVector<T>>
LayerSpecification<T>::WeightsToBasicVector() const {
  if (get_layer_type() == LayerType::FullyConnected) {
    return MatrixToBasicVector(matrix_weights_);
  } else if (get_layer_type() == LayerType::Convolutional) {
    return FilterBankToBasicVector(filter_bank_);
  } else {
    // TODO(nikos-tri) do something better here
    DRAKE_DEMAND(false);
    return nullptr;  // compiler can't tell that control doesn't reach here
  }
}

template <typename T>
std::unique_ptr<systems::BasicVector<T>>
LayerSpecification<T>::BiasToBasicVector() const {
  if (get_layer_type() == LayerType::FullyConnected) {
  	
#ifndef NDEBUG
		std::cout <<"encoding as: " << std::endl << *(VectorToBasicVector(vector_bias_)) << std::endl;
#endif
    return VectorToBasicVector(vector_bias_);
  } else if (get_layer_type() == LayerType::Convolutional) {
    return MatrixToBasicVector(matrix_bias_);
  } else {
    // TODO(nikos-tri) do something better here
    DRAKE_DEMAND(false);
    return nullptr;  // compiler can't tell that control doesn't reach here
  }
}

template <typename T>
std::unique_ptr<MatrixX<T>> LayerSpecification<T>::WeightsMatrixFromBasicVector(
    const systems::BasicVector<T>& basic_vector) const {
  DRAKE_THROW_UNLESS(is_fully_connected());
  return MatrixFromBasicVector(basic_vector, matrix_weights_.rows(),
                               matrix_weights_.cols() );
}
template <typename T>
std::vector<std::unique_ptr<Tensor<T, 3>>>
LayerSpecification<T>::FilterBankFromBasicVector(
    const systems::BasicVector<T>& basic_vector) const {
  DRAKE_THROW_UNLESS(is_convolutional());
  //DRAKE_THROW_UNLESS(num_weights_ == basic_vector.size());
  std::vector<std::unique_ptr<Tensor<T,3>>> recovered_filter_bank;
  int index_offset = 0;
  for ( decltype(filter_bank_.size()) i = 0; i < filter_bank_.size(); i++ ) {
  	auto these_dimensions = filter_bank_[i].dimensions();
  	recovered_filter_bank.push_back(
  	Tensor3FromBasicVector(basic_vector, these_dimensions[0],
  															these_dimensions[1],
  															these_dimensions[2],
  															index_offset ));
  	index_offset += filter_bank_[i].size();
	}
	return recovered_filter_bank;
}

template <typename T>
std::unique_ptr<VectorX<T>> LayerSpecification<T>::BiasVectorFromBasicVector(
    const systems::BasicVector<T>& basic_vector) const {
  DRAKE_THROW_UNLESS(is_fully_connected());
  return VectorFromBasicVector(basic_vector);
}
template <typename T>
std::unique_ptr<MatrixX<T>> LayerSpecification<T>::BiasMatrixFromBasicVector(
    const systems::BasicVector<T>& basic_vector) const {
  DRAKE_THROW_UNLESS(is_convolutional());
  return MatrixFromBasicVector(basic_vector, matrix_bias_.rows(), matrix_bias_.cols());
}

// getters
template <typename T>
bool LayerSpecification<T>::is_convolutional() const {
  return (get_layer_type() == LayerType::Convolutional);
}
template <typename T>
bool LayerSpecification<T>::is_fully_connected() const {
  return (get_layer_type() == LayerType::FullyConnected);
}
template <typename T>
LayerType LayerSpecification<T>::get_layer_type() const {
  return layer_type_;
}
template <typename T>
NonlinearityType LayerSpecification<T>::get_nonlinearity_type() const { return nonlinearity_type_; }
template <typename T>
int LayerSpecification<T>::get_num_outputs() const {
	// Unsupported for convolutional
	DRAKE_THROW_UNLESS( is_fully_connected() );
	return matrix_weights_.rows();
}

template class LayerSpecification<double>;
//template class LayerSpecification<AutoDiffXd>;

}  // namespace perception
}  // namespace drake
