#include "drake/perception/dev/feedforward_neural_network.h"

namespace drake {

using drake::systems::Context;
using drake::systems::InputPortDescriptor;
using drake::systems::OutputPort;
using drake::systems::System;
using drake::systems::BasicVector;

using Eigen::AutoDiffScalar;
using Eigen::Tensor;

using std::cout;
using std::endl;

namespace perception {

template <typename T>
FeedforwardNeuralNetwork<T>::FeedforwardNeuralNetwork(
    std::vector<LayerSpecification<T>> specs)
    : input_index_{this->DeclareAbstractInputPort().get_index()},
      output_index_{this->DeclareVectorOutputPort(
                            BasicVector<T>(specs.back().get_num_outputs()),
                            &FeedforwardNeuralNetwork::DoCalcOutput)
                        .get_index()} {
  // Ensure that the last layer is not convolutional. This complicates our
  // calculation of the number of outputs. Also, architectures with a
  // convolutional layer at the end are rare enough that it is not worth
  // supporting at this point.
  DRAKE_THROW_UNLESS(specs.back().get_layer_type() != LayerType::Convolutional);

  num_outputs_ = specs.back().get_num_outputs();
  num_layers_ = specs.size();

  typedef typename vector<LayerSpecification<T>>::size_type sz;
  for (sz i = 0; i < specs.size(); i++) {
    weight_indices_.push_back(
        this->DeclareNumericParameter(*(specs[i].WeightsToBasicVector())));

    bias_indices_.push_back(
        this->DeclareNumericParameter(*(specs[i].BiasToBasicVector())));
  }
  layer_specifications_ = specs;
}

template <typename T>
void FeedforwardNeuralNetwork<T>::DoCalcOutput(const Context<T>& context,
                                               BasicVector<T>* output) const {
  // Read the input
  VectorX<T> input_value = ReadInput(context);

  // Evaluate each layer
  LayerResult<T> intermediate_value;
  intermediate_value.v = input_value;
  for (int i = 0; i < num_layers_; i++) {
    const BasicVector<T>& weights_vector =
        get_parameter_vector(weight_indices_[i], context);
    const BasicVector<T>& bias_vector =
        get_parameter_vector(bias_indices_[i], context);
    intermediate_value = layer_specifications_[i].Evaluate(
        intermediate_value, weights_vector, bias_vector);
  }

  // Write output
  WriteOutput(intermediate_value.v, output);
}

// template <typename T>
// VectorX<T> FeedforwardNeuralNetwork<T>::EvaluateLayer(
//    const VectorX<T>& layerInput, MatrixX<T> Weights, VectorX<T> bias,
//    LayerType layer, NonlinearityType nonlinearity) const {
//  // Only suppports fully-connected RELU at this time
//  DRAKE_ASSERT(layer == LayerType::FullyConnected);
//  DRAKE_ASSERT(nonlinearity == NonlinearityType::Relu);
//  VectorX<T> layer_output = relu(Weights * layerInput + bias);
//
//  return layer_output;
//}

// TODO(nikos-tri) fix this
// template <typename T>
// FeedforwardNeuralNetwork<AutoDiffXd>*
// FeedforwardNeuralNetwork<T>::DoToAutoDiffXd() const {
//  // ?
//  // vector<MatrixX<AutoDiffScalar<MatrixX<T>>>> W_autodiff;
//  // ?
//  vector<MatrixX<AutoDiffXd>> W_autodiff;
//  vector<VectorX<AutoDiffXd>> b_autodiff;
//
//  typedef typename vector<MatrixX<T>>::size_type sz;
//  for (sz i = 0; i < weights_matrices_.size(); i++) {
//    MatrixX<T> this_W = weights_matrices_[i];
//    // ?
//    // W_autodiff.push_back(this_W.template
//    cast<AutoDiffScalar<MatrixX<T>>>());
//    // ?
//    W_autodiff.push_back(this_W.template cast<AutoDiffXd>());
//
//    VectorX<T> this_b = bias_vectors_[i];
//    b_autodiff.push_back(this_b.template cast<AutoDiffXd>());
//  }
//
//  return new FeedforwardNeuralNetwork<AutoDiffXd>(W_autodiff, b_autodiff,
//                                                  layers_, nonlinearities_);
//}

template <typename T>
int FeedforwardNeuralNetwork<T>::get_num_layers() const {
  return num_layers_;
}
// template <typename T>
// int FeedforwardNeuralNetwork<T>::get_num_inputs() const {
//  return num_inputs_;
//}
template <typename T>
int FeedforwardNeuralNetwork<T>::get_num_outputs() const {
  return num_outputs_;
}

template <typename T>
// std::unique_ptr<BasicVector<T>> FeedforwardNeuralNetwork<T>
// get_parameter_vector(
const BasicVector<T>& FeedforwardNeuralNetwork<T>::get_parameter_vector(
    int index, const Context<T>& context) const {
  const BasicVector<T>& parameters =
      this->template GetNumericParameter<BasicVector>(context, index);
  return parameters;
}

// template <typename T>
// std::unique_ptr<MatrixX<T>> FeedforwardNeuralNetwork<T>::get_weight_matrix(
//    int index, const Context<T>& context) const {
//  DRAKE_THROW_UNLESS((0 <= index) &&
//                     ((vector<int>::size_type)index <
//                     weight_indices_.size()));
//
//  const BasicVector<T>& encodedMatrix =
//      this->template GetNumericParameter<BasicVector>(context,
//                                                      weight_indices_[index]);
//
//  return DecodeWeightsFromBasicVector(rows_[index], cols_[index],
//                                      encodedMatrix);
//}
//
// template <typename T>
// std::unique_ptr<VectorX<T>> FeedforwardNeuralNetwork<T>::get_bias_vector(
//    int index, const Context<T>& context) const {
//  DRAKE_THROW_UNLESS((0 <= index) &&
//                     ((vector<int>::size_type)index < bias_indices_.size()));
//
//  const BasicVector<T>& encoded_vector =
//      this->template GetNumericParameter<BasicVector>(context,
//                                                      bias_indices_[index]);
//
//  VectorX<T>* bias_vector = new VectorX<T>(encoded_vector.get_value());
//  std::unique_ptr<VectorX<T>> uptr(bias_vector);
//  return uptr;
//}

template <typename T>
const InputPortDescriptor<T>& FeedforwardNeuralNetwork<T>::input() const {
  return System<T>::get_input_port(input_index_);
}

template <typename T>
const OutputPort<T>& FeedforwardNeuralNetwork<T>::output() const {
  return System<T>::get_output_port(output_index_);
}

template <typename T>
const VectorX<T> FeedforwardNeuralNetwork<T>::ReadInput(
    const Context<T>& context) const {
  const BasicVector<T>* input =
      this->template EvalVectorInput<BasicVector>(context, input_index_);
  DRAKE_ASSERT((input != nullptr));
  return input->get_value();
}
template <typename T>
void FeedforwardNeuralNetwork<T>::WriteOutput(const VectorX<T> value,
                                              BasicVector<T>* output) const {
  output->set_value(value);
}

template class FeedforwardNeuralNetwork<double>;
// template class FeedforwardNeuralNetwork<AutoDiffXd>;

}  // namespace automotive
}  // namespace drake
