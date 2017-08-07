#include "drake/perception/dev/feedforward_neural_network.h"
#include <stdlib.h>
#include <iostream>
#include <memory>
#include <Eigen/Dense>
#include <gtest/gtest.h>
#include "drake/common/eigen_matrix_compare.h"
#include "drake/systems/framework/basic_vector.h"

namespace drake {
using std::abs;
using std::unique_ptr;
using systems::SystemOutput;
using systems::Context;
using systems::BasicVector;

using Eigen::MatrixXd;
using Eigen::VectorXd;

using std::cout;
using std::endl;
namespace perception {
namespace {

typedef Tensor<double,3> Tensor3d;

/********************************
 * Helper function declarations
 * (definitions at end of file)
 ********************************/
// Helper functions to generate a new matrix/vector/tensor with specified 
// dimensions, initialize the entries. We don't care about what the entries are;
// this is meant to be a way to get some variety of entries without using 
// Matrix::Random
VectorXd NewVector(int rows);
MatrixXd NewMatrix(int rows, int columns);
Tensor3d NewTensor3(int rows, int columns, int depth);
// Helper function to generate a vector of Connected layers
//std::vector<LayerSpecification> GenerateFullyConnectedLayers(int num_layers);
// Helper function to generate a vector of Relu nonlinearities
//std::vector<NonlinearityType> GenerateReluNonlinearities(int num_layers);
// Helper function to check IO behavior of a double FeedforwardNeuralNetwork
template <typename T>
void TestIO( const std::vector<LayerSpecification<T>>& layers,
            VectorX<T> input, VectorX<T> expected_output);
// Helper function to compare a MatrixX<double> and a MatrixX<AutoDiffXd>
// TODO(nikos-tri) Generalize this function
//bool CompareMats(const MatrixX<double>& Mdouble,
//                 const MatrixX<AutoDiffXd>& Mautodiff, double tolerance = 0);
bool Compare( const Tensor3d& t1, const Tensor3d& t2, double tolerance = 0);

/********************************
 * Tests
 ********************************/

// Test that the matrices are being encoded and decoded correctly from
// BasicVectors
GTEST_TEST(FeedforwardNeuralNetworkTest, WeightsCoderDecoder) {
  MatrixXd weights1 = NewMatrix(3, 3);
  VectorXd bias1 = NewVector(3);
  LayerSpecification<double> L1(weights1, bias1);

  VectorXd bias2 = NewVector(7);
  MatrixXd weights2 = NewMatrix(7, 3);
  LayerSpecification<double> L2(weights2, bias2);

 	Tensor3d weights3 = NewTensor3(5, 2, 8); 
  MatrixXd bias3 = NewMatrix(5,13);
  std::vector<Tensor3d> weights_list3; weights_list3.push_back(weights3);
  LayerSpecification<double> L3(weights_list3, bias3);

	
#ifndef NDEBUG
	cout << "Encoding weights..." << endl;
#endif
  unique_ptr<BasicVector<double>> encoded_weights1 = L1.WeightsToBasicVector();
  unique_ptr<BasicVector<double>> encoded_weights2 = L2.WeightsToBasicVector();
#ifndef NDEBUG
	cout << "Encoding tensor..." << endl;
#endif
  unique_ptr<BasicVector<double>> encoded_weights3 = L3.WeightsToBasicVector();

#ifndef NDEBUG
	cout << "Encoding biases..." << endl;
#endif
  unique_ptr<BasicVector<double>> encoded_bias1 = L1.BiasToBasicVector();
  unique_ptr<BasicVector<double>> encoded_bias2 = L2.BiasToBasicVector();
  unique_ptr<BasicVector<double>> encoded_bias3 = L3.BiasToBasicVector();

#ifndef NDEBUG
	cout << "Decoding weights..." << endl;
#endif
  unique_ptr<MatrixXd> recovered_weights1 = L1.WeightsMatrixFromBasicVector(
  																													*encoded_weights1 );
  unique_ptr<MatrixXd> recovered_weights2 = L2.WeightsMatrixFromBasicVector(
  																													*encoded_weights2 );
  std::vector<unique_ptr<Tensor3d>> recovered_weights3 = L3.FilterBankFromBasicVector(
  																													*encoded_weights3 );

#ifndef NDEBUG
	cout << "Decoding biases..." << endl;
#endif
	unique_ptr<VectorXd> recovered_bias1 = L1.BiasVectorFromBasicVector(
																														*encoded_bias1 );
	unique_ptr<VectorXd> recovered_bias2 = L2.BiasVectorFromBasicVector(
																														*encoded_bias2 );
	unique_ptr<MatrixXd> recovered_bias3 = L3.BiasMatrixFromBasicVector(
																														*encoded_bias3 );
  EXPECT_EQ(weights1, *recovered_weights1);
  EXPECT_EQ(weights2, *recovered_weights2);
  EXPECT_TRUE( Compare(weights3, *(recovered_weights3[0])) );

  EXPECT_EQ(bias1, *recovered_bias1);
  EXPECT_EQ(bias2, *recovered_bias2);
  EXPECT_EQ(bias3, *recovered_bias3);
}

//// Test that the NN is correctly loading and extracting its parameters from
//// the Context
//GTEST_TEST(FeedforwardNeuralNetworkTest, ParameterStorage) {
//  MatrixXd W1 = NewMatrix(3, 3);
//  MatrixXd W2 = NewMatrix(7, 3);
//  MatrixXd W3 = NewMatrix(2, 7);
//  std::vector<MatrixXd> W;
//  W.push_back(W1);
//  W.push_back(W2);
//  W.push_back(W3);
//
//  VectorXd B1 = NewVector(3);
//  VectorXd B2 = NewVector(7);
//  VectorXd B3 = NewVector(2);
//  std::vector<VectorXd> B;
//  B.push_back(B1);
//  B.push_back(B2);
//  B.push_back(B3);
//
//  FeedforwardNeuralNetwork<double> dut(W, B, GenerateFullyConnectedLayers(3),
//                                       GenerateReluNonlinearities(3));
//
//  unique_ptr<Context<double>> context = dut.CreateDefaultContext();
//
//  EXPECT_EQ(W1, *(dut.get_weight_matrix(0, *context)));
//  EXPECT_EQ(W2, *(dut.get_weight_matrix(1, *context)));
//  EXPECT_EQ(W3, *(dut.get_weight_matrix(2, *context)));
//
//  EXPECT_EQ(B1, *(dut.get_bias_vector(0, *context)));
//  EXPECT_EQ(B2, *(dut.get_bias_vector(1, *context)));
//  EXPECT_EQ(B3, *(dut.get_bias_vector(2, *context)));
//}

// Test that the NN can figure out what "shape" it should be:
// - num inputs,
// - num outputs,
// - num layers
GTEST_TEST(FeedforwardNeuralNetworkTest, ShapeParameters) {
	std::vector<LayerSpecification<double>> layers;
	
  MatrixXd weights1 = NewMatrix(10, 19);
  VectorXd bias1 = NewVector(10);
  LayerSpecification<double> layer1( weights1, bias1, NonlinearityType::Relu );
  layers.push_back( layer1 );

  MatrixXd weights2 = NewMatrix(7, 10);
  VectorXd bias2 = NewVector(7);
  LayerSpecification<double> layer2( weights2, bias2, NonlinearityType::Relu );
  layers.push_back( layer2 );

  MatrixXd weights3 = NewMatrix(2, 7);
  VectorXd bias3 = NewVector(2);
  LayerSpecification<double> layer3( weights3, bias3, NonlinearityType::Relu );
  layers.push_back( layer3 );

  FeedforwardNeuralNetwork<double> dut( layers );

  EXPECT_EQ(dut.get_num_layers(), 3);
  EXPECT_EQ(dut.get_num_outputs(), 2);
  // Inputs are not so simple anymore now that there is convolution. We can
  // convolve with an input that is significantly larger, so the size of the
  // tensor doesn't tell us the size of the input
  //EXPECT_EQ(dut.get_num_inputs(), 19);

}

GTEST_TEST(FeedforwardNeuralNetworkTest, ConvolutionTest) {
	Tensor3d t(3,3,3); t.setZero();
	std::vector<Tensor3d> filter_bank; filter_bank.push_back(t);
	MatrixXd bias = MatrixXd::Zero(3,3);
	LayerSpecification<double> layer( filter_bank, bias );
	
	LayerResult<double> input; input.t = NewTensor3(9,9,3);
	Tensor3d result = layer.Convolve( input.t, filter_bank );

#ifndef NDEBUG
	auto input_dimensions = input.t.dimensions();
	cout << "Input input_dimensions are: ( " << input_dimensions[0] << ", " << input_dimensions[1] << ", " << input_dimensions[2] << " )" << endl;
	auto result_dimensions = result.dimensions();
	cout << "Result result_dimensions are: ( " << result_dimensions[0] << ", " << result_dimensions[1] << ", " << result_dimensions[2] << " )" << endl;
#endif
	EXPECT_TRUE( true );
	
}


// TODO(nikos-tri) Fix this
//GTEST_TEST(FeedforwardNeuralNetworkTest, ToAutoDiffTest) {
//  MatrixXd W1 = NewMatrix(10, 19);
//  MatrixXd W2 = NewMatrix(7, 10);
//  MatrixXd W3 = NewMatrix(2, 7);
//  std::vector<MatrixXd> W;
//  W.push_back(W1);
//  W.push_back(W2);
//  W.push_back(W3);
//
//  VectorXd B1 = NewVector(10);
//  VectorXd B2 = NewVector(7);
//  VectorXd B3 = NewVector(2);
//  std::vector<VectorXd> B;
//  B.push_back(B1);
//  B.push_back(B2);
//  B.push_back(B3);
//
//  FeedforwardNeuralNetwork<double> dut(W, B, GenerateFullyConnectedLayers(3),
//                                       GenerateReluNonlinearities(3));
//  unique_ptr<Context<double>> context = dut.CreateDefaultContext();
//
//  unique_ptr<FeedforwardNeuralNetwork<AutoDiffXd>> ad_ffnn = dut.ToAutoDiffXd();
//  unique_ptr<Context<AutoDiffXd>> ad_context = ad_ffnn->CreateDefaultContext();
//  ad_context->SetTimeStateAndParametersFrom(*context);
//
//  unique_ptr<MatrixX<AutoDiffXd>> W1recovered =
//      ad_ffnn->get_weight_matrix(0, *ad_context);
//  unique_ptr<MatrixX<AutoDiffXd>> W2recovered =
//      ad_ffnn->get_weight_matrix(1, *ad_context);
//  unique_ptr<MatrixX<AutoDiffXd>> W3recovered =
//      ad_ffnn->get_weight_matrix(2, *ad_context);
//
//  EXPECT_TRUE(CompareMats(W1, *W1recovered));
//  EXPECT_TRUE(CompareMats(W2, *W2recovered));
//  EXPECT_TRUE(CompareMats(W3, *W3recovered));
//}

// Very basic sanity test of input-output behavior with identity weight
// matrices
GTEST_TEST(FeedforwardNeuralNetworkTest, BasicSanity) {
	std::vector<LayerSpecification<double>> layers;

	#ifndef NDEBUG
		cout << "Initializing first layer" << endl;
	#endif

  MatrixXd identity1 = MatrixXd::Identity(3, 3);
  VectorXd bias1 = VectorXd::Zero(3);
  LayerSpecification<double> layer1( identity1, bias1, NonlinearityType::Relu );
  layers.push_back( layer1 );

	#ifndef NDEBUG
		cout << "Initializing second layer" << endl;
	#endif

  MatrixXd identity2 = MatrixXd::Identity(3, 3);
  VectorXd bias2 = VectorXd::Zero(3);
  LayerSpecification<double> layer2( identity2, bias2, NonlinearityType::Relu );
  layers.push_back( layer2 );

	#ifndef NDEBUG
		cout << "Initializing third layer" << endl;
	#endif

  MatrixXd identity3 = MatrixXd::Identity(3, 3);
  VectorXd bias3 = VectorXd::Zero(3);
  LayerSpecification<double> layer3( identity3, bias3, NonlinearityType::Relu );
  layers.push_back( layer3 );

	#ifndef NDEBUG
		cout << "Creating test vectors" << endl;
	#endif

  // A few test vectors
  VectorXd input1(3);
  input1 << 1, 2, 3;
  VectorXd expected_output1 = input1;

  VectorXd input2(3);
  input2 << -1, -2, -3;
  VectorXd expected_output2(3);
  expected_output2 << 0, 0, 0;  // ReLU zeroes them out because they are negative

  VectorXd input3(3);
  input3 << -1, 0, 1;
  VectorXd expected_output3(3);
  expected_output3 << 0, 0, 1;  // ReLU zeroes out the negative

	#ifndef NDEBUG
		cout << "Starting first sanity test" << endl;
	#endif
  TestIO(layers, input1, expected_output1);
  TestIO(layers, input2, expected_output2);
  TestIO(layers, input3, expected_output3);
	#ifndef NDEBUG
		cout << "Sanity test completed" << endl;
	#endif
}

/********************************
 * Helper function definitions
 ********************************/
// Helper functions to generate a new matrix/vector with specified dimensions,
// initialize the entries. We don't care about what the entries are; this is
// meant to be a way to get some variety of entries without using Matrix::Random
VectorXd NewVector(int rows) {
  VectorXd vector(rows);

  for (int i = 0; i < rows; i++) {
    vector(i) = i * i;
  }
  return vector;
}
MatrixXd NewMatrix(int rows, int columns) {
  MatrixXd matrix(rows, columns);

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < columns; j++) {
      matrix(i, j) = i + j;  // Initialize each entry to whatever number
    }
  }
  return matrix;
}
Tensor3d NewTensor3(int rows, int columns, int depth) {
	Tensor3d tensor3( rows, columns, depth );

	for ( int i = 0; i < rows; i++ ) {
		for ( int j = 0; j < columns; j++ ) {
			for ( int k = 0; k < depth; k++ ) {
				tensor3(i, j, k) = i + j + k;
			}
		}
	}
	return tensor3;
}

// TODO(nikos-tri) probably delete this
//// Helper function to generate a vector of Connected layers
//std::vector<LayerType> GenerateFullyConnectedLayers(int num_layers) {
//  std::vector<LayerType> layers;
//  LayerType fullyConnected = LayerType::FullyConnected;
//  for (int i = 0; i < num_layers; i++) {
//    layers.push_back(fullyConnected);
//  }
//  return layers;
//}
//
//// Helper function to generate a vector of Relu nonlinearities
//std::vector<NonlinearityType> GenerateReluNonlinearities(int num_layers) {
//  std::vector<NonlinearityType> nonlinearities;
//  NonlinearityType relu = NonlinearityType::Relu;
//  for (int i = 0; i < num_layers; i++) {
//    nonlinearities.push_back(relu);
//  }
//  return nonlinearities;
//}

// Helper function to compare a MatrixX<double> and a MatrixX<AutoDiffXd>
//bool CompareMats(const MatrixX<double>& Mdouble,
//                 const MatrixX<AutoDiffXd>& Mautodiff,
//                 double tolerance /* default is 0, see declaration*/) {
//  if ((Mdouble.rows() != Mautodiff.rows()) ||
//      (Mdouble.cols() != Mautodiff.cols())) {
//    return false;
//  }
//
//  for (int i = 0; i < Mdouble.rows(); i++) {
//    for (int j = 0; j < Mdouble.rows(); j++) {
//      double thisAdValue = (Mautodiff(i, j)).value();
//
//      if (abs(thisAdValue - Mdouble(i, j)) > tolerance) {
//        return false;
//      }
//    }
//  }
//  return true;
//}

bool Compare( const Tensor3d& t1, const Tensor3d& t2, double tolerance /*= 0*/ ) {
	auto dims1 = t1.dimensions();
	auto dims2 = t2.dimensions();
	if ( (dims1[0] != dims2[0]) || (dims1[1] != dims2[1]) || (dims1[2] != dims2[2]) ) {
		#ifndef NDEBUG
		  cout << "Tensor dimensions differ." << endl;
			cout << "Dimensions of t1 are: " << "(" << dims1[0] << ", " << dims1[1] << ", " << dims1[2] << ")" << endl;
			cout << "Dimensions of t1 are: " << "(" << dims2[0] << ", " << dims2[1] << ", " << dims2[2] << ")" << endl;
		#endif
		return false;
	}

	#ifndef NDEBUG	
		cout << "Tensor dimensions are the same." << endl;
	#endif

	for ( int i = 0; i < dims1[0]; i++ ) {
		for ( int j = 0; j < dims1[1]; j++ ) {
			for ( int k = 0; k < dims1[2]; k++ ) {
				if ( abs( t1(i,j,k) - t2(i,j,k) ) > tolerance ) {
					
				#ifndef NDEBUG
					cout << "Tensors differ at location: ";
					cout << "(" << i << ", " << j << ", " << k << ")" << endl;
					cout << "t1: " << t1(i, j, k) << endl;
					cout << "t2: " << t2(i, j, k) << endl;
				#endif
					return false;
				}
			}
		}
	}

	return true;
}


// Check that a FeedforwardNeuralNetwork<T> gives the expected result
template <typename T>
void TestIO( const std::vector<LayerSpecification<T>>& layers,
            VectorX<T> input, VectorX<T> expected_output) {

  FeedforwardNeuralNetwork<double> dut( layers );
  unique_ptr<Context<double>> context = dut.CreateDefaultContext();
  unique_ptr<SystemOutput<double>> output = dut.AllocateOutput(*context);

	#ifndef NDEBUG
		cout << "Fixing input..." << endl;
	#endif
  context->FixInputPort(dut.input().get_index(), input);
	#ifndef NDEBUG
		cout << "Calculating output..." << endl;
	#endif
  dut.CalcOutput(*context, output.get());
	#ifndef NDEBUG
		cout << "Getting output..." << endl;
	#endif
  VectorXd computed_output =
      output->get_vector_data(dut.output().get_index())->get_value();
	#ifndef NDEBUG
		cout << "Checking output..." << endl;
		cout << "Output is: " << endl << computed_output << endl;
		cout << "Expected is: " << endl << expected_output << endl;
	#endif

  EXPECT_EQ(expected_output, computed_output);
}

}  // namespace
}  // namespace perception
}  // namespace drake
