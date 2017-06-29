#include "drake/automotive/dropout_filter.h"

namespace drake {

using std::cout;
using std::endl;

static constexpr int kDropoutFilterParamsIndex{0};

namespace automotive {

template <typename T>
DropoutFilter<T>::DropoutFilter( T error_period )
	: traffic_input_index_{this->DeclareAbstractInputPort().get_index()},
	traffic_output_index_{this->DeclareAbstractOutputPort( 
	&DropoutFilter::MakeTrafficOutput,
	&DropoutFilter::CalcTrafficOutput).get_index() } {

		error_period_ = error_period;
		error_duty_cycle_ = error_duty_cycle;

		this->DeclareDiscreteState( 1 );
		this->DeclareDiscreteUpdatePeriodSec( (double)error_period/100 );

		//std::unique_ptr<BasicVector<T>> parameters = BasicVector<T>::Make( error_period, error_duty_cycle );
		this->DeclareNumericParameter( DropoutFilterParameters<T>() );
}

template <typename T>
const systems::InputPortDescriptor<T>& DropoutFilter<T>::traffic_input() const {
	return systems::System<T>::get_input_port( traffic_input_index_ );
}

template <typename T>
const systems::OutputPort<T>& DropoutFilter<T>::traffic_output() const {
	return systems::System<T>::get_output_port( traffic_output_index_ );
}

template <typename T>
std::unique_ptr<DropoutFilter<AutoDiffXd>> DropoutFilter<T>::ToAutoDiffXd() const {
	return std::unique_ptr<DropoutFilter<AutoDiffXd>>( DoToAutoDiffXd() );
}

template <typename T>
DropoutFilter<AutoDiffXd>* DropoutFilter<T>::DoToAutoDiffXd() const {
	return new DropoutFilter<AutoDiffXd>( (AutoDiffXd)(this->error_period_), 
																				(AutoDiffXd)(this->error_duty_cycle_) );
}

template <typename T>
PoseBundle<T> DropoutFilter<T>::MakeTrafficOutput() const {
	return PoseBundle<T>( 0 );
}

template <typename T>
void DropoutFilter<T>::CalcTrafficOutput( const systems::Context<T>& context,
                                    PoseBundle<T>* output_traffic_poses ) const {

	// TODO(nikos-tri) delete these crude debugging tools
	static int dropped; static int passed;

	//if ( /*state >= (100 - error_duty_cycle_)*/ 
	if ( is_time_to_drop_frame( context ) ) {
//		cout << "Dropping frame!" << endl;
		*output_traffic_poses = PoseBundle<T>(0);
		dropped++;
	} else {
//		cout << "Forwarding posebundle without errors" << endl;
		const PoseBundle<T>* const input_traffic_poses = this->template EvalInputValue<PoseBundle<T>>( 
																																context,
																																traffic_input_index_ );
		DRAKE_ASSERT( input_traffic_poses != nullptr );
		DRAKE_ASSERT( output_traffic_poses != nullptr );
		PoseBundle<T> copy_of_input( *input_traffic_poses );
		*output_traffic_poses = copy_of_input;
		passed++;
	}

	//cout << "dropped: " << dropped << " passed: " << passed << endl;
}

template <typename T>
bool DropoutFilter<T>::is_time_to_drop_frame( Context<T>& context ) const {
	//BasicVector<T>& parameters = this->template GetNumericParameter<BasicVector<T>>( context, 0);
	//error_duty_cycle = parameters->GetAtIndex(1);
	const DropoutFilterParameters<T>& filter_params =
			this->template GetNumericParameter<DropoutFilterParameters>(context,
																																	kDropoutFilterParamsIndex );
			DRAKE_DEMAND( filter_params.IsValid() );
	T state = context.get_discrete_state(0)->GetAtIndex(0);
	return (state >= (100 - error_duty_cycle ) );
}

template <typename T>
void DropoutFilter<T>::DoCalcDiscreteVariableUpdates( const Context<T>& context,
																		DiscreteValues<T>* discrete_state_update ) const {

	T current_state = context.get_discrete_state(0)->GetAtIndex(0);
	//cout << "Updating state from: " << current_state;
	T next_state = current_state + 1;

	if ( next_state > 100 ) {
		next_state = 0;
	}

	//cout << " to: " << next_state << endl;
	(*discrete_state_update)[0] = next_state;
}
																		

template class DropoutFilter<double>;
template class DropoutFilter<AutoDiffXd>;

} // namespace automotive
} // namespace drake
