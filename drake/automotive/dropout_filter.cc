#include "drake/automotive/dropout_filter.h"

namespace drake {

using std::cout;
using std::endl;

namespace automotive {

template <typename T>
DropoutFilter<T>::DropoutFilter( double error_period, double error_duty_cycle )
	: traffic_input_index_{this->DeclareAbstractInputPort().get_index()},
	traffic_output_index_{this->DeclareAbstractOutputPort( 
	&DropoutFilter::MakeTrafficOutput,
	&DropoutFilter::CalcTrafficOutput).get_index() } {


		error_period_ = error_period;
		error_duty_cycle_ = error_duty_cycle;

		this->DeclareDiscreteState( 1 );
		this->DeclareDiscreteUpdatePeriodSec( error_period/100 );

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
PoseBundle<T> DropoutFilter<T>::MakeTrafficOutput() const {
	return PoseBundle<T>( 0 );
}

template <typename T>
void DropoutFilter<T>::CalcTrafficOutput( const systems::Context<T>& context,
                                    PoseBundle<T>* output_traffic_poses ) const {

	// TODO(nikos-tri) delete these crude debugging tools
	static int dropped; static int passed;

	T current_state = context.get_discrete_state(0)->GetAtIndex(0);
//	cout << "-----------------------------------------------------------------------"<<endl;
//	cout << "current state is: " << current_state << endl;
//	cout << "error_duty_cycle is: " << error_duty_cycle_ << endl;
	if ( current_state >= (100 - error_duty_cycle_) ) {
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
void DropoutFilter<T>::DoCalcDiscreteVariableUpdates( const Context<T>& context,
																		DiscreteValues<T>* discrete_state_update ) const {

	double current_state = context.get_discrete_state(0)->GetAtIndex(0);
//	cout << "*********************************************************************"<<endl;
//	cout << "Updating state from: " << current_state << endl;
	double next_state = current_state + 1;

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
