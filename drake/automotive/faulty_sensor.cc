#include "drake/automotive/faulty_sensor.h"

namespace drake {

using std::cout;
using std::endl;

namespace automotive {

template <typename T>
FaultySensor<T>::FaultySensor( double error_period, double error_duty_cycle ) 
	: traffic_input_index_{this->DeclareAbstractInputPort().get_index()},
	traffic_output_index_{this->DeclareAbstractOutputPort( 
	&FaultySensor::MakeTrafficOutput,
	&FaultySensor::CalcTrafficOutput).get_index() } { 


		error_period_ = error_period;
		error_duty_cycle_ = error_duty_cycle;

		this->DeclareDiscreteState( 1 );
		this->DeclareDiscreteUpdatePeriodSec( error_period/100 );


}

template <typename T>
const systems::InputPortDescriptor<T>& FaultySensor<T>::traffic_input() const {
	return systems::System<T>::get_input_port( traffic_input_index_ );
}

template <typename T>
const systems::OutputPort<T>& FaultySensor<T>::traffic_output() const {
	return systems::System<T>::get_output_port( traffic_output_index_ );
}

template <typename T>
PoseBundle<T> FaultySensor<T>::MakeTrafficOutput() const {
	return PoseBundle<T>( 0 );
}

template <typename T>
void FaultySensor<T>::CalcTrafficOutput( const systems::Context<T>& context,
                                    PoseBundle<T>* output_traffic_poses ) const {


	double current_state = context.get_discrete_state(0)->GetAtIndex(0);
	if ( current_state >= (100 - error_duty_cycle_) ) {
		cout << "Dropping frame!" << endl;
		*output_traffic_poses = PoseBundle<T>(0);
	} else {
		cout << "Forwarding posebundle without errors" << endl;
		const PoseBundle<T>* const input_traffic_poses = this->template EvalInputValue<PoseBundle<T>>( 
																																context,
																																traffic_input_index_ );
		DRAKE_ASSERT( input_traffic_poses != nullptr );
		DRAKE_ASSERT( output_traffic_poses != nullptr );
		PoseBundle<T> copy_of_input( *input_traffic_poses );
		*output_traffic_poses = copy_of_input;
	}
}

template <typename T>
void FaultySensor<T>::DoCalcDiscreteVariableUpdates( const Context<T>& context,
																		DiscreteValues<T>* discrete_state_update ) const {

	double current_state = context.get_discrete_state(0)->GetAtIndex(0);
	cout << "Updating state from: " << current_state;
	double next_state = current_state + 1;

	if ( next_state > 100 ) {
		next_state = 100;
	}

	cout << " to: " << next_state << endl;
	(*discrete_state_update)[0] = next_state;
}
																		

template class FaultySensor<double>;

} // namespace automotive
} // namespace drake
