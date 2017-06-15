#include "drake/automotive/faulty_sensor.h"

namespace drake {

using systems::rendering::PoseBundle;

namespace automotive {

template <typename T>
FaultySensor<T>::FaultySensor() 
	: traffic_input_index_{this->DeclareAbstractInputPort().get_index()},
	traffic_output_index_{this->DeclareAbstractOutputPort( &FaultySensor::CalcTrafficOutput).get_index() } { 
	// Nothing else is needed at this point.
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
void FaultySensor<T>::CalcTrafficOutput( const systems::Context<T>& context,
                                    PoseBundle<T>* output_traffic_poses ) const {


	const PoseBundle<T>* const input_traffic_poses = this->template EvalInputValue<PoseBundle<T>>( 
																															context,
																															traffic_input_index_ );
	DRAKE_ASSERT( input_traffic_poses != nullptr );

//	PoseBundle<T>* output_traffic_poses = &output->GetMutableData( traffic_output_index_ )
//	                                             ->template GetMutableValue<PoseBundle<T>>();
	DRAKE_ASSERT( output_traffic_poses != nullptr );

	PoseBundle<T> copy_of_input( *input_traffic_poses );
	*output_traffic_poses = copy_of_input;
}

template class FaultySensor<double>;

} // namespace automotive
} // namespace drake
