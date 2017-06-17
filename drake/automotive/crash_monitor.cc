#include "drake/automotive/fixed_step_crash_monitor.h"

namespace drake {

using drake::systems::BasicVector;
using drake::systems::Context;
using drake::systems::DiscreteValues;

using std::cout;
using std::endl;

namespace automotive {

template <typename T>
CrashMonitor<T>::CrashMonitor( T epsilon, T update_period ) 
	:	epsilon_{ epsilon }, 
		traffic_input_index_{ this->DeclareAbstractInputPort().get_index() },
	  crash_count_index_{ this->DeclareVectorOutputPort( 
	  															BasicVector<T>(1),
	  															&CrashMonitor::CalcNumberOfCrashes ) } {

	  // to keep track of the crashes so far
	  this->DeclareDiscreteState(1);
	  this->DeclareDiscreteUpdatePeriodSec( update_period );
		// ... additional setup for witness functions use?
}

template <typename T>
const systems::InputPortDescriptor<T>& CrashMonitor<T>::traffic_input() const {
	return systems::System<T>::get_input_port( traffic_input_index_ );
}

template <typename T>
const systems::OutputPort<T>& CrashMonitor<T>::crash_count() const {
	return systems::System<T>::get_output_port( crash_count_index_ );	
}

template <typename T>
void CrashMonitor<T>::CalcNumberOfCrashes( const systems::Context<T>& context,
																						BasicVector<T>* crash_count ) const {
	const PoseVector<T>* traffic =
		this->template EvalVectorInput<PoseVector<T>>( context, traffic_input_index_ );
	DRAKE_ASSERT( traffic != nullptr );

	crash_count->SetFromVector( context.get_discrete_state(0) );

}

template <typename T>
void CrashMonitor<T>::DoCalcDiscreteVariableUpdates( const Context<T>& context,
																											DiscreteValues<T>* state_updates ) const {

	T crash_count = context.get_discrete_state(0)->GetAtIndex(0);

	const PoseVector<T>* traffic_input =
		this->template EvalVectorInput<PoseVector<T>>( context, traffic_input_index_ );
	DRAKE_ASSERT( traffic_input != nullptr );

	if ( has_crash( traffic_input ) ) {
		(*state_updates) = crash_count + 1;
	} else {
		(*state_updates) = crash_count;
	}
}

template <typename T>
bool has_crash( const PoseVector<T>& traffic ) {
	cout << "Checking for crashes...";	

	for ( int k = 0; k < traffic.get_num_poses(); k++ ) {
		Isometry3<T>& this_pose = traffic.get_pose( k );
		cout << "Inspecting pose: " << k << endl;
		cout << this_pose << endl;
	}

	return true;
}
// TODO: Write the has_crash function)


} // namespace automotive
} // namespace drake
