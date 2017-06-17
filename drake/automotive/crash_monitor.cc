#include "drake/automotive/crash_monitor.h"

namespace drake {


using std::cout;
using std::endl;

namespace automotive {

template <typename T>
CrashMonitor<T>::CrashMonitor( T epsilon, T update_period ) 
	:	epsilon_{ epsilon }, 
		traffic_input_index_{ this->DeclareAbstractInputPort().get_index() },
	  crash_count_index_{ this->DeclareVectorOutputPort( 
	  															BasicVector<T>(1),
	  															&CrashMonitor::CalcNumberOfCrashes ).get_index() } {

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
	const PoseBundle<T>* traffic =
		this->template EvalInputValue<PoseBundle<T>>( context, traffic_input_index_ );
	DRAKE_ASSERT( traffic != nullptr );

	T crashes_counted = context.get_discrete_state(0)->GetAtIndex(0);

	BasicVector<T> copy_of_state( crashes_counted );
  crash_count->get_mutable_value() << crashes_counted;

}

template <typename T>
void CrashMonitor<T>::DoCalcDiscreteVariableUpdates( const Context<T>& context,
																											DiscreteValues<T>* state_updates ) const {

	T crash_count = context.get_discrete_state(0)->GetAtIndex(0);

	const PoseBundle<T>* traffic_input =
		this->template EvalInputValue<PoseBundle<T>>( context, traffic_input_index_ );
	DRAKE_ASSERT( traffic_input != nullptr );

	if ( has_crash( *traffic_input ) ) {
		(*state_updates)[0] = crash_count + 1;
	} else {
		(*state_updates)[0] = crash_count;
	}
}

template <typename T>
bool CrashMonitor<T>::has_crash( const PoseBundle<T>& traffic ) const {
	cout << "Checking for crashes...";	

	for ( int k = 0; k < traffic.get_num_poses(); k++ ) {
		Isometry3<T> this_pose = traffic.get_pose( k );
		cout << "Inspecting pose: " << k << endl;
		//cout << this_pose << endl;
	}

	return true;
}

template class CrashMonitor<double>;


} // namespace automotive
} // namespace drake
