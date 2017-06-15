#include "drake/automotive/fixed_step_crash_monitor.h"

namespace drake {
namespace automotive {

template <typename T>
FixedStepCrashMonitor<T>::FixedStepCrashMonitor( T epsilon, T update_rate ) 
	: traffic_input_index_{ ... },
	  crash_count_index_{ ... } {

		epsilon_ = epsilon;
		// Declare update rate
		// Declare discrete internal state: crash_count_
}

template <typename T>
const systems::InputPortDescriptor<T>& FixedStepCrashMonitor<T>::traffic_input() {

}

template <typename T>
const systems::OutputPort<T>& FixedStepCrashMonitor<T>::crash_count() {

}

template <typename T>

} // namespace automotive
} // namespace drake
