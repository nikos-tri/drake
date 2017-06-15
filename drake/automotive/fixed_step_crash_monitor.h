#pragma once

#include "drake/common/drake_copyable.h"
#include "drake/systems/framework/leaf_system.h"
#include "drake/systems/rendering/pose_bundle.h"

namespace drake {
namespace automotive {

template <typename T>
class FixedStepCrashMonitor : public systems::LeafSystem<T> {

	public:
	DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN( FixedStepCrashMonitor )

	// TODO(nikos-tri) Improve documentation
	// Takes as input a PoseBundle, checks if two cars are within epsilon_
	// distance of each other. If yes, updates its state, which represents crash
	// count. Outputs number of crashes
	explicit FixedStepCrashMonitor( T epsilon_, T update_rate_ );

	const systems::InputPortDescriptor<T>& traffic_input() const;
	const systems::OutputPort<T>& crash_count() const;

	private:
	T epsilon_;
	const int traffic_input_index_;
	const int crash_count_index_;


};

} // namespace automotive
} // namespace drake
