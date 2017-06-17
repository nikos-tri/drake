#pragma once

#include "drake/common/drake_copyable.h"
#include "drake/systems/framework/leaf_system.h"
#include "drake/systems/rendering/pose_bundle.h"

namespace drake {

using drake::systems::Context;
using drake::systems::DiscreteValues;

using std::cout;
using std::endl;

namespace automotive {

template <typename T>
class CrashMonitor : public systems::LeafSystem<T> {

	public:
	DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN( CrashMonitor )

	// TODO(nikos-tri) Improve documentation
	// Takes as input a PoseBundle, checks if two cars are within epsilon_
	// distance of each other. If yes, updates its state, which represents crash
	// count. Outputs number of crashes
	explicit CrashMonitor( T epsilon_, T update_period );

	const systems::InputPortDescriptor<T>& traffic_input() const;
	const systems::OutputPort<T>& crash_count() const;

	private:
	void DoCalcDiscreteVariableUpdates( const Context<T>& context,
																			DiscreteValues<T>* state_updates ) const;
	bool has_crash( const PoseVector<T>& traffic ) const;


	const T epsilon_;
	const int traffic_input_index_;
	const int crash_count_index_;


};

} // namespace automotive
} // namespace drake
