#pragma once


#include "drake/common/drake_copyable.h"
#include "drake/systems/framework/leaf_system.h"
#include "drake/systems/rendering/pose_bundle.h"


namespace drake {

using systems::rendering::PoseBundle;

namespace automotive {
	
	template <typename T>
	class FaultySensor : public systems::LeafSystem<T> {
		public:
		DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(FaultySensor)
		explicit FaultySensor();

		const systems::InputPortDescriptor<T>& traffic_input() const;
		const systems::OutputPort<T>& traffic_output() const;

		private:
		void CalcTrafficOutput( const systems::Context<T>& context,
											PoseBundle<T>* output ) const;

		// Indices for the input/output ports
		const int traffic_input_index_;
		const int traffic_output_index_;
	};

} // namespace automotive
} // namespace drake
