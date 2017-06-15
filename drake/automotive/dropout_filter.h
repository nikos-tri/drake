#pragma once


#include "drake/common/drake_copyable.h"
#include "drake/systems/framework/leaf_system.h"
#include "drake/systems/rendering/pose_bundle.h"


namespace drake {

using systems::rendering::PoseBundle;
using systems::DiscreteValues;
using systems::Context;

namespace automotive {
	
	template <typename T>
	class DropoutFilter : public systems::LeafSystem<T> {
		public:
		DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(DropoutFilter)
		explicit DropoutFilter( double update_period, double error_threshold );

		const systems::InputPortDescriptor<T>& traffic_input() const;
		const systems::OutputPort<T>& traffic_output() const;

		private:
		PoseBundle<T> MakeTrafficOutput() const;
		void CalcTrafficOutput( const systems::Context<T>& context,
											PoseBundle<T>* output ) const;
		void DoCalcDiscreteVariableUpdates( const Context<T>& context,
																				DiscreteValues<T>* discrete_state_update ) const override;

		// Indices for the input/output ports
		const int traffic_input_index_;
		const int traffic_output_index_;

		double error_period_;
		double error_duty_cycle_;
	};

} // namespace automotive
} // namespace drake
