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
		explicit DropoutFilter( T update_period, T error_duty_cycle );

		const systems::InputPortDescriptor<T>& traffic_input() const;
		const systems::OutputPort<T>& traffic_output() const;
		std::unique_ptr<DropoutFilter<AutoDiffXd>> ToAutoDiffXd() const;

		protected:
		// System<T> override. Returns DropoutFilter<AutoDiffXd> with the same
		// update period and error duty cycle as this DropoutFilter
		DropoutFilter<AutoDiffXd>* DoToAutoDiffXd() const override;

		private:
		PoseBundle<T> MakeTrafficOutput() const;
		void CalcTrafficOutput( const systems::Context<T>& context,
											PoseBundle<T>* output ) const;
		void DoCalcDiscreteVariableUpdates( const Context<T>& context,
																				DiscreteValues<T>* discrete_state_update ) const override;
		bool DropoutFilter<T>::is_time_to_drop_frame( Context<T>& context ) const;

		// Indices for the input/output ports
		const int traffic_input_index_;
		const int traffic_output_index_;

		T error_period_;
		T error_duty_cycle_;
	};

} // namespace automotive
} // namespace drake
