import { ModuleWorkspace } from '../shared/ModuleWorkspace';

export function SwarmingPage() {
  return (
    <ModuleWorkspace
      title="Colony Swarming Prediction"
      target="swarming_happened_1"
      responsibilities={[
        'Future swarming-event target construction',
        'Swarming-specific temporal and regime features',
        'Imbalance-aware model evaluation',
        'Live swarming-risk and warning interface',
      ]}
    />
  );
}
