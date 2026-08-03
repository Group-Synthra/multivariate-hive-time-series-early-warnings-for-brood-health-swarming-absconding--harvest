import { ModuleWorkspace } from '../shared/ModuleWorkspace';

export function BroodHealthPage() {
  return (
    <ModuleWorkspace
      title="Brood Health Prediction"
      target="brood_health_healthy_1"
      responsibilities={[
        'Module-specific EDA and health-target definition',
        'Brood Health Score, BHSI and RoD',
        'Chronological model training and evaluation',
        'Live IoT prediction and early-warning interface',
      ]}
    />
  );
}
