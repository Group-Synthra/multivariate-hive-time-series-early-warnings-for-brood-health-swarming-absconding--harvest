import { ModuleWorkspace } from '../shared/ModuleWorkspace';

export function HarvestingPage() {
  return (
    <ModuleWorkspace
      title="Honey Harvesting Prediction"
      target="honey_harvested_1"
      responsibilities={[
        'Future harvest-window target construction',
        'Weight-accumulation and stability features',
        'Harvest-readiness model evaluation',
        'Live readiness and recommendation interface',
      ]}
    />
  );
}
