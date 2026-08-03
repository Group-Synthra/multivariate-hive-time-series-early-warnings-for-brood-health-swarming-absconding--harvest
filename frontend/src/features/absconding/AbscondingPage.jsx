import { ModuleWorkspace } from '../shared/ModuleWorkspace';

export function AbscondingPage() {
  return (
    <ModuleWorkspace
      title="Absconding Prediction"
      target="absconding_happened_1"
      responsibilities={[
        'Future absconding-window definition',
        'Long-term deterioration and stress features',
        'Rare-event and anomaly baseline evaluation',
        'Live absconding-risk interface',
      ]}
    />
  );
}
