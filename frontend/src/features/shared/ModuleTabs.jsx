import { BarChart3, BrainCircuit, RadioTower } from 'lucide-react';

const MODULE_TABS = [
  {
    id: 'exploratory-analysis',
    label: 'Exploratory Analysis',
    icon: BarChart3,
  },
  {
    id: 'model-training',
    label: 'Model Training',
    icon: BrainCircuit,
  },
  {
    id: 'live-early-warning',
    label: 'Live Early Warning (IoT)',
    icon: RadioTower,
  },
];

export function ModuleTabs({ activeTab, onChange }) {
  return (
    <div className="module-tabs" role="tablist" aria-label="Module workspace sections">
      {MODULE_TABS.map(({ id, label, icon: Icon }) => {
        const isActive = activeTab === id;

        return (
          <button
            key={id}
            type="button"
            role="tab"
            aria-selected={isActive}
            className={`module-tab ${isActive ? 'active' : ''}`}
            onClick={() => onChange(id)}
          >
            <Icon size={17} aria-hidden="true" />
            <span>{label}</span>
          </button>
        );
      })}
    </div>
  );
}

export { MODULE_TABS };
