import { useState } from 'react';
import { Construction } from 'lucide-react';
import { Panel } from '../../components/common/Panel';
import { ModuleTabs } from './ModuleTabs';

export function ModuleWorkspace({ member, title, target, responsibilities }) {
  const [activeModuleTab, setActiveModuleTab] = useState('exploratory-analysis');

  return (
    <div className="page-stack">
      <section className="hero compact">
        <div>
          <span className="eyebrow">{member}</span>
          <h2>{title}</h2>
        </div>
        <Construction size={42} />
      </section>

      <ModuleTabs
        activeTab={activeModuleTab}
        onChange={setActiveModuleTab}
      />
    </div>
  );
}
