import { useState } from 'react';
import { Activity, HeartPulse, RadioTower, ShieldCheck } from 'lucide-react';
import { ModuleTabs } from '../shared/ModuleTabs';
import { BroodExploratoryTab } from './components/BroodExploratoryTab';
import { BroodTrainingTab } from './components/BroodTrainingTab';
import { BroodIoTTab } from './components/BroodIoTTab';
import './brood-health.css';

export function BroodHealthPage() {
  const [activeTab, setActiveTab] = useState('exploratory-analysis');

  return (
    <div className="page-stack brood-health-page">
      <section className="hero brood-hero">
        <div>
          <span className="eyebrow">BROOD HEALTH INTELLIGENCE</span>
          <h2>Brood Health Monitoring & Six-Hour Forecast</h2>
          <p>Monitor current health, future health, forecast stability and deterioration in one dashboard.</p>
          <div className="brood-hero-tags">
            <span><HeartPulse size={15} /> Current score 1–100</span>
            <span><Activity size={15} /> Exact score at +6 h</span>
            <span><ShieldCheck size={15} /> Safety minimum over 1–6 h</span>
            <span><RadioTower size={15} /> PostgreSQL IoT integration</span>
          </div>
        </div>
        <div className="brood-hero-target">
          <small>FORECAST HORIZON</small>
          <code>Exact +6 hours</code>
          <span>Critical · Poor · Good · Excellent</span>
          <small className="brood-secondary-target">Forecast BHSI · Forecast RoD · Composite alert</small>
        </div>
      </section>

      <ModuleTabs activeTab={activeTab} onChange={setActiveTab} />

      <div role="tabpanel" aria-label={activeTab}>
        {activeTab === 'exploratory-analysis' && <BroodExploratoryTab />}
        {activeTab === 'model-training' && <BroodTrainingTab />}
        {activeTab === 'live-early-warning' && <BroodIoTTab />}
      </div>
    </div>
  );
}
