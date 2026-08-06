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
          <h2>Current Condition and Exact Six-Hour Brood Health Forecast</h2>
          <p>
            A transparent sensor-derived score, leakage-safe multi-horizon forecasting,
            environmental stability analysis and live Sri Lankan IoT early warning.
          </p>
          <div className="brood-hero-tags">
            <span><HeartPulse size={15} /> Current score 1–100</span>
            <span><Activity size={15} /> Exact score at +6 h</span>
            <span><ShieldCheck size={15} /> Safety minimum over 1–6 h</span>
            <span><RadioTower size={15} /> PostgreSQL IoT integration</span>
          </div>
        </div>
        <div className="brood-hero-target">
          <small>PRIMARY MODEL OUTPUT</small>
          <code>brood_health_score_t_plus_6h</code>
          <span>Critical · Poor · Good · Excellent</span>
          <small className="brood-secondary-target">
            Secondary warning: minimum of the predicted 1–6 h trajectory
          </small>
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
