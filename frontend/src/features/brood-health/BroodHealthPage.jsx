import { useState } from 'react';
import { Activity, HeartPulse, RadioTower } from 'lucide-react';
import { ModuleTabs } from '../shared/ModuleTabs';
import { BroodExploratoryTab } from './components/BroodExploratoryTab';
import { BroodTrainingTab } from './components/BroodTrainingTab';
import { BroodIoTTab } from './components/BroodIoTTab';

export function BroodHealthPage() {
  const [activeTab, setActiveTab] = useState('exploratory-analysis');
  return (
    <div className="page-stack brood-health-page">
      <section className="hero brood-hero">
        <div>
          <span className="eyebrow">BROOD HEALTH INTELLIGENCE</span>
          <h2>Current and Future Brood Health Score Prediction</h2>
          <p>Historical exploratory analysis, brood-specific feature engineering, unseen-hive model validation and live Sri Lankan IoT prediction in one reproducible workflow.</p>
          <div className="brood-hero-tags">
            <span><HeartPulse size={15} /> Current score 1–100</span>
            <span><Activity size={15} /> Future minimum score</span>
            <span><RadioTower size={15} /> PostgreSQL IoT integration</span>
            <span>BHSI and RoD</span>
          </div>
        </div>
        <div className="brood-hero-target">
          <small>MODEL TARGET</small>
          <code>future_minimum_brood_health_score</code>
          <span>Critical · Poor · Good · Excellent</span>
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
