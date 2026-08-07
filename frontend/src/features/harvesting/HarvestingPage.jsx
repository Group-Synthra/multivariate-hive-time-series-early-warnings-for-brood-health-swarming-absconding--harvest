import { useState } from "react";
import {
  PackageCheck,
  RadioTower,
} from "lucide-react";

import HarvestingEdaTab from "./eda";
import HarvestingModelTrainingTab from "./model";
import { Panel } from "../../components/common/Panel";
import { ModuleTabs } from "../shared/ModuleTabs";
import ClassifierDerivedHuiPredictionTab from "./live/ClassifierDerivedHuiPredictionTab";
import LiveIoTHuiPredictionTab from "./live/LiveIoTHuiPredictionTab";
// function LiveWarningPlaceholder() {
//   return (
//     <Panel
//       title="Live Early Warning (IoT)"
//       subtitle="This tab will use the calibrated model to produce HUI, HRSI, HRRoC and the candidate harvest window."
//     >
//       <div className="empty-state">
//         <div>
//           <RadioTower size={34} aria-hidden="true" />
//           <h3>Live prediction is not connected yet</h3>
//           <p>
//             The next stage is training-only probability
//             calibration, followed by readiness scoring and the
//             prediction API.
//           </p>
//         </div>
//       </div>
//     </Panel>
//   );
// }

export function HarvestingPage() {
  const [activeModuleTab, setActiveModuleTab] = useState(
    "exploratory-analysis",
  );

  return (
    <div className="page-stack">
      <section className="hero compact">
        <div>
          <span className="eyebrow">
            MODULE 4 · TIME-OPTIMAL HONEY HARVESTING
          </span>
          <h2>Honey Harvesting Decision Support</h2>
          <p>
            Reviewed-event analysis and 72-hour probable-harvest
            forecasting from hive weight and environmental
            time-series data.
          </p>
        </div>
        <PackageCheck size={42} aria-hidden="true" />
      </section>

      <ModuleTabs
        activeTab={activeModuleTab}
        onChange={setActiveModuleTab}
      />

      {activeModuleTab === "exploratory-analysis" && (
        <HarvestingEdaTab />
      )}

      {activeModuleTab === "model-training" && (
        <HarvestingModelTrainingTab />
      )}

      {activeModuleTab === "live-early-warning" && (
         <ClassifierDerivedHuiPredictionTab />
      )}

      {activeModuleTab === "live-iot-prediction" && (
        <LiveIoTHuiPredictionTab />
      )}
    </div>
  );
}
