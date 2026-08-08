import { useState } from "react";
import {
  PackageCheck,
  RadioTower,
} from "lucide-react";

import HarvestingEdaTab from "./eda";
import HarvestingModelTrainingTab from "./model";
import { Panel } from "../../components/common/Panel";
import HarvestingModuleTabs from "./HarvestingModuleTabs";
import ClassifierDerivedHuiPredictionTab from "./live/ClassifierDerivedHuiPredictionTab";
import LiveIoTHuiPredictionTab from "./live/LiveIoTHuiPredictionTab";
import "../../styles/harvesting.css";

export function HarvestingPage() {
  const [activeModuleTab, setActiveModuleTab] = useState(
    "exploratory-analysis",
  );

  return (
    <div className="page-stack">
      <section className="hero compact">
        <div>
          <span className="eyebrow">
            MODULE 4 Â· TIME-OPTIMAL HONEY HARVESTING
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

      <section
        className="harvesting-video-panel"
        aria-label="Time-optimal honey harvesting demonstration video"
      >
        <video
          className="harvesting-autoplay-video"
          src="/videos/harvest.mp4"
          autoPlay
          unmuted
          loop
          playsInline
          preload="auto"
          disablePictureInPicture
          controlsList="nodownload noplaybackrate"
        >
          Your browser does not support the video element.
        </video>
      </section>

      <HarvestingModuleTabs
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
