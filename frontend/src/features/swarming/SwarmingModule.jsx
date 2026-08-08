import React, { useState } from "react";
import { Zap } from "lucide-react";

import SwarmPrediction from "./SwarmPrediction";
import SwarmTraining from "./SwarmTraining";
import SwarmExploratory from "./SwarmExploratory";
import { ModuleTabs } from "../shared/ModuleTabs";

export default function SwarmingModule() {
  const [activeTab, setActiveTab] = useState("exploratory-analysis");

  const renderActiveTab = () => {
    switch (activeTab) {
      case "model-training":
        return <SwarmTraining />;

      case "live-early-warning":
        return <SwarmPrediction />;

      case "exploratory-analysis":
      default:
        return <SwarmExploratory />;
    }
  };

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: "1.5rem",
      }}
    >
    {/* Header Section */}
<div className="dashboard-grid">
  <div
    className="card welcome-card"
    style={{
      border: "1px solid #dbe4f0",
      borderLeft: "4px solid #2563eb",
      borderRadius: "16px",
      background:
        "linear-gradient(135deg, #ffffff 0%, #f8fafc 60%, #eff6ff 100%)",
      boxShadow: "0 8px 24px rgba(15, 23, 42, 0.07)",
    }}
  >
    <div className="welcome-content">
      <div className="welcome-text">
        <h2
          style={{
            marginBottom: "8px",
            color: "#0f172a",
          }}
        >
          Module 2: Colony Swarming Prediction
        </h2>

        <p
          style={{
            margin: 0,
            color: "#64748b",
            lineHeight: 1.6,
          }}
        >
          Swarming is the natural reproduction mechanism where half of
          the worker colony leaves with the old queen to establish a new
          home.
        </p>
      </div>

      <div
        style={{
          width: "58px",
          height: "58px",
          flexShrink: 0,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: "#eff6ff",
          border: "1px solid #bfdbfe",
          borderRadius: "14px",
          boxShadow: "0 6px 16px rgba(37, 99, 235, 0.12)",
        }}
      >
        <Zap size={32} color="#2563eb" />
      </div>
    </div>
  </div>
</div>

      {/* Same navigation style used by the other modules */}
      <ModuleTabs
        activeTab={activeTab}
        onChange={setActiveTab}
      />

      {/* Selected Tab Content */}
      <div role="tabpanel">
        {renderActiveTab()}
      </div>
    </div>
  );
}