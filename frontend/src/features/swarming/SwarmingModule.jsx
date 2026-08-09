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
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: "1.5rem",
          padding: "1.5rem",
          background: "#ffffff",
          border: "1px solid #e2e8f0",
          borderRadius: "14px",
          boxShadow: "0 4px 12px rgba(15, 23, 42, 0.05)",
        }}
      >
        <div>
          <p
            style={{
              margin: "0 0 6px",
              color: "#2563eb",
              fontSize: "0.75rem",
              fontWeight: 700,
              letterSpacing: "0.08em",
              textTransform: "uppercase",
            }}
          >
            Module 02
          </p>

          <h2
            style={{
              margin: "0 0 8px",
              color: "#0f172a",
              fontSize: "1.6rem",
              fontWeight: 700,
            }}
          >
            Colony Swarming Prediction
          </h2>

          <p
            style={{
              margin: 0,
              color: "#64748b",
              lineHeight: 1.6,
            }}
          >
        Early identification of colony behavioural changes, 72-hour swarming
        forecasts, current risk assessment and continuous per-hive monitoring.
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

      {/* Swarming Video */}
      <section
        aria-label="Honeybee swarming behavior video"
        style={{
          width: "100%",
          overflow: "hidden",
          backgroundColor: "#0f172a",
          border: "1px solid #dbe3ec",
          borderRadius: "14px",
          boxShadow: "0 4px 14px rgba(15, 23, 42, 0.08)",
        }}
      >
        <video
          src="/videos/swarming.mp4"
          autoPlay
          muted
          loop
          playsInline
          preload="auto"
          disablePictureInPicture
          controlsList="nodownload noplaybackrate"
          style={{
            display: "block",
            width: "100%",
            height: "auto",
            maxHeight: "520px",
            objectFit: "cover",
            backgroundColor: "#0f172a",
          }}
        >
          Your browser does not support the video element.
        </video>
      </section>

      {/* Module Navigation */}
      <ModuleTabs activeTab={activeTab} onChange={setActiveTab} />

      {/* Selected Tab Content */}
      <div role="tabpanel">{renderActiveTab()}</div>
    </div>
  );
}