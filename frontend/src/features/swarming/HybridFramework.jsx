import React from "react";

const STEPS = [
  { label: "Raw Sensor Data", icon: "📡" },
  { label: "Preprocessing", icon: "⚙️" },
  { label: "PELT Change Detection", icon: "📍" },
  { label: "Feature Engineering", icon: "⚙️" },
  { label: "RF · XGB · LSTM", icon: "🧠" },
];

function StepCard({ label, icon }) {
  return (
    <div
      style={{
        minWidth: "150px",
        minHeight: "105px",
        flex: "1 1 150px",
        padding: "16px",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        gap: "9px",
        background: "#ffffff",
        border: "1px solid #dbe4f0",
        borderRadius: "12px",
        boxShadow: "0 5px 15px rgba(15, 23, 42, 0.06)",
        color: "#334155",
        textAlign: "center",
        fontSize: "13px",
        fontWeight: 600,
      }}
    >
      <span style={{ fontSize: "24px" }}>{icon}</span>
      <span>{label}</span>
    </div>
  );
}

function Arrow() {
  return (
    <span
      aria-hidden="true"
      style={{
        alignSelf: "center",
        color: "#94a3b8",
        fontSize: "22px",
        fontWeight: 700,
      }}
    >
      →
    </span>
  );
}

export default function HybridFramework() {
  return (
    <section
      style={{
        marginTop: "28px",
        padding: "24px",
        background: "#f8fafc",
        border: "1px solid #dbe4f0",
        borderRadius: "15px",
        boxShadow: "0 8px 24px rgba(15, 23, 42, 0.06)",
      }}
    >
      <h2
        style={{
          margin: "0 0 8px",
          color: "#1d4ed8",
          fontSize: "18px",
        }}
      >
        Hybrid PELT–LSTM Prediction Framework
      </h2>

      <p
        style={{
          margin: "0 0 22px",
          color: "#475569",
          fontSize: "13px",
          lineHeight: 1.65,
        }}
      >
        Behavioural change points detected by the PELT algorithm are
        transformed into temporal features and combined with hive sensor
        measurements. The LSTM achieved the best predictive performance and
        was selected as the final swarming prediction model.
      </p>

      <div
        style={{
          display: "flex",
          alignItems: "stretch",
          gap: "10px",
          flexWrap: "wrap",
        }}
      >
        {STEPS.map((step, index) => (
          <React.Fragment key={step.label}>
            <StepCard {...step} />
            {index < STEPS.length - 1 && <Arrow />}
          </React.Fragment>
        ))}

        <Arrow />

        <div
          style={{
            minWidth: "150px",
            minHeight: "105px",
            flex: "1 1 150px",
            padding: "16px",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            gap: "7px",
            background: "#ecfdf5",
            border: "2px solid #22c55e",
            borderRadius: "12px",
            color: "#166534",
            textAlign: "center",
          }}
        >
          <span style={{ fontSize: "24px" }}>🏆</span>
          <strong style={{ fontSize: "13px" }}>Best Model</strong>
          <strong style={{ color: "#16a34a", fontSize: "16px" }}>LSTM</strong>
        </div>
      </div>
    </section>
  );
}