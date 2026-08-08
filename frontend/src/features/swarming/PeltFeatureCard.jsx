import React from "react";

export default function PeltFeatureCard({
  title,
  icon,
  description,
  color = "#2563eb",
}) {
  return (
    <div
      style={{
        height: "100%",
        padding: "20px",
        background: "#ffffff",
        border: "1px solid #dbe4f0",
        borderTop: `4px solid ${color}`,
        borderRadius: "14px",
        boxShadow: "0 6px 18px rgba(15, 23, 42, 0.07)",
        transition: "transform 0.2s ease, box-shadow 0.2s ease",
      }}
      onMouseEnter={(event) => {
        event.currentTarget.style.transform = "translateY(-3px)";
        event.currentTarget.style.boxShadow =
          "0 10px 24px rgba(15, 23, 42, 0.12)";
      }}
      onMouseLeave={(event) => {
        event.currentTarget.style.transform = "translateY(0)";
        event.currentTarget.style.boxShadow =
          "0 6px 18px rgba(15, 23, 42, 0.07)";
      }}
    >
      <div
        style={{
          width: "48px",
          height: "48px",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          marginBottom: "14px",
          background: `${color}12`,
          border: `1px solid ${color}40`,
          borderRadius: "12px",
          fontSize: "25px",
        }}
      >
        {icon}
      </div>

      <h3
        style={{
          margin: "0 0 9px",
          color,
          fontSize: "15px",
          fontWeight: 700,
        }}
      >
        {title}
      </h3>

      <p
        style={{
          margin: 0,
          color: "#475569",
          fontSize: "13px",
          lineHeight: 1.6,
        }}
      >
        {description}
      </p>
    </div>
  );
}