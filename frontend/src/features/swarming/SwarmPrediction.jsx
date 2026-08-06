import React, { useState, useEffect, useCallback, useRef, useMemo } from "react";
import {
  AreaChart, Area, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, Legend,
} from "recharts";

// ─────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────
const API_BASE = ""; // Vite proxy forwards /api/* → http://localhost:5000
const REFRESH_INTERVAL = 600_000; // 10 minutes

function secondsUntilRefresh(dataTimestamp) {
  const dataTime = new Date(dataTimestamp).getTime();
  if (!Number.isFinite(dataTime)) {
    return REFRESH_INTERVAL / 1000;
  }

  const nextRefreshTime = dataTime + REFRESH_INTERVAL;
  return Math.max(0, Math.ceil((nextRefreshTime - Date.now()) / 1000));
}

const RISK_CONFIG = {
  LOW: { color: "#22c55e", bg: "#ecfdf5", border: "#16a34a", emoji: "🟢", label: "LOW RISK", statusDesc: "Hive behaviour is normal. No immediate swarming risk detected." },
  MEDIUM: { color: "#eab308", bg: "#fffbeb", border: "#ca8a04", emoji: "🟡", label: "MEDIUM RISK", statusDesc: "Increased activity detected — monitor closely." },
  HIGH: { color: "#ef4444", bg: "#fef2f2", border: "#dc2626", emoji: "🔴", label: "HIGH RISK", statusDesc: "Immediate hive inspection recommended!" },
};

// Sensor display metadata
const SENSOR_META = [
  { key: "internal_temperature_c", label: "Internal Temp", unit: "°C", icon: "🌡️", color: "#dc2626", bg: "#fff7f7", border: "#fecaca" },
  { key: "internal_humidity_pct", label: "Internal Humidity", unit: "%", icon: "💧", color: "#2563eb", bg: "#eff6ff", border: "#bfdbfe" },
  { key: "co2_ppm", label: "CO₂", unit: "ppm", icon: "💨", color: "#7c3aed", bg: "#f5f3ff", border: "#ddd6fe" },
  { key: "hive_weight_kg", label: "Hive Weight", unit: "kg", icon: "⚖️", color: "#d97706", bg: "#fffbeb", border: "#fde68a" },
  // { key: "external_temperature_c", label: "External Temp", unit: "°C", icon: "☀️", color: "#ea580c", bg: "#fff7ed", border: "#fed7aa" },
  // { key: "external_humidity_pct", label: "External Humidity", unit: "%", icon: "🌤️", color: "#0891b2", bg: "#ecfeff", border: "#a5f3fc" },
  // { key: "battery_voltage", label: "Battery Voltage", unit: "V", icon: "🔋", color: "#16a34a", bg: "#f0fdf4", border: "#bbf7d0" },
];

const TIME_RANGES = [
  { label: "1H", minutes: 60 },
  { label: "6H", minutes: 360 },
  { label: "12H", minutes: 720 },
  { label: "24H", minutes: 1440 },
  { label: "7D", minutes: 10080 },
];

const RISK_TIME_RANGES = [
  { label: "30M", minutes: 30 },
  ...TIME_RANGES,
];

const TREND_SENSORS = [
  { key: "internal_temperature_c", label: "Internal Temperature", unit: "°C", color: "#ef4444", icon: "🌡️" },
  { key: "internal_humidity_pct", label: "Internal Humidity", unit: "%", color: "#2563eb", icon: "💧" },
  { key: "co2_ppm", label: "CO₂", unit: "ppm", color: "#8b5cf6", icon: "💨" },
  { key: "hive_weight_kg", label: "Hive Weight", unit: "kg", color: "#f59e0b", icon: "⚖️" },
  { key: "external_temperature_c", label: "External Temperature", unit: "°C", color: "#f97316", icon: "☀️" },
  { key: "external_humidity_pct", label: "External Humidity", unit: "%", color: "#06b6d4", icon: "🌤️" },
  { key: "battery_voltage", label: "Battery Voltage", unit: "V", color: "#16a34a", icon: "🔋" },
];

const RISK_THRESHOLD = 70;
const MEDIUM_THRESHOLD = 35;

// ─────────────────────────────────────────────────────────────────────
// Sub-components
// ─────────────────────────────────────────────────────────────────────

/** Animated circular risk gauge */
function RiskGauge({ percentage, riskLevel, label }) {
  const cfg = RISK_CONFIG[riskLevel] || RISK_CONFIG.LOW;
  const radius = 72;
  const circumference = 2 * Math.PI * radius;
  const strokeDash = (percentage / 100) * circumference;
  const isHigh = riskLevel === "HIGH";

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "8px" }}>
      <div style={{ position: "relative", width: 180, height: 180 }}>
        <svg width={180} height={180} style={{ transform: "rotate(-90deg)" }}>
          <circle cx={90} cy={90} r={radius} fill="none" stroke="#dbe4f0" strokeWidth={14} />
          <circle
            cx={90} cy={90} r={radius} fill="none"
            stroke={cfg.color} strokeWidth={14}
            strokeDasharray={`${strokeDash} ${circumference}`}
            strokeLinecap="round"
            style={{ transition: "stroke-dasharray 1.2s ease-in-out, stroke 0.5s" }}
          />
        </svg>
        <div style={{
          position: "absolute", inset: 0,
          display: "flex", flexDirection: "column",
          alignItems: "center", justifyContent: "center",
        }}>
          <span style={{
            fontSize: "2.2rem", fontWeight: 800, lineHeight: 1,
            color: cfg.color, fontFamily: "'Outfit', sans-serif",
            animation: isHigh ? "pulseNumber 1.4s ease-in-out infinite" : "none",
          }}>
            {Number(percentage ?? 0).toFixed(2)}
          </span>
          <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>%</span>
        </div>
      </div>
      <div style={{
        display: "inline-flex", alignItems: "center", gap: "6px",
        background: cfg.bg, border: `1.5px solid ${cfg.border}`,
        borderRadius: "20px", padding: "5px 16px",
        fontSize: "0.8rem", fontWeight: 700, color: cfg.color,
        animation: isHigh ? "pulseBadge 1.4s ease-in-out infinite" : "none",
      }}>
        {cfg.emoji} {cfg.label}
      </div>
      {label && (
        <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", marginTop: "4px" }}>
          {label}
        </div>
      )}
    </div>
  );
}

/** Real-time sensor value display card */
function SensorCard({ meta, value, isLive }) {
  const displayVal = value != null ? value : "—";
  return (
    <div style={{
      background: meta.bg,
      border: `1px solid ${meta.border}`,
      borderLeft: `4px solid ${meta.color}`,
      borderRadius: "10px", padding: "10px 12px",
      position: "relative",
      boxShadow: isLive ? `0 5px 14px ${meta.color}18` : "none",
    }}>
      {isLive && (
        <span style={{
          position: "absolute", top: 6, right: 8,
          width: 6, height: 6, borderRadius: "50%",
          background: "#22c55e",
          boxShadow: "0 0 6px #22c55e",
          display: "inline-block",
        }} />
      )}
      <div style={{
        display: "flex", alignItems: "center", gap: "6px",
        fontSize: "0.72rem", color: "var(--text-secondary)", marginBottom: "6px",
      }}>
        <span style={{ fontSize: "1rem" }}>{meta.icon}</span>
        <span>{meta.label}</span>
        <span style={{ color: "var(--text-muted)", marginLeft: "auto" }}>{meta.unit}</span>
      </div>
      <div style={{
        color: meta.color, fontSize: "1.1rem", fontWeight: 700,
        fontFamily: "'Outfit', sans-serif",
      }}>
        {displayVal !== "—" ? Number(displayVal).toFixed(meta.key === "co2_ppm" ? 0 : 1) : "—"}
      </div>
    </div>
  );
}

/** 3-Day Forecast Card */
function ForecastCard({ forecast }) {
  if (!forecast || forecast.length === 0) return null;

  const getRiskEmoji = (level) => {
    const map = { LOW: "🟢", MEDIUM: "🟡", HIGH: "🔴" };
    return map[level] || "⚪";
  };

  const today = new Date();
  const dayLabels = [1, 2, 3].map((d) => {
    const dt = new Date(today);
    dt.setDate(today.getDate() + d);
    return dt.toLocaleDateString("en-US", { weekday: "short", month: "short", day: "numeric" });
  });

  return (
    <div style={{
      background: "#eff6ff",
      borderLeft: "4px solid #2563eb",
      borderRadius: "10px",
      padding: "14px 16px",
      marginTop: "8px",
    }}>
      <h4 style={{ margin: "0 0 12px", fontSize: "0.85rem", color: "#2563eb" }}>
        📊 3-Day Swarming Forecast
      </h4>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "10px" }}>
        {forecast.map((d, i) => (
          <div key={d.day} style={{
            background: "#f8fafc", borderRadius: "8px",
            padding: "10px 8px", textAlign: "center",
            border: `1px solid ${d.color}33`,
          }}>
            <div style={{ fontSize: "0.65rem", color: "var(--text-muted)", marginBottom: "4px" }}>
              {dayLabels[i]}
            </div>
            <div style={{ fontSize: "1.4rem", fontWeight: 800, color: d.color, fontFamily: "'Outfit',sans-serif" }}>
              {d.risk}%
            </div>
            <div style={{ fontSize: "0.7rem", color: d.color, marginTop: "2px" }}>
              {getRiskEmoji(d.level)} {d.level}
            </div>
          </div>
        ))}
      </div>
      <div style={{
        marginTop: "10px", fontSize: "0.68rem", color: "var(--text-muted)",
        textAlign: "center", borderTop: "1px solid #dbe4f0", paddingTop: "6px",
      }}>
        Forecast based on current hive trend · Derived from LSTM risk output
      </div>
    </div>
  );
}

/** High Risk Alert Box */
function HighRiskAlert({ riskLevel, percentage }) {
  if (riskLevel !== "HIGH") return null;
  return (
    <div style={{
      background: "linear-gradient(135deg, #fef2f2, #fee2e2)",
      border: "2px solid #dc2626",
      borderRadius: "12px",
      padding: "16px 20px",
      marginTop: "8px",
      animation: "pulseAlert 1.5s ease-in-out infinite",
    }}>
      <style>{`
        @keyframes pulseAlert {
          0%, 100% { box-shadow: 0 0 0 0 rgba(220, 38, 38, 0.3); }
          50% { box-shadow: 0 0 20px 8px rgba(220, 38, 38, 0.2); }
        }
      `}</style>
      <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
        <span style={{ fontSize: "2rem" }}>🚨</span>
        <div style={{ flex: 1 }}>
          <div style={{ fontWeight: 700, color: "#ef4444", fontSize: "1rem" }}>
            HIGH SWARMING RISK DETECTED!
          </div>
          <div style={{ color: "#b91c1c", fontSize: "0.85rem" }}>
            Current risk: {percentage.toFixed(1)}% — Immediate hive inspection recommended!
          </div>
        </div>
      </div>
      <div style={{
        marginTop: "8px", padding: "8px 12px",
        background: "#f8fafc", borderRadius: "6px",
        fontSize: "0.8rem", color: "#b91c1c",
      }}>
        ⚠️ Action Required: Check for queen cells, reduce overcrowding, or consider hive splitting.
      </div>
    </div>
  );
}

/** PELT feature snapshot card */
function PeltSnapshot({ snapshot }) {
  if (!snapshot) return null;
  const items = [
    { label: "Breakpoint Detected", value: snapshot.breakpoint ? "Yes ⚠️" : "No ✅",
      color: snapshot.breakpoint ? "#ef4444" : "#22c55e" },
    { label: "Days Since Breakpoint", value: `${snapshot.days_since_breakpoint} readings`, color: "#2563eb" },
    { label: "Breakpoint Density", value: `${snapshot.breakpoint_density} / 24h`, color: "#a78bfa" },
    { label: "Segment Duration", value: `${snapshot.segment_duration} steps`, color: "#f59e0b" },
  ];
  return (
    <div style={{
      background: "#eff6ff", 
      borderLeft: "4px solid #2563eb",
      borderRadius: "10px", 
      padding: "14px 16px",
      marginTop: "8px",
    }}>
      <h4 style={{ margin: "0 0 10px", fontSize: "0.85rem", color: "#2563eb" }}>
        🔎 PELT Change-Point Snapshot
      </h4>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "8px" }}>
        {items.map(({ label, value, color }) => (
          <div key={label} style={{
            background: "#f8fafc", borderRadius: "8px", padding: "8px 10px",
          }}>
            <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", marginBottom: "2px" }}>{label}</div>
            <div style={{ fontSize: "0.9rem", fontWeight: 700, color }}>{value}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── PELT Change-Point Timeline ────────────────────────────────────────

function PeltTimeline({ peltHistory }) {
  const [activeRange, setActiveRange] = useState("6H");

  // Filter ONLY breakpoint events
  const breakpointData = useMemo(() => {
    if (!peltHistory || peltHistory.length === 0) return [];

    const rangeObj = TIME_RANGES.find((r) => r.label === activeRange) || TIME_RANGES[1];
    const cutoffMs = rangeObj.minutes * 60 * 1000;
    const now = Date.now();

    // Filter breakpoints only - ensure strict comparison
    const breakpoints = peltHistory.filter((item) => {
      const ts = item.timestamp ? new Date(item.timestamp).getTime() : 0;
      return Number(item.breakpoint) === 1 && (now - ts <= cutoffMs);
    });

    console.log('🔍 PELT History:', peltHistory.length, 'entries');
    console.log('✅ Breakpoints found:', breakpoints.length);

    return breakpoints.map((item) => {
      const date = item.timestamp ? new Date(item.timestamp) : new Date();
      return {
        time: date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
        fullTimestamp: date.toLocaleString(),
        breakpoint: 1,
        density: item.breakpoint_density || 0,
        daysSince: item.days_since_breakpoint || 0,
        segmentDuration: item.segment_duration || 0,
        isBreakpoint: true,
        fullTs: date.getTime()
      };
    });
  }, [peltHistory, activeRange]);

  const CustomTooltip = ({ active, payload, label }) => {
    if (!active || !payload?.length) return null;
    
    const data = payload[0]?.payload;
    if (!data) return null;

    return (
      <div style={{
        background: "#ffffff",
        border: "2px solid #ef4444",
        borderRadius: "10px",
        padding: "12px 16px",
        fontSize: "0.78rem",
        color: "#0f172a",
        boxShadow: "0 8px 24px rgba(15,23,42,0.12)",
        minWidth: "200px",
      }}>
        <div style={{ 
          color: "#ef4444", 
          marginBottom: "4px", 
          fontWeight: 700,
          fontSize: "0.85rem"
        }}>
          ⚠️ {data.fullTimestamp} - BREAKPOINT!
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: "4px", marginTop: "6px" }}>
          <div style={{ display: "flex", justifyContent: "space-between" }}>
            <span style={{ color: "#94a3b8" }}>Breakpoint:</span>
            <span style={{ color: "#ef4444", fontWeight: 600 }}>⚠️ Yes</span>
          </div>
          <div style={{ display: "flex", justifyContent: "space-between" }}>
            <span style={{ color: "#94a3b8" }}>Density:</span>
            <span style={{ color: "#a78bfa", fontWeight: 600 }}>{data.density} / 24h</span>
          </div>
          <div style={{ display: "flex", justifyContent: "space-between" }}>
            <span style={{ color: "#94a3b8" }}>Days Since:</span>
            <span style={{ color: "#2563eb", fontWeight: 600 }}>{data.daysSince} readings</span>
          </div>
          <div style={{ display: "flex", justifyContent: "space-between" }}>
            <span style={{ color: "#94a3b8" }}>Segment:</span>
            <span style={{ color: "#f59e0b", fontWeight: 600 }}>{data.segmentDuration} steps</span>
          </div>
        </div>
      </div>
    );
  };

  const renderDot = (props) => {
    const { cx, cy, payload } = props;
    if (payload && payload.isBreakpoint) {
      return (
        <g>
          <circle
            cx={cx}
            cy={cy}
            r={14}
            fill="#ef4444"
            opacity="0.15"
          />
          <circle
            cx={cx}
            cy={cy}
            r={10}
            fill="#ef4444"
            opacity="0.3"
          />
          <circle
            cx={cx}
            cy={cy}
            r={8}
            fill="#ef4444"
            stroke="#ffffff"
            strokeWidth={2}
            style={{ cursor: "pointer" }}
          />
          <circle
            cx={cx}
            cy={cy}
            r={3}
            fill="#ffffff"
            opacity="0.8"
          />
          <text
            x={cx}
            y={cy + 20}
            textAnchor="middle"
            fill="#ef4444"
            fontSize="7"
            fontWeight="600"
            fontFamily="'Outfit', sans-serif"
            style={{ pointerEvents: "none" }}
          >
            {payload.time}
          </text>
        </g>
      );
    }
    return null;
  };

  const hasBreakpoints = breakpointData.length > 0;

  return (
    <div style={{
      background: "#ffffff",
      border: "1px solid #dbe4f0",
      borderTop: "4px solid #ef4444",
      borderRadius: "16px",
      padding: "20px 24px 14px",
      boxShadow: "0 6px 20px rgba(15,23,42,0.06)",
      width: "100%",
      marginTop: "8px",
    }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "18px", flexWrap: "wrap", gap: "8px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <span style={{ fontSize: "1.1rem" }}>🔎</span>
          <span style={{
            fontFamily: "'Outfit', sans-serif",
            fontWeight: 700,
            fontSize: "1rem",
            color: "#1e3a5f",
          }}>
            PELT Change-Point Timeline
          </span>
          <span style={{
            fontSize: "0.6rem",
            padding: "2px 8px",
            borderRadius: "10px",
            background: "#eff6ff",
            color: "#2563eb",
            border: "1px solid #2563eb33",
          }}>
            Breakpoint Events
          </span>
          {hasBreakpoints && (
            <span style={{
              fontSize: "0.6rem",
              padding: "2px 10px",
              borderRadius: "10px",
              background: "#fef2f2",
              color: "#ef4444",
              border: "1px solid #ef444433",
            }}>
              ⚠️ {breakpointData.length} Breakpoint{breakpointData.length > 1 ? 's' : ''}
            </span>
          )}
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
          <span style={{ fontSize: "0.6rem", color: "#64748b" }}>Legend:</span>
          <span style={{ fontSize: "0.55rem", display: "flex", alignItems: "center", gap: "4px" }}>
            <span style={{ width: 12, height: 12, borderRadius: "50%", background: "#ef4444", display: "inline-block", border: "1px solid white" }} />
            <span style={{ color: "#94a3b8" }}>Breakpoint Event</span>
          </span>
        </div>
      </div>

      {/* Breakpoint Events List with Full Timestamps */}
      {hasBreakpoints && (
        <div style={{
          display: "flex",
          flexWrap: "wrap",
          gap: "8px",
          marginBottom: "12px",
          padding: "10px 14px",
          background: "rgba(239,68,68,0.08)",
          borderRadius: "8px",
          border: "1px solid rgba(239,68,68,0.15)",
        }}>
          <span style={{ fontSize: "0.7rem", color: "#b91c1c", fontWeight: 600, marginRight: "4px" }}>
            ⚠️ Breakpoints detected at:
          </span>
          {breakpointData.map((event, index) => (
            <span
              key={index}
              style={{
                fontSize: "0.6rem",
                padding: "3px 12px",
                borderRadius: "12px",
                background: "#fef2f2",
                color: "#ef4444",
                border: "1px solid #ef444433",
                fontWeight: 600,
              }}
            >
              {event.fullTimestamp}
            </span>
          ))}
        </div>
      )}

      {hasBreakpoints ? (
        <ResponsiveContainer width="100%" height={200}>
          <AreaChart data={breakpointData} margin={{ top: 20, right: 10, left: -10, bottom: 0 }}>
            <defs>
              <linearGradient id="peltDensityGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#ef4444" stopOpacity={0.35} />
                <stop offset="95%" stopColor="#ef4444" stopOpacity={0.02} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="rgba(255,255,255,0.05)" strokeDasharray="3 3" vertical={false} />
            <XAxis
              dataKey="time"
              tick={{ fill: "#64748b", fontSize: 9, fontFamily: "'Outfit', sans-serif" }}
              axisLine={{ stroke: "#dbe4f0" }}
              tickLine={false}
              interval="preserveStartEnd"
            />
            <YAxis
              domain={[0, 1.5]}
              tick={{ fill: "#64748b", fontSize: 9, fontFamily: "'Outfit', sans-serif" }}
              axisLine={false}
              tickLine={false}
              tickFormatter={() => ''}
            />
            <Tooltip content={<CustomTooltip />} />

            {/* Breakpoint Events with density as fill */}
            <Area
              type="step"
              dataKey="breakpoint"
              stroke="#ef4444"
              strokeWidth={0}
              fill="url(#peltDensityGrad)"
              dot={renderDot}
              activeDot={{ r: 10, fill: "#ef4444", stroke: "#ffffff", strokeWidth: 2 }}
              isAnimationActive
              animationDuration={600}
            />

            <Legend
              verticalAlign="bottom"
              height={28}
              formatter={() => (
                <span style={{ color: "#ef4444", fontSize: "0.68rem" }}>
                  ● Breakpoint Events with Timestamp
                </span>
              )}
            />
          </AreaChart>
        </ResponsiveContainer>
      ) : (
        <div style={{
          height: 200, 
          display: "flex", 
          alignItems: "center", 
          justifyContent: "center",
          flexDirection: "column", 
          gap: "8px",
          background: "#f8fafc",
          borderRadius: "8px",
        }}>
          <span style={{ fontSize: "2rem" }}>✅</span>
          <span style={{ color: "#22c55e", fontSize: "0.9rem", fontWeight: 600 }}>
            No Breakpoints Detected
          </span>
          <span style={{ fontSize: "0.7rem", color: "#475569" }}>
            The timeline will show breakpoint events when they occur
          </span>
        </div>
      )}

      {/* Time Range Selector */}
      <div style={{ display: "flex", gap: "6px", justifyContent: "center", marginTop: "12px" }}>
        {TIME_RANGES.map(({ label }) => {
          const isActive = activeRange === label;
          return (
            <button
              key={label}
              onClick={() => setActiveRange(label)}
              style={{
                padding: "4px 12px",
                borderRadius: "16px",
                border: `1px solid ${isActive ? "#2563eb" : "#dbe4f0"}`,
                background: isActive ? "#2563eb" : "transparent",
                color: isActive ? "#0f172a" : "#64748b",
                fontSize: "0.65rem",
                fontWeight: 600,
                fontFamily: "'Outfit', sans-serif",
                cursor: "pointer",
                transition: "all 0.18s ease",
              }}
            >
              {label}
            </button>
          );
        })}
      </div>
    </div>
  );
}

/** Selectable historical IoT sensor timeline */
function SensorTrendsTimeline({ readings }) {
  const [activeRange, setActiveRange] = useState("6H");
  const [selectedSensor, setSelectedSensor] = useState("internal_temperature_c");

  const sensor = TREND_SENSORS.find((item) => item.key === selectedSensor) || TREND_SENSORS[0];

  const chartData = useMemo(() => {
    if (!Array.isArray(readings) || readings.length === 0) return [];

    const validReadings = readings
      .map((reading) => {
        const timestamp = reading.recorded_at || reading.reading_at || reading.timestamp;
        return {
          timestamp,
          timestampMs: timestamp ? new Date(timestamp).getTime() : NaN,
          value: Number(reading[selectedSensor]),
        };
      })
      .filter((item) => Number.isFinite(item.timestampMs) && Number.isFinite(item.value))
      .sort((a, b) => a.timestampMs - b.timestampMs);

    if (validReadings.length === 0) return [];

    const range = TIME_RANGES.find((item) => item.label === activeRange) || TIME_RANGES[1];
    const cutoffMs = range.minutes * 60 * 1000;
    const latestDataTime = validReadings[validReadings.length - 1].timestampMs;

    return validReadings
      .filter((item) => latestDataTime - item.timestampMs <= cutoffMs)
      .map((item) => {
        const date = new Date(item.timestampMs);
        return {
          ...item,
          time: date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
          fullTimestamp: date.toLocaleString(),
        };
      });
  }, [readings, selectedSensor, activeRange]);

  const SensorTooltip = ({ active, payload }) => {
    if (!active || !payload?.length) return null;
    const item = payload[0].payload;
    return (
      <div style={{
        minWidth: "190px", padding: "11px 14px", background: "#ffffff",
        border: `1px solid ${sensor.color}`, borderRadius: "9px",
        boxShadow: "0 8px 24px rgba(15,23,42,0.12)",
      }}>
        <div style={{ marginBottom: "6px", color: "#64748b", fontSize: "0.72rem" }}>
          {item.fullTimestamp}
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", gap: "18px", color: "#334155", fontSize: "0.8rem" }}>
          <span>{sensor.label}</span>
          <strong style={{ color: sensor.color }}>{item.value.toFixed(2)} {sensor.unit}</strong>
        </div>
      </div>
    );
  };

  return (
    <div style={{
      width: "100%", marginTop: "8px", padding: "20px 24px 16px",
      background: "#ffffff", border: "1px solid #dbe4f0", borderRadius: "16px",
      boxShadow: "0 6px 20px rgba(15,23,42,0.06)",
    }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: "12px", marginBottom: "18px" }}>
        <div>
          <h3 style={{ margin: 0, color: "#1e3a5f", fontSize: "1rem", fontWeight: 700 }}>
            📈 Sensor Value Trends Timeline
          </h3>
          <p style={{ margin: "5px 0 0", color: "#64748b", fontSize: "0.72rem" }}>
            Historical IoT sensor readings for the selected hive
          </p>
        </div>

        <select value={selectedSensor} onChange={(event) => setSelectedSensor(event.target.value)}
          style={{
            minWidth: "210px", padding: "8px 12px", color: "#334155", background: "#ffffff",
            border: "1px solid #cbd5e1", borderRadius: "8px", fontSize: "0.78rem",
            fontWeight: 600, outline: "none", cursor: "pointer",
          }}>
          {TREND_SENSORS.map((item) => (
            <option key={item.key} value={item.key}>{item.icon} {item.label}</option>
          ))}
        </select>
      </div>

      {chartData.length > 0 ? (
        <ResponsiveContainer width="100%" height={240}>
          <AreaChart data={chartData} margin={{ top: 10, right: 20, left: 5, bottom: 5 }}>
            <defs>
              <linearGradient id="sensorTrendGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={sensor.color} stopOpacity={0.3} />
                <stop offset="95%" stopColor={sensor.color} stopOpacity={0.02} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="#1e3a5f" strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="time" tick={{ fill: "#64748b", fontSize: 10 }} axisLine={{ stroke: "#cbd5e1" }} tickLine={false} interval="preserveStartEnd" />
            <YAxis domain={["auto", "auto"]} tick={{ fill: "#64748b", fontSize: 10 }} axisLine={false} tickLine={false} width={50} tickFormatter={(value) => Number(value).toFixed(1)} />
            <Tooltip content={<SensorTooltip />} />
            <Area type="monotone" dataKey="value" name={sensor.label} stroke={sensor.color}
              strokeWidth={2.5} fill="url(#sensorTrendGradient)" dot={false}
              activeDot={{ r: 6, fill: sensor.color, stroke: "#ffffff", strokeWidth: 2 }}
              isAnimationActive animationDuration={500} />
          </AreaChart>
        </ResponsiveContainer>
      ) : (
        <div style={{
          height: "240px", display: "flex", flexDirection: "column", alignItems: "center",
          justifyContent: "center", gap: "8px", color: "#64748b", background: "#f8fafc",
          border: "1px dashed #cbd5e1", borderRadius: "10px",
        }}>
          <span style={{ fontSize: "2rem" }}>📊</span>
          <strong>No sensor data available</strong>
          <span style={{ fontSize: "0.72rem" }}>Select another time range or refresh the prediction.</span>
        </div>
      )}

      <div style={{ display: "flex", justifyContent: "center", flexWrap: "wrap", gap: "7px", marginTop: "14px" }}>
        {TIME_RANGES.map(({ label }) => {
          const isActive = activeRange === label;
          return (
            <button key={label} type="button" onClick={() => setActiveRange(label)}
              style={{
                padding: "5px 13px", color: isActive ? "#ffffff" : "#64748b",
                background: isActive ? "#2563eb" : "#ffffff",
                border: `1px solid ${isActive ? "#2563eb" : "#dbe4f0"}`,
                borderRadius: "18px", fontSize: "0.68rem", fontWeight: 600, cursor: "pointer",
              }}>
              {label}
            </button>
          );
        })}
      </div>
    </div>
  );
}

/** Live multi-sensor trend chart */
function RealtimeSensorTrendsTimeline({ readings }) {
  const [activeRange, setActiveRange] = useState("6H");

  const chartData = useMemo(() => {
    if (!Array.isArray(readings) || readings.length === 0) return [];

    const validReadings = readings
      .map((reading) => {
        const timestamp = reading.recorded_at || reading.reading_at || reading.timestamp;
        const timestampMs = timestamp ? new Date(timestamp).getTime() : NaN;
        return {
          timestampMs,
          fullTimestamp: timestamp ? new Date(timestamp).toLocaleString() : "—",
          internalTemperature: Number(reading.internal_temperature_c),
          humidity: Number(reading.internal_humidity_pct),
          co2: Number(reading.co2_ppm),
          weight: Number(reading.hive_weight_kg),
        };
      })
      .filter((item) =>
        Number.isFinite(item.timestampMs) &&
        Number.isFinite(item.internalTemperature) &&
        Number.isFinite(item.humidity) &&
        Number.isFinite(item.co2) &&
        Number.isFinite(item.weight)
      )
      .sort((a, b) => a.timestampMs - b.timestampMs);

    if (validReadings.length === 0) return [];

    const range = TIME_RANGES.find((item) => item.label === activeRange) || TIME_RANGES[1];
    const cutoffMs = range.minutes * 60 * 1000;
    const latestTimestamp = validReadings[validReadings.length - 1].timestampMs;

    return validReadings
      .filter((item) => latestTimestamp - item.timestampMs <= cutoffMs)
      .map((item) => ({
        ...item,
        time: new Date(item.timestampMs).toLocaleTimeString([], {
          hour: "2-digit",
          minute: "2-digit",
        }),
      }));
  }, [readings, activeRange]);

  const LiveTooltip = ({ active, payload }) => {
    if (!active || !payload?.length) return null;
    const item = payload[0].payload;
    const rows = [
      ["CO₂", item.co2, "ppm", "#f59e0b"],
      ["Humidity", item.humidity, "%", "#2563eb"],
      ["Temperature", item.internalTemperature, "°C", "#ef4444"],
      ["Weight", item.weight, "kg", "#16a34a"],
    ];

    return (
      <div style={{
        minWidth: "205px", padding: "11px 14px", background: "#ffffff",
        border: "1px solid #dbe4f0", borderRadius: "9px",
        boxShadow: "0 8px 24px rgba(15,23,42,0.14)",
      }}>
        <div style={{ color: "#64748b", fontSize: "0.7rem", marginBottom: "7px" }}>
          {item.fullTimestamp}
        </div>
        {rows.map(([label, value, unit, color]) => (
          <div key={label} style={{ display: "flex", justifyContent: "space-between", gap: "18px", marginTop: "4px", fontSize: "0.75rem" }}>
            <span style={{ color }}>{label}</span>
            <strong style={{ color: "#334155" }}>{Number(value).toFixed(2)} {unit}</strong>
          </div>
        ))}
      </div>
    );
  };

  const legendFormatter = (value) => {
    const labels = {
      co2: "CO₂ (ppm)",
      humidity: "Humidity (%)",
      internalTemperature: "Temp (°C)",
      weight: "Weight (kg)",
    };
    return <span style={{ color: "#475569", fontSize: "0.68rem" }}>{labels[value] || value}</span>;
  };

  return (
    <div style={{
      width: "100%", marginTop: "8px", padding: "18px 20px 14px",
      background: "#ffffff", border: "1px solid #dbe4f0", borderRadius: "16px",
      borderTop: "4px solid #8b5cf6",
      boxShadow: "0 6px 20px rgba(15,23,42,0.06)",
    }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "12px" }}>
        <div>
          <h3 style={{ margin: 0, color: "#1e3a5f", fontSize: "1rem", fontWeight: 700 }}>
            Sensor Trend (Live)
          </h3>
          <p style={{ margin: "4px 0 0", color: "#64748b", fontSize: "0.7rem" }}>
            Real-time IoT readings from the selected hive
          </p>
        </div>
        <span style={{ display: "flex", alignItems: "center", gap: "6px", color: "#16a34a", fontSize: "0.68rem", fontWeight: 700 }}>
          <span style={{ width: 8, height: 8, borderRadius: "50%", background: "#22c55e", boxShadow: "0 0 0 4px rgba(34,197,94,0.12)" }} />
          LIVE
        </span>
      </div>

      {chartData.length > 0 ? (
        <ResponsiveContainer width="100%" height={270}>
          <LineChart data={chartData} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
            <CartesianGrid stroke="#1e3a5f" strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="time" tick={{ fill: "#64748b", fontSize: 9 }} axisLine={{ stroke: "#cbd5e1" }} tickLine={false} interval="preserveStartEnd" />
            <YAxis yAxisId="sensor" orientation="left" domain={["auto", "auto"]} tick={{ fill: "#64748b", fontSize: 9 }} axisLine={false} tickLine={false} width={42} />
            <YAxis yAxisId="co2" orientation="right" domain={["auto", "auto"]} tick={{ fill: "#f59e0b", fontSize: 9 }} axisLine={false} tickLine={false} width={48} />
            <Tooltip content={<LiveTooltip />} />
            <Legend verticalAlign="bottom" height={32} formatter={legendFormatter} iconType="circle" iconSize={7} />
            <Line yAxisId="co2" type="monotone" dataKey="co2" stroke="#f59e0b" strokeWidth={2} dot={false} activeDot={{ r: 4 }} isAnimationActive={false} />
            <Line yAxisId="sensor" type="monotone" dataKey="humidity" stroke="#2563eb" strokeWidth={2} dot={false} activeDot={{ r: 4 }} isAnimationActive={false} />
            <Line yAxisId="sensor" type="monotone" dataKey="internalTemperature" stroke="#ef4444" strokeWidth={2} dot={false} activeDot={{ r: 4 }} isAnimationActive={false} />
            <Line yAxisId="sensor" type="monotone" dataKey="weight" stroke="#16a34a" strokeWidth={2} dot={false} activeDot={{ r: 4 }} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      ) : (
        <div style={{ height: 240, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: "8px", color: "#64748b", background: "#f8fafc", border: "1px dashed #cbd5e1", borderRadius: "10px" }}>
          <span style={{ fontSize: "2rem" }}>📊</span>
          <strong>No live sensor data available</strong>
          <span style={{ fontSize: "0.72rem" }}>Refresh the prediction or choose another time range.</span>
        </div>
      )}

      <div style={{ display: "flex", justifyContent: "center", flexWrap: "wrap", gap: "7px", marginTop: "8px" }}>
        {TIME_RANGES.map(({ label }) => {
          const isActive = activeRange === label;
          return (
            <button key={label} type="button" onClick={() => setActiveRange(label)} style={{
              padding: "5px 13px", color: isActive ? "#ffffff" : "#64748b",
              background: isActive ? "#2563eb" : "#ffffff",
              border: `1px solid ${isActive ? "#2563eb" : "#dbe4f0"}`,
              borderRadius: "18px", fontSize: "0.68rem", fontWeight: 600, cursor: "pointer",
            }}>
              {label}
            </button>
          );
        })}
      </div>
    </div>
  );
}

/** Countdown bar to next refresh */
function CountdownBar({ secondsLeft, total }) {
  const safeSeconds = Math.max(0, Math.min(total, Math.ceil(secondsLeft)));
  const pct = Math.min(100, Math.max(0, ((total - safeSeconds) / total) * 100));
  const minutesLeft = Math.floor(safeSeconds / 60);
  const secondsRemain = safeSeconds % 60;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
      <span style={{ fontSize: "0.7rem", color: "var(--text-muted)", whiteSpace: "nowrap" }}>
        Next refresh in {minutesLeft}m {secondsRemain}s
      </span>
      <div style={{ flex: 1, height: "3px", background: "#dbe4f0", borderRadius: "3px" }}>
        <div style={{
          height: "100%", background: "var(--accent-cyan)",
          borderRadius: "3px", width: `${pct}%`,
          transition: "width 1s linear",
        }} />
      </div>
    </div>
  );
}

/** Connection status badge */
function ConnectionBadge({ status }) {
  const cfg = {
    connected: { color: "#22c55e", bg: "#ecfdf5", border: "#16a34a", icon: "🟢", label: "Live IoT Data" },
    loading: { color: "#eab308", bg: "#fffbeb", border: "#ca8a04", icon: "⏳", label: "Connecting…" },
    error: { color: "#ef4444", bg: "#fef2f2", border: "#dc2626", icon: "❌", label: "Connection Error" },
    no_data: { color: "#f59e0b", bg: "#fffbeb", border: "#ca8a04", icon: "📭", label: "No Data" },
  };
  const info = cfg[status] || cfg.loading;
  return (
    <div style={{
      display: "inline-flex", alignItems: "center", gap: "6px",
      background: info.bg, border: `1px solid ${info.border}`,
      borderRadius: "20px", padding: "4px 12px",
      fontSize: "0.75rem", fontWeight: 600, color: info.color,
    }}>
      {info.icon} {info.label}
    </div>
  );
}

// ─── Risk Level Card ──────────────────────────────────────────────────────

function RiskLevelCard({ percentage, riskLevel }) {
  const cfg = RISK_CONFIG[riskLevel] || RISK_CONFIG.LOW;
  const clamped = Math.min(100, Math.max(0, percentage ?? 0));

  const subtitleMap = {
    LOW: "Normal Risk",
    MEDIUM: "Elevated Risk",
    HIGH: "Critical Risk",
  };

  const messageMap = {
    LOW: "Conditions are currently stable.",
    MEDIUM: "Increased activity detected — monitor closely.",
    HIGH: "Immediate hive inspection recommended.",
  };

  return (
    <div style={{
      background: `linear-gradient(180deg, #ffffff 0%, ${cfg.bg} 100%)`,
      border: `1px solid ${cfg.border}66`,
      borderTop: `4px solid ${cfg.border}`,
      borderRadius: "16px",
      padding: "18px 20px",
      display: "flex",
      flexDirection: "column",
      gap: "14px",
      width: "100%",
      boxSizing: "border-box",
    }}>
      <div style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        marginBottom: "4px",
      }}>
        <span style={{
          fontSize: "0.85rem",
          fontWeight: 700,
          color: "var(--text-secondary)",
          letterSpacing: "0.3px",
        }}>
          Risk Level
        </span>
        <span style={{
          width: "18px",
          height: "18px",
          borderRadius: "50%",
          border: "1.5px solid #cbd5e1",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontSize: "0.65rem",
          color: "var(--text-muted)",
          fontStyle: "italic",
          fontFamily: "serif",
          cursor: "help",
        }}>
          i
        </span>
      </div>

      <div style={{ textAlign: "center" }}>
        <div style={{
          fontSize: "2.1rem",
          fontWeight: 800,
          color: cfg.color,
          fontFamily: "'Outfit', sans-serif",
          lineHeight: 1.1,
          animation: riskLevel === "HIGH" ? "pulseNumber 1.4s ease-in-out infinite" : "none",
        }}>
          {riskLevel}
        </div>
        <div style={{
          fontSize: "0.8rem",
          color: "var(--text-muted)",
          marginTop: "2px",
          fontWeight: 500,
        }}>
          {subtitleMap[riskLevel] || subtitleMap.LOW}
        </div>
      </div>

      <div style={{ padding: "4px 2px 0" }}>
        <div style={{ position: "relative", height: "8px" }}>
          <div style={{
            position: "absolute",
            top: "-7px",
            left: `calc(${clamped}% - 6px)`,
            width: 0,
            height: 0,
            borderLeft: "6px solid transparent",
            borderRight: "6px solid transparent",
            borderTop: `7px solid ${cfg.color}`,
            transition: "left 0.8s ease-in-out",
            zIndex: 2,
          }} />
          <div style={{
            width: "100%",
            height: "8px",
            borderRadius: "6px",
            background: "linear-gradient(90deg, #22c55e 0%, #eab308 50%, #ef4444 100%)",
          }} />
        </div>
        <div style={{
          display: "flex",
          justifyContent: "space-between",
          marginTop: "8px",
          fontSize: "0.65rem",
          fontWeight: 700,
          color: "var(--text-muted)",
          letterSpacing: "0.03em",
        }}>
          <span>LOW</span>
          <span>MEDIUM</span>
          <span>HIGH</span>
        </div>
      </div>

      <div style={{
        background: cfg.bg,
        border: `1px solid ${cfg.border}`,
        borderRadius: "10px",
        padding: "10px 12px",
        fontSize: "0.78rem",
        color: cfg.color,
        textAlign: "center",
        marginTop: "4px",
      }}>
        {messageMap[riskLevel] || messageMap.LOW}
      </div>
    </div>
  );
}
// ─── Recommended Actions Card ────────────────────────────────────────────

function RecommendedActions({ riskLevel }) {
  const actions = {
    LOW: {
      color: "#4ade80",
      bg: "#ecfdf5",
      border: "#22c55e",
      icon: "🟢",
      title: "Recommended Actions",
      list: [
        "Continue routine hive monitoring.",
        "Maintain normal hive ventilation.",
        "Ensure adequate food and water availability.",
        "Review sensor readings during the next update."
      ]
    },
    MEDIUM: {
      color: "#fbbf24",
      bg: "#fffbeb",
      border: "#eab308",
      icon: "🟡",
      title: "Recommended Actions",
      list: [
        "Inspect the colony within the next 24 hours.",
        "Check for developing queen cells.",
        "Monitor colony congestion and brood pattern.",
        "Increase monitoring frequency."
      ]
    },
    HIGH: {
      color: "#f87171",
      bg: "#fef2f2",
      border: "#ef4444",
      icon: "🔴",
      title: "Immediate Recommended Actions",
      list: [
        "Perform an immediate hive inspection.",
        "Inspect and remove excess queen cells if appropriate.",
        "Reduce overcrowding or split the colony.",
        "Prepare swarm traps or additional hive boxes.",
        "Continue close monitoring until risk decreases."
      ]
    }
  };

  const cfg = actions[riskLevel] || actions.LOW;

  return (
    <div style={{ 
      background: cfg.bg, 
      border: `2px solid ${cfg.border}`, 
      borderRadius: "14px", 
      padding: "18px", 
      animation: "fadeSlide .4s ease" 
    }}>
      <div style={{ 
        display: "flex", 
        alignItems: "center", 
        gap: "8px", 
        marginBottom: "14px" 
      }}>
        <span style={{ fontSize: "1.2rem" }}>{cfg.icon}</span>
        <span style={{ 
          color: cfg.color, 
          fontWeight: 700, 
          fontSize: "0.95rem" 
        }}>
          {cfg.title}
        </span>
      </div>

      <div style={{ 
        display: "flex", 
        flexDirection: "column", 
        gap: "10px" 
      }}>
        {cfg.list.map((item, index) => (
          <div 
            key={index} 
            style={{ 
              display: "flex", 
              alignItems: "flex-start", 
              gap: "10px", 
              background: "rgba(255,255,255,.65)",
              border: `1px solid ${cfg.border}22`,
              borderRadius: "8px", 
              padding: "10px" 
            }}
          >
            <span style={{ 
              color: cfg.color, 
              fontWeight: "bold",
              fontSize: "0.9rem",
              minWidth: "20px",
            }}>
              ✓
            </span>
            <span style={{ 
              color: "#475569",
              fontSize: ".85rem", 
              lineHeight: 1.6, 
              fontWeight: 400,
              letterSpacing: "0.2px",
            }}>
              {item}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── 3-Day Forecast Risk Timeline ────────────────────────────────────────

function ForecastRiskTimeline({ forecast, currentProbability }) {
  const chartData = useMemo(() => {
    if (!forecast || forecast.length === 0) return [];
    
    return [
      { dayStr: "Today", risk: currentProbability ?? 0 },
      ...forecast.map(f => ({
        dayStr: `Day ${f.day}`,
        risk: f.risk
      }))
    ];
  }, [forecast, currentProbability]);

  const CustomTooltip = ({ active, payload, label }) => {
    if (!active || !payload?.length) return null;
    const val = payload[0]?.value;
    const color = val >= RISK_THRESHOLD ? "#ef4444" : val >= MEDIUM_THRESHOLD ? "#eab308" : "#22c55e";
    return (
      <div style={{
        background: "#ffffff",
        border: "1px solid #cbd5e1",
        borderRadius: "10px",
        padding: "10px 14px",
        fontSize: "0.78rem",
        color: "#0f172a",
        boxShadow: "0 8px 24px rgba(15,23,42,0.12)",
      }}>
        <div style={{ color: "#94a3b8", marginBottom: "4px" }}>{label}</div>
        <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
          <span style={{ width: 8, height: 8, borderRadius: "50%", background: color, display: "inline-block" }} />
          <span>Forecast Risk:</span>
          <strong style={{ color }}>{val != null ? `${Number(val).toFixed(1)}%` : "—"}</strong>
        </div>
      </div>
    );
  };

  return (
    <div style={{
      background: "#ffffff",
      border: "1px solid #dbe4f0",
      borderRadius: "16px",
      padding: "20px 24px 14px",
      boxShadow: "0 6px 20px rgba(15,23,42,0.06)",
      width: "100%",
      height: "100%",
      minHeight: "280px",
    }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "18px", flexWrap: "wrap", gap: "8px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <span style={{ fontSize: "1.1rem" }}>🔮</span>
          <span style={{
            fontFamily: "'Outfit', sans-serif",
            fontWeight: 700,
            fontSize: "1rem",
            color: "#1e3a5f",
          }}>
            3-Day Forecast Timeline
          </span>
        </div>
      </div>

      {chartData.length > 0 ? (
        <ResponsiveContainer width="100%" height={180}>
          <AreaChart data={chartData} margin={{ top: 8, right: 10, left: -10, bottom: 0 }}>
            <defs>
              <linearGradient id="forecastGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#2563eb" stopOpacity={0.35} />
                <stop offset="95%" stopColor="#2563eb" stopOpacity={0.02} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="rgba(255,255,255,0.05)" strokeDasharray="3 3" vertical={false} />
            <XAxis
              dataKey="dayStr"
              tick={{ fill: "#64748b", fontSize: 10, fontFamily: "'Outfit', sans-serif" }}
              axisLine={{ stroke: "#dbe4f0" }}
              tickLine={false}
            />
            <YAxis
              domain={[0, 100]}
              tick={{ fill: "#64748b", fontSize: 10, fontFamily: "'Outfit', sans-serif" }}
              axisLine={false}
              tickLine={false}
              tickFormatter={(v) => `${v}`}
            />
            <Tooltip content={<CustomTooltip />} />
            <ReferenceLine
              y={RISK_THRESHOLD}
              stroke="#ef4444"
              strokeDasharray="6 4"
              strokeWidth={1.5}
            />
            <ReferenceLine
              y={MEDIUM_THRESHOLD}
              stroke="#f59e0b"
              strokeDasharray="6 4"
              strokeWidth={1.5}
            />
            <Area
              type="monotone"
              dataKey="risk"
              stroke="#2563eb"
              strokeWidth={2}
              fill="url(#forecastGrad)"
              dot={{ r: 4, fill: "#2563eb", strokeWidth: 0 }}
              activeDot={{ r: 6, fill: "#2563eb", strokeWidth: 0 }}
              isAnimationActive
              animationDuration={600}
            />
          </AreaChart>
        </ResponsiveContainer>
      ) : (
        <div style={{
          height: 180, display: "flex", alignItems: "center", justifyContent: "center",
          color: "#475569", fontSize: "0.85rem",
          flexDirection: "column", gap: "8px",
        }}>
          <span style={{ fontSize: "2rem" }}>🔮</span>
          <span>No forecast available</span>
        </div>
      )}
    </div>
  );
}

// ─── Swarming Risk Timeline Chart ───────────────────────────────────────

function SwarmingRiskTimeline({ history, currentRisk, riskLevel }) {
  const [activeRange, setActiveRange] = useState("6H");

  const chartData = useMemo(() => {
    if (!Array.isArray(history) || history.length === 0) return [];

    const rangeObj = RISK_TIME_RANGES.find((r) => r.label === activeRange) || RISK_TIME_RANGES[2];
    const cutoffMs = rangeObj.minutes * 60 * 1000;
    const ordered = [...history]
      .map((item) => ({
        ...item,
        timestampMs: new Date(item.predicted_at).getTime(),
      }))
      .filter((item) => Number.isFinite(item.timestampMs))
      .sort((a, b) => a.timestampMs - b.timestampMs);

    if (ordered.length === 0) return [];
    const latestTimestamp = ordered[ordered.length - 1].timestampMs;
    const filtered = ordered.filter(
      (item) => latestTimestamp - item.timestampMs <= cutoffMs
    );

    const step = Math.max(1, Math.floor(filtered.length / 120));
    const sampled = filtered.filter((_, i) => i % step === 0);

    return sampled.map((item) => {
      const date = new Date(item.timestampMs);
      const label = date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
      return {
        time: label,
        probability: Number(item.combined_risk_percentage),
        lstmRisk: Number(item.lstm_risk_percentage),
        peltRisk: Number(item.pelt_risk_percentage),
        fullTs: item.timestampMs,
      };
    });
  }, [history, activeRange]);

  const displayData = useMemo(() => {
    if (chartData.length > 0) return chartData;
    if (currentRisk == null) return [];
    return [{
      time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      probability: Number(currentRisk),
      fullTs: Date.now(),
    }];
  }, [chartData, currentRisk]);

  const lineColor = riskLevel === "HIGH" ? "#ef4444" : riskLevel === "MEDIUM" ? "#eab308" : "#22c55e";
  const gradientId = "swarmGrad";

  const CustomTooltip = ({ active, payload, label }) => {
    if (!active || !payload?.length) return null;
    const val = payload[0]?.value;
    return (
      <div style={{
        background: "#ffffff",
        border: "1px solid #cbd5e1",
        borderRadius: "10px",
        padding: "10px 14px",
        fontSize: "0.78rem",
        color: "#0f172a",
        boxShadow: "0 8px 24px rgba(15,23,42,0.12)",
      }}>
        <div style={{ color: "#94a3b8", marginBottom: "4px" }}>{label}</div>
        <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
          <span style={{ width: 8, height: 8, borderRadius: "50%", background: lineColor, display: "inline-block" }} />
          <span>Swarming Risk:</span>
          <strong style={{ color: lineColor }}>{val != null ? `${Number(val).toFixed(2)}%` : "—"}</strong>
        </div>
        {val >= RISK_THRESHOLD && (
          <div style={{ color: "#ef4444", marginTop: "4px", fontSize: "0.7rem" }}>⚠️ Above risk threshold</div>
        )}
        {val >= MEDIUM_THRESHOLD && val < RISK_THRESHOLD && (
          <div style={{ color: "#eab308", marginTop: "4px", fontSize: "0.7rem" }}>⚡ Elevated — monitor closely</div>
        )}
      </div>
    );
  };

  return (
    <div style={{
      background: "#ffffff",
      border: "1px solid #dbe4f0",
      borderRadius: "16px",
      padding: "20px 24px 14px",
      boxShadow: "0 6px 20px rgba(15,23,42,0.06)",
      width: "100%",
      height: "100%",
      minHeight: "280px",
    }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "18px", flexWrap: "wrap", gap: "8px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <span style={{ fontSize: "1.1rem" }}>📈</span>
          <span style={{
            fontFamily: "'Outfit', sans-serif",
            fontWeight: 700,
            fontSize: "1rem",
            color: "#1e3a5f",
          }}>
            Swarming Risk Timeline
          </span>
          {riskLevel && (
            <span style={{
              fontSize: "0.65rem",
              fontWeight: 700,
              padding: "2px 10px",
              borderRadius: "20px",
              background: riskLevel === "HIGH" ? "#fef2f2" : riskLevel === "MEDIUM" ? "#fffbeb" : "#ecfdf5",
              color: lineColor,
              border: `1px solid ${lineColor}`,
              letterSpacing: "0.5px",
            }}>
              {riskLevel}
            </span>
          )}
        </div>
        <div style={{
          width: 22, height: 22, borderRadius: "50%",
          border: "1.5px solid #cbd5e1",
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: "0.65rem", color: "#94a3b8", fontStyle: "italic", fontFamily: "serif",
          cursor: "help",
        }} title="LSTM swarming probability over time">
          i
        </div>
      </div>

      {displayData.length > 0 ? (
        <ResponsiveContainer width="100%" height={180}>
          <AreaChart data={displayData} margin={{ top: 8, right: 10, left: -10, bottom: 0 }}>
            <defs>
              <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={lineColor} stopOpacity={0.35} />
                <stop offset="95%" stopColor={lineColor} stopOpacity={0.02} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="rgba(255,255,255,0.05)" strokeDasharray="3 3" vertical={false} />
            <XAxis
              dataKey="time"
              tick={{ fill: "#64748b", fontSize: 10, fontFamily: "'Outfit', sans-serif" }}
              axisLine={{ stroke: "#dbe4f0" }}
              tickLine={false}
              interval="preserveStartEnd"
            />
            <YAxis
              domain={[0, 100]}
              tick={{ fill: "#64748b", fontSize: 10, fontFamily: "'Outfit', sans-serif" }}
              axisLine={false}
              tickLine={false}
              tickFormatter={(v) => `${v}`}
            />
            <Tooltip content={<CustomTooltip />} />
            <ReferenceLine
              y={RISK_THRESHOLD}
              stroke="#ef4444"
              strokeDasharray="6 4"
              strokeWidth={1.5}
              label={{
                value: `Risk Threshold (${RISK_THRESHOLD / 100})`,
                position: "insideTopRight",
                fill: "#ef4444",
                fontSize: 10,
                fontFamily: "'Outfit', sans-serif",
                dy: -4,
              }}
            />
            <ReferenceLine
              y={MEDIUM_THRESHOLD}
              stroke="#f59e0b"
              strokeDasharray="6 4"
              strokeWidth={1.5}
              label={{
                value: `Moderate Threshold (${MEDIUM_THRESHOLD / 100})`,
                position: "insideTopRight",
                fill: "#f59e0b",
                fontSize: 10,
                fontFamily: "'Outfit', sans-serif",
                dy: -4,
              }}
            />
            <Area
              type="monotone"
              dataKey="probability"
              stroke={lineColor}
              strokeWidth={2}
              fill={`url(#${gradientId})`}
              dot={false}
              activeDot={{ r: 5, fill: lineColor, strokeWidth: 0 }}
              connectNulls
              isAnimationActive
              animationDuration={600}
            />
            <Legend
              verticalAlign="bottom"
              height={28}
              formatter={() => (
                <span style={{ color: lineColor, fontSize: "0.72rem", fontFamily: "'Outfit', sans-serif" }}>
                  ◈ Swarming Probability
                </span>
              )}
            />
          </AreaChart>
        </ResponsiveContainer>
      ) : (
        <div style={{
          height: 180, display: "flex", alignItems: "center", justifyContent: "center",
          color: "#475569", fontSize: "0.85rem",
          flexDirection: "column", gap: "8px",
        }}>
          <span style={{ fontSize: "2rem" }}>📊</span>
          <span>No data available for this time range</span>
          <span style={{ fontSize: "0.7rem", color: "#334155" }}>Fetch a prediction first</span>
        </div>
      )}

      <div style={{ display: "flex", gap: "6px", justifyContent: "center", marginTop: "12px" }}>
        {RISK_TIME_RANGES.map(({ label }) => {
          const isActive = activeRange === label;
          return (
            <button
              key={label}
              onClick={() => setActiveRange(label)}
              style={{
                padding: "5px 14px",
                borderRadius: "20px",
                border: `1px solid ${isActive ? "#2563eb" : "#dbe4f0"}`,
                background: isActive ? "#2563eb" : "transparent",
                color: isActive ? "#0f172a" : "#64748b",
                fontSize: "0.72rem",
                fontWeight: 700,
                fontFamily: "'Outfit', sans-serif",
                cursor: "pointer",
                transition: "all 0.18s ease",
              }}
            >
              {label}
            </button>
          );
        })}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────
// Main SwarmPrediction Component
// ─────────────────────────────────────────────────────────────────────

const SwarmPrediction = () => {
  const [hiveList, setHiveList] = useState([]);
  const [selectedHive, setSelectedHive] = useState(null);
  const [sensorValues, setSensorValues] = useState(null);
  const [result, setResult] = useState(null);
  const [forecast, setForecast] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [countdown, setCountdown] = useState(REFRESH_INTERVAL / 1000);
  const [connStatus, setConnStatus] = useState("loading");
  const [modelHealth, setModelHealth] = useState(null);
  const [readingsCache, setReadingsCache] = useState([]);
  const [riskHistory, setRiskHistory] = useState([]);
  const [dataTimestamp, setDataTimestamp] = useState(null);
  const [peltHistory, setPeltHistory] = useState([]);

  const countdownRef = useRef(null);
  const refreshTriggeredForTimestampRef = useRef(null);

  // ── Fetch list of available hives ──────────────────────────────────────
  useEffect(() => {
    fetch(`${API_BASE}/api/iot/devices`)
      .then((r) => r.json())
      .then((data) => {
        const devices = data.devices || [];
        setHiveList(devices);
        if (devices.length > 0 && !selectedHive) {
          setSelectedHive(devices[0]);
        } else if (devices.length === 0) {
          setConnStatus("no_data");
        }
      })
      .catch(() => setConnStatus("error"));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── Check model health on mount ──────────────────────────────────────
  useEffect(() => {
    fetch(`${API_BASE}/api/swarming/live-prediction/health`)
      .then((r) => r.json())
      .then(setModelHealth)
      .catch(() => setModelHealth({ all_ready: false }));
  }, []);

  // ── Main: fetch IoT data + run prediction ────────────────────────────
  const fetchLivePrediction = useCallback(async (hive) => {
    if (!hive) return;
    setLoading(true);
    setError(null);
    setConnStatus("loading");
    try {
      const res = await fetch(
        `${API_BASE}/api/swarming/predict-from-iot?device_id=${encodeURIComponent(hive)}&limit=432`
      );
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.error || `HTTP ${res.status}`);
      }
      const data = await res.json();

      setSensorValues(data.latest_sensor);
      setResult(data.prediction);
      setForecast(data.forecast);
      
      if (data.readings) {
        setReadingsCache(data.readings);
      }

      // Load genuine stored risks. These values replace the previous
      // Math.random()-generated timeline.
      try {
        const historyResponse = await fetch(
          `${API_BASE}/api/swarming/risk-history?device_id=${encodeURIComponent(hive)}&minutes=10080&limit=5000`
        );
        if (historyResponse.ok) {
          const historyData = await historyResponse.json();
          setRiskHistory(Array.isArray(historyData.history) ? historyData.history : []);
        } else {
          setRiskHistory([]);
        }
      } catch {
        setRiskHistory([]);
      }
      
      // ── FIX: Store PELT history - ALWAYS store breakpoint entries ──
      if (data.prediction?.pelt_snapshot) {
        const breakpointValue = Number(data.prediction.pelt_snapshot.breakpoint) || 0;
        const densityValue = Number(data.prediction.pelt_snapshot.breakpoint_density) || 0;
        
        const peltEntry = {
          timestamp: data.prediction.timestamp || new Date().toISOString(),
          breakpoint: breakpointValue,
          breakpoint_density: densityValue,
          days_since_breakpoint: Number(data.prediction.pelt_snapshot.days_since_breakpoint) || 0,
          segment_duration: Number(data.prediction.pelt_snapshot.segment_duration) || 0,
        };
        
        // Debug logging
        console.log('📊 PELT Entry from API:', peltEntry);
        
        if (peltEntry.breakpoint === 1) {
          console.log('⚠️ BREAKPOINT DETECTED!', peltEntry);
        }
        
        setPeltHistory(prev => {
          // ALWAYS add to history if breakpoint is detected
          if (peltEntry.breakpoint === 1) {
            console.log('✅ Adding breakpoint to history');
            const newHistory = [...prev, peltEntry];
            if (newHistory.length > 500) {
              return newHistory.slice(-500);
            }
            return newHistory;
          }
          
          // For non-breakpoint entries, check if data changed
          const lastEntry = prev[prev.length - 1];
          const getEntryKey = (entry) => 
            `${entry.breakpoint}-${entry.breakpoint_density}-${entry.days_since_breakpoint}-${entry.segment_duration}`;
          
          const currentKey = getEntryKey(peltEntry);
          const lastKey = lastEntry ? getEntryKey(lastEntry) : null;
          
          // Only add if data changed
          if (currentKey !== lastKey) {
            console.log('📊 Adding new PELT entry (data changed)');
            const newHistory = [...prev, peltEntry];
            if (newHistory.length > 500) {
              return newHistory.slice(-500);
            }
            return newHistory;
          }
          
          // Update timestamp only
          if (lastEntry) {
            console.log('⏭️ Updating timestamp (data unchanged)');
            const updatedHistory = [...prev];
            updatedHistory[updatedHistory.length - 1] = {
              ...lastEntry,
              timestamp: peltEntry.timestamp,
            };
            return updatedHistory;
          }
          
          return prev;
        });
      }
      
      let latestDataTimestamp = data.latest_sensor?.recorded_at || null;

      if (!latestDataTimestamp && data.readings?.length > 0) {
        const latestReading = data.readings[data.readings.length - 1];
        latestDataTimestamp = latestReading?.reading_at || latestReading?.recorded_at || null;
      }

      if (latestDataTimestamp) {
        setDataTimestamp(latestDataTimestamp);
        setCountdown(secondsUntilRefresh(latestDataTimestamp));
      }
      
      setLastUpdated(new Date());
      setConnStatus("connected");
    } catch (e) {
      setError(e.message);
      setConnStatus("error");
    } finally {
      setLoading(false);
    }
  }, []);

  // ── Fetch immediately when the selected hive changes ───────────────
  useEffect(() => {
    if (!selectedHive) return;

    refreshTriggeredForTimestampRef.current = null;
    fetchLivePrediction(selectedHive);
  }, [selectedHive, fetchLivePrediction]);

  // ── Countdown synced to 10 minutes after recorded data time ─────────
  useEffect(() => {
    clearInterval(countdownRef.current);

    if (!dataTimestamp) {
      setCountdown(REFRESH_INTERVAL / 1000);
      return undefined;
    }

    const updateCountdown = () => {
      const remaining = secondsUntilRefresh(dataTimestamp);
      setCountdown(remaining);

      // Only refresh once for a given database timestamp. If the database
      // has not received a newer reading yet, this prevents a request loop.
      if (
        remaining === 0 &&
        selectedHive &&
        refreshTriggeredForTimestampRef.current !== dataTimestamp
      ) {
        refreshTriggeredForTimestampRef.current = dataTimestamp;
        fetchLivePrediction(selectedHive);
      }
    };

    updateCountdown();
    countdownRef.current = setInterval(() => {
      updateCountdown();
    }, 1000);

    return () => clearInterval(countdownRef.current);
  }, [dataTimestamp, selectedHive, fetchLivePrediction]);

  const handleHiveChange = (h) => {
    setSelectedHive(h);
    setReadingsCache([]);
    setRiskHistory([]);
    setDataTimestamp(null);
    setPeltHistory([]);
    setCountdown(REFRESH_INTERVAL / 1000);
    refreshTriggeredForTimestampRef.current = null;
  };
  
  const handleRefreshNow = () => {
    refreshTriggeredForTimestampRef.current = null;
    fetchLivePrediction(selectedHive);
  };

  // ── Derived values ────────────────────────────────────────────────────
  const riskLevel = result?.risk_level || "LOW";
  const riskPercentage = result?.risk_percentage || 0;
  const riskCfg = RISK_CONFIG[riskLevel] || RISK_CONFIG.LOW;
  const showPrediction = result !== null && sensorValues !== null;

  const formatTimestamp = (timestamp) => {
    if (!timestamp) return "N/A";
    try {
      const date = new Date(timestamp);
      return date.toLocaleString();
    } catch {
      return timestamp;
    }
  };

  // ──────────────────────────────────────────────────────────────────────
  return (
    <div style={{
      padding: "24px 28px",
      minHeight: "100vh",
      color: "#0f172a",
      background: "#f8fafc",
    }}>

      {/* ── CSS animations ── */}
      <style>{`
        @keyframes pulseNumber {
          0%,100% { opacity:1; transform:scale(1); }
          50%     { opacity:0.85; transform:scale(1.06); }
        }
        @keyframes pulseBadge {
          0%,100% { box-shadow: 0 0 0 0 rgba(239,68,68,0); }
          50%     { box-shadow: 0 0 12px 4px rgba(239,68,68,0.4); }
        }
        @keyframes spin { to { transform:rotate(360deg); } }
        @keyframes fadeSlide {
          from { opacity:0; transform:translateY(10px); }
          to   { opacity:1; transform:translateY(0); }
        }
        @keyframes livePulse {
          0%,100% { opacity:1; }
          50%     { opacity:0.4; }
        }
        .pred-btn {
          cursor:pointer; transition:all 0.2s;
          border:none; border-radius:10px; padding:10px 20px;
          font-family:'Outfit',sans-serif; font-weight:700; font-size:0.9rem;
        }
        .pred-btn:hover { filter:brightness(1.15); transform:translateY(-1px); }
        .pred-btn:active { transform:translateY(0); }
        .pred-btn:disabled { opacity:0.5; cursor:not-allowed; transform:none; }
      `}</style>

      {/* ── Page header ── */}
      <div style={{
        marginBottom: "20px",
        padding: "18px 20px",
        background: "linear-gradient(135deg,#ffffff 0%,#eff6ff 58%,#f5f3ff 100%)",
        border: "1px solid #dbe4f0",
        borderLeft: "5px solid #2563eb",
        borderRadius: "14px",
        boxShadow: "0 6px 18px rgba(37,99,235,0.08)",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "6px" }}>
          <span style={{ fontSize: "1.6rem" }}>🐝</span>
          <h2 style={{ margin: 0, fontSize: "1.4rem", fontFamily: "'Outfit',sans-serif",
            color: "#1d4ed8" }}>
            Live Swarming Prediction
          </h2>
          <div style={{ marginLeft: "auto" }}>
            <ConnectionBadge status={connStatus} />
          </div>
        </div>
        <p style={{ margin: 0, color: "var(--text-secondary)", fontSize: "0.82rem" }}>
          {/* Real IoT data from Supabase · LSTM model · PELT change-point features · Threshold 0.70 · Auto-refresh every 2 min */}
        </p>
      </div>

      {/* ── Model health bar ── */}
      {modelHealth && (
        <div style={{
          display: "flex", alignItems: "center", gap: "8px",
          background: modelHealth.all_ready ? "#ecfdf5" : "#fef2f2",
          border: `1px solid ${modelHealth.all_ready ? "#16a34a" : "#dc2626"}`,
          borderRadius: "8px", padding: "8px 14px", marginBottom: "16px",
          fontSize: "0.8rem",
        }}>
          <span>{modelHealth.all_ready ? "✅" : "❌"}</span>
          <span style={{ color: modelHealth.all_ready ? "#22c55e" : "#ef4444", fontWeight: 600 }}>
            {modelHealth.all_ready ? "LSTM model loaded & ready" : "Model files missing — run training first"}
          </span>
          {modelHealth.all_ready && (
            <span style={{ color: "var(--text-muted)", marginLeft: "auto", fontSize: "0.72rem" }}>
              {/* best_lstm.keras · lstm_scaler.pkl · label_encoder.pkl */}
            </span>
          )}
        </div>
      )}

      {/* ── Countdown bar ── */}
      {lastUpdated && (
        <div style={{ marginBottom: "16px" }}>
          <CountdownBar secondsLeft={countdown} total={REFRESH_INTERVAL / 1000} />
          <div style={{ fontSize: "0.7rem", color: "var(--text-muted)", marginTop: "4px" }}>
            Last updated: {lastUpdated.toLocaleTimeString()} · Data from: {dataTimestamp
              ? formatTimestamp(dataTimestamp)
              : "—"}
          </div>
        </div>
      )}

      <div style={{ display: "grid", gridTemplateColumns: "1fr 360px", gap: "20px", alignItems: "start" }}>

        {/* ── LEFT: Controls + sensor readings ── */}
        <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>

          {/* Hive selector + control row */}
          <div style={{
            background: "#ffffff", borderRadius: "14px",
            padding: "16px 18px", border: "1px solid #dbe4f0",
            borderTop: "4px solid #2563eb",
            boxShadow: "0 5px 16px rgba(37,99,235,0.06)",
          }}>
            <h3 style={{ margin: "0 0 12px", fontSize: "0.95rem", color: "#2563eb" }}>
              🏠 Hive Selection & Control
            </h3>
            <div style={{ display: "flex", gap: "12px", alignItems: "center", flexWrap: "wrap" }}>
              <select
                id="hive-selector"
                value={selectedHive || ""}
                onChange={(e) => handleHiveChange(e.target.value)}
                disabled={hiveList.length === 0}
                style={{
                  background: "#ffffff", color: "#334155", border: "1px solid #cbd5e1",
                  borderRadius: "8px", padding: "8px 14px", fontSize: "0.9rem",
                  fontFamily: "'Outfit',sans-serif", cursor: "pointer", minWidth: "160px",
                }}
              >
                {hiveList.length === 0
                  ? <option value="">Loading devices…</option>
                  : hiveList.map((h) => <option key={h} value={h}>🐝 {h}</option>)
                }
              </select>

              <button
                id="btn-refresh-now"
                className="pred-btn"
                onClick={handleRefreshNow}
                disabled={loading || !selectedHive}
                style={{
                  background: loading
                    ? "rgba(56,189,248,0.3)"
                    : "linear-gradient(135deg,#0ea5e9,#2563eb)",
                  color: "white",
                  display: "flex", alignItems: "center", gap: "8px",
                }}
              >
                {loading ? (
                  <>
                    <span style={{ display:"inline-block", width:14, height:14,
                      border:"2px solid rgba(255,255,255,0.3)", borderTopColor:"white",
                      borderRadius:"50%", animation:"spin 0.8s linear infinite" }} />
                    Fetching…
                  </>
                ) : "🔄 Refresh Now"}
              </button>

              {lastUpdated && (
                <span style={{ fontSize: "0.72rem", color: "var(--text-muted)", marginLeft: "auto" }}>
                  Updated: {lastUpdated.toLocaleTimeString()}
                </span>
              )}
            </div>
          </div>

          {/* Latest Sensor Readings */}
          <div style={{
            background: "#ffffff", borderRadius: "14px",
            padding: "16px 18px", border: "1px solid #dbe4f0",
            borderTop: "4px solid #06b6d4",
            boxShadow: "0 5px 16px rgba(6,182,212,0.06)",
          }}>
            <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "12px" }}>
              <h3 style={{ margin: 0, fontSize: "0.95rem", color: "#2563eb" }}>
                📡 Latest Sensor Readings
              </h3>
              {connStatus === "connected" && (
                <span style={{
                  fontSize: "0.65rem", color: "#22c55e",
                  display: "flex", alignItems: "center", gap: "4px",
                }}>
                  <span style={{
                    width: 6, height: 6, borderRadius: "50%", background: "#22c55e",
                    display: "inline-block", animation: "livePulse 1.5s ease-in-out infinite",
                  }} />
                  LIVE
                </span>
              )}
              <span style={{ fontSize: "0.68rem", color: "var(--text-muted)", marginLeft: "auto" }}>
                {/* Real-time from Supabase · read-only */}
              </span>
            </div>

            {loading && !sensorValues && (
              <div style={{ textAlign: "center", padding: "20px 0", color: "var(--text-muted)", fontSize: "0.85rem" }}>
                <span style={{
                  display:"inline-block", width:20, height:20,
                  border:"2px solid #dbe4f0", borderTopColor:"var(--accent-gold)",
                  borderRadius:"50%", animation:"spin 1s linear infinite", marginRight: "8px",
                  verticalAlign: "middle",
                }} />
                Loading IoT data…
              </div>
            )}

            {!loading && !sensorValues && !error && (
              <div style={{ textAlign: "center", padding: "20px 0", color: "var(--text-muted)", fontSize: "0.85rem" }}>
                📭 No sensor data loaded yet. Select a hive above.
              </div>
            )}

            {sensorValues && (
              <>
                <div style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))",
                  gap: "10px",
                }}>
                  {SENSOR_META.map((meta) => (
                    <SensorCard
                      key={meta.key}
                      meta={meta}
                      value={sensorValues[meta.key]}
                      isLive={connStatus === "connected"}
                    />
                  ))}
                </div>
                {dataTimestamp && (
                  <div style={{
                    fontSize: "0.65rem",
                    color: (() => {
                      try {
                        const diff = (new Date() - new Date(dataTimestamp)) / (1000 * 60 * 60);
                        return diff > 2 ? "#ef4444" : diff > 1 ? "#f59e0b" : "#22c55e";
                      } catch {
                        return "var(--text-muted)";
                      }
                    })(),
                    marginTop: "8px",
                    textAlign: "right",
                    paddingRight: "4px",
                  }}>
                    📊 Data from: {formatTimestamp(dataTimestamp)}
                  </div>
                )}
              </>
            )}
          </div>

          {/* 3-Day Forecast */}
          {forecast && forecast.length > 0 && (
            <div style={{ animation: "fadeSlide 0.4s ease-out" }}>
              <ForecastCard forecast={forecast} />
            </div>
          )}

          {/* TWO COLUMN LAYOUT: Swarming Risk Timeline + 3-Day Forecast Timeline */}
          <div style={{ 
            display: "grid", 
            gridTemplateColumns: "1fr 1fr", 
            gap: "16px",
            alignItems: "stretch",
          }}>
            {/* Swarming Risk Timeline - Left Column */}
            <div style={{ animation: "fadeSlide 0.4s ease-out" }}>
              <SwarmingRiskTimeline
                history={riskHistory}
                currentRisk={result ? Number(result.risk_percentage) : 0}
                riskLevel={riskLevel}
              />
            </div>
            
            {/* 3-Day Forecast Risk Timeline - Right Column */}
            {forecast && forecast.length > 0 && (
              <div style={{ animation: "fadeSlide 0.4s ease-out" }}>
                <ForecastRiskTimeline
                  forecast={forecast}
                  currentProbability={result ? Number(result.risk_percentage) : 0}
                />
              </div>
            )}
          </div>

          {/* Sensor Value Trends Timeline (Full Width) */}
          <div style={{ animation: "fadeSlide 0.4s ease-out", marginTop: "8px" }}>
            <RealtimeSensorTrendsTimeline readings={readingsCache} />
          </div>

          {/* High Risk Alert */}
          <HighRiskAlert riskLevel={riskLevel} percentage={riskPercentage} />

          {/* Error state */}
          {error && (
            <div style={{
              background: "#fef2f2", border: "1px solid #dc2626",
              borderRadius: "10px", padding: "12px 16px",
              color: "#b91c1c", fontSize: "0.85rem",
            }}>
              <strong>❌ Error:</strong> {error}
              <div style={{ marginTop: "6px", color: "var(--text-muted)", fontSize: "0.78rem" }}>
                Make sure the Flask backend is running on port 5000 and the database is reachable.
              </div>
            </div>
          )}
        </div>

        {/* ── RIGHT: Prediction result panel ── */}
        <div style={{ display: "flex", flexDirection: "column", gap: "14px" }}>

          {/* Risk Level Card */}
          <RiskLevelCard percentage={riskPercentage} riskLevel={riskLevel} />

          {/* Result card */}
          <div style={{
            background: showPrediction ? riskCfg.bg : "#1e293b",
            border: `2px solid ${showPrediction ? riskCfg.border : "#dbe4f0"}`,
            borderRadius: "16px", padding: "22px 18px",
            display: "flex", flexDirection: "column", alignItems: "center", gap: "16px",
            animation: showPrediction ? "fadeSlide 0.5s ease-out" : "none",
            transition: "border-color 0.5s, background 0.5s",
          }}>

            {loading && !result && (
              <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "12px", padding: "30px 0" }}>
                <span style={{
                  display: "inline-block", width: 40, height: 40,
                  border: "3px solid #dbe4f0", borderTopColor: "var(--accent-gold)",
                  borderRadius: "50%", animation: "spin 1s linear infinite",
                }} />
                <span style={{ color: "var(--text-muted)", fontSize: "0.85rem" }}>Fetching IoT data & running prediction…</span>
              </div>
            )}

            {!loading && !result && !error && (
              <div style={{ padding: "30px 0", textAlign: "center" }}>
                <div style={{ fontSize: "2.5rem", marginBottom: "8px" }}>🐝</div>
                <div style={{ color: "var(--text-muted)", fontSize: "0.85rem" }}>
                  Select a hive to load live prediction…
                </div>
              </div>
            )}

            {showPrediction && (
              <>
                <RiskGauge
                  percentage={Number(result.risk_percentage ?? 0)}
                  riskLevel={riskLevel}
                  label="Current Swarming Risk"
                />

                <div style={{
                  width: "100%", background: "#f8fafc",
                  borderRadius: "10px", padding: "12px 14px",
                  display: "grid", gridTemplateColumns: "1fr 1fr", gap: "8px",
                }}>
                  {[
                 {
                  label: "Swarming Risk",
                  value: `${Number(result.risk_percentage ?? 0).toFixed(2)}%`,
                  color: riskCfg.color,
                },
                    { label: "Risk Level", value: result.risk_level, color: riskCfg.color },
                    { label: "Predicted Class", value: result.predicted_class, color: result.predicted_class === "Swarming" ? "#ef4444" : "#22c55e" },
                    { label: "Decision Threshold", value: result.threshold_used, color: "var(--text-secondary)" },
                  ].map(({ label, value, color }) => (
                    <div key={label} style={{ textAlign: "center" }}>
                      <div style={{ fontSize: "0.65rem", color: "var(--text-muted)", marginBottom: "2px" }}>{label}</div>
                      <div style={{ fontSize: "1rem", fontWeight: 700, color, fontFamily: "'Outfit',sans-serif" }}>{value}</div>
                    </div>
                  ))}
                </div>

                <div style={{
                  width: "100%", background: `${riskCfg.bg}cc`,
                  border: `1px solid ${riskCfg.border}`,
                  borderRadius: "10px", padding: "10px 14px",
                  fontSize: "0.82rem", color: riskCfg.color,
                  textAlign: "center", lineHeight: 1.5,
                }}>
                  {riskCfg.emoji} {result.warning || riskCfg.statusDesc}
                </div>

                <div style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>
  <div style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>
  {/* 🕒{" "}
  {formatTimestamp(dataTimestamp)} */}
</div>
</div>
              </>
            )}
          </div>

          {/* PELT Change-Point Snapshot - On Right Side */}
          {result?.pelt_snapshot && (
            <PeltSnapshot snapshot={result.pelt_snapshot} />
          )}

          {/* Recommended Actions - Now under PELT on Right Side */}
          {showPrediction && (
            <div style={{ animation: "fadeSlide 0.4s ease-out" }}>
              <RecommendedActions riskLevel={riskLevel} />
            </div>
          )}

        </div>
      </div>

    </div>
  );
};

export default SwarmPrediction;