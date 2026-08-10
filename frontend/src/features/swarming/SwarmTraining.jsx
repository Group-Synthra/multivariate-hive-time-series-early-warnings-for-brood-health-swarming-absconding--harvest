import React, { useEffect, useState } from "react";
import {
    Bar,
    BarChart,
    CartesianGrid,
    Legend,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis,
} from "recharts";
import {
    Activity,
    ArrowRight,
    BarChart3,
    BrainCircuit,
    CheckCircle2,
    Clock3,
    Database,
    GitBranch,
    Layers3,
    Settings2,
    Timer,
    Trophy,
} from "lucide-react";
// import PipelineCard from "./PipelineCard";

const scaleErrorMetric = (value) => {
    if (value === null || value === undefined || value === "") return null;

    const numericValue = Number(value);
    return Number.isFinite(numericValue) ? numericValue * 100 : null;
};

const formatErrorMetric = (value) => {
    const scaledValue = scaleErrorMetric(value);
    return scaledValue === null ? "—" : scaledValue.toFixed(4);
};

// ============================================================
// Research-style PELT feature card
// ============================================================
function PeltFeatureCard({ index, icon: Icon, title, points }) {
    return (
        <article
            style={{
                minHeight: "160px",
                padding: "16px",
                background: "#ffffff",
                border: "1px solid #cbd5e1",
                borderRadius: "8px",
            }}
        >
            <div
                style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "9px",
                    marginBottom: "12px",
                }}
            >
                <div
                    style={{
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        width: "32px",
                        height: "32px",
                        flexShrink: 0,
                        color: "#1e3a5f",
                        background: "#f1f5f9",
                        border: "1px solid #dbe3ec",
                        borderRadius: "6px",
                    }}
                >
                    <Icon size={17} strokeWidth={1.8} />
                </div>
                <div>
                    <span
                        style={{
                            display: "block",
                            color: "#64748b",
                            fontSize: "10px",
                            fontWeight: 700,
                            letterSpacing: "0.06em",
                        }}
                    >
                        {index}
                    </span>
                    <h3
                        style={{
                            margin: 0,
                            color: "#0f172a",
                            fontSize: "14px",
                            fontWeight: 650,
                        }}
                    >
                        {title}
                    </h3>
                </div>
            </div>

            <ul
                style={{
                    display: "grid",
                    gap: "8px",
                    margin: 0,
                    padding: 0,
                    listStyle: "none",
                    color: "#475569",
                    fontSize: "12px",
                    lineHeight: 1.45,
                }}
            >
                {points.map((point) => (
                    <li
                        key={point}
                        style={{
                            display: "flex",
                            alignItems: "flex-start",
                            gap: "7px",
                        }}
                    >
                        <CheckCircle2
                            size={14}
                            color="#475569"
                            strokeWidth={1.8}
                            style={{ flexShrink: 0, marginTop: "2px" }}
                        />
                        <span>{point}</span>
                    </li>
                ))}
            </ul>
        </article>
    );
}

function HybridFramework() {
    const pipelineSteps = [
        {
            icon: Database,
            title: "Raw Sensor Data",
            subtitle: "Hive observations",
        },
        {
            icon: Settings2,
            title: "Preprocessing",
            subtitle: "Clean and scale data",
        },
        {
            icon: GitBranch,
            title: "PELT Change Detection",
            subtitle: "Detect behavioural shifts",
        },
        {
            icon: Layers3,
            title: "Feature Engineering",
            subtitle: "Create temporal features",
        },
        {
            icon: BrainCircuit,
            title: "RF · XGB · LSTM",
            subtitle: "Compare trained models",
        },
        {
            icon: Trophy,
            title: "Best Model",
            subtitle: "Selected for prediction",
            highlighted: true,
        },
    ];

    return (
        <section
            aria-labelledby="hybrid-framework-title"
            style={{
                marginTop: "28px",
                padding: "22px",
                background: "#ffffff",
                border: "1px solid #cbd5e1",
                borderRadius: "12px",
            }}
        >
            <div style={{ marginBottom: "20px" }}>
                <span
                    style={{
                        display: "block",
                        marginBottom: "5px",
                        color: "#64748b",
                        fontSize: "10px",
                        fontWeight: 700,
                        letterSpacing: "0.08em",
                        textTransform: "uppercase",
                    }}
                >
                    Analytical workflow
                </span>

                <h2
                    id="hybrid-framework-title"
                    style={{
                        margin: "0 0 8px",
                        color: "#0f172a",
                        fontSize: "17px",
                        fontWeight: 650,
                    }}
                >
                    Hybrid PELT–LSTM Prediction Framework
                </h2>

                <p
                    style={{
                        maxWidth: "980px",
                        margin: 0,
                        color: "#475569",
                        fontSize: "13px",
                        lineHeight: 1.6,
                    }}
                >
                    Behavioural change points detected by the PELT algorithm are
                    transformed into temporal features and combined with hive
                    sensor measurements. The LSTM achieved the best predictive
                    performance and was selected as the final swarming
                    prediction model.
                </p>
            </div>

            <div
                style={{
                    display: "flex",
                    alignItems: "stretch",
                    gap: "10px",
                    padding: "18px",
                    overflowX: "auto",
                    background: "#f8fafc",
                    border: "1px solid #dbe3ec",
                    borderRadius: "10px",
                }}
            >
                {pipelineSteps.map((step, index) => {
                    const StepIcon = step.icon;

                    return (
                        <React.Fragment key={step.title}>
                            <article
                                style={{
                                    display: "flex",
                                    minWidth: "145px",
                                    flex: "1 0 145px",
                                    flexDirection: "column",
                                    alignItems: "center",
                                    justifyContent: "center",
                                    minHeight: "128px",
                                    padding: "14px 12px",
                                    textAlign: "center",
                                    background: step.highlighted
                                        ? "#f0fdf4"
                                        : "#ffffff",
                                    border: step.highlighted
                                        ? "1px solid #86b89a"
                                        : "1px solid #cbd5e1",
                                    borderRadius: "8px",
                                }}
                            >
                                <div
                                    style={{
                                        display: "flex",
                                        alignItems: "center",
                                        justifyContent: "center",
                                        width: "38px",
                                        height: "38px",
                                        marginBottom: "10px",
                                        color: step.highlighted
                                            ? "#166534"
                                            : "#1e3a5f",
                                        background: step.highlighted
                                            ? "#dcfce7"
                                            : "#f1f5f9",
                                        border: step.highlighted
                                            ? "1px solid #bbddc5"
                                            : "1px solid #dbe3ec",
                                        borderRadius: "7px",
                                    }}
                                >
                                    <StepIcon size={19} strokeWidth={1.8} />
                                </div>

                                <strong
                                    style={{
                                        color: step.highlighted
                                            ? "#166534"
                                            : "#0f172a",
                                        fontSize: "12.5px",
                                        fontWeight: 650,
                                        lineHeight: 1.35,
                                    }}
                                >
                                    {step.title}
                                </strong>

                                <span
                                    style={{
                                        marginTop: "5px",
                                        color: "#64748b",
                                        fontSize: "10.5px",
                                        lineHeight: 1.35,
                                    }}
                                >
                                    {step.subtitle}
                                </span>
                            </article>

                            {index < pipelineSteps.length - 1 && (
                                <div
                                    aria-hidden="true"
                                    style={{
                                        display: "flex",
                                        alignItems: "center",
                                        justifyContent: "center",
                                        flex: "0 0 22px",
                                        color: "#64748b",
                                    }}
                                >
                                    <ArrowRight size={20} strokeWidth={1.7} />
                                </div>
                            )}
                        </React.Fragment>
                    );
                })}
            </div>
        </section>
    );
}

// ============================================================
// Research-style model performance comparison
// ============================================================
function PerformanceComparisonChart({ comparison = [] }) {
    if (!Array.isArray(comparison) || comparison.length === 0) {
        return (
            <p
                style={{
                    padding: "30px",
                    color: "#64748b",
                    textAlign: "center",
                    fontSize: "14px",
                }}
            >
                Model-comparison data are not available.
            </p>
        );
    }

    const modelColors = ["#1d4ed8", "#475569", "#0f766e", "#7c3aed"];

    const getNumber = (value) => {
        const number = Number(value);
        return Number.isFinite(number) ? number : null;
    };

    const classificationData = [
        {
            metric: "Precision",
            ...Object.fromEntries(
                comparison.map((row) => [row.Model, getNumber(row.Precision)])
            ),
        },
        {
            metric: "Recall",
            ...Object.fromEntries(
                comparison.map((row) => [row.Model, getNumber(row.Recall)])
            ),
        },
        {
            metric: "F1-score",
            ...Object.fromEntries(
                comparison.map((row) => [
                    row.Model,
                    getNumber(row["F1-Score"]),
                ])
            ),
        },
    ];

    const errorData = [
        {
            metric: "RMSE",
            ...Object.fromEntries(
                comparison.map((row) => [
                    row.Model,
                    scaleErrorMetric(row.RMSE ?? row.rmse),
                ])
            ),
        },
        {
            metric: "MAE",
            ...Object.fromEntries(
                comparison.map((row) => [
                    row.Model,
                    scaleErrorMetric(row.MAE ?? row.mae),
                ])
            ),
        },
    ];

    const tooltipFormatter = (value, modelName) => [
        value === null || value === undefined
            ? "Not available"
            : Number(value).toFixed(4),
        modelName,
    ];

    const chartPanelStyle = {
        minWidth: 0,
        padding: "20px",
        background: "#ffffff",
        border: "1px solid #d9e2ec",
        borderRadius: "10px",
    };

    const chartTitleStyle = {
        margin: "0 0 4px",
        color: "#0f172a",
        fontSize: "15px",
        fontWeight: 600,
    };

    const chartDescriptionStyle = {
        margin: "0 0 18px",
        color: "#64748b",
        fontSize: "12px",
    };

    const renderBars = () =>
        comparison.map((row, index) => (
            <Bar
                key={row.Model}
                dataKey={row.Model}
                name={row.Model}
                fill={modelColors[index % modelColors.length]}
                radius={[2, 2, 0, 0]}
                maxBarSize={42}
            />
        ));

    return (
        <section
            aria-label="Model performance comparison charts"
            style={{
                padding: "22px",
                background: "#f8fafc",
                border: "1px solid #cbd5e1",
                borderRadius: "12px",
            }}
        >
            <div style={{ marginBottom: "20px" }}>
                <h2
                    style={{
                        margin: "0 0 6px",
                        color: "#0f172a",
                        fontSize: "17px",
                        fontWeight: 600,
                    }}
                >
                    Model Performance Comparison
                </h2>
                <p
                    style={{
                        margin: 0,
                        color: "#64748b",
                        fontSize: "13px",
                        lineHeight: 1.5,
                    }}
                >
                    Comparative evaluation of classification performance and
                    prediction error across the trained models.
                </p>
            </div>

            <div
                style={{
                    display: "grid",
                    gridTemplateColumns:
                        "repeat(auto-fit, minmax(min(100%, 420px), 1fr))",
                    gap: "20px",
                }}
            >
                <div style={chartPanelStyle}>
                    <h3 style={chartTitleStyle}>Classification Performance</h3>
                    <p style={chartDescriptionStyle}>
                        Higher values indicate stronger performance.
                    </p>
                    <div style={{ width: "100%", height: "350px" }}>
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart
                                data={classificationData}
                                margin={{ top: 10, right: 20, left: 5, bottom: 10 }}
                                barCategoryGap="25%"
                            >
                                <CartesianGrid
                                    stroke="#e2e8f0"
                                    strokeDasharray="3 3"
                                    vertical={false}
                                />
                                <XAxis
                                    dataKey="metric"
                                    tick={{ fill: "#334155", fontSize: 12 }}
                                    axisLine={{ stroke: "#94a3b8" }}
                                    tickLine={false}
                                />
                                <YAxis
                                    domain={[0, 1]}
                                    ticks={[0, 0.2, 0.4, 0.6, 0.8, 1]}
                                    tickFormatter={(value) => value.toFixed(1)}
                                    tick={{ fill: "#475569", fontSize: 11 }}
                                    axisLine={{ stroke: "#94a3b8" }}
                                    tickLine={false}
                                    width={42}
                                    label={{
                                        value: "Score",
                                        angle: -90,
                                        position: "insideLeft",
                                        fill: "#475569",
                                        fontSize: 12,
                                    }}
                                />
                                <Tooltip
                                    formatter={tooltipFormatter}
                                    contentStyle={{
                                        background: "#ffffff",
                                        border: "1px solid #cbd5e1",
                                        borderRadius: "6px",
                                        fontSize: "12px",
                                    }}
                                />
                                <Legend
                                    verticalAlign="top"
                                    align="right"
                                    wrapperStyle={{
                                        paddingBottom: "16px",
                                        fontSize: "12px",
                                    }}
                                />
                                {renderBars()}
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                <div style={chartPanelStyle}>
                    <h3 style={chartTitleStyle}>Prediction Error</h3>
                    <p style={chartDescriptionStyle}>
                        Lower RMSE and MAE values indicate stronger performance.
                    </p>
                    <div style={{ width: "100%", height: "350px" }}>
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart
                                data={errorData}
                                margin={{ top: 10, right: 20, left: 5, bottom: 10 }}
                                barCategoryGap="25%"
                            >
                                <CartesianGrid
                                    stroke="#e2e8f0"
                                    strokeDasharray="3 3"
                                    vertical={false}
                                />
                                <XAxis
                                    dataKey="metric"
                                    tick={{ fill: "#334155", fontSize: 12 }}
                                    axisLine={{ stroke: "#94a3b8" }}
                                    tickLine={false}
                                />
                                <YAxis
                                    domain={[0, "auto"]}
                                    tickFormatter={(value) =>
                                        Number(value).toFixed(2)
                                    }
                                    tick={{ fill: "#475569", fontSize: 11 }}
                                    axisLine={{ stroke: "#94a3b8" }}
                                    tickLine={false}
                                    width={50}
                                    label={{
                                        value: "Error",
                                        angle: -90,
                                        position: "insideLeft",
                                        fill: "#475569",
                                        fontSize: 12,
                                    }}
                                />
                                <Tooltip
                                    formatter={tooltipFormatter}
                                    contentStyle={{
                                        background: "#ffffff",
                                        border: "1px solid #cbd5e1",
                                        borderRadius: "6px",
                                        fontSize: "12px",
                                    }}
                                />
                                <Legend
                                    verticalAlign="top"
                                    align="right"
                                    wrapperStyle={{
                                        paddingBottom: "16px",
                                        fontSize: "12px",
                                    }}
                                />
                                {renderBars()}
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            </div>

            <p
                style={{
                    margin: "16px 0 0",
                    color: "#64748b",
                    fontSize: "12px",
                    lineHeight: 1.5,
                }}
            >
          
            </p>
        </section>
    );
}

// ============================================================
// PELT Card Component - UPDATED
// ============================================================
function PeltCard({ data }) {
    if (!data || !data.pelt) return null;

    const tableStyle = {
        width: "100%",
        borderCollapse: "collapse",
        background: "#f8fafc",
        borderRadius: "10px",
        overflow: "hidden",
        fontSize: "12px",
    };

    const thStyle = {
        background: "#2563eb",
        color: "white",
        padding: "8px",
        textAlign: "center",
    };

    const tdStyle = {
        padding: "6px 8px",
        textAlign: "center",
        borderBottom: "1px solid #e2e8f0",
        color: "#475569",
    };

    // Helper to get all features as a flat array (including temporal)
    const getAllFeatures = () => {
        const features = [];
        const gen = data.pelt.generated_features;
        
        // Handle both old (array) and new (object) formats
        if (Array.isArray(gen)) {
            return gen;
        }
        
        // New format: object with categories
        if (gen.existing) features.push(...gen.existing);
        if (gen.per_variable) features.push(...gen.per_variable);
        if (gen.alignment) features.push(...gen.alignment);
        if (gen.temporal) features.push(...gen.temporal);
        
        return features;
    };

    const allFeatures = getAllFeatures();

    return (
        <div className="pelt-card">
            <h2>🔎 PELT Change Point Detection</h2>

            <p>
                <strong>Hives Processed:</strong> {data.pelt.total_hives_processed}
            </p>

            <p>
                <strong>Change Points Detected:</strong> {data.pelt.total_change_points}
            </p>

            <p>
                <strong>Total Features Generated:</strong> {data.pelt.total_features || allFeatures.length}
            </p>

            {/* Feature Categories */}
            {!Array.isArray(data.pelt.generated_features) && (
                <div style={{ marginTop: "10px" }}>
                    <p><strong>Generated Features:</strong></p>
                    
                    {data.pelt.generated_features.existing && (
                        <div style={{ marginBottom: "6px" }}>
                            <span style={{ color: "#2563eb", fontWeight: "bold" }}>Existing:</span>
                            <ul style={{ margin: "2px 0 6px 20px" }}>
                                {data.pelt.generated_features.existing.map((f, i) => (
                                    <li key={i} style={{ fontSize: "13px" }}>{f}</li>
                                ))}
                            </ul>
                        </div>
                    )}
                    
                    {data.pelt.generated_features.per_variable && (
                        <div style={{ marginBottom: "6px" }}>
                            <span style={{ color: "#22c55e", fontWeight: "bold" }}>Per-Variable:</span>
                            <ul style={{ margin: "2px 0 6px 20px" }}>
                                {data.pelt.generated_features.per_variable.slice(0, 6).map((f, i) => (
                                    <li key={i} style={{ fontSize: "13px" }}>{f}</li>
                                ))}
                                {data.pelt.generated_features.per_variable.length > 6 && (
                                    <li style={{ fontSize: "13px", color: "#64748b" }}>
                                        +{data.pelt.generated_features.per_variable.length - 6} more
                                    </li>
                                )}
                            </ul>
                        </div>
                    )}
                    
                    {data.pelt.generated_features.alignment && (
                        <div style={{ marginBottom: "6px" }}>
                            <span style={{ color: "#f59e0b", fontWeight: "bold" }}>Alignment:</span>
                            <ul style={{ margin: "2px 0 6px 20px" }}>
                                {data.pelt.generated_features.alignment.map((f, i) => (
                                    <li key={i} style={{ fontSize: "13px" }}>{f}</li>
                                ))}
                            </ul>
                        </div>
                    )}
                    
                    {data.pelt.generated_features.temporal && (
                        <div style={{ marginBottom: "6px" }}>
                            <span style={{ color: "#8b5cf6", fontWeight: "bold" }}>Temporal:</span>
                            <ul style={{ margin: "2px 0 6px 20px" }}>
                                {data.pelt.generated_features.temporal.map((f, i) => (
                                    <li key={i} style={{ fontSize: "13px" }}>{f}</li>
                                ))}
                            </ul>
                        </div>
                    )}
                </div>
            )}

            {/* Fallback for old format (array) */}
            {Array.isArray(data.pelt.generated_features) && (
                <>
                    <p><strong>Generated Features:</strong></p>
                    <ul>
                        {data.pelt.generated_features.map((feature, index) => (
                            <li key={index}>{feature}</li>
                        ))}
                    </ul>
                </>
            )}

            {/* PELT Summary Table */}
            {data.pelt.summary && data.pelt.summary.length > 0 && (
                <div style={{ marginTop: "16px" }}>
                    <h4 style={{ color: "#2563eb", fontSize: "14px", marginBottom: "8px" }}>
                        Breakpoint Summary
                    </h4>
                    <table style={tableStyle}>
                        <thead>
                            <tr>
                                <th style={thStyle}>Hive</th>
                                <th style={thStyle}>Records</th>
                                <th style={thStyle}>Breakpoints</th>
                                <th style={thStyle}>Avg Density</th>
                                <th style={thStyle}>Max Density</th>
                                <th style={thStyle}>per 100h</th>
                            </tr>
                        </thead>
                        <tbody>
                            {data.pelt.summary.slice(0, 10).map((row, idx) => (
                                <tr key={idx}>
                                    <td style={tdStyle}>{row.Hive}</td>
                                    <td style={tdStyle}>{row.Records}</td>
                                    <td style={tdStyle}>{row.Breakpoints}</td>
                                    <td style={tdStyle}>{row["Avg Density"]}</td>
                                    <td style={tdStyle}>{row["Max Density"]}</td>
                                    <td style={tdStyle}>{row["per 100h"]}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                    {data.pelt.summary.length > 10 && (
                        <p style={{ color: "#64748b", fontSize: "12px", marginTop: "6px" }}>
                            Showing top 10 of {data.pelt.summary.length} hives
                        </p>
                    )}
                </div>
            )}
        </div>
    );
}

// ============================================================
// Metric Card Component
// ============================================================
function MetricCard({ title, metrics, isBest }) {
    if (!metrics) return null;

    // Format as decimal with 4 decimal places
    const toDecimal = (value) => value.toFixed(4);

    // Color coding based on value
    const getColor = (value, type) => {
        if (type === "precision") return value >= 0.70 ? "#22c55e" : value >= 0.40 ? "#eab308" : "#ef4444";
        if (type === "recall") return value >= 0.85 ? "#22c55e" : value >= 0.60 ? "#eab308" : "#ef4444";
        if (type === "f1") return value >= 0.70 ? "#22c55e" : value >= 0.40 ? "#eab308" : "#ef4444";
        if (type === "rmse") return value <= 0.30 ? "#22c55e" : value <= 0.50 ? "#eab308" : "#ef4444";
        if (type === "mae") return value <= 0.15 ? "#22c55e" : value <= 0.30 ? "#eab308" : "#ef4444";
        return "#22c55e";
    };

    // Get metrics with fallbacks
    const precision = metrics.Precision || 0;
    const recall = metrics.Recall || 0;
    const f1 = metrics["F1-Score"] || 0;
    const rmse = metrics.RMSE ?? metrics.rmse ?? null;
    const mae = metrics.MAE ?? metrics.mae ?? null;

    return (
        <div
            style={{
                background: isBest ? "#ecfdf5" : "#ffffff",
                borderRadius: "15px",
                padding: "18px 20px",
                boxShadow: "0 8px 24px rgba(15,23,42,.08)",
                border: isBest ? "2px solid #16a34a" : "1px solid #dbe4f0",
            }}
        >
            <h2
                style={{
                    marginBottom: "16px",
                    color: "#2563eb",
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    fontSize: "16px",
                }}
            >
                {title}
                {isBest && (
                    <span
                        style={{
                            fontSize: "11px",
                            background: "#16a34a",
                            color: "#ffffff",
                            padding: "2px 10px",
                            borderRadius: "20px",
                            fontWeight: "bold",
                        }}
                    >
                        ⭐ BEST
                    </span>
                )}
            </h2>

            {/* Precision */}
            <div
                style={{
                    display: "flex",
                    justifyContent: "space-between",
                    padding: "6px 0",
                    borderBottom: "1px solid #e2e8f0",
                    fontSize: "14px",
                }}
            >
                <span>Precision</span>
                <strong style={{ color: getColor(precision, "precision"), fontSize: "15px" }}>
                    {toDecimal(precision)}
                </strong>
            </div>

            {/* Recall */}
            <div
                style={{
                    display: "flex",
                    justifyContent: "space-between",
                    padding: "6px 0",
                    borderBottom: "1px solid #e2e8f0",
                    fontSize: "14px",
                }}
            >
                <span>Recall</span>
                <strong style={{ color: getColor(recall, "recall"), fontSize: "15px" }}>
                    {toDecimal(recall)}
                </strong>
            </div>

            {/* F1-Score */}
            <div
                style={{
                    display: "flex",
                    justifyContent: "space-between",
                    padding: "6px 0",
                    borderBottom: "1px solid #e2e8f0",
                    fontSize: "14px",
                }}
            >
                <span>F1 Score</span>
                <strong style={{ color: getColor(f1, "f1"), fontSize: "15px" }}>
                    {toDecimal(f1)}
                </strong>
            </div>

            {/* RMSE ↓ - Regression Metric */}
            {rmse !== null && rmse !== undefined && (
                <div
                    style={{
                        display: "flex",
                        justifyContent: "space-between",
                        padding: "6px 0",
                        borderBottom: "1px solid #e2e8f0",
                        fontSize: "14px",
                    }}
                >
                    <span>RMSE ↓</span>
                    <strong style={{ color: getColor(rmse, "rmse"), fontSize: "15px" }}>
                        {formatErrorMetric(rmse)}
                    </strong>
                </div>
            )}

            {/* MAE ↓ - Regression Metric */}
            {mae !== null && mae !== undefined && (
                <div
                    style={{
                        display: "flex",
                        justifyContent: "space-between",
                        padding: "6px 0",
                        borderBottom: "1px solid #e2e8f0",
                        fontSize: "14px",
                    }}
                >
                    <span>MAE ↓</span>
                    <strong style={{ color: getColor(mae, "mae"), fontSize: "15px" }}>
                        {formatErrorMetric(mae)}
                    </strong>
                </div>
            )}

            {/* Additional metrics */}
            {metrics.ROC_AUC && (
                <div
                    style={{
                        display: "flex",
                        justifyContent: "space-between",
                        padding: "6px 0",
                        borderBottom: "none",
                        fontSize: "12px",
                        color: "#64748b",
                        borderTop: "1px solid #e2e8f0",
                        marginTop: "6px",
                        paddingTop: "8px",
                    }}
                >
                    <span>ROC-AUC</span>
                    <strong style={{ color: "#64748b" }}>
                        {metrics.ROC_AUC.toFixed(4)}
                    </strong>
                </div>
            )}
        </div>
    );
}

// ============================================================
// Swarm Training Component
// ============================================================
function SwarmTraining() {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        fetch("/api/swarming/model-training")
            .then((res) => {
                if (!res.ok) throw new Error(`HTTP error! Status: ${res.status}`);
                return res.json();
            })
            .then((result) => {
                setData(result);
                setLoading(false);
            })
            .catch((err) => {
                console.error("Error fetching data:", err);
                setError(err.message);
                setLoading(false);
            });
    }, []);

    if (loading) {
        return (
            <div
                style={{
                    padding: "30px",
                    background: "#f8fafc",
                    minHeight: "100vh",
                    color: "#0f172a",
                }}
            >
                <h2 style={{ color: "#0f172a", padding: "30px", fontSize: "18px" }}>
                    ⏳ Loading Model Training Results...
                </h2>
            </div>
        );
    }

    if (error) {
        return (
            <div
                style={{
                    padding: "30px",
                    background: "#f8fafc",
                    minHeight: "100vh",
                    color: "#0f172a",
                }}
            >
                <h2 style={{ color: "#ef4444", padding: "30px", fontSize: "18px" }}>
                    ❌ Error: {error}
                </h2>
                <p style={{ color: "#64748b" }}>Please make sure the backend server is running.</p>
            </div>
        );
    }

    // Helper: Format as decimal with 4 decimal places
    const toDecimal = (val) => val.toFixed(4);

    // ============================================================
    // Styling Constants
    // ============================================================
    const tableStyle = {
        width: "100%",
        borderCollapse: "collapse",
        background: "#ffffff",
        borderRadius: "12px",
        overflow: "hidden",
        fontSize: "13px",
    };

    const thStyle = {
        background: "#2563eb",
        color: "white",
        padding: "10px 12px",
        textAlign: "center",
    };

    const tdStyle = {
        padding: "8px 12px",
        textAlign: "center",
        borderBottom: "1px solid #e2e8f0",
        color: "#334155",
    };

    // Helper to get regression metrics from comparison row
    const getRegMetric = (row, key) => {
        return row[key] || row[key.toLowerCase()] || null;
    };

    const hasRMSE = data.comparison.some(
        (row) => scaleErrorMetric(row.RMSE ?? row.rmse) !== null
    );
    const hasMAE = data.comparison.some(
        (row) => scaleErrorMetric(row.MAE ?? row.mae) !== null
    );

    return (
        <div
            style={{
                padding: "24px 30px",
                background: "#f8fafc",
                minHeight: "100vh",
                color: "#0f172a",
                border: "1px solid #dbe4f0",
                borderRadius: "16px",
                boxShadow: "0 10px 30px rgba(15,23,42,.06)",
            }}
        >
            {/* ============================================================
                HEADER - Model Training Completed
            ============================================================ */}
            <div
                style={{
                    background: "#ecfdf5",
                    padding: "14px 20px",
                    borderRadius: "12px",
                    marginBottom: "20px",
                    borderLeft: "6px solid #16a34a",
                }}
            >
                <h3 style={{ margin: 0, fontSize: "15px" }}>✅ Model Training Completed</h3>
                {/* <p style={{ marginTop: "4px", fontSize: "14px" }}>
                    Best Model: &nbsp;
                    <strong style={{ color: "#16a34a" }}>
                        {data.best_model.Model}
                    </strong>
                    &nbsp; with F1-Score:{" "}
                    <strong style={{ color: "#16a34a" }}>
                        {toDecimal(data.best_model["F1-Score"])}
                    </strong>
                    {data.best_model.RMSE && (
                        <span style={{ color: "#64748b", fontSize: "12px", marginLeft: "12px" }}>
                            RMSE: {data.best_model.RMSE.toFixed(4)}
                        </span>
                    )}
                </p> */}
            </div>

            {/* ============================================================
                PIPELINE & PELT CARDS
            ============================================================ */}
            {/* <PipelineCard />
            <PeltCard data={data} /> */}

            {/* ============================================================
                PELT FEATURES - 4 COLUMN GRID
            ============================================================ */}
            <section
                aria-labelledby="pelt-features-title"
                style={{ marginTop: "28px" }}
            >
                <div style={{ marginBottom: "16px" }}>
                    <span
                        style={{
                            display: "block",
                            marginBottom: "5px",
                            color: "#64748b",
                            fontSize: "10px",
                            fontWeight: 700,
                            letterSpacing: "0.08em",
                            textTransform: "uppercase",
                        }}
                    >
                        Behavioural indicators
                    </span>
                    <h2
                        id="pelt-features-title"
                        style={{
                            margin: "0 0 6px",
                            color: "#0f172a",
                            fontSize: "17px",
                            fontWeight: 650,
                        }}
                    >
                        Generated PELT Features
                    </h2>
                </div>

                <div
                    style={{
                        display: "grid",
                        gridTemplateColumns:
                            "repeat(auto-fit, minmax(min(100%, 230px), 1fr))",
                        gap: "14px",
                    }}
                >
                    <PeltFeatureCard
                        index="P1"
                        icon={GitBranch}
                        title="Breakpoint"
                        points={[
                            "0: no change detected",
                            "1: behavioural change detected",
                        ]}
                    />
                    <PeltFeatureCard
                        index="P2"
                        icon={Timer}
                        title="Days Since Breakpoint"
                        points={[
                            "Time since the latest change",
                            "Lower value: more recent change",
                        ]}
                    />
                    <PeltFeatureCard
                        index="P3"
                        icon={BarChart3}
                        title="Breakpoint Density"
                        points={[
                            "Changes within the last 24 hours",
                            "Higher value: frequent changes",
                        ]}
                    />
                    <PeltFeatureCard
                        index="P4"
                        icon={Clock3}
                        title="Segment Duration"
                        points={[
                            "Length of the current state",
                            "Longer duration: greater stability",
                        ]}
                    />
                </div>
            </section>

            <HybridFramework />

            {/* ============================================================
                METRIC CARDS (3 columns)
            ============================================================ */}
            <div
                style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(3, 1fr)",
                    gap: "18px",
                    marginTop: "28px",
                }}
            >
                <MetricCard
                    title="Random Forest"
                    metrics={data.rf}
                    isBest={data.best_model.Model === "Random Forest"}
                />
                <MetricCard
                    title="XGBoost"
                    metrics={data.xgb}
                    isBest={data.best_model.Model === "XGBoost"}
                />
                <MetricCard
                    title="LSTM"
                    metrics={data.lstm}
                    isBest={data.best_model.Model === "LSTM"}
                />
            </div>

            {/* ============================================================
                BEST MODEL CARD
            ============================================================ */}
            <section
                aria-labelledby="selected-model-title"
                style={{
                    marginTop: "28px",
                    overflow: "hidden",
                    background: "#ffffff",
                    border: "1px solid #cbd5e1",
                    borderRadius: "12px",
                    boxShadow: "0 3px 10px rgba(15, 23, 42, 0.05)",
                }}
            >
                <div
                    style={{
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "space-between",
                        gap: "18px",
                        padding: "18px 20px",
                        borderBottom: "1px solid #e2e8f0",
                    }}
                >
                    <div
                        style={{
                            display: "flex",
                            alignItems: "center",
                            gap: "12px",
                        }}
                    >
                        <div
                            style={{
                                display: "flex",
                                alignItems: "center",
                                justifyContent: "center",
                                width: "40px",
                                height: "40px",
                                flexShrink: 0,
                                color: "#1e3a5f",
                                background: "#f1f5f9",
                                border: "1px solid #cbd5e1",
                                borderRadius: "8px",
                            }}
                        >
                            <Trophy size={20} strokeWidth={1.8} />
                        </div>

                        <div>
                            <span
                                style={{
                                    display: "block",
                                    marginBottom: "3px",
                                    color: "#64748b",
                                    fontSize: "10px",
                                    fontWeight: 700,
                                    letterSpacing: "0.08em",
                                    textTransform: "uppercase",
                                }}
                            >
                                Model selection result
                            </span>
                            <h2
                                id="selected-model-title"
                                style={{
                                    margin: 0,
                                    color: "#0f172a",
                                    fontSize: "17px",
                                    fontWeight: 650,
                                }}
                            >
                                Selected Best Model
                            </h2>
                        </div>
                    </div>

                    <span
                        style={{
                            padding: "7px 12px",
                            color: "#1e3a5f",
                            background: "#eff6ff",
                            border: "1px solid #bfdbfe",
                            borderRadius: "6px",
                            fontSize: "14px",
                            fontWeight: 700,
                        }}
                    >
                        {data.best_model.Model}
                    </span>
                </div>

                <div style={{ padding: "20px" }}>
                    <p
                        style={{
                            margin: "0 0 18px",
                            color: "#475569",
                            fontSize: "13px",
                            lineHeight: 1.55,
                        }}
                    >
                        This model achieved the highest overall performance
                        based on the evaluation metrics.
                    </p>

                    <div
                        style={{
                            display: "grid",
                            gridTemplateColumns:
                                "repeat(auto-fit, minmax(140px, 1fr))",
                            overflow: "hidden",
                            background: "#cbd5e1",
                            border: "1px solid #cbd5e1",
                            borderRadius: "8px",
                            gap: "1px",
                        }}
                    >
                        {[
                            {
                                label: "Precision",
                                value: data.best_model.Precision,
                                note: "Higher is better",
                            },
                            {
                                label: "Recall",
                                value: data.best_model.Recall,
                                note: "Higher is better",
                            },
                            {
                                label: "F1-Score",
                                value: data.best_model["F1-Score"],
                                note: "Higher is better",
                            },
                            {
                                label: "RMSE",
                                value:
                                    data.best_model.RMSE ??
                                    data.best_model.rmse,
                                note: "Lower is better",
                                isErrorMetric: true,
                            },
                            {
                                label: "MAE",
                                value:
                                    data.best_model.MAE ??
                                    data.best_model.mae,
                                note: "Lower is better",
                                isErrorMetric: true,
                            },
                        ].map((metricItem) => {
                            const numericValue = Number(metricItem.value);
                            const formattedValue = metricItem.isErrorMetric
                                ? formatErrorMetric(metricItem.value)
                                : Number.isFinite(numericValue)
                                  ? numericValue.toFixed(4)
                                  : "—";

                            return (
                                <div
                                    key={metricItem.label}
                                    style={{
                                        minHeight: "102px",
                                        padding: "15px",
                                        background: "#f8fafc",
                                    }}
                                >
                                    <span
                                        style={{
                                            display: "block",
                                            marginBottom: "8px",
                                            color: "#64748b",
                                            fontSize: "11px",
                                            fontWeight: 600,
                                        }}
                                    >
                                        {metricItem.label}
                                    </span>

                                    <strong
                                        style={{
                                            display: "block",
                                            color: "#0f172a",
                                            fontSize: "21px",
                                            fontWeight: 700,
                                            fontVariantNumeric: "tabular-nums",
                                            lineHeight: 1.1,
                                        }}
                                    >
                                        {formattedValue}
                                    </strong>

                                    <span
                                        style={{
                                            display: "block",
                                            marginTop: "7px",
                                            color: "#94a3b8",
                                            fontSize: "10px",
                                        }}
                                    >
                                        {metricItem.note}
                                    </span>
                                </div>
                            );
                        })}
                    </div>
                </div>
            </section>

            {/* ============================================================
                MODEL COMPARISON TABLE
            ============================================================ */}
            <div style={{ marginTop: "28px" }}>
                <h2 style={{ marginBottom: "14px", fontSize: "17px" }}>
                    📊 Model Performance Comparison
                </h2>

                <table style={tableStyle}>
                    <thead>
                        <tr>
                            <th style={thStyle}>Model</th>
                            <th style={thStyle}>Precision</th>
                            <th style={thStyle}>Recall</th>
                            <th style={thStyle}>F1 Score</th>
                            {hasRMSE && (
                                <th style={thStyle}>RMSE ↓</th>
                            )}
                            {hasMAE && (
                                <th style={thStyle}>MAE ↓</th>
                            )}
                            {data.comparison[0]?.ROC_AUC && (
                                <th style={thStyle}>ROC-AUC</th>
                            )}
                        </tr>
                    </thead>

                    <tbody>
                        {data.comparison.map((row, index) => {
                            const isBest = row.Model === data.best_model.Model;
                            return (
                                <tr
                                    key={index}
                                    style={
                                        isBest
                                            ? {
                                                  background: "#dcfce7",
                                                  fontWeight: "bold",
                                              }
                                            : {}
                                    }
                                >
                                    <td style={tdStyle}>
                                        {isBest && "⭐ "}
                                        {row.Model}
                                    </td>
                                    <td style={tdStyle}>
                                        {toDecimal(row.Precision)}
                                    </td>
                                    <td style={tdStyle}>
                                        {toDecimal(row.Recall)}
                                    </td>
                                    <td style={tdStyle}>
                                        {toDecimal(row["F1-Score"])}
                                    </td>
                                    {hasRMSE && (
                                        <td style={tdStyle}>
                                            {formatErrorMetric(row.RMSE ?? row.rmse)}
                                        </td>
                                    )}
                                    {hasMAE && (
                                        <td style={tdStyle}>
                                            {formatErrorMetric(row.MAE ?? row.mae)}
                                        </td>
                                    )}
                                    {row.ROC_AUC && (
                                        <td style={tdStyle}>
                                            {row.ROC_AUC.toFixed(4)}
                                        </td>
                                    )}
                                </tr>
                            );
                        })}
                    </tbody>
                </table>
            </div>

            {/* ============================================================
                PERFORMANCE COMPARISON CHART
            ============================================================ */}
            <div style={{ marginTop: "28px" }}>
                <PerformanceComparisonChart comparison={data.comparison} />
            </div>

            {/* ============================================================
                CSS Styles (for PELT card)
            ============================================================ */}
            <style>{`
                .pelt-card {
                    background: #eff6ff;
                    padding: 20px 24px;
                    border-radius: 15px;
                    margin-bottom: 24px;
                    border: 1px solid #bfdbfe;
                    border-left: 6px solid #2563eb;
                }

                .pelt-card h2 {
                    color: #2563eb;
                    margin-top: 0;
                    font-size: 17px;
                }

                .pelt-card p {
                    color: #475569;
                    margin: 4px 0;
                    font-size: 14px;
                }

                .pelt-card ul {
                    color: #475569;
                    padding-left: 20px;
                    margin: 6px 0;
                    font-size: 14px;
                }

                .pelt-card ul li {
                    margin: 2px 0;
                }

                .pelt-card table {
                    width: 100%;
                    border-collapse: collapse;
                    background: #f8fafc;
                    border-radius: 10px;
                    overflow: hidden;
                    font-size: 12px;
                }

                .pelt-card table th {
                    background: #2563eb;
                    color: white;
                    padding: 8px;
                    text-align: center;
                }

                .pelt-card table td {
                    padding: 6px 8px;
                    text-align: center;
                    border-bottom: 1px solid #e2e8f0;
                    color: #475569;
                }

                .pelt-card table tr:hover td {
                    background: #ffffff;
                }
            `}</style>
        </div>
    );
}

export default SwarmTraining;