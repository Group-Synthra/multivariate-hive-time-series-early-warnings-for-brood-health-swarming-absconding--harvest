import React, { useEffect, useState } from "react";
// import PipelineCard from "./PipelineCard";
import PeltFeatureCard from "./PeltFeatureCard";
import HybridFramework from "./HybridFramework";

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
// Metric Card Component (With RMSE, MAE, R²)
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
        if (type === "r2") return value >= 0.80 ? "#22c55e" : value >= 0.60 ? "#eab308" : "#ef4444";
        return "#22c55e";
    };

    // Get metrics with fallbacks
    const precision = metrics.Precision || 0;
    const recall = metrics.Recall || 0;
    const f1 = metrics["F1-Score"] || 0;
    const rmse = metrics.RMSE || metrics.rmse || null;
    const mae = metrics.MAE || metrics.mae || null;
    const r2 = metrics.R2 || metrics.r2 || metrics["R²"] || null;

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
                        {rmse.toFixed(4)}
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
                        {mae.toFixed(4)}
                    </strong>
                </div>
            )}

            {/* R² ↑ - Regression Metric */}
            {r2 !== null && r2 !== undefined && (
                <div
                    style={{
                        display: "flex",
                        justifyContent: "space-between",
                        padding: "6px 0",
                        borderBottom: "none",
                        fontSize: "14px",
                    }}
                >
                    <span>R² ↑</span>
                    <strong style={{ color: getColor(r2, "r2"), fontSize: "15px" }}>
                        {r2.toFixed(4)}
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
                    {data.best_model.R2 && (
                        <span style={{ color: "#64748b", fontSize: "12px", marginLeft: "12px" }}>
                            R²: {data.best_model.R2.toFixed(4)}
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
            <h2
                style={{
                    marginTop: 28,
                    marginBottom: 16,
                    fontSize: "17px",
                    color: "#2563eb",
                }}
            >
                Generated PELT Features
            </h2>

            <div
                style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(4, 1fr)",
                    gap: "14px",
                }}
            >
                <PeltFeatureCard
                    icon="📍"
                    title="Breakpoint"
                    color="#ef4444"
                    description="Identifies sudden behavioural changes in the hive using the PELT change-point detection algorithm."
                />
                <PeltFeatureCard
                    icon="⏱"
                    title="Days Since Breakpoint"
                    color="#3b82f6"
                    description="Measures the elapsed time since the most recent behavioural change was detected."
                />
                <PeltFeatureCard
                    icon="📊"
                    title="Breakpoint Density"
                    color="#22c55e"
                    description="Counts the number of detected behavioural changes within the previous 24-hour observation window."
                />
                <PeltFeatureCard
                    icon="📈"
                    title="Segment Duration"
                    color="#f59e0b"
                    description="Represents the duration of stable hive behaviour between two consecutive change points."
                />
            </div>

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
            <div
                style={{
                    marginTop: "28px",
                    background: "#ecfdf5",
                    padding: "20px 24px",
                    borderRadius: "15px",
                    textAlign: "center",
                    borderLeft: "8px solid #16a34a",
                }}
            >
                <h3 style={{ margin: 0, fontSize: "15px" }}>🏆 Selected Best Model</h3>
                <h1
                    style={{
                        fontSize: "30px",
                        color: "#16a34a",
                        marginTop: "6px",
                    }}
                >
                    {data.best_model.Model}
                </h1>
                <p style={{ margin: 0, fontSize: "13px", color: "#64748b" }}>
                    This model achieved the highest overall performance based on the
                    evaluation metrics.
                </p>
                <div
                    style={{
                        display: "flex",
                        justifyContent: "center",
                        gap: "28px",
                        marginTop: "14px",
                        flexWrap: "wrap",
                    }}
                >
                    <div>
                        <span style={{ color: "#64748b", fontSize: "12px" }}>Precision</span>
                        <br />
                        <strong style={{ color: "#22c55e", fontSize: "17px" }}>
                            {toDecimal(data.best_model.Precision)}
                        </strong>
                    </div>
                    <div>
                        <span style={{ color: "#64748b", fontSize: "12px" }}>Recall</span>
                        <br />
                        <strong style={{ color: "#22c55e", fontSize: "17px" }}>
                            {toDecimal(data.best_model.Recall)}
                        </strong>
                    </div>
                    <div>
                        <span style={{ color: "#64748b", fontSize: "12px" }}>F1-Score</span>
                        <br />
                        <strong style={{ color: "#22c55e", fontSize: "17px" }}>
                            {toDecimal(data.best_model["F1-Score"])}
                        </strong>
                    </div>
                    {data.best_model.RMSE && (
                        <div>
                            <span style={{ color: "#64748b", fontSize: "12px" }}>RMSE ↓</span>
                            <br />
                            <strong style={{ color: "#22c55e", fontSize: "17px" }}>
                                {data.best_model.RMSE.toFixed(4)}
                            </strong>
                        </div>
                    )}
                    {data.best_model.MAE && (
                        <div>
                            <span style={{ color: "#64748b", fontSize: "12px" }}>MAE ↓</span>
                            <br />
                            <strong style={{ color: "#22c55e", fontSize: "17px" }}>
                                {data.best_model.MAE.toFixed(4)}
                            </strong>
                        </div>
                    )}
                    {data.best_model.R2 && (
                        <div>
                            <span style={{ color: "#64748b", fontSize: "12px" }}>R² ↑</span>
                            <br />
                            <strong style={{ color: "#22c55e", fontSize: "17px" }}>
                                {data.best_model.R2.toFixed(4)}
                            </strong>
                        </div>
                    )}
                </div>
            </div>

            {/* ============================================================
                MODEL COMPARISON TABLE (With RMSE, MAE, R²)
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
                            {data.comparison[0]?.RMSE && (
                                <th style={thStyle}>RMSE ↓</th>
                            )}
                            {data.comparison[0]?.MAE && (
                                <th style={thStyle}>MAE ↓</th>
                            )}
                            {data.comparison[0]?.R2 && (
                                <th style={thStyle}>R² ↑</th>
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
                                    {row.RMSE && (
                                        <td style={tdStyle}>{row.RMSE.toFixed(4)}</td>
                                    )}
                                    {row.MAE && (
                                        <td style={tdStyle}>{row.MAE.toFixed(4)}</td>
                                    )}
                                    {row.R2 && (
                                        <td style={tdStyle}>{row.R2.toFixed(4)}</td>
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
                <h2 style={{ marginBottom: "14px", fontSize: "17px" }}>
                    📈 Performance Comparison Chart
                </h2>
                <img
                    src="/api/swarming/model-training/chart"
                    alt="Comparison Chart"
                    style={{
                        width: "100%",
                        borderRadius: "15px",
                        background: "#ffffff",
                        border: "1px solid #dbe4f0",
                        padding: "10px",
                        boxShadow: "0 8px 24px rgba(15,23,42,.08)",
                    }}
                    onError={(e) => {
                        e.target.style.display = "none";
                        e.target.parentElement.innerHTML =
                            '<p style="color:#64748b;padding:30px;text-align:center;font-size:14px;">⚠️ Chart not available. Please run the model training first.</p>';
                    }}
                />
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