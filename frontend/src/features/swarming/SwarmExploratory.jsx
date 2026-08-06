import React, { useState, useEffect, useCallback } from 'react';
import {
  BarChart2, Activity, TrendingUp, TrendingDown, AlertTriangle,
  CheckCircle, Info, ChevronDown, ChevronUp, RefreshCw, Image,
  FileText, Zap, Database, Layers, Star, XCircle, Loader
} from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, Legend
} from 'recharts';

// ─── colour palette ──────────────────────────────────────────────────────────
const C = {
  crimson:  '#ef4444',
  gold:     '#f59e0b',
  emerald:  '#10b981',
  cyan:     '#06b6d4',
  purple:   '#a78bfa',
  indigo:   '#6366f1',
};

// ─── small helpers ────────────────────────────────────────────────────────────
const fmt = (v, d = 2) => (typeof v === 'number' ? v.toFixed(d) : '—');
const pct = v => (typeof v === 'number' ? `${v.toFixed(2)}%` : '—');

function StatCard({ label, value, sub, color = C.cyan, icon: Icon }) {
  return (
    <div style={{
      background: `linear-gradient(135deg, ${color}18, transparent)`,
      border: `1px solid ${color}30`,
      borderLeft: `4px solid ${color}`,
      borderRadius: '10px',
      padding: '0.9rem 1rem',
      display: 'flex', flexDirection: 'column', gap: '0.2rem',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem',
        fontSize: '0.7rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
        {Icon && <Icon size={13} color={color} />}{label}
      </div>
      <div style={{ fontSize: '1.4rem', fontWeight: 700, color }}>{value}</div>
      {sub && <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>{sub}</div>}
    </div>
  );
}

function Section({ title, icon: Icon, color = C.cyan, children, defaultOpen = true }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="card" style={{ padding: 0, overflow: 'hidden' }}>
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          padding: '0.85rem 1rem', background: `${color}0d`,
          border: 'none', borderBottom: open ? `1px solid ${color}25` : 'none',
          cursor: 'pointer', color: 'var(--text-primary)',
        }}
      >
        <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 600, fontSize: '0.9rem' }}>
          {Icon && <Icon size={16} color={color} />}{title}
        </span>
        {open ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
      </button>
      {open && <div style={{ padding: '1rem' }}>{children}</div>}
    </div>
  );
}

function ImageViewer({ filename }) {
  const src = `/api/eda-swarming/images/${filename}`;
  const label = filename.replace(/_/g, ' ').replace('.png', '');
  const [expanded, setExpanded] = useState(false);
  return (
    <div style={{
      border: '1px solid rgba(255,255,255,0.08)', borderRadius: '8px',
      overflow: 'hidden', background: 'rgba(0,0,0,0.25)',
    }}>
      <div style={{
        padding: '0.4rem 0.7rem', fontSize: '0.7rem',
        color: 'var(--text-secondary)', textTransform: 'capitalize',
        background: 'rgba(255,255,255,0.04)',
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
      }}>
        <span><Image size={11} style={{ marginRight: '0.3rem', verticalAlign: 'middle' }} />{label}</span>
        <button
          onClick={() => setExpanded(e => !e)}
          style={{ background: 'none', border: 'none', cursor: 'pointer',
            fontSize: '0.65rem', color: C.cyan }}
        >{expanded ? 'Collapse' : 'Expand'}</button>
      </div>
      <img
        src={src}
        alt={label}
        style={{
          width: '100%',
          maxHeight: expanded ? '800px' : '280px',
          objectFit: 'contain',
          display: 'block',
          transition: 'max-height 0.35s ease',
          cursor: 'zoom-in',
        }}
        onClick={() => setExpanded(e => !e)}
      />
    </div>
  );
}

// ─── Main component ────────────────────────────────────────────────────────────
export default function SwarmExploratory() {
  const [edaData,   setEdaData]   = useState(null);
  const [images,    setImages]    = useState([]);
  const [report,    setReport]    = useState('');
  const [loading,   setLoading]   = useState(true);
  const [error,     setError]     = useState(null);

  const fetchAll = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [dataRes, imagesRes, reportRes] = await Promise.all([
        fetch('/api/eda-swarming'),
        fetch('/api/eda-swarming/images'),
        fetch('/api/eda-swarming/report'),
      ]);

      if (!dataRes.ok) throw new Error(`Dashboard JSON: ${dataRes.status} ${dataRes.statusText}`);
      const data    = await dataRes.json();
      const imgList = imagesRes.ok ? await imagesRes.json() : [];
      const rpt     = reportRes.ok ? (await reportRes.json()).report : '';

      setEdaData(data);
      setImages(imgList);
      setReport(rpt);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchAll(); }, [fetchAll]);

  // ── Loading ────────────────────────────────────────────────────────────────
  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '4rem 1rem' }}>
        <Loader size={36} color={C.gold} style={{ animation: 'spin 1.2s linear infinite' }} />
        <p style={{ color: 'var(--text-secondary)', marginTop: '1rem' }}>
          Loading Swarming EDA results…
        </p>
        <style>{`@keyframes spin { from { transform:rotate(0deg) } to { transform:rotate(360deg) } }`}</style>
      </div>
    );
  }

  // ── Error ──────────────────────────────────────────────────────────────────
  if (error) {
    return (
      <div className="card" style={{ textAlign: 'center', padding: '3rem 1rem' }}>
        <XCircle size={40} color={C.crimson} />
        <h3 style={{ color: C.crimson, marginTop: '0.75rem' }}>Could not load EDA data</h3>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginTop: '0.5rem' }}>{error}</p>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.8rem', marginTop: '0.5rem' }}>
          Make sure the Flask backend is running and <code>backend/outputs/eda_swarming/dashboard.json</code> exists.
        </p>
        <button
          onClick={fetchAll}
          style={{ marginTop: '1rem', padding: '0.5rem 1.25rem', borderRadius: '6px',
            background: C.cyan, color: '#0f172a', border: 'none', cursor: 'pointer', fontWeight: 600 }}
        >
          <RefreshCw size={14} style={{ marginRight: '0.4rem', verticalAlign: 'middle' }} />
          Retry
        </button>
      </div>
    );
  }

  // ── Data shortcuts ─────────────────────────────────────────────────────────
  const summary      = edaData?.summary         || {};
  const dquality     = edaData?.data_quality    || {};
  const distStats    = edaData?.distribution_stats?.numeric || {};
  const catStats     = edaData?.distribution_stats?.categorical || {};
  const swarmAnal    = edaData?.swarming_analysis || {};
  const corrAnal     = edaData?.correlation_analysis?.top_features || [];
  const featSugg     = edaData?.feature_suggestions || [];
  const outliers     = dquality?.outliers || {};
  const byHive       = swarmAnal?.by_hive || [];
  const byStock      = swarmAnal?.by_bee_stock || [];
  const swarmDist    = swarmAnal?.distribution || {};

  // bar-chart data
  const corrChartData = corrAnal.map(f => ({
    feature: f.feature.replace(/_/g, ' '),
    rawFeature: f.feature,
    value: parseFloat(f.correlation.toFixed(4)),
    abs: Math.abs(f.correlation),
  }));

  const hiveChartData = byHive.slice(0, 12).map(h => ({
    hive: h.hive_id,
    swarmRate: parseFloat(h.swarm_rate.toFixed(3)),
    swarmEvents: h.swarm_events,
  }));

  const stockChartData = byStock.map(s => ({
    stock: s.bee_stock,
    rate: parseFloat(s.swarm_rate.toFixed(3)),
    events: s.swarm_events,
  }));

  const outlierChartData = Object.entries(outliers).map(([feat, info]) => ({
    feature: feat.replace(/_/g, ' '),
    pct: parseFloat((info.percentage || 0).toFixed(2)),
  }));

  // feature suggestion type colours
  const suggColor = { temporal: C.cyan, derived: C.gold, rolling: C.emerald, regime: C.purple };

  // ─── render ─────────────────────────────────────────────────────────────────
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>

      {/* ── Header Banner ──────────────────────────────────────────────── */}
      <div className="card" style={{
        background: 'linear-gradient(135deg, rgba(20,26,40,0.85) 0%, rgba(16,185,129,0.10) 100%)',
        borderLeft: '4px solid var(--accent-emerald)',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        flexWrap: 'wrap', gap: '0.75rem',
      }}>
        <div>
          <h2 style={{ margin: 0, color: C.emerald, fontSize: '1.1rem' }}>
            🐝 Swarming EDA — Exploratory Data Analysis
          </h2>
          {/* <p style={{ margin: '0.3rem 0 0', color: 'var(--text-secondary)', fontSize: '0.8rem' }}>
            Results from <code>eda_analysis_swarming.py</code> →
            <code style={{ marginLeft: '0.25rem' }}>backend/outputs/eda_swarming/</code>
          </p> */}
        </div>
        <button
          onClick={fetchAll}
          style={{ padding: '0.45rem 1rem', borderRadius: '6px',
            background: C.emerald, color: '#0f172a', border: 'none',
            cursor: 'pointer', fontWeight: 600, fontSize: '0.8rem',
            display: 'flex', alignItems: 'center', gap: '0.4rem' }}
        >
          <RefreshCw size={14} /> Refresh
        </button>
      </div>

      {/* ── Summary Stats ──────────────────────────────────────────────── */}
      {/* <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: '0.75rem' }}>
        <StatCard label="Total Records"  value={(summary.total_records || 0).toLocaleString()}
          sub="Hive sensor readings"    color={C.cyan}    icon={Database} />
        <StatCard label="Total Hives"   value={summary.total_hives || '—'}
          sub="Monitored colonies"       color={C.emerald} icon={Activity} />
        <StatCard label="Dataset Span"  value={`${summary.time_days || '—'} days`}
          sub="Time range covered"       color={C.gold}    icon={BarChart2} />
        <StatCard label="Swarm Rate (72h)" value={pct(summary.swarm_rate)}
          sub="Positive label rate"      color={C.crimson} icon={Zap} />
        <StatCard label="EDA Images"   value={images.length}
          sub="Generated plots"          color={C.purple}  icon={Image} />
      </div> */}

      {/* ── Swarming Event Distribution ────────────────────────────────── */}
      <Section title="Swarming Event Distribution" icon={Zap} color={C.crimson}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '0.75rem', marginBottom: '1rem' }}>
          {[
            // { label: 'Swarming Events (actual)',    d: swarmDist?.swarming_event,    color: C.crimson },
            // { label: 'Swarming Label (next 72 h)',  d: swarmDist?.swarming_72h,      color: C.gold },
          ].map(({ label, d, color }) => d && (
            <div key={label} style={{ background: `${color}12`, borderRadius: '8px', padding: '0.75rem',
              border: `1px solid ${color}30` }}>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: '0.5rem', fontWeight: 600 }}>{label}</div>
              <div style={{ display: 'flex', gap: '1rem' }}>
                <div>
                  <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>Positive</div>
                  <div style={{ color, fontWeight: 700, fontSize: '1.1rem' }}>{(d.count?.positive || 0).toLocaleString()}</div>
                  <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>{pct(d.rate?.positive)}</div>
                </div>
                <div>
                  <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>Negative</div>
                  <div style={{ color: C.emerald, fontWeight: 700, fontSize: '1.1rem' }}>{(d.count?.negative || 0).toLocaleString()}</div>
                  <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>{pct(d.rate?.negative)}</div>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Swarm rate by hive bar chart */}
        {hiveChartData.length > 0 && (
          <>
            <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>
              Swarm event rate by hive (top 12)
            </p>
            <div style={{ height: 220 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={hiveChartData} margin={{ top: 4, right: 10, left: -18, bottom: 40 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="hive" stroke="var(--text-secondary)" tick={{ fontSize: 10 }} angle={-35} textAnchor="end" interval={0} />
                  <YAxis stroke="var(--text-secondary)" tick={{ fontSize: 10 }} unit="%" />
                  <Tooltip
                    contentStyle={{ background: '#1e293b', border: '1px solid rgba(255,255,255,0.1)', fontSize: '0.78rem' }}
                    formatter={(v, n) => [`${v}%`, 'Swarm Rate']}
                  />
                  <Bar dataKey="swarmRate" name="Swarm Rate %" radius={[4, 4, 0, 0]}>
                    {hiveChartData.map((_, i) => (
                      <Cell key={i} fill={i < 3 ? C.crimson : i < 6 ? C.gold : C.cyan} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </>
        )}

        {/* By bee stock */}
        {stockChartData.length > 0 && (
          <>
            <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', margin: '1rem 0 0.5rem' }}>
              Swarm event rate by bee stock
            </p>
            <div style={{ height: 180 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={stockChartData} margin={{ top: 4, right: 10, left: -18, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="stock" stroke="var(--text-secondary)" tick={{ fontSize: 11 }} />
                  <YAxis stroke="var(--text-secondary)" tick={{ fontSize: 10 }} unit="%" />
                  <Tooltip
                    contentStyle={{ background: '#1e293b', border: '1px solid rgba(255,255,255,0.1)', fontSize: '0.78rem' }}
                    formatter={(v) => [`${v}%`, 'Swarm Rate']}
                  />
                  <Bar dataKey="rate" name="Swarm Rate %" radius={[4, 4, 0, 0]}>
                    {stockChartData.map((_, i) => (
                      <Cell key={i} fill={[C.emerald, C.cyan, C.crimson][i % 3]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </>
        )}
      </Section>

      {/* ── Feature Correlations with Swarming ──────────────────────────
      {corrChartData.length > 0 && (
        <Section title="Feature Correlations with Swarming (Next 72h)" icon={TrendingUp} color={C.gold}>
          <div style={{ height: 220, marginBottom: '0.75rem' }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={corrChartData} layout="vertical"
                margin={{ top: 4, right: 20, left: 140, bottom: 4 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" horizontal={false} />
                <XAxis type="number" stroke="var(--text-secondary)" tick={{ fontSize: 10 }} />
                <YAxis type="category" dataKey="feature" width={138} stroke="var(--text-secondary)" tick={{ fontSize: 10 }} />
                <Tooltip
                  contentStyle={{ background: '#1e293b', border: '1px solid rgba(255,255,255,0.1)', fontSize: '0.78rem' }}
                  formatter={(v) => [v.toFixed(4), 'Pearson r']}
                />
                <Bar dataKey="value" name="Correlation" radius={[0, 4, 4, 0]}>
                  {corrChartData.map((d, i) => (
                    <Cell key={i} fill={d.value >= 0 ? C.emerald : C.crimson} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div style={{ display: 'flex', gap: '0.75rem', fontSize: '0.7rem', color: 'var(--text-muted)' }}>
            <span><span style={{ color: C.emerald }}>■</span> Positive correlation</span>
            <span><span style={{ color: C.crimson }}>■</span> Negative correlation</span>
          </div>
        </Section>
      )} */}

      {/* ── Sensor Distribution Stats ────────────────────────────────────
      {Object.keys(distStats).length > 0 && (
        <Section title="Sensor Feature Distribution Statistics" icon={BarChart2} color={C.cyan} defaultOpen={false}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{
              width: '100%', borderCollapse: 'collapse', fontSize: '0.75rem',
              color: 'var(--text-secondary)',
            }}>
              <thead>
                <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
                  {['Feature','Mean','Median','Std','Skewness','Kurtosis','Min','Max','Q25','Q75'].map(h => (
                    <th key={h} style={{ padding: '0.4rem 0.6rem', textAlign: 'right',
                      textTransform: 'uppercase', fontSize: '0.65rem', color: 'var(--text-muted)',
                      whiteSpace: 'nowrap' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Object.entries(distStats).map(([feat, s]) => (
                  <tr key={feat} style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                    <td style={{ padding: '0.35rem 0.6rem', fontFamily: 'monospace',
                      fontSize: '0.7rem', color: C.cyan, whiteSpace: 'nowrap' }}>
                      {feat}
                    </td>
                    {[s.mean, s.median, s.std, s.skewness, s.kurtosis,
                      s.min, s.max, s.q25, s.q75].map((v, i) => (
                      <td key={i} style={{ padding: '0.35rem 0.6rem', textAlign: 'right',
                        color: 'var(--text-primary)' }}>
                        {fmt(v, 3)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Section>
      )} */}

      {/* ── Data Quality — Outliers ──────────────────────────────────────
      {outlierChartData.length > 0 && (
        <Section title="Data Quality — Outlier Analysis (IQR Method)" icon={AlertTriangle} color={C.gold} defaultOpen={false}>
          <div style={{ height: 180, marginBottom: '0.75rem' }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={outlierChartData} margin={{ top: 4, right: 10, left: -10, bottom: 40 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="feature" stroke="var(--text-secondary)" tick={{ fontSize: 10 }}
                  angle={-30} textAnchor="end" interval={0} />
                <YAxis stroke="var(--text-secondary)" tick={{ fontSize: 10 }} unit="%" />
                <Tooltip
                  contentStyle={{ background: '#1e293b', border: '1px solid rgba(255,255,255,0.1)', fontSize: '0.78rem' }}
                  formatter={(v) => [`${v}%`, 'Outlier %']}
                />
                <Bar dataKey="pct" name="Outlier %" radius={[4, 4, 0, 0]}>
                  {outlierChartData.map((d, i) => (
                    <Cell key={i} fill={d.pct > 10 ? C.crimson : d.pct > 5 ? C.gold : C.emerald} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.74rem',
              color: 'var(--text-secondary)' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
                  {['Feature','Outlier Count','Outlier %','Lower Bound','Upper Bound','Min','Max'].map(h => (
                    <th key={h} style={{ padding: '0.35rem 0.6rem', textAlign: 'right',
                      textTransform: 'uppercase', fontSize: '0.65rem', color: 'var(--text-muted)' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Object.entries(outliers).map(([feat, info]) => {
                  const bad = (info.percentage || 0) > 10;
                  return (
                    <tr key={feat} style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                      <td style={{ padding: '0.35rem 0.6rem', fontFamily: 'monospace',
                        fontSize: '0.7rem', color: C.cyan, whiteSpace: 'nowrap' }}>{feat}</td>
                      <td style={{ textAlign: 'right', padding: '0.35rem 0.6rem',
                        color: bad ? C.crimson : 'var(--text-primary)' }}>
                        {(info.count || 0).toLocaleString()}
                      </td>
                      <td style={{ textAlign: 'right', padding: '0.35rem 0.6rem',
                        color: bad ? C.crimson : 'var(--text-primary)' }}>
                        {pct(info.percentage)}
                      </td>
                      <td style={{ textAlign: 'right', padding: '0.35rem 0.6rem' }}>{fmt(info.lower_bound)}</td>
                      <td style={{ textAlign: 'right', padding: '0.35rem 0.6rem' }}>{fmt(info.upper_bound)}</td>
                      <td style={{ textAlign: 'right', padding: '0.35rem 0.6rem' }}>{fmt(info.min)}</td>
                      <td style={{ textAlign: 'right', padding: '0.35rem 0.6rem' }}>{fmt(info.max)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Section>
      )} */}

      {/* ── Categorical Distribution ─────────────────────────────────────
      {Object.keys(catStats).length > 0 && (
        <Section title="Categorical Feature Distributions" icon={Layers} color={C.purple} defaultOpen={false}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '0.75rem' }}>
            {Object.entries(catStats).map(([feat, info]) => {
              const total = Object.values(info.value_counts || {}).reduce((a, b) => a + b, 0) || 1;
              return (
                <div key={feat} style={{ background: 'rgba(255,255,255,0.03)',
                  borderRadius: '8px', padding: '0.75rem', border: '1px solid rgba(255,255,255,0.07)' }}>
                  <div style={{ fontSize: '0.72rem', color: C.purple, fontWeight: 600,
                    marginBottom: '0.5rem', fontFamily: 'monospace' }}>{feat}</div>
                  {Object.entries(info.value_counts || {}).map(([val, cnt]) => (
                    <div key={val} style={{ marginBottom: '0.3rem' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between',
                        fontSize: '0.7rem', marginBottom: '0.15rem' }}>
                        <span style={{ color: 'var(--text-secondary)', maxWidth: '70%',
                          overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{val}</span>
                        <span style={{ color: 'var(--text-primary)' }}>{pct(cnt / total * 100)}</span>
                      </div>
                      <div style={{ height: '4px', background: 'rgba(255,255,255,0.08)', borderRadius: '2px' }}>
                        <div style={{
                          height: '100%', borderRadius: '2px',
                          width: `${(cnt / total * 100).toFixed(1)}%`,
                          background: C.purple, transition: 'width 0.4s ease',
                        }} />
                      </div>
                    </div>
                  ))}
                </div>
              );
            })}
          </div>
        </Section>
      )} */}

      {/* ── Feature Engineering Suggestions ────────────────────────────
      {featSugg.length > 0 && (
        <Section title={`Feature Engineering Suggestions (${featSugg.length})`} icon={Star} color={C.indigo} defaultOpen={false}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))', gap: '0.6rem' }}>
            {featSugg.map((s, i) => {
              const col = suggColor[s.type] || C.cyan;
              return (
                <div key={i} style={{ background: `${col}10`, border: `1px solid ${col}28`,
                  borderRadius: '7px', padding: '0.65rem 0.8rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem',
                    marginBottom: '0.25rem' }}>
                    <span style={{ fontSize: '0.6rem', background: col, color: '#0f172a',
                      borderRadius: '4px', padding: '0.1rem 0.35rem', fontWeight: 700,
                      textTransform: 'uppercase' }}>{s.type}</span>
                    <span style={{ fontFamily: 'monospace', fontSize: '0.72rem', color: col }}>{s.name}</span>
                  </div>
                  <p style={{ fontSize: '0.7rem', color: 'var(--text-secondary)', margin: 0 }}>{s.description}</p>
                </div>
              );
            })}
          </div>
        </Section>
      )} */}

      {/* ── EDA Plots Gallery ─────────────────────────────────────────── */}
      {images.length > 0 && (
        <Section title={`EDA Plot Gallery (${images.length} images)`} icon={Image} color={C.cyan} defaultOpen={true}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: '0.75rem' }}>
            {images.map(img => (
              <ImageViewer key={img} filename={img} />
            ))}
          </div>
          {/* <p style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '0.5rem' }}>
            💡 Click any image to expand/collapse. Images served from{' '}
            <code>backend/outputs/eda_swarming/images/</code>
          </p> */}
        </Section>
      )}

      {/* ── Feature Analysis Report ──────────────────────────────────────
      {report && (
        <Section title="Feature Analysis Summary Report" icon={FileText} color={C.emerald} defaultOpen={false}>
          <pre style={{
            whiteSpace: 'pre-wrap', wordBreak: 'break-word',
            fontFamily: 'monospace', fontSize: '0.75rem',
            color: 'var(--text-secondary)', lineHeight: 1.7,
            background: 'rgba(0,0,0,0.25)', borderRadius: '6px',
            padding: '0.75rem', maxHeight: '400px', overflowY: 'auto',
          }}>
            {report}
          </pre>
        </Section>
      )} */}

    </div>
  );
}