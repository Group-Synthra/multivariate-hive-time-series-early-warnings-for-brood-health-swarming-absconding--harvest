import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from 'recharts';
import { EmptyState } from '../../../components/common/EmptyState';
import { SENSOR_META } from '../utils/broodHealth';

const STATUS_COLORS = { Healthy: '#0f766e', Unhealthy: '#dc2626' };
const SENSOR_COLORS = {
  temperature_c: '#dc2626',
  humidity_pct: '#0284c7',
  co2_ppm: '#7c3aed',
  weight_kg: '#ca8a04',
};

function ChartShell({ data, children, height = 320, message = 'Chart data are not available.' }) {
  if (!data?.length) return <EmptyState message={message} />;
  return <div className="brood-chart" style={{ height }}>{children}</div>;
}

export function TargetBalanceChart({ data }) {
  return (
    <ChartShell data={data}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 12, right: 16, left: 10, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="label" />
          <YAxis tickFormatter={(value) => Number(value).toLocaleString()} />
          <Tooltip formatter={(value) => Number(value).toLocaleString()} />
          <Bar dataKey="count" name="Records" radius={[7, 7, 0, 0]}>
            {data?.map((row) => <Cell key={row.label} fill={STATUS_COLORS[row.label] || '#2563eb'} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </ChartShell>
  );
}

export function SensorDistributionChart({ data, sensor }) {
  const meta = SENSOR_META[sensor] || { label: sensor, unit: '' };
  return (
    <ChartShell data={data} message="Sensor-distribution data are unavailable.">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 10, right: 20, left: 5, bottom: 10 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="bin_mid" tickFormatter={(v) => Number(v).toFixed(1)} />
          <YAxis unit="%" />
          <Tooltip
            labelFormatter={(v) => `${meta.label}: ${Number(v).toFixed(2)} ${meta.unit}`}
            formatter={(v, name) => [`${Number(v).toFixed(2)}%`, name]}
          />
          <Legend />
          <Area type="monotone" dataKey="healthy_percentage" name="Healthy" stroke="#0f766e" fill="#0f766e" fillOpacity={0.20} />
          <Area type="monotone" dataKey="unhealthy_percentage" name="Unhealthy" stroke="#dc2626" fill="#dc2626" fillOpacity={0.16} />
        </AreaChart>
      </ResponsiveContainer>
    </ChartShell>
  );
}

export function TemporalHealthyRateChart({ data, xKey, labelFormatter }) {
  return (
    <ChartShell data={data}>
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={data} margin={{ top: 12, right: 20, left: 6, bottom: 10 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey={xKey} tickFormatter={labelFormatter} minTickGap={18} />
          <YAxis yAxisId="rate" domain={[0, 100]} unit="%" />
          <YAxis yAxisId="count" orientation="right" />
          <Tooltip labelFormatter={labelFormatter} />
          <Legend />
          <Bar yAxisId="count" dataKey="unhealthy_count" name="Unhealthy records" fill="#fecaca" opacity={0.72} />
          <Line yAxisId="rate" type="monotone" dataKey="healthy_rate" name="Healthy rate (%)" stroke="#0f766e" strokeWidth={3} dot={false} />
        </ComposedChart>
      </ResponsiveContainer>
    </ChartShell>
  );
}

export function HiveHealthyRateChart({ data }) {
  const chartData = [...(data || [])].sort((a, b) => Number(a.healthy_rate) - Number(b.healthy_rate));
  return (
    <ChartShell data={chartData} height={Math.max(420, chartData.length * 23)}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={chartData} layout="vertical" margin={{ top: 8, right: 18, left: 62, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" horizontal={false} />
          <XAxis type="number" domain={[0, 100]} unit="%" />
          <YAxis type="category" dataKey="hive_id" width={78} interval={0} />
          <Tooltip formatter={(value) => `${Number(value).toFixed(2)}%`} />
          <Bar dataKey="healthy_rate" name="Healthy rate" fill="#2563eb" radius={[0, 5, 5, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </ChartShell>
  );
}

export function SensorEffectChart({ data }) {
  const chartData = (data || []).map((row) => ({ ...row, effect: Math.abs(Number(row.cohens_d || 0)) }));
  return (
    <ChartShell data={chartData}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={chartData} margin={{ top: 10, right: 12, left: 5, bottom: 15 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="label" />
          <YAxis />
          <Tooltip formatter={(v, name, item) => [Number(v).toFixed(3), `${name} (${item.payload.mean_difference >= 0 ? 'higher when healthy' : 'lower when healthy'})`]} />
          <Bar dataKey="effect" name="Absolute Cohen's d" fill="#7c3aed" radius={[6, 6, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </ChartShell>
  );
}

export function ConditionLevelChart({ data }) {
  const colors = { Critical: '#b91c1c', Poor: '#ea580c', Good: '#2563eb', Excellent: '#0f766e' };
  return (
    <ChartShell data={data}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 10, right: 15, left: 5, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="level" />
          <YAxis unit="%" />
          <Tooltip formatter={(v) => `${Number(v).toFixed(2)}%`} />
          <Bar dataKey="percentage" name="Records" radius={[6, 6, 0, 0]}>
            {data?.map((row) => <Cell key={row.level} fill={colors[row.level] || '#64748b'} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </ChartShell>
  );
}

export function PrecursorChart({ data, sensor }) {
  const filtered = (data || []).filter((row) => row.sensor === sensor);
  const meta = SENSOR_META[sensor] || { label: sensor, unit: '' };
  return (
    <ChartShell data={filtered}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={filtered} margin={{ top: 10, right: 15, left: 10, bottom: 25 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="window" angle={-12} textAnchor="end" height={55} />
          <YAxis unit={meta.unit} />
          <Tooltip formatter={(v) => [`${Number(v).toFixed(3)} ${meta.unit}`, 'Change from baseline']} />
          <ReferenceLine y={0} stroke="#475569" />
          <Bar dataKey="delta_from_baseline" name="Change from 48–96 h baseline" fill="#d97706" radius={[5, 5, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </ChartShell>
  );
}

export function RelationshipScatterChart({ data, xSensor, ySensor }) {
  const xMeta = SENSOR_META[xSensor];
  const yMeta = SENSOR_META[ySensor];
  const healthy = (data || []).filter((row) => Number(row.brood_health_healthy_1) === 1);
  const unhealthy = (data || []).filter((row) => Number(row.brood_health_healthy_1) === 0);
  return (
    <ChartShell data={data} height={360}>
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 10, right: 18, left: 5, bottom: 15 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis type="number" dataKey={xSensor} name={xMeta?.label} unit={xMeta?.unit} />
          <YAxis type="number" dataKey={ySensor} name={yMeta?.label} unit={yMeta?.unit} />
          <ZAxis range={[16, 16]} />
          <Tooltip cursor={{ strokeDasharray: '3 3' }} />
          <Legend />
          <Scatter name="Healthy" data={healthy} fill="#0f766e" fillOpacity={0.45} />
          <Scatter name="Unhealthy" data={unhealthy} fill="#dc2626" fillOpacity={0.50} />
        </ScatterChart>
      </ResponsiveContainer>
    </ChartShell>
  );
}

export function ModelComparisonChart({ data }) {
  const chartData = (data || []).filter((row) => row.status === 'ok').map((row) => ({
    model: row.model,
    transitionAccuracy: Number(row.test?.transition_level_accuracy || 0) * 100,
    overallAccuracy: Number(row.test?.health_level_accuracy || 0) * 100,
    criticalRecall: Number(row.test?.critical_recall || 0) * 100,
  }));
  return (
    <ChartShell data={chartData} height={370} message="Train the models to display unseen-hive performance.">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={chartData} margin={{ top: 10, right: 15, left: 3, bottom: 75 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="model" angle={-24} textAnchor="end" interval={0} height={90} />
          <YAxis domain={[0, 100]} unit="%" />
          <Tooltip formatter={(v) => `${Number(v).toFixed(2)}%`} />
          <Legend />
          <Bar dataKey="transitionAccuracy" name="Transition accuracy" fill="#2563eb" />
          <Bar dataKey="overallAccuracy" name="Overall level accuracy" fill="#0f766e" />
          <Bar dataKey="criticalRecall" name="Critical recall" fill="#d97706" />
        </BarChart>
      </ResponsiveContainer>
    </ChartShell>
  );
}

export function ModelErrorComparisonChart({ data }) {
  const chartData = (data || []).filter((row) => row.status === 'ok').map((row) => ({
    model: row.model,
    mae: Number(row.test?.test_mae || 0),
    transitionMae: Number(row.test?.transition_mae || 0),
    cvMae: Number(row.test?.cv_mae_mean || 0),
  }));
  return (
    <ChartShell data={chartData} height={370} message="Train the models to display score errors.">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={chartData} margin={{ top: 10, right: 15, left: 3, bottom: 75 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="model" angle={-24} textAnchor="end" interval={0} height={90} />
          <YAxis unit=" pts" />
          <Tooltip formatter={(v) => `${Number(v).toFixed(3)} points`} />
          <Legend />
          <Bar dataKey="mae" name="Overall MAE" fill="#2563eb" />
          <Bar dataKey="transitionMae" name="Transition MAE" fill="#dc2626" />
          <Bar dataKey="cvMae" name="Group-CV MAE" fill="#7c3aed" />
        </BarChart>
      </ResponsiveContainer>
    </ChartShell>
  );
}

export function PersistenceComparisonChart({ model, persistence }) {
  const data = [
    {
      name: 'Overall level accuracy',
      model: Number(model?.health_level_accuracy || 0) * 100,
      persistence: Number(persistence?.health_level_accuracy || 0) * 100,
    },
    {
      name: 'Transition accuracy',
      model: Number(model?.transition_level_accuracy || 0) * 100,
      persistence: Number(persistence?.transition_level_accuracy || 0) * 100,
    },
    {
      name: 'Critical recall',
      model: Number(model?.critical_recall || 0) * 100,
      persistence: Number(persistence?.critical_recall || 0) * 100,
    },
  ];
  return (
    <ChartShell data={data} height={320}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ top: 10, right: 20, left: 105, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" horizontal={false} />
          <XAxis type="number" domain={[0, 100]} unit="%" />
          <YAxis type="category" dataKey="name" width={125} />
          <Tooltip formatter={(v) => `${Number(v).toFixed(2)}%`} />
          <Legend />
          <Bar dataKey="model" name="Selected model" fill="#2563eb" />
          <Bar dataKey="persistence" name="Current-score persistence" fill="#94a3b8" />
        </BarChart>
      </ResponsiveContainer>
    </ChartShell>
  );
}

export function ActualPredictedScoreChart({ data }) {
  return (
    <ChartShell data={data} height={320} message="Prediction samples are not available.">
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 12, right: 20, left: 8, bottom: 15 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis type="number" dataKey="actual" name="Actual" domain={[1, 100]} unit="/100" />
          <YAxis type="number" dataKey="predicted" name="Predicted" domain={[1, 100]} unit="/100" />
          <ZAxis range={[20, 20]} />
          <Tooltip cursor={{ strokeDasharray: '3 3' }} formatter={(v) => Number(v).toFixed(2)} />
          <ReferenceLine segment={[{ x: 1, y: 1 }, { x: 100, y: 100 }]} stroke="#64748b" strokeDasharray="4 4" />
          <Scatter name="Test predictions" data={data} fill="#2563eb" fillOpacity={0.48} />
        </ScatterChart>
      </ResponsiveContainer>
    </ChartShell>
  );
}

export function CurveChart({ data, kind = 'roc' }) {
  const isRoc = kind === 'roc';
  return (
    <ChartShell data={data} height={300}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 10, right: 15, left: 5, bottom: 10 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis type="number" dataKey={isRoc ? 'false_positive_rate' : 'recall'} domain={[0, 1]} />
          <YAxis type="number" dataKey={isRoc ? 'true_positive_rate' : 'precision'} domain={[0, 1]} />
          <Tooltip formatter={(v) => Number(v).toFixed(4)} />
          {isRoc && <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#94a3b8" strokeDasharray="4 4" />}
          <Line type="monotone" dataKey={isRoc ? 'true_positive_rate' : 'precision'} stroke={isRoc ? '#2563eb' : '#0f766e'} strokeWidth={3} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </ChartShell>
  );
}

export function FeatureImportanceChart({ data }) {
  const chartData = [...(data || [])].slice(0, 15).reverse();
  return (
    <ChartShell data={chartData} height={420}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={chartData} layout="vertical" margin={{ top: 8, right: 18, left: 135, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" horizontal={false} />
          <XAxis type="number" unit="%" />
          <YAxis type="category" dataKey="feature" width={145} interval={0} />
          <Tooltip formatter={(v) => `${Number(v).toFixed(2)}%`} />
          <Bar dataKey="importance_percentage" name="Relative importance" fill="#7c3aed" radius={[0, 5, 5, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </ChartShell>
  );
}

export function IoTSensorHistoryChart({ data, sensor }) {
  const meta = SENSOR_META[sensor] || { label: sensor, unit: '' };
  return (
    <ChartShell data={data} height={330} message="No hourly IoT history is available.">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 10, right: 20, left: 8, bottom: 12 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="timestamp" tickFormatter={(v) => new Date(v).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit' })} minTickGap={40} />
          <YAxis unit={meta.unit} />
          <Tooltip labelFormatter={(v) => new Date(v).toLocaleString()} />
          <Line type="monotone" dataKey={sensor} name={meta.label} stroke={SENSOR_COLORS[sensor] || '#2563eb'} strokeWidth={2.5} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </ChartShell>
  );
}

export function HealthHistoryChart({ data }) {
  return (
    <ChartShell data={data} height={350} message="No prediction history is available.">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 10, right: 20, left: 5, bottom: 12 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="timestamp" tickFormatter={(v) => new Date(v).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit' })} minTickGap={40} />
          <YAxis domain={[0, 100]} unit="%" />
          <Tooltip labelFormatter={(v) => new Date(v).toLocaleString()} formatter={(v) => `${Number(v).toFixed(2)}`} />
          <Legend />
          <ReferenceLine y={80} stroke="#0f766e" strokeDasharray="4 4" />
          <ReferenceLine y={60} stroke="#2563eb" strokeDasharray="4 4" />
          <ReferenceLine y={40} stroke="#dc2626" strokeDasharray="4 4" />
          <Line type="monotone" dataKey="forecast_score" name="Predicted future minimum score" stroke="#2563eb" strokeWidth={3} dot={false} />
          <Line type="monotone" dataKey="condition_score" name="Current condition score" stroke="#d97706" strokeWidth={2.5} dot={false} />
          <Line type="monotone" dataKey="bhsi" name="Stability index" stroke="#7c3aed" strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </ChartShell>
  );
}

export function HealthScoreComparisonChart({ currentScore, predictedScore }) {
  const data = [
    { name: 'Current', score: Number(currentScore || 0) },
    { name: 'Predicted minimum', score: Number(predictedScore || 0) },
  ];
  const colors = ['#d97706', '#2563eb'];
  return (
    <ChartShell data={data} height={280}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ top: 12, right: 20, left: 75, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" horizontal={false} />
          <XAxis type="number" domain={[0, 100]} />
          <YAxis type="category" dataKey="name" width={90} />
          <Tooltip formatter={(value) => `${Number(value).toFixed(1)} / 100`} />
          <ReferenceLine x={40} stroke="#dc2626" strokeDasharray="4 4" />
          <ReferenceLine x={60} stroke="#d97706" strokeDasharray="4 4" />
          <ReferenceLine x={80} stroke="#0f766e" strokeDasharray="4 4" />
          <Bar dataKey="score" radius={[0, 8, 8, 0]}>
            {data.map((item, index) => <Cell key={item.name} fill={colors[index]} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </ChartShell>
  );
}

export function LiveEarlyWarningTimeline({ data }) {
  return (
    <ChartShell data={data} height={360} message="No live timeline is available.">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 10, right: 24, left: 5, bottom: 12 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="timestamp" tickFormatter={(v) => new Date(v).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })} minTickGap={45} />
          <YAxis yAxisId="score" domain={[0, 100]} />
          <YAxis yAxisId="rod" orientation="right" />
          <Tooltip labelFormatter={(v) => new Date(v).toLocaleString()} formatter={(v) => Number(v).toFixed(2)} />
          <Legend />
          <ReferenceLine yAxisId="score" y={40} stroke="#dc2626" strokeDasharray="4 4" />
          <ReferenceLine yAxisId="score" y={60} stroke="#d97706" strokeDasharray="4 4" />
          <ReferenceLine yAxisId="score" y={80} stroke="#0f766e" strokeDasharray="4 4" />
          <Line yAxisId="score" type="monotone" dataKey="condition_score" name="Current health" stroke="#d97706" strokeWidth={2.5} dot={false} />
          <Line yAxisId="score" type="monotone" dataKey="forecast_score" name="Predicted minimum health" stroke="#2563eb" strokeWidth={3} dot={false} />
          <Line yAxisId="score" type="monotone" dataKey="bhsi" name="BHSI" stroke="#7c3aed" strokeWidth={2} dot={false} />
          <Line yAxisId="rod" type="monotone" dataKey="rod_points_per_hour" name="RoD" stroke="#dc2626" strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </ChartShell>
  );
}
