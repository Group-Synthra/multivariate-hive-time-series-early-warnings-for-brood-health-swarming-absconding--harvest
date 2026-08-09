import { useId } from 'react';
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Legend,
  LabelList,
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
import { SENSOR_META, clampScore, healthLevelFromScore } from '../utils/broodHealth';

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
  const chartData = (data || []).map((row) => ({
    ...row,
    percentage: Number(row.percentage || 0),
  }));

  return (
    <ChartShell data={chartData}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={chartData} margin={{ top: 28, right: 16, left: 10, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="label" />
          <YAxis tickFormatter={(value) => Number(value).toLocaleString()} />
          <Tooltip
            formatter={(value, name, item) => [
              `${Number(value).toLocaleString()} records (${Number(item?.payload?.percentage || 0).toFixed(2)}%)`,
              name,
            ]}
          />
          <Bar dataKey="count" name="Records" radius={[7, 7, 0, 0]}>
            <LabelList
              dataKey="percentage"
              position="top"
              formatter={(value) => `${Number(value).toFixed(1)}%`}
              fill="#334155"
              fontSize={11}
              fontWeight={800}
            />
            {chartData.map((row) => <Cell key={row.label} fill={STATUS_COLORS[row.label] || '#2563eb'} />)}
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
  const chartData = (data || []).map((row) => ({
    ...row,
    healthy_rate: Number(row.healthy_rate || 0),
    unhealthy_rate: 100 - Number(row.healthy_rate || 0),
  }));

  return (
    <ChartShell data={chartData}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData} margin={{ top: 18, right: 20, left: 6, bottom: 10 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey={xKey} tickFormatter={labelFormatter} minTickGap={18} />
          <YAxis domain={[0, 100]} unit="%" />
          <Tooltip
            labelFormatter={labelFormatter}
            formatter={(value, name) => [`${Number(value).toFixed(2)}%`, name]}
          />
          <Legend />
          <Line type="monotone" dataKey="healthy_rate" name="Healthy records" stroke="#0f766e" strokeWidth={3} dot={false} />
          <Line type="monotone" dataKey="unhealthy_rate" name="Unhealthy records" stroke="#dc2626" strokeWidth={2.5} dot={false} />
        </LineChart>
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
          <Bar dataKey="healthy_rate" name="Healthy rate" fill="#2563eb" radius={[0, 5, 5, 0]}>
            <LabelList
              dataKey="healthy_rate"
              position="right"
              formatter={(value) => `${Number(value).toFixed(1)}%`}
              fill="#334155"
              fontSize={10}
              fontWeight={750}
            />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </ChartShell>
  );
}

export function SensorEffectChart({ data }) {
  const chartData = (data || []).map((row) => {
    const healthy = Number(row.healthy_mean || 0);
    const unhealthy = Number(row.unhealthy_mean || 0);
    const percentDifference = Math.abs(unhealthy) > 1e-9
      ? ((healthy - unhealthy) / Math.abs(unhealthy)) * 100
      : 0;
    return {
      ...row,
      effect: Math.abs(Number(row.cohens_d || 0)),
      percentDifference,
    };
  });

  return (
    <ChartShell data={chartData}>
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={chartData} margin={{ top: 24, right: 20, left: 5, bottom: 15 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="label" />
          <YAxis yAxisId="effect" />
          <YAxis yAxisId="percent" orientation="right" unit="%" />
          <Tooltip
            formatter={(value, name) => [
              name === 'Healthy vs unhealthy mean difference'
                ? `${Number(value).toFixed(1)}%`
                : Number(value).toFixed(3),
              name,
            ]}
          />
          <Legend />
          <Bar yAxisId="effect" dataKey="effect" name="Absolute Cohen's d" fill="#7c3aed" radius={[6, 6, 0, 0]} />
          <Line
            yAxisId="percent"
            type="monotone"
            dataKey="percentDifference"
            name="Healthy vs unhealthy mean difference"
            stroke="#0f766e"
            strokeWidth={2.5}
          >
            <LabelList
              dataKey="percentDifference"
              position="top"
              formatter={(v) => `${Number(v).toFixed(1)}%`}
              fontSize={9}
              fill="#0f766e"
            />
          </Line>
        </ComposedChart>
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
            <LabelList
              dataKey="percentage"
              position="top"
              formatter={(value) => `${Number(value).toFixed(1)}%`}
              fill="#334155"
              fontSize={11}
              fontWeight={800}
            />
            {data?.map((row) => <Cell key={row.level} fill={colors[row.level] || '#64748b'} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </ChartShell>
  );
}

export function PrecursorChart({ data, sensor }) {
  const filtered = (data || [])
    .filter((row) => row.sensor === sensor)
    .map((row) => ({
      ...row,
      change_percentage: Math.abs(Number(row.baseline_mean || 0)) > 1e-9
        ? Number(row.delta_from_baseline || 0) / Math.abs(Number(row.baseline_mean)) * 100
        : 0,
    }));
  const meta = SENSOR_META[sensor] || { label: sensor, unit: '' };
  return (
    <ChartShell data={filtered}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={filtered} margin={{ top: 24, right: 15, left: 10, bottom: 25 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="window" angle={-12} textAnchor="end" height={55} />
          <YAxis unit={meta.unit} />
          <Tooltip
            formatter={(v, name, item) => [
              `${Number(v).toFixed(3)} ${meta.unit} (${Number(item?.payload?.change_percentage || 0).toFixed(2)}%)`,
              'Change from baseline',
            ]}
          />
          <ReferenceLine y={0} stroke="#475569" />
          <Bar dataKey="delta_from_baseline" name="Change from earlier baseline" fill="#d97706" radius={[5, 5, 0, 0]}>
            <LabelList
              dataKey="change_percentage"
              position="top"
              formatter={(v) => `${Number(v).toFixed(1)}%`}
              fontSize={9}
              fill="#334155"
            />
          </Bar>
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
  const chartData = (data || [])
    .filter((row) => row.status === 'ok')
    .map((row) => ({
      model: row.model,
      exactAccuracy: Number(row.test?.exact_horizon?.health_level_accuracy || 0) * 100,
      transitionAccuracy: Number(row.test?.transition?.health_level_accuracy || 0) * 100,
      deteriorationRecall: Number(row.test?.deterioration?.recall || 0) * 100,
    }));
  return (
    <ChartShell data={chartData} height={380} message="Train the models to display unseen-hive performance.">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={chartData} margin={{ top: 10, right: 15, left: 3, bottom: 78 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="model" angle={-24} textAnchor="end" interval={0} height={92} />
          <YAxis domain={[0, 100]} unit="%" />
          <Tooltip formatter={(value) => `${Number(value).toFixed(2)}%`} />
          <Legend />
          <Bar dataKey="exactAccuracy" name="Accuracy ↑" fill="#2563eb">
            <LabelList dataKey="exactAccuracy" position="top" formatter={(v) => `${Number(v).toFixed(1)}%`} fontSize={9} fill="#334155" />
          </Bar>
          <Bar dataKey="transitionAccuracy" name="Transition Accuracy ↑" fill="#d97706">
            <LabelList dataKey="transitionAccuracy" position="top" formatter={(v) => `${Number(v).toFixed(1)}%`} fontSize={9} fill="#334155" />
          </Bar>
          <Bar dataKey="deteriorationRecall" name="Deterioration Recall ↑" fill="#0f766e">
            <LabelList dataKey="deteriorationRecall" position="top" formatter={(v) => `${Number(v).toFixed(1)}%`} fontSize={9} fill="#334155" />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </ChartShell>
  );
}


export function ModelErrorComparisonChart({ data }) {
  const chartData = (data || [])
    .filter((row) => row.status === 'ok')
    .map((row) => ({
      model: row.model,
      exactMae: Number(row.test?.exact_horizon?.mae || 0),
      exactRmse: Number(row.test?.exact_horizon?.rmse || 0),
      transitionMae: Number(row.test?.transition?.mae || 0),
      groupCvMae: Number(row.test?.cv_mae_mean || 0),
    }));

  return (
    <ChartShell data={chartData} height={380} message="Train the models to display score errors.">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={chartData} margin={{ top: 30, right: 15, left: 3, bottom: 78 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="model" angle={-24} textAnchor="end" interval={0} height={92} />
          <YAxis tickFormatter={(value) => `${Number(value).toFixed(0)}%`} />
          <Tooltip
            formatter={(value, name) => [
              `${Number(value).toFixed(2)} points (${Number(value).toFixed(2)}% of the 100-point scale)`,
              name,
            ]}
          />
          <Legend />
          <Bar dataKey="exactMae" name="MAE ↓" fill="#2563eb">
            <LabelList dataKey="exactMae" position="top" formatter={(v) => `${Number(v).toFixed(2)}%`} fontSize={9} fill="#334155" />
          </Bar>
          <Bar dataKey="exactRmse" name="RMSE ↓" fill="#7c3aed">
            <LabelList dataKey="exactRmse" position="top" formatter={(v) => `${Number(v).toFixed(2)}%`} fontSize={9} fill="#334155" />
          </Bar>
          <Bar dataKey="transitionMae" name="Transition MAE ↓" fill="#dc2626">
            <LabelList dataKey="transitionMae" position="top" formatter={(v) => `${Number(v).toFixed(2)}%`} fontSize={9} fill="#334155" />
          </Bar>
          <Bar dataKey="groupCvMae" name="Cross-validation MAE ↓" fill="#0f766e">
            <LabelList dataKey="groupCvMae" position="top" formatter={(v) => `${Number(v).toFixed(2)}%`} fontSize={9} fill="#334155" />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </ChartShell>
  );
}


export function PersistenceComparisonChart({ model, persistence }) {
  const exact = model?.exact_horizon || {};
  const transition = model?.transition || {};
  const deterioration = model?.deterioration || {};
  const persistenceExact = persistence?.exact_horizon || {};
  const persistenceTransition = persistence?.transition || {};
  const persistenceDeterioration = persistence?.deterioration || {};
  const data = [
    {
      name: 'Accuracy ↑',
      model: Number(exact.health_level_accuracy || 0) * 100,
      persistence: Number(persistenceExact.health_level_accuracy || 0) * 100,
    },
    {
      name: 'Transition Accuracy ↑',
      model: Number(transition.health_level_accuracy || 0) * 100,
      persistence: Number(persistenceTransition.health_level_accuracy || 0) * 100,
    },
    {
      name: 'Deterioration Recall ↑',
      model: Number(deterioration.recall || 0) * 100,
      persistence: Number(persistenceDeterioration.recall || 0) * 100,
    },
  ];
  return (
    <ChartShell data={data} height={320}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ top: 10, right: 20, left: 108, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" horizontal={false} />
          <XAxis type="number" domain={[0, 100]} unit="%" />
          <YAxis type="category" dataKey="name" width={130} />
          <Tooltip formatter={(value) => `${Number(value).toFixed(2)}%`} />
          <Legend />
          <Bar dataKey="model" name="Selected model" fill="#2563eb">
            <LabelList dataKey="model" position="right" formatter={(v) => `${Number(v).toFixed(2)}%`} fontSize={10} fill="#334155" />
          </Bar>
          <Bar dataKey="persistence" name="Repeat current score" fill="#94a3b8">
            <LabelList dataKey="persistence" position="right" formatter={(v) => `${Number(v).toFixed(2)}%`} fontSize={10} fill="#334155" />
          </Bar>
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
          <Tooltip
            cursor={{ strokeDasharray: '3 3' }}
            formatter={(v, name) => [`${Number(v).toFixed(2)} / 100`, name]}
          />
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
          <Bar dataKey="importance_percentage" name="Relative importance" fill="#7c3aed" radius={[0, 5, 5, 0]}>
            <LabelList dataKey="importance_percentage" position="right" formatter={(v) => `${Number(v).toFixed(1)}%`} fontSize={10} fill="#334155" />
          </Bar>
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
          <XAxis dataKey="timestamp" tickFormatter={(value) => new Date(value).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit' })} minTickGap={40} />
          <YAxis domain={[0, 100]} />
          <Tooltip labelFormatter={(value) => new Date(value).toLocaleString()} formatter={(value) => Number(value).toFixed(2)} />
          <Legend />
          <ReferenceLine y={80} stroke="#0f766e" strokeDasharray="4 4" />
          <ReferenceLine y={60} stroke="#d97706" strokeDasharray="4 4" />
          <ReferenceLine y={40} stroke="#dc2626" strokeDasharray="4 4" />
          <Line type="monotone" dataKey="exact_forecast_score" name="Exact +6 h score" stroke="#2563eb" strokeWidth={3} dot={false} />
          <Line type="monotone" dataKey="condition_score" name="Current score" stroke="#d97706" strokeWidth={2.5} dot={false} />
          <Line type="monotone" dataKey="bhsi" name="BHSI" stroke="#7c3aed" strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </ChartShell>
  );
}


export function HealthScoreComparisonChart({
  currentScore,
  exactScore,
  safetyScore,
  forecastHorizonHours = 6,
  currentTimestamp,
  forecastTimestamp,
}) {
  const comparisonId = useId().replaceAll(':', '');
  const trackClipId = `${comparisonId}-track`;
  const shadowId = `${comparisonId}-shadow`;

  const healthColors = {
    Critical: '#dc2626',
    Poor: '#d97706',
    Good: '#2563eb',
    Excellent: '#0f766e',
  };

  const current = clampScore(currentScore);
  const predicted = clampScore(exactScore);
  const safety = Number.isFinite(Number(safetyScore))
    ? clampScore(safetyScore)
    : null;

  const currentLevel = healthLevelFromScore(current);
  const predictedLevel = healthLevelFromScore(predicted);
  const safetyLevel = safety === null ? null : healthLevelFromScore(safety);
  const difference = predicted - current;
  const currentTimeLabel = currentTimestamp
    ? new Date(currentTimestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : 'Now';
  const forecastTimeLabel = forecastTimestamp
    ? new Date(forecastTimestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : `+${forecastHorizonHours} h`;

  const trendColor = difference < -0.05
    ? '#dc2626'
    : difference > 0.05
      ? '#0f766e'
      : '#64748b';

  const trendText = difference < -0.05
    ? 'Predicted decline'
    : difference > 0.05
      ? 'Predicted improvement'
      : 'No meaningful score change';

  const currentColor = healthColors[currentLevel];
  const predictedColor = healthColors[predictedLevel];
  const safetyColor = safetyLevel ? healthColors[safetyLevel] : '#7c3aed';

  const scaleStart = 70;
  const scaleWidth = 620;
  const scoreToX = (score) => scaleStart + (score / 100) * scaleWidth;
  const currentX = scoreToX(current);
  const predictedX = scoreToX(predicted);
  const movement = predictedX - currentX;
  const direction = movement >= 0 ? 1 : -1;
  const hasVisibleMovement = Math.abs(movement) > 10;

  const healthSegments = [
    { label: 'Critical', minimum: 0, maximum: 40, color: healthColors.Critical },
    { label: 'Poor', minimum: 40, maximum: 60, color: healthColors.Poor },
    { label: 'Good', minimum: 60, maximum: 80, color: healthColors.Good },
    { label: 'Excellent', minimum: 80, maximum: 100, color: healthColors.Excellent },
  ];

  const lineStartX = currentX + direction * 19;
  const lineEndX = predictedX - direction * 19;
  const arrowBaseX = lineEndX - direction * 12;
  const arrowPoints = `${lineEndX},84 ${arrowBaseX},77 ${arrowBaseX},91`;

  return (
    <div className="brood-old-style-comparison">
      <div className="brood-comparison-heading">
        <div>
          <h4>Current and Predicted Health Comparison</h4>
          <p>
            {trendText}:{' '}
            <strong style={{ color: trendColor }}>
              {difference > 0 ? '+' : ''}{difference.toFixed(2)} points
            </strong>
          </p>
        </div>
        <span className={`brood-comparison-trend ${difference < -0.05 ? 'decline' : difference > 0.05 ? 'improve' : 'stable'}`}>
          {difference < -0.05 ? '↓' : difference > 0.05 ? '↑' : '→'}
          {currentTimeLabel} → {forecastTimeLabel}
        </span>
      </div>

      <div className="brood-comparison-svg-wrap">
        <svg
          viewBox="0 0 760 310"
          width="100%"
          height="100%"
          preserveAspectRatio="xMidYMid meet"
          role="img"
          aria-label={`Current Brood Health Score ${current.toFixed(2)} and exact ${forecastHorizonHours}-hour score ${predicted.toFixed(2)}`}
        >
          <defs>
            <filter id={shadowId} x="-50%" y="-50%" width="200%" height="200%">
              <feDropShadow dx="0" dy="3" stdDeviation="4" floodColor="#0f172a" floodOpacity="0.20" />
            </filter>
            <clipPath id={trackClipId}>
              <rect x={scaleStart} y="112" width={scaleWidth} height="42" rx="9" />
            </clipPath>
          </defs>

          <rect
            x={scaleStart}
            y="112"
            width={scaleWidth}
            height="42"
            rx="9"
            fill="#e2e8f0"
          />

          {healthSegments.map((segment, index) => {
            const x = scoreToX(segment.minimum);
            const width = ((segment.maximum - segment.minimum) / 100) * scaleWidth;
            return (
              <g key={segment.label}>
                <rect
                  x={x}
                  y="112"
                  width={width}
                  height="42"
                  fill={segment.color}
                  opacity="0.94"
                  clipPath={`url(#${trackClipId})`}
                >
                  <title>{segment.label}: {segment.minimum}–{segment.maximum}</title>
                </rect>
                <text
                  x={x + width / 2}
                  y="138"
                  textAnchor="middle"
                  fill="#ffffff"
                  fontSize="12"
                  fontWeight="800"
                >
                  {segment.label}
                </text>
              </g>
            );
          })}

          {[0, 20, 40, 60, 80, 100].map((value) => {
            const x = scoreToX(value);
            return (
              <g key={value}>
                <line
                  x1={x}
                  y1="106"
                  x2={x}
                  y2="161"
                  stroke={value === 40 || value === 60 || value === 80 ? '#ffffff' : 'rgba(255,255,255,0.65)'}
                  strokeWidth={value === 40 || value === 60 || value === 80 ? 2 : 1}
                />
                <text
                  x={x}
                  y="178"
                  textAnchor="middle"
                  fill="#64748b"
                  fontSize="11"
                  fontWeight="650"
                >
                  {value}
                </text>
              </g>
            );
          })}

          {hasVisibleMovement ? (
            <g>
              <line
                x1={lineStartX}
                y1="84"
                x2={lineEndX}
                y2="84"
                stroke={trendColor}
                strokeWidth="4"
                strokeLinecap="round"
              />
              <polygon points={arrowPoints} fill={trendColor} />
            </g>
          ) : (
            <g>
              <line
                x1={currentX - 18}
                y1="84"
                x2={currentX + 18}
                y2="84"
                stroke={trendColor}
                strokeWidth="4"
                strokeLinecap="round"
              />
              <circle cx={currentX} cy="84" r="5" fill={trendColor} />
            </g>
          )}

          <line
            x1={currentX}
            y1="56"
            x2={currentX}
            y2="112"
            stroke={currentColor}
            strokeWidth="3"
            strokeDasharray="5 4"
          />

          <g className="brood-comparison-marker" filter={`url(#${shadowId})`} tabIndex="0">
            <circle
              cx={currentX}
              cy="56"
              r="18"
              fill={currentColor}
              stroke="#ffffff"
              strokeWidth="3"
            >
              <title>Current score: {current.toFixed(2)} ({currentLevel})</title>
            </circle>
            <text
              x={currentX}
              y="60"
              textAnchor="middle"
              fill="#ffffff"
              fontSize="10"
              fontWeight="800"
              pointerEvents="none"
            >
              NOW
            </text>
          </g>

          <line
            x1={predictedX}
            y1="154"
            x2={predictedX}
            y2="223"
            stroke={predictedColor}
            strokeWidth="3"
            strokeDasharray="5 4"
          />

          <g className="brood-comparison-marker" filter={`url(#${shadowId})`} tabIndex="0">
            <circle
              cx={predictedX}
              cy="223"
              r="19"
              fill={predictedColor}
              stroke="#ffffff"
              strokeWidth="3"
            >
              <title>Exact +{forecastHorizonHours} h score: {predicted.toFixed(2)} ({predictedLevel})</title>
            </circle>
            <text
              x={predictedX}
              y="227"
              textAnchor="middle"
              fill="#ffffff"
              fontSize="9"
              fontWeight="800"
              pointerEvents="none"
            >
              +{forecastHorizonHours}H
            </text>
          </g>

          <text
            x={currentX}
            y="20"
            textAnchor="middle"
            fill={currentColor}
            fontSize="19"
            fontWeight="850"
          >
            {current.toFixed(2)}
          </text>
          <text
            x={currentX}
            y="39"
            textAnchor="middle"
            fill="#475569"
            fontSize="11"
            fontWeight="650"
          >
            Current · {currentLevel}
          </text>

          <text
            x={predictedX}
            y="266"
            textAnchor="middle"
            fill={predictedColor}
            fontSize="19"
            fontWeight="850"
          >
            {predicted.toFixed(2)}
          </text>
          <text
            x={predictedX}
            y="286"
            textAnchor="middle"
            fill="#475569"
            fontSize="11"
            fontWeight="650"
          >
            {forecastHorizonHours}h Forecast · {predictedLevel}
          </text>
        </svg>
      </div>

      <div className="brood-comparison-footer">
        <span style={{ color: currentColor }}>
          <i style={{ background: currentColor }} />
          Current: {current.toFixed(2)} ({currentLevel})
        </span>
        <b style={{ color: trendColor }}>
          {difference < -0.05 ? '↓' : difference > 0.05 ? '↑' : '→'}
        </b>
        <span style={{ color: predictedColor }}>
          <i style={{ background: predictedColor }} />
          +{forecastHorizonHours} h: {predicted.toFixed(2)} ({predictedLevel})
        </span>
      </div>

      {safety !== null && (
        <div className="brood-safety-minimum-note">
          <span style={{ background: safetyColor }} />
          <div>
            <strong>Safety minimum: {safety.toFixed(2)} ({safetyLevel})</strong>
            <small>
              Lowest predicted point inside the 1–{forecastHorizonHours} hour trajectory.
              The primary future output above remains the exact +{forecastHorizonHours}-hour score.
            </small>
          </div>
        </div>
      )}
    </div>
  );
}


export function LiveEarlyWarningTimeline({ data }) {
  return (
    <ChartShell data={data} height={380} message="No live timeline is available.">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 10, right: 24, left: 5, bottom: 12 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="timestamp" tickFormatter={(value) => new Date(value).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit' })} minTickGap={45} />
          <YAxis yAxisId="score" domain={[0, 100]} />
          <YAxis yAxisId="rod" orientation="right" />
          <Tooltip labelFormatter={(value) => new Date(value).toLocaleString()} formatter={(value) => Number(value).toFixed(2)} />
          <Legend />
          <ReferenceLine yAxisId="score" y={40} stroke="#dc2626" strokeDasharray="4 4" />
          <ReferenceLine yAxisId="score" y={60} stroke="#d97706" strokeDasharray="4 4" />
          <ReferenceLine yAxisId="score" y={80} stroke="#0f766e" strokeDasharray="4 4" />
          <Line yAxisId="score" type="monotone" dataKey="condition_score" name="Current score" stroke="#d97706" strokeWidth={2.5} dot={false} />
          <Line yAxisId="score" type="monotone" dataKey="exact_forecast_score" name="Exact +6 h score" stroke="#2563eb" strokeWidth={3} dot={false} />
          <Line yAxisId="score" type="monotone" dataKey="safety_minimum_score" name="Safety minimum" stroke="#7c3aed" strokeWidth={2.2} dot={false} />
          <Line yAxisId="score" type="monotone" dataKey="forecast_bhsi" name="Forecast BHSI" stroke="#0f766e" strokeWidth={2.4} dot={false} />
          <Line yAxisId="rod" type="monotone" dataKey="forecast_rod_points_per_hour" name="Forecast RoD" stroke="#dc2626" strokeWidth={2.4} dot={false} />
          <Line yAxisId="score" type="monotone" dataKey="bhsi" name="Observed BHSI" stroke="#64748b" strokeWidth={1.4} strokeDasharray="4 4" dot={false} />
          <Line yAxisId="rod" type="monotone" dataKey="rod_points_per_hour" name="Observed RoD" stroke="#94a3b8" strokeWidth={1.4} strokeDasharray="4 4" dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </ChartShell>
  );
}



export function HorizonErrorChart({ data }) {
  return (
    <ChartShell data={data} height={320} message="Per-horizon metrics are unavailable.">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 26, right: 28, left: 5, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="horizon_hours" tickFormatter={(value) => `+${value} h`} />
          <YAxis />
          <Tooltip
            labelFormatter={(value) => `Forecast horizon: +${value} hours`}
            formatter={(value, name) => [
              `${Number(value).toFixed(2)} score points`,
              name,
            ]}
          />
          <Legend />
          <Line type="monotone" dataKey="mae" name="MAE ↓" stroke="#2563eb" strokeWidth={3}>
            <LabelList dataKey="mae" position="top" formatter={(v) => Number(v).toFixed(2)} fontSize={9} fill="#2563eb" />
          </Line>
          <Line type="monotone" dataKey="rmse" name="RMSE ↓" stroke="#dc2626" strokeWidth={2.5}>
            <LabelList dataKey="rmse" position="bottom" formatter={(v) => Number(v).toFixed(2)} fontSize={9} fill="#dc2626" />
          </Line>
        </LineChart>
      </ResponsiveContainer>
    </ChartShell>
  );
}

export function ForecastTrajectoryChart({
  data,
  currentScore,
  exactHorizon = 6,
  anchorTimestamp,
  targetTimestamp,
}) {
  const source = data || [];
  const alreadyContainsCurrent = Number(source?.[0]?.offset_minutes) === 0;
  const chartData = alreadyContainsCurrent
    ? source
    : [
      {
        offset_minutes: 0,
        horizon_hours: 0,
        score: Number(currentScore || 0),
        level: 'Current',
        forecast_timestamp: anchorTimestamp,
        is_native_model_point: true,
        value_kind: 'current_observation',
      },
      ...source.map((row) => ({
        ...row,
        offset_minutes: Number(
          row.offset_minutes ?? Number(row.horizon_hours || 0) * 60,
        ),
      })),
    ];

  const futureRows = chartData.filter((row) => Number(row.offset_minutes) > 0);
  const minimum = futureRows.length
    ? Math.min(...futureRows.map((row) => Number(row.score || 100)))
    : 0;
  const targetMinutes = Number(exactHorizon) * 60;
  const ticks = Array.from(
    { length: Number(exactHorizon) + 1 },
    (_, index) => index * 60,
  );

  const renderPoint = (props) => {
    const { cx, cy, payload } = props;
    const nativePoint = Boolean(payload?.is_native_model_point);
    return (
      <circle
        cx={cx}
        cy={cy}
        r={nativePoint ? 5 : 2.4}
        fill={nativePoint ? '#2563eb' : '#93c5fd'}
        stroke="#ffffff"
        strokeWidth={nativePoint ? 2 : 1}
      >
        <title>
          {payload?.value_kind === 'display_interpolation'
            ? 'Display interpolation between hourly model outputs'
            : payload?.value_kind === 'current_observation'
              ? 'Current rolling condition score'
              : 'Native hourly model output'}
        </title>
      </circle>
    );
  };

  return (
    <ChartShell data={chartData} height={380} message="Forecast trajectory is unavailable.">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData} margin={{ top: 18, right: 28, left: 8, bottom: 16 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            type="number"
            dataKey="offset_minutes"
            domain={[0, targetMinutes]}
            ticks={ticks}
            tickFormatter={(value) => value === 0 ? 'Now' : `+${value / 60} h`}
          />
          <YAxis domain={[0, 100]} />
          <Tooltip
            labelFormatter={(_, payload) => {
              const row = payload?.[0]?.payload;
              const time = row?.forecast_timestamp
                ? new Date(row.forecast_timestamp).toLocaleString()
                : `+${row?.offset_minutes || 0} minutes`;
              const kind = row?.value_kind === 'display_interpolation'
                ? 'interpolated display point'
                : row?.value_kind === 'current_observation'
                  ? 'current score'
                  : 'native hourly model output';
              return `${time} · ${kind}`;
            }}
            formatter={(value) => [`${Number(value).toFixed(2)} / 100`, 'Brood Health Score']}
          />
          <Legend />
          <ReferenceLine y={40} stroke="#dc2626" strokeDasharray="4 4" />
          <ReferenceLine y={60} stroke="#d97706" strokeDasharray="4 4" />
          <ReferenceLine y={80} stroke="#0f766e" strokeDasharray="4 4" />
          <ReferenceLine
            x={targetMinutes}
            stroke="#2563eb"
            strokeDasharray="5 5"
            label={{
              value: targetTimestamp
                ? `Exact target ${new Date(targetTimestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`
                : `Exact +${exactHorizon} h`,
              position: 'insideTopRight',
            }}
          />
          {futureRows.length > 0 && (
            <ReferenceLine
              y={minimum}
              stroke="#7c3aed"
              strokeDasharray="3 3"
              label={{
                value: `Safety minimum ${minimum.toFixed(2)}`,
                position: 'insideBottomRight',
              }}
            />
          )}
          <Line
            type="linear"
            dataKey="score"
            name="Current-to-future Brood Health Score"
            stroke="#2563eb"
            strokeWidth={3.2}
            dot={renderPoint}
            activeDot={{ r: 7 }}
            isAnimationActive
          />
        </LineChart>
      </ResponsiveContainer>
      <p className="chart-footnote">
        Large points are native hourly model outputs. Small points are ten-minute
        display interpolation. The exact +{exactHorizon}-hour score remains a native
        model output; interpolation does not create extra training information.
      </p>
    </ChartShell>
  );
}
