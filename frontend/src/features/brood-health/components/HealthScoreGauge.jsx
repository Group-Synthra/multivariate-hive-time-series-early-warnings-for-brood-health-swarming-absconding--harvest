import { useEffect, useId, useMemo, useRef, useState } from 'react';
import {
  Activity,
  Gauge,
  TrendingDown,
  TrendingUp,
} from 'lucide-react';
import {
  healthClass,
  numberValue,
  stabilityClass,
} from '../utils/broodHealth';

const HEALTH_SEGMENTS = [
  { start: 0, end: 40, label: 'Critical', rangeLabel: '0–<40', color: '#dc2626' },
  { start: 40, end: 60, label: 'Poor', rangeLabel: '40–<60', color: '#d97706' },
  { start: 60, end: 80, label: 'Good', rangeLabel: '60–<80', color: '#2563eb' },
  { start: 80, end: 100, label: 'Excellent', rangeLabel: '80–100', color: '#0f766e' },
];

const STABILITY_SEGMENTS = [
  { start: 0, end: 40, label: 'Low', rangeLabel: '0–<40', color: '#dc2626' },
  { start: 40, end: 70, label: 'Moderate', rangeLabel: '40–<70', color: '#d97706' },
  { start: 70, end: 100, label: 'High', rangeLabel: '70–100', color: '#0f766e' },
];

const HEALTH_COLORS = {
  Critical: '#dc2626',
  Poor: '#d97706',
  Good: '#2563eb',
  Excellent: '#0f766e',
};

const STABILITY_COLORS = {
  Low: '#dc2626',
  Moderate: '#d97706',
  High: '#0f766e',
};

function clamp(value, minimum, maximum) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return minimum;
  return Math.min(maximum, Math.max(minimum, numeric));
}

function healthLevelFromScore(value) {
  const score = clamp(value, 0, 100);
  if (score >= 80) return 'Excellent';
  if (score >= 60) return 'Good';
  if (score >= 40) return 'Poor';
  return 'Critical';
}

function stabilityLevelFromScore(value) {
  const score = clamp(value, 0, 100);
  if (score >= 70) return 'High';
  if (score >= 40) return 'Moderate';
  return 'Low';
}

function pointForScore(cx, cy, radius, score) {
  const angleDegrees = 180 - clamp(score, 0, 100) * 1.8;
  const angleRadians = (angleDegrees * Math.PI) / 180;

  return {
    x: cx + radius * Math.cos(angleRadians),
    y: cy - radius * Math.sin(angleRadians),
  };
}

function arcPath(cx, cy, radius, startScore, endScore) {
  const start = pointForScore(cx, cy, radius, startScore);
  const end = pointForScore(cx, cy, radius, endScore);

  return [
    `M ${start.x.toFixed(3)} ${start.y.toFixed(3)}`,
    `A ${radius} ${radius} 0 0 1 ${end.x.toFixed(3)} ${end.y.toFixed(3)}`,
  ].join(' ');
}

function useAnimatedValue(targetValue, duration = 850) {
  const target = clamp(targetValue, 0, 100);
  const [displayValue, setDisplayValue] = useState(target);
  const currentValueRef = useRef(target);

  useEffect(() => {
    const startValue = currentValueRef.current;
    const endValue = target;

    if (
      typeof window === 'undefined'
      || window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
      || Math.abs(endValue - startValue) < 0.01
    ) {
      currentValueRef.current = endValue;
      setDisplayValue(endValue);
      return undefined;
    }

    const startTime = window.performance.now();
    let frameId = 0;

    const animate = (time) => {
      const progress = Math.min(1, (time - startTime) / duration);
      const eased = 1 - ((1 - progress) ** 3);
      const nextValue = startValue + (endValue - startValue) * eased;

      currentValueRef.current = nextValue;
      setDisplayValue(nextValue);

      if (progress < 1) {
        frameId = window.requestAnimationFrame(animate);
      }
    };

    frameId = window.requestAnimationFrame(animate);
    return () => window.cancelAnimationFrame(frameId);
  }, [target, duration]);

  return displayValue;
}

function SemicircleGauge({
  value,
  segments,
  ticks,
  accent,
  valueLabel,
  secondaryLabel,
  ariaLabel,
}) {
  const animatedValue = useAnimatedValue(value);
  const shadowId = useId().replaceAll(':', '');
  const cx = 160;
  const cy = 158;
  const arcRadius = 104;
  const needleRadius = 83;
  const tailRadius = 17;

  const needleTip = pointForScore(cx, cy, needleRadius, animatedValue);
  const directionX = (needleTip.x - cx) / needleRadius;
  const directionY = (needleTip.y - cy) / needleRadius;
  const needleTail = {
    x: cx - directionX * tailRadius,
    y: cy - directionY * tailRadius,
  };

  const tickElements = useMemo(
    () => ticks.map((tick) => {
      const inner = pointForScore(cx, cy, 118, tick.value);
      const outer = pointForScore(cx, cy, 126, tick.value);
      const text = pointForScore(cx, cy, 141, tick.value);

      return (
        <g key={`${tick.value}-${tick.label}`}>
          <line
            x1={inner.x}
            y1={inner.y}
            x2={outer.x}
            y2={outer.y}
            className="brood-gauge-tick"
          />
          <text
            x={text.x}
            y={text.y + 4}
            textAnchor="middle"
            className="brood-gauge-tick-label"
          >
            {tick.label}
          </text>
        </g>
      );
    }),
    [ticks],
  );

  return (
    <svg
      className="brood-svg-gauge"
      viewBox="0 0 320 215"
      role="meter"
      aria-label={ariaLabel}
      aria-valuemin="0"
      aria-valuemax="100"
      aria-valuenow={clamp(value, 0, 100)}
    >
      <defs>
        <filter id={shadowId} x="-50%" y="-50%" width="200%" height="200%">
          <feDropShadow dx="0" dy="2" stdDeviation="2.5" floodColor="#0f172a" floodOpacity="0.25" />
        </filter>
      </defs>

      <path
        d={arcPath(cx, cy, arcRadius, 0, 100)}
        fill="none"
        stroke="#e2e8f0"
        strokeWidth="24"
        strokeLinecap="round"
      />

      {segments.map((segment) => (
        <path
          key={`${segment.start}-${segment.end}`}
          d={arcPath(cx, cy, arcRadius, segment.start, segment.end)}
          fill="none"
          stroke={segment.color}
          strokeWidth="20"
          strokeLinecap="butt"
        >
          <title>{segment.label}: {segment.start}–{segment.end}</title>
        </path>
      ))}

      {tickElements}

      <line
        x1={needleTail.x}
        y1={needleTail.y}
        x2={needleTip.x}
        y2={needleTip.y}
        stroke="#14213d"
        strokeWidth="5"
        strokeLinecap="round"
        filter={`url(#${shadowId})`}
      />

      <circle
        cx={needleTip.x}
        cy={needleTip.y}
        r="4.5"
        fill={accent}
        stroke="#ffffff"
        strokeWidth="2"
      />

      <circle cx={cx} cy={cy} r="14" fill="#ffffff" stroke="#94a3b8" strokeWidth="2" />
      <circle cx={cx} cy={cy} r="8" fill="#14213d" />
      <circle cx={cx} cy={cy} r="3" fill={accent} />

      <text x={cx} y="111" textAnchor="middle" className="brood-gauge-value" fill={accent}>
        {numberValue(animatedValue, 1)}
      </text>
      <text x={cx} y="133" textAnchor="middle" className="brood-gauge-unit">
        {valueLabel}
      </text>
      <text x={cx} y="199" textAnchor="middle" className="brood-gauge-secondary">
        {secondaryLabel}
      </text>
    </svg>
  );
}

function SegmentLegend({ segments }) {
  return (
    <div className="brood-gauge-legend" aria-label="Gauge ranges">
      {segments.map((segment) => (
        <span key={segment.label}>
          <i style={{ background: segment.color }} />
          {segment.label}
          <small>{segment.rangeLabel || `${segment.start}–${segment.end}`}</small>
        </span>
      ))}
    </div>
  );
}

export function HealthScoreGauge({
  score,
  level,
  label,
  detail,
  badge,
}) {
  const value = clamp(score, 1, 100);
  const resolvedLevel = healthLevelFromScore(value);
  const accent = HEALTH_COLORS[resolvedLevel];
  const suppliedLevelMismatch = Boolean(level && level !== resolvedLevel);

  return (
    <article
      className={`brood-analogue-card brood-realistic-gauge-card ${healthClass(resolvedLevel)}`}
      style={{ '--brood-gauge-accent': accent }}
    >
      <header>
        <span><Gauge size={17} /> {label}</span>
        {badge && <small>{badge}</small>}
      </header>

      <SemicircleGauge
        value={value}
        segments={HEALTH_SEGMENTS}
        ticks={[
          { value: 0, label: '0' },
          { value: 20, label: '20' },
          { value: 40, label: '40' },
          { value: 60, label: '60' },
          { value: 80, label: '80' },
          { value: 100, label: '100' },
        ]}
        accent={accent}
        valueLabel="/ 100"
        secondaryLabel={`${resolvedLevel} health`}
        ariaLabel={`${label}: ${value.toFixed(1)} out of 100, ${resolvedLevel}`}
      />

      <div className="brood-gauge-result">
        <strong>{resolvedLevel}</strong>
        {suppliedLevelMismatch && (
          <small className="brood-gauge-consistency-note">
            Display level recalculated from the score; API supplied “{level}”.
          </small>
        )}
        {detail && <p>{detail}</p>}
      </div>

      <SegmentLegend segments={HEALTH_SEGMENTS} />
    </article>
  );
}

export function StabilityGauge({
  score,
  level,
  detail,
  label = 'Brood Health Stability Index',
  badge = 'Previous 6 hours',
}) {
  const value = clamp(score, 0, 100);
  const resolvedLevel = stabilityLevelFromScore(value);
  const accent = STABILITY_COLORS[resolvedLevel];
  const suppliedLevelMismatch = Boolean(level && level !== resolvedLevel);

  return (
    <article
      className={`brood-analogue-card brood-realistic-gauge-card stability-${stabilityClass(resolvedLevel)}`}
      style={{ '--brood-gauge-accent': accent }}
    >
      <header>
        <span><Activity size={17} /> {label}</span>
        <small>{badge}</small>
      </header>

      <SemicircleGauge
        value={value}
        segments={STABILITY_SEGMENTS}
        ticks={[
          { value: 0, label: '0' },
          { value: 40, label: '40' },
          { value: 70, label: '70' },
          { value: 100, label: '100' },
        ]}
        accent={accent}
        valueLabel="/ 100"
        secondaryLabel={`${resolvedLevel} stability`}
        ariaLabel={`BHSI: ${value.toFixed(1)} out of 100, ${resolvedLevel} stability`}
      />

      <div className="brood-gauge-result">
        <strong>{resolvedLevel} stability</strong>
        {suppliedLevelMismatch && (
          <small className="brood-gauge-consistency-note">
            Display level recalculated from BHSI; API supplied “{level}”.
          </small>
        )}
        <p>
          {detail
            || 'Lower six-hour variability in internal temperature, humidity and CO₂ produces a higher value.'}
        </p>
      </div>

      <SegmentLegend segments={STABILITY_SEGMENTS} />
    </article>
  );
}

function rodTone(value) {
  if (value < -3) return { label: 'Rapid Declining', tone: 'rapid-decline', color: '#dc2626' };
  if (value < -0.5) return { label: 'Slow Declining', tone: 'slow-decline', color: '#d97706' };
  if (value <= 0.5) return { label: 'Stable', tone: 'stable', color: '#64748b' };
  if (value <= 3) return { label: 'Slow Improving', tone: 'slow-improve', color: '#10b981' };
  return { label: 'Rapid Improving', tone: 'rapid-improve', color: '#0f766e' };
}

export function RoDMeter({
  value,
  label,
  title = 'Rate of Deterioration',
  badge = 'Score points/hour',
  detail,
}) {
  const numeric = Number.isFinite(Number(value)) ? Number(value) : 0;
  const clipped = clamp(numeric, -6, 6);
  const position = ((clipped + 6) / 12) * 100;
  const resolved = rodTone(numeric);
  const displayLabel = label || resolved.label;
  const DirectionIcon = numeric < -0.5
    ? TrendingDown
    : numeric > 0.5
      ? TrendingUp
      : Activity;

  return (
    <article
      className={`brood-analogue-card brood-realistic-gauge-card rod-${resolved.tone}`}
      style={{ '--brood-gauge-accent': resolved.color }}
    >
      <header>
        <span><DirectionIcon size={17} /> {title}</span>
        <small>{badge}</small>
      </header>

      <div
        className="brood-realistic-rod"
        role="meter"
        aria-label={`Rate of Deterioration: ${numeric.toFixed(2)} score points per hour, ${displayLabel}`}
        aria-valuemin="-6"
        aria-valuemax="6"
        aria-valuenow={clipped}
      >
        <div className="brood-realistic-rod-readout">
          <strong style={{ color: resolved.color }}>
            {numeric > 0 ? '+' : ''}{numberValue(numeric, 2)}
          </strong>
          <span>points/hour</span>
        </div>

        <div className="brood-realistic-rod-track">
          <span className="rapid-decline" />
          <span className="slow-decline" />
          <span className="stable" />
          <span className="slow-improve" />
          <span className="rapid-improve" />
          <i
            className="brood-realistic-rod-pointer"
            style={{ left: `${position}%` }}
            title={`${numeric.toFixed(2)} points/hour`}
          />
        </div>

        <div className="brood-realistic-rod-ticks">
          <span>−6</span>
          <span>−3</span>
          <span>−0.5</span>
          <span>0</span>
          <span>+0.5</span>
          <span>+3</span>
          <span>+6</span>
        </div>

        <div className="brood-realistic-rod-zones">
          <span>Rapid decline</span>
          <span>Slow decline</span>
          <span>Stable</span>
          <span>Slow improve</span>
          <span>Rapid improve</span>
        </div>
      </div>

      <div className="brood-gauge-result">
        <strong>{displayLabel}</strong>
        <p>{detail || 'Negative values indicate deterioration. Positive values indicate improvement.'}</p>
      </div>
    </article>
  );
}
