import { useMemo } from 'react';
import { BookOpen, Calculator, Gauge, TrendingDown } from 'lucide-react';
import { asPercent, numberValue } from '../utils/broodHealth';

function clipText(value, fallback) {
  return value === null || value === undefined || value === '' ? fallback : value;
}

function FormulaLine({ symbol, children }) {
  return (
    <div className="brood-formula-line">
      <span>{symbol}</span>
      <code>{children}</code>
    </div>
  );
}

export function BroodFormulaReference({
  scoreDefinition,
  stabilityReference,
  weightCalibration,
}) {
  const config = scoreDefinition?.config || {};
  const components = scoreDefinition?.components || [];

  const weights = useMemo(() => {
    const byKey = Object.fromEntries(
      components.map((item) => [item.key, Number(item.weight || 0)]),
    );

    return {
      temperature: Number(byKey.temperature ?? config.temperature_weight ?? 0.45),
      humidity: Number(byKey.humidity ?? config.humidity_weight ?? 0.25),
      co2: Number(byKey.co2 ?? config.co2_weight ?? 0.20),
      weight: Number(
        byKey.weight_stability
          ?? config.weight_stability_weight
          ?? 0.10,
      ),
    };
  }, [components, config]);

  const temperatureCentre = Number(config.temperature_centre ?? 35);
  const temperatureScale = Number(config.temperature_scale ?? 4);
  const humidityCentre = Number(config.humidity_centre ?? 65);
  const humidityScale = Number(config.humidity_scale ?? 16);
  const co2Good = Number(config.co2_good_max ?? 3000);
  const co2Warning = Number(config.co2_warning_max ?? 10000);
  const co2Critical = Number(config.co2_critical_max ?? 30000);
  const weightHours = Number(config.weight_reference_hours ?? 24);
  const weightPenalty = Number(config.weight_penalty_per_percentage_point ?? 8);

  const residualScale = Number(
    stabilityReference?.residual_rmse_scale ?? 3,
  );
  const stepScale = Number(
    stabilityReference?.step_change_std_scale ?? 2,
  );
  const calibrationPercent = Number(
    stabilityReference?.calibration_quantile ?? 0.90,
  );

  return (
    <div className="brood-formula-reference">
      <details className="brood-formula-details">
        <summary>
          <span className="brood-formula-summary-icon"><Calculator size={18} /></span>
          <span>
            <strong>Score, BHSI and RoD formulas</strong>
          </span>
        </summary>

        <div className="brood-formula-content">
          <section className="brood-formula-section">
            <div className="brood-formula-section-title">
              <Gauge size={18} />
              <div>
                <h4>1. Brood Health Score (1–100)</h4>
                <p>
                  Four sensor sub-scores are combined. A higher value means the measured
                  hive condition is closer to the defined favourable condition.
                </p>
              </div>
            </div>

            <div className="brood-formula-highlight">
              <FormulaLine symbol="BHS =">
                {weights.temperature.toFixed(2)}·Sₜ + {weights.humidity.toFixed(2)}·Sₕ + {weights.co2.toFixed(2)}·SCO₂ + {weights.weight.toFixed(2)}·Sᵥ
              </FormulaLine>
            </div>

            <div className="brood-formula-weight-table table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Input</th>
                    <th>Contribution</th>
                    <th>Meaning in the score</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td><strong>Temperature</strong></td>
                    <td>{asPercent(weights.temperature, 0)}</td>
                    <td>How close internal temperature is to the reference brood temperature.</td>
                  </tr>
                  <tr>
                    <td><strong>Humidity</strong></td>
                    <td>{asPercent(weights.humidity, 0)}</td>
                    <td>How close internal humidity is to the reference humidity.</td>
                  </tr>
                  <tr>
                    <td><strong>CO₂</strong></td>
                    <td>{asPercent(weights.co2, 0)}</td>
                    <td>Penalises increasingly high CO₂ conditions.</td>
                  </tr>
                  <tr>
                    <td><strong>Relative weight stability</strong></td>
                    <td>{asPercent(weights.weight, 0)}</td>
                    <td>Uses relative 24-hour weight change rather than absolute hive weight.</td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div className="brood-formula-grid">
              <article>
                <strong>Temperature sub-score</strong>
                <FormulaLine symbol="Sₜ =">
                  100 × exp[-((T − {temperatureCentre}) / {temperatureScale})²]
                </FormulaLine>
                <p>Maximum score occurs close to {numberValue(temperatureCentre, 1)} °C.</p>
              </article>

              <article>
                <strong>Humidity sub-score</strong>
                <FormulaLine symbol="Sₕ =">
                  100 × exp[-((H − {humidityCentre}) / {humidityScale})²]
                </FormulaLine>
                <p>Maximum score occurs close to {numberValue(humidityCentre, 1)}% humidity.</p>
              </article>

              <article>
                <strong>CO₂ sub-score</strong>
                <FormulaLine symbol="SCO₂ =">
                  100 when CO₂ ≤ {numberValue(co2Good, 0)} ppm; then progressively reduced to 0 by {numberValue(co2Critical, 0)} ppm
                </FormulaLine>
                <p>
                  The implemented piecewise penalty changes slope again at
                  {' '}{numberValue(co2Warning, 0)} ppm.
                </p>
              </article>

              <article>
                <strong>Relative-weight sub-score</strong>
                <FormulaLine symbol="ΔW%">
                  = 100 × (Wₜ − Wₜ₋{weightHours}h) / max(|Wₜ₋{weightHours}h|, 1)
                </FormulaLine>
                <FormulaLine symbol="Sᵥ =">
                  clip(100 − {weightPenalty} × |ΔW%|, 0, 100)
                </FormulaLine>
              </article>
            </div>

            <div className="brood-formula-note">
              <BookOpen size={17} />
              <span>
                The active weights sum to 100%. They were evaluated using training-hive
                sensitivity analysis under the ordering constraint
                {' '}temperature ≥ humidity ≥ CO₂ ≥ weight stability.
                {weightCalibration?.candidate_count
                  ? ` ${numberValue(weightCalibration.candidate_count, 0)} candidate weight sets were evaluated.`
                  : ''}
              </span>
            </div>

            <div className="brood-formula-levels">
              <span><b>Critical</b> 1–&lt;40</span>
              <span><b>Poor</b> 40–&lt;60</span>
              <span><b>Good</b> 60–&lt;80</span>
              <span><b>Excellent</b> 80–100</span>
            </div>
          </section>

          <section className="brood-formula-section">
            <div className="brood-formula-section-title">
              <Gauge size={18} />
              <div>
                <h4>2. Forecast BHSI (0–100)</h4>
                <p>
                  BHSI describes how smooth or unstable the predicted health-score path is
                  from the current score through the next six hours.
                </p>
              </div>
            </div>

            <div className="brood-formula-grid">
              <article>
                <strong>Step A — fit the six-hour trend</strong>
                <FormulaLine symbol="ŷᵢ =">a + b·tᵢ</FormulaLine>
                <p>A straight trend is fitted through the current score and +1 to +6 hour predictions.</p>
              </article>
              <article>
                <strong>Step B — measure fluctuation</strong>
                <FormulaLine symbol="R =">√mean[(yᵢ − ŷᵢ)²]</FormulaLine>
                <FormulaLine symbol="D =">SD(yᵢ − yᵢ₋₁)</FormulaLine>
              </article>
              <article>
                <strong>Step C — normalise instability</strong>
                <FormulaLine symbol="I =">
                  0.60·(R/{numberValue(residualScale, 2)}) + 0.40·(D/{numberValue(stepScale, 2)})
                </FormulaLine>
                <p>
                  The two scale values were calibrated from the
                  {' '}{(calibrationPercent * 100).toFixed(0)}th percentile of training-hive trajectories.
                </p>
              </article>
              <article>
                <strong>Step D — convert to 0–100</strong>
                <FormulaLine symbol="BHSI =">100 × exp(−I)</FormulaLine>
                <p>Higher BHSI = smoother predicted path. Direction is reported separately by RoD.</p>
              </article>
            </div>

            <div className="brood-formula-levels">
              <span><b>Low</b> 0–&lt;40</span>
              <span><b>Moderate</b> 40–&lt;70</span>
              <span><b>High</b> 70–100</span>
            </div>
          </section>

          <section className="brood-formula-section">
            <div className="brood-formula-section-title">
              <TrendingDown size={18} />
              <div>
                <h4>3. Forecast Rate of Deterioration (RoD)</h4>
                <p>
                  RoD is the slope of the predicted current-to-+6-hour health-score line,
                  measured in score points per hour.
                </p>
              </div>
            </div>

            <div className="brood-formula-highlight">
              <FormulaLine symbol="RoD =">
                Σ[(tᵢ − t̄)(yᵢ − ȳ)] / Σ[(tᵢ − t̄)²]
              </FormulaLine>
            </div>

            <div className="brood-formula-levels brood-formula-levels-five">
              <span><b>Rapid decline</b> &lt; −3</span>
              <span><b>Slow decline</b> −3 to &lt;−0.5</span>
              <span><b>Stable</b> −0.5 to 0.5</span>
              <span><b>Slow improve</b> &gt;0.5 to 3</span>
              <span><b>Rapid improve</b> &gt;3</span>
            </div>

            <p className="chart-footnote">
              Example: RoD = −2 means the predicted Brood Health Score is falling by
              about 2 points per hour across the six-hour forecast path.
            </p>
          </section>
        </div>
      </details>
    </div>
  );
}
