import { useEffect, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  FlaskConical,
  ShieldCheck,
} from "lucide-react";

import { Panel } from "../../../components/common/Panel";
import { loadClassifierDerivedHuiDashboard } from "../../../services/classifierDerivedHuiService";

import "./HarvestingFinalResearchPanel.css";

export default function HarvestingFinalResearchPanel({ dashboard }) {
  const [huiDashboard, setHuiDashboard] = useState(null);

  useEffect(() => {
    let cancelled = false;

    loadClassifierDerivedHuiDashboard()
      .then((payload) => {
        if (!cancelled) {
          setHuiDashboard(payload);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setHuiDashboard(null);
        }
      });

    return () => {
      cancelled = true;
    };
  }, []);

  const decision = dashboard?.decision;
  const classification = decision?.classification_branch ?? {};
  const calibrationGate = huiDashboard?.calibration?.gate;
  const futureGate = huiDashboard?.future_hui_regression?.gate;

  return (
    <Panel
      title="Final Viva Research Status"
      subtitle="Classification, calibration assessment and future-HUI regression are reported as separate evidence stages."
    >
      <div className="harvest-final-research">
        <div className="harvest-final-banner">
          <FlaskConical size={22} aria-hidden="true" />
          <div>
            <strong>Classifier-derived HUI research prototype</strong>
            <p>
              The 72-hour classifier provides the risk signal used to
              construct the current HUI. Multi-horizon regression forecasts
              the HUI after 24, 48 and 72 hours. The system is suitable for
              the final viva dashboard, but independent operational and
              biological validation remains future work.
            </p>
          </div>
        </div>

        <div className="harvest-final-grid">
          <div className="is-complete">
            <CheckCircle2 size={18} aria-hidden="true" />
            <span>Classification comparison</span>
            <strong>Complete</strong>
          </div>
          <div className="is-limited">
            <AlertTriangle size={18} aria-hidden="true" />
            <span>Temporal alert policy</span>
            <strong>
              {classification.unchanged_test_event_supported
                ? "Test supported"
                : "Test event missed"}
            </strong>
          </div>
          <div className="is-limited">
            <ShieldCheck size={18} aria-hidden="true" />
            <span>Probability calibration</span>
            <strong>
              {calibrationGate?.gate_passed
                ? "Research gate passed"
                : "Research-stage only"}
            </strong>
          </div>
          <div className="is-complete">
            <CheckCircle2 size={18} aria-hidden="true" />
            <span>Future-HUI regression</span>
            <strong>
              {futureGate?.gate_passed ? "3/3 horizons passed" : "Pending"}
            </strong>
          </div>
        </div>

        <div className="harvest-final-test-note">
          The stricter smoothed temporal alert policy detected{" "}
          <strong>
            {classification.validation_detected_event_count ?? 0}/
            {classification.validation_event_count ?? 0}
          </strong>{" "}
          validation events and{" "}
          <strong>
            {classification.test_detected_event_count ?? 0}/
            {classification.test_event_count ?? 0}
          </strong>{" "}
          held-out test events. This limitation remains visible and is not
          replaced by the later HUI regression results.
        </div>

        <div className="harvest-final-hui-note">
          Future-HUI regression improved validation MAE over persistence at
          all three horizons. The final dashboard is therefore supported as
          a decision-support research prototype, not as an autonomous harvest
          instruction system.
        </div>
      </div>
    </Panel>
  );
}
