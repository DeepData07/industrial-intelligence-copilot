"""Early-warning and incident generation for live operational twin states."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from industrial_copilot.config import get_settings
from industrial_copilot.simulation.schemas import TelemetryEvent
from industrial_copilot.simulation.state import (
    EngineeredTelemetry,
    OperationalTwinState,
    RuleMargins,
)

IncidentSeverity = Literal["WARNING", "INCIDENT"]
IncidentStatus = Literal["ACTIVE", "CLEARED"]


class MonitoringPolicy(BaseModel):
    """Configurable product-policy thresholds for live warning generation."""

    model_config = ConfigDict(frozen=True)

    warning_risk_threshold: float = Field(default=0.20, ge=0, le=1)
    incident_risk_threshold: float = Field(default=0.35, ge=0, le=1)
    osf_warning_margin_min_nm: float = Field(default=1000.0, ge=0)
    hdf_temperature_margin_k: float = Field(default=0.5, ge=0)
    hdf_rpm_margin: float = Field(default=50.0, ge=0)
    pwf_power_margin_w: float = Field(default=500.0, ge=0)

    @classmethod
    def from_settings(cls) -> MonitoringPolicy:
        """Load alert-policy thresholds from application settings."""

        settings = get_settings()
        return cls(
            warning_risk_threshold=settings.live_warning_risk_threshold,
            incident_risk_threshold=settings.live_incident_risk_threshold,
            osf_warning_margin_min_nm=settings.live_osf_warning_margin_min_nm,
            hdf_temperature_margin_k=settings.live_hdf_temperature_margin_k,
            hdf_rpm_margin=settings.live_hdf_rpm_margin,
            pwf_power_margin_w=settings.live_pwf_power_margin_w,
        )


class IncidentContextPackage(BaseModel):
    """Compact event context consumed by the incident copilot."""

    model_config = ConfigDict(frozen=True)

    current_telemetry: TelemetryEvent
    engineered: EngineeredTelemetry
    rule_margins: RuleMargins
    failure_risk_probability: float | None
    triggered_rules: tuple[str, ...]
    evidence: tuple[str, ...]
    limitations: tuple[str, ...]


class Incident(BaseModel):
    """One active operational event created from monitored evidence."""

    model_config = ConfigDict(frozen=True)

    incident_id: str
    asset_id: str
    session_id: str
    first_cycle: int
    latest_cycle: int
    severity: IncidentSeverity
    status: IncidentStatus = "ACTIVE"
    title: str
    primary_reason: str
    suggested_investigation: str
    context: IncidentContextPackage


class IncidentEvaluation(BaseModel):
    """Result of evaluating one twin state against the incident policy."""

    model_config = ConfigDict(frozen=True)

    incident: Incident | None
    created_new_incident: bool = False
    cleared_active_incident: bool = False
    evidence: tuple[str, ...] = ()


class IncidentEngine:
    """Stateful early-warning engine with duplicate active-incident debouncing."""

    def __init__(self, policy: MonitoringPolicy | None = None) -> None:
        self._policy = policy or MonitoringPolicy.from_settings()
        self._incident_counter = 1
        self._active_incident: Incident | None = None

    @property
    def active_incident(self) -> Incident | None:
        """Return the current active incident, if any."""

        return self._active_incident

    def reset(self) -> None:
        """Clear active incident state and restart local incident numbering."""

        self._incident_counter = 1
        self._active_incident = None

    def evaluate(self, twin_state: OperationalTwinState) -> IncidentEvaluation:
        """Evaluate the latest machine state and create/update/clear an incident."""

        if (
            twin_state.current_telemetry is None
            or twin_state.engineered is None
            or twin_state.rule_margins is None
        ):
            return IncidentEvaluation(incident=self._active_incident)

        severity, evidence = self._severity_and_evidence(twin_state)
        if severity is None:
            cleared = self._active_incident is not None
            self._active_incident = None
            return IncidentEvaluation(
                incident=None,
                cleared_active_incident=cleared,
                evidence=evidence,
            )

        if self._active_incident is None:
            incident = self._create_incident(twin_state, severity, evidence)
            self._active_incident = incident
            self._incident_counter += 1
            return IncidentEvaluation(
                incident=incident,
                created_new_incident=True,
                evidence=evidence,
            )

        updated = self._active_incident.model_copy(
            update={
                "latest_cycle": twin_state.current_cycle,
                "severity": _max_severity(self._active_incident.severity, severity),
                "context": self._context_package(twin_state, evidence),
            }
        )
        self._active_incident = updated
        return IncidentEvaluation(incident=updated, evidence=evidence)

    def _severity_and_evidence(
        self,
        twin_state: OperationalTwinState,
    ) -> tuple[IncidentSeverity | None, tuple[str, ...]]:
        evidence: list[str] = []
        severity: IncidentSeverity | None = None

        if twin_state.risk is not None:
            probability = twin_state.risk.failure_probability
            if probability >= self._policy.incident_risk_threshold:
                severity = "INCIDENT"
                evidence.append(
                    f"Failure risk {probability:.1%} exceeded incident policy threshold "
                    f"{self._policy.incident_risk_threshold:.0%}."
                )
            elif probability >= self._policy.warning_risk_threshold:
                severity = "WARNING"
                evidence.append(
                    f"Failure risk {probability:.1%} exceeded warning policy threshold "
                    f"{self._policy.warning_risk_threshold:.0%}."
                )

        margin_severity, margin_evidence = self._rule_margin_evidence(twin_state.rule_margins)
        if margin_severity is not None:
            severity = _max_severity(severity, margin_severity)
            evidence.extend(margin_evidence)

        return severity, tuple(evidence)

    def _rule_margin_evidence(
        self,
        margins: RuleMargins,
    ) -> tuple[IncidentSeverity | None, tuple[str, ...]]:
        evidence: list[str] = []
        severity: IncidentSeverity | None = None

        if margins.osf_remaining_margin_min_nm <= self._policy.osf_warning_margin_min_nm:
            severity = "WARNING"
            evidence.append(
                "Overstrain margin is narrow "
                f"({margins.osf_remaining_margin_min_nm:.0f} min Nm remaining)."
            )
        if "OSF" in margins.triggered_rules:
            severity = "WARNING"
            evidence.append("Current telemetry is inside the documented OSF rule condition.")

        approaching_hdf = (
            margins.hdf_temperature_delta_margin_k <= self._policy.hdf_temperature_margin_k
            and margins.hdf_rpm_margin <= self._policy.hdf_rpm_margin
        )
        if approaching_hdf:
            severity = "WARNING"
            evidence.append("Temperature-delta and RPM margins are near the HDF envelope.")
        if "HDF" in margins.triggered_rules:
            severity = "WARNING"
            evidence.append("Current telemetry is inside the documented HDF rule condition.")

        near_low_power = 0 <= margins.pwf_low_power_margin_w <= self._policy.pwf_power_margin_w
        near_high_power = 0 <= margins.pwf_high_power_margin_w <= self._policy.pwf_power_margin_w
        if near_low_power or near_high_power:
            severity = "WARNING"
            evidence.append("Mechanical power is near a documented PWF boundary.")
        if "PWF" in margins.triggered_rules:
            severity = "WARNING"
            evidence.append("Current telemetry is inside the documented PWF rule condition.")

        return severity, tuple(evidence)

    def _create_incident(
        self,
        twin_state: OperationalTwinState,
        severity: IncidentSeverity,
        evidence: tuple[str, ...],
    ) -> Incident:
        primary_reason = evidence[0] if evidence else "Operating condition crossed monitoring policy."
        return Incident(
            incident_id=f"INC-{self._incident_counter:04d}",
            asset_id=twin_state.asset_id,
            session_id=twin_state.session_id,
            first_cycle=twin_state.current_cycle,
            latest_cycle=twin_state.current_cycle,
            severity=severity,
            title="Operating condition changed",
            primary_reason=primary_reason,
            suggested_investigation=_suggested_investigation(twin_state.rule_margins),
            context=self._context_package(twin_state, evidence),
        )

    def _context_package(
        self,
        twin_state: OperationalTwinState,
        evidence: tuple[str, ...],
    ) -> IncidentContextPackage:
        if (
            twin_state.current_telemetry is None
            or twin_state.engineered is None
            or twin_state.rule_margins is None
        ):
            raise ValueError("Cannot build incident context without current telemetry evidence.")
        return IncidentContextPackage(
            current_telemetry=twin_state.current_telemetry,
            engineered=twin_state.engineered,
            rule_margins=twin_state.rule_margins,
            failure_risk_probability=(
                twin_state.risk.failure_probability if twin_state.risk is not None else None
            ),
            triggered_rules=twin_state.rule_margins.triggered_rules,
            evidence=evidence,
            limitations=twin_state.limitations,
        )


def _max_severity(
    first: IncidentSeverity | None,
    second: IncidentSeverity | None,
) -> IncidentSeverity | None:
    if first == "INCIDENT" or second == "INCIDENT":
        return "INCIDENT"
    if first == "WARNING" or second == "WARNING":
        return "WARNING"
    return None


def _suggested_investigation(margins: RuleMargins | None) -> str:
    if margins is None:
        return "Review current telemetry against recent healthy operation."
    if "OSF" in margins.triggered_rules or margins.osf_remaining_margin_min_nm <= 1000:
        return "Inspect tool condition and verify whether the torque increase is expected."
    if "HDF" in margins.triggered_rules:
        return "Check cooling/thermal conditions and confirm whether the low-RPM state is expected."
    if "PWF" in margins.triggered_rules:
        return "Review RPM-torque operating point and confirm the power demand is expected."
    return "Compare current telemetry with recent baseline and similar historical conditions."
