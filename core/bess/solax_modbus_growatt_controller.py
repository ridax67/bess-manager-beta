"""Growatt MIN inverter controller using solax_modbus with VPP remote power control.

This controller replaces the previous single-segment TOU approach with Growatt's
VPP remote power control registers, giving per-period power control without any
TOU segment management.

VPP control entities (via solax_modbus HA integration):
    select.growatt_inverter_vpp_remote_control  — enable/disable remote control
    select.growatt_inverter_vpp_status          — enable/disable VPP status
    select.growatt_inverter_vpp_allow_ac_charging — enable AC charging from grid
    number.growatt_inverter_vpp_power           — power % (-100..100)
    number.growatt_inverter_vpp_time            — fallback duration in minutes

Enable sequence:  Remote Control → wait 1s → Status
Disable sequence: Status → wait 1s → Remote Control

VPP Time is written every period (20 min) to reset the fallback timer.
If BESS stops writing for any reason, the inverter returns to load_first
after 20 minutes automatically.

Intent → VPP power mapping:
    BATTERY_EXPORT → negative % (discharge to grid)
    GRID_CHARGING  → positive % (charge from grid, AC charging enabled)
    SOLAR_STORAGE  → 0 (inverter handles solar naturally)
    LOAD_SUPPORT   → 0 (inverter handles load support naturally)
    IDLE           → 0
"""

import logging
import time

from . import time_utils
from .dp_schedule import DPSchedule
from .growatt_min_controller import GrowattMinController
from .health_check import perform_health_check
from .settings import BatterySettings

logger = logging.getLogger(__name__)

# VPP entity IDs (via solax_modbus HA integration)
VPP_REMOTE_CONTROL_ENTITY = "select.growatt_inverter_vpp_remote_control"
VPP_STATUS_ENTITY = "select.growatt_inverter_vpp_status"
VPP_ALLOW_AC_CHARGING_ENTITY = "select.growatt_inverter_vpp_allow_ac_charging"
VPP_POWER_ENTITY = "number.growatt_inverter_vpp_power"
VPP_TIME_ENTITY = "number.growatt_inverter_vpp_time"

VPP_ENABLE = "Enabled"
VPP_DISABLE = "Disabled"

# Fallback duration in minutes — inverter returns to load_first if BESS
# stops writing. Must be > 15 (period length) to avoid spurious fallback
# during normal operation.
VPP_FALLBACK_MINUTES = 20


class SolaxModbusGrowattController(GrowattMinController):
    """Growatt MIN controller using VPP remote power control.

    Manages per-period charge/discharge power via VPP registers instead of
    TOU segments. Schedule creation and comparison logic is inherited from
    GrowattMinController via strategic intents.
    """

    # Class-level VPP state — shared across all instances so that when
    # battery_system_manager replaces the controller with a new instance
    # each optimization cycle, the VPP enable state is preserved and
    # flash registers are not written unnecessarily.
    _class_vpp_enabled: bool = False
    _class_vpp_status_enabled: bool = False
    _class_ac_charging_enabled: bool = False
    _class_last_written_vpp_power: int | None = None

    def __init__(self, battery_settings: BatterySettings) -> None:
        """Initialize the VPP controller."""
        super().__init__(battery_settings)
        # Keep TOU interval lists for API/display compatibility with parent
        self._active_tou_intervals: list[dict] = []

        # VPP controls charge/discharge power directly — disable the separate
        # EMS charging rate register to avoid conflicting with VPP commands.
        self.supports_charge_rate_control = False

    @property
    def _vpp_enabled(self) -> bool:
        return SolaxModbusGrowattController._class_vpp_enabled

    @_vpp_enabled.setter
    def _vpp_enabled(self, value: bool) -> None:
        SolaxModbusGrowattController._class_vpp_enabled = value

    @property
    def _vpp_status_enabled(self) -> bool:
        return SolaxModbusGrowattController._class_vpp_status_enabled

    @_vpp_status_enabled.setter
    def _vpp_status_enabled(self, value: bool) -> None:
        SolaxModbusGrowattController._class_vpp_status_enabled = value

    @property
    def _ac_charging_enabled(self) -> bool:
        return SolaxModbusGrowattController._class_ac_charging_enabled

    @_ac_charging_enabled.setter
    def _ac_charging_enabled(self, value: bool) -> None:
        SolaxModbusGrowattController._class_ac_charging_enabled = value

    @property
    def _last_written_vpp_power(self) -> int | None:
        return SolaxModbusGrowattController._class_last_written_vpp_power

    @_last_written_vpp_power.setter
    def _last_written_vpp_power(self, value: int | None) -> None:
        SolaxModbusGrowattController._class_last_written_vpp_power = value

    # ── Abstract property (required by parent) ───────────────────────────────

    @property
    def active_tou_intervals(self) -> list[dict]:
        return self._active_tou_intervals

    @active_tou_intervals.setter
    def active_tou_intervals(self, value: list[dict]) -> None:
        self._active_tou_intervals = value

    # ── Schedule creation ────────────────────────────────────────────────────

    def create_schedule(
        self,
        schedule: DPSchedule,
        current_period: int = 0,
        previous_tou_intervals: list[dict] | None = None,
    ) -> None:
        """Store strategic intents — VPP power is applied per-period.

        No TOU segment computation needed. Strategic intents are stored
        and used directly in apply_period each quarter-hour.

        Args:
            schedule: DPSchedule containing strategic_intent list.
            current_period: Current 15-minute period (0-95).
            previous_tou_intervals: Unused for VPP approach.
        """
        logger.info("Creating VPP schedule from strategic intents")

        self.strategic_intents = schedule.original_dp_results["strategic_intent"]
        self.current_schedule = schedule

        logger.info(
            "VPP: %d strategic intents loaded",
            len(self.strategic_intents),
        )

        # Log intent transitions for debugging
        for period in range(1, len(self.strategic_intents)):
            if self.strategic_intents[period] != self.strategic_intents[period - 1]:
                logger.info(
                    "Intent transition at period %d: %s -> %s",
                    period,
                    self.strategic_intents[period - 1],
                    self.strategic_intents[period],
                )

        # Build display state for API/UI consumption
        self._update_vpp_display_state()

    # ── VPP enable/disable ───────────────────────────────────────────────────

    def _enable_vpp(self, controller) -> None:
        """Enable VPP control.

        VPP Status is written once (or if it was disabled) followed by 1s pause.
        VPP Remote Control is written every period together with power commands.
        """
        if not self._vpp_status_enabled:
            logger.info("HARDWARE: VPP Status -> Enabled")
            controller._service_call_with_retry(
                "select",
                "select_option",
                operation="VPP enable status",
                entity_id=VPP_STATUS_ENTITY,
                option=VPP_ENABLE,
            )
            self._vpp_status_enabled = True
            logger.info("HARDWARE: Waiting 1s after enabling VPP Status")
            time.sleep(1)

        logger.info("HARDWARE: VPP Remote Control -> Enabled")
        controller._service_call_with_retry(
            "select",
            "select_option",
            operation="VPP enable remote control",
            entity_id=VPP_REMOTE_CONTROL_ENTITY,
            option=VPP_ENABLE,
        )
        self._vpp_enabled = True

    def _disable_vpp(self, controller) -> None:
        """Disable VPP remote control with required 1s delay between steps.

        Sequence: Status → wait 1s → Remote Control
        """
        logger.info("HARDWARE: VPP Status -> Disabled")
        controller._service_call_with_retry(
            "select",
            "select_option",
            operation="VPP disable status",
            entity_id=VPP_STATUS_ENTITY,
            option=VPP_DISABLE,
        )
        logger.info("HARDWARE: Waiting 1s before disabling VPP Remote Control")
        time.sleep(1)
        logger.info("HARDWARE: VPP Remote Control -> Disabled")
        controller._service_call_with_retry(
            "select",
            "select_option",
            operation="VPP disable remote control",
            entity_id=VPP_REMOTE_CONTROL_ENTITY,
            option=VPP_DISABLE,
        )
        self._vpp_enabled = False
        self._vpp_status_enabled = False
        logger.info("VPP remote control disabled")

    def deinitialize_hardware(self, controller) -> None:
        """Disable VPP control cleanly on BESS shutdown.

        Mirrors initialize_hardware — should be called from battery_system_manager
        shutdown hook, e.g. via SIGTERM handler in the addon entry point:

            signal.signal(signal.SIGTERM, lambda s, f: manager.deinitialize_hardware())

        Without this call, the inverter remains in VPP mode until the 20-minute
        fallback timer expires and returns to load_first. TOU control will NOT
        resume until VPP Remote Control is explicitly Disabled.
        """
        if self._vpp_enabled or self._vpp_status_enabled:
            try:
                self._disable_vpp(controller)
                logger.info("VPP shutdown complete")
            except Exception as e:
                logger.error("FAILED: VPP shutdown: %s", e)
        else:
            logger.info("VPP already disabled, no shutdown action needed")

    # ── Intent → VPP power ───────────────────────────────────────────────────

    def _intent_to_vpp_power(
        self, intent: str, discharge_rate: int, charge_rate: int
    ) -> int:
        """Convert a strategic intent and rates to a VPP power percentage.

        Args:
            intent: Strategic intent (e.g. BATTERY_EXPORT)
            discharge_rate: Discharge rate 0-100%
            charge_rate: Charge rate 0-100%

        Returns:
            VPP power %: negative=discharge to grid, positive=charge, 0=idle
        """
        if intent == "BATTERY_EXPORT":
            return -min(discharge_rate, 100)
        elif intent == "GRID_CHARGING":
            return min(charge_rate, 100)
        elif intent == "LOAD_SUPPORT":
            return -min(discharge_rate, 100)
        else:
            # SOLAR_STORAGE, IDLE
            return 0

    # ── Hardware interface ────────────────────────────────────────────────────

    def apply_period(
        self, controller, grid_charge: bool, discharge_rate: int
    ) -> tuple[bool, str]:
        """Write VPP power setting for the current period.

        Called every 15 minutes by BESS. Always resets the fallback timer
        by writing VPP Time, so the inverter returns to load_first if BESS
        stops for any reason.

        Args:
            controller: HomeAssistantAPIController instance
            grid_charge: Whether grid charging is active this period
            discharge_rate: Discharge power rate (0-100%)

        Returns:
            Tuple of (success, error_message).
        """
        errors = []
        now = time_utils.now()
        current_period = now.hour * 4 + now.minute // 15

        intent = "IDLE"
        if current_period < len(self.strategic_intents):
            intent = self.strategic_intents[current_period]

        charge_rate = self.battery_settings.charging_power_rate if grid_charge else 0
        vpp_power = self._intent_to_vpp_power(intent, discharge_rate, charge_rate)

        logger.info(
            "Period %d (%02d:%02d): intent=%s vpp_power=%d%% "
            "(discharge_rate=%d%% charge_rate=%d%%)",
            current_period,
            now.hour,
            now.minute,
            intent,
            vpp_power,
            discharge_rate,
            charge_rate,
        )

        # VPP Status written once; Remote Control written every period
        try:
            self._enable_vpp(controller)
        except Exception as e:
            logger.error("FAILED: Enable VPP: %s", e)
            errors.append(str(e))

        # Always reset fallback timer every period
        try:
            logger.info(
                "HARDWARE: VPP Time -> %d min (fallback timer reset)",
                VPP_FALLBACK_MINUTES,
            )
            controller._service_call_with_retry(
                "number",
                "set_value",
                operation="VPP reset fallback timer",
                entity_id=VPP_TIME_ENTITY,
                value=VPP_FALLBACK_MINUTES,
            )
        except Exception as e:
            logger.error("FAILED: Reset VPP timer: %s", e)
            errors.append(str(e))

        # Enable/disable AC charging based on grid_charge parameter —
        # battery_system_manager is the authoritative source for this,
        # matching how the TOU implementation works.
        ac_charging_needed = grid_charge
        if ac_charging_needed != self._ac_charging_enabled:
            try:
                option = VPP_ENABLE if ac_charging_needed else VPP_DISABLE
                controller._service_call_with_retry(
                    "select",
                    "select_option",
                    operation=f"VPP set AC charging -> {option}",
                    entity_id=VPP_ALLOW_AC_CHARGING_ENTITY,
                    option=option,
                )
                self._ac_charging_enabled = ac_charging_needed
                logger.info(
                    "HARDWARE: VPP Allow AC charging -> %s (intent=%s)",
                    option,
                    intent,
                )
            except Exception as e:
                logger.error("FAILED: Set VPP Allow AC charging: %s", e)
                errors.append(str(e))

        # Write VPP power — only on change
        if vpp_power != self._last_written_vpp_power:
            try:
                controller._service_call_with_retry(
                    "number",
                    "set_value",
                    operation=f"VPP set power -> {vpp_power}%",
                    entity_id=VPP_POWER_ENTITY,
                    value=vpp_power,
                )
                logger.info(
                    "HARDWARE: VPP power %s%% -> %d%%",
                    self._last_written_vpp_power,
                    vpp_power,
                )
                self._last_written_vpp_power = vpp_power
            except Exception as e:
                logger.error("FAILED: Set VPP power to %d%%: %s", vpp_power, e)
                errors.append(str(e))
        else:
            logger.debug("VPP power unchanged at %d%%, skipping write", vpp_power)

        if errors:
            return False, "; ".join(errors)
        return True, ""

    def write_schedule_to_hardware(
        self,
        controller,
        effective_period: int,
        current_tou: list,
    ) -> tuple[int, int]:
        """Enable VPP and write initial power for the current period.

        Called when a new schedule is activated. No batch TOU write needed —
        subsequent periods are handled per-period in apply_period.

        Returns:
            Tuple of (writes, disables) — disables always 0 for VPP.
        """
        now = time_utils.now()
        current_period = now.hour * 4 + now.minute // 15

        intent = "IDLE"
        if current_period < len(self.strategic_intents):
            intent = self.strategic_intents[current_period]

        # Derive rates from schedule actions if available
        discharge_rate = 0
        charge_rate = 0
        if self.current_schedule:
            actions = self.current_schedule.original_dp_results.get("action", [])
            if current_period < len(actions):
                action_kwh = actions[current_period]
                max_kw = self.battery_settings.max_discharge_power_kw
                if max_kw > 0:
                    if action_kwh < 0:
                        discharge_rate = int(
                            min(abs(action_kwh * 4) / max_kw * 100, 100)
                        )
                    else:
                        charge_rate = int(min(action_kwh * 4 / max_kw * 100, 100))

        vpp_power = self._intent_to_vpp_power(intent, discharge_rate, charge_rate)

        try:
            self._enable_vpp(controller)
            logger.info(
                "HARDWARE: VPP Time -> %d min (fallback timer)",
                VPP_FALLBACK_MINUTES,
            )
            controller._service_call_with_retry(
                "number",
                "set_value",
                operation="VPP set fallback timer (initial)",
                entity_id=VPP_TIME_ENTITY,
                value=VPP_FALLBACK_MINUTES,
            )
            logger.info("HARDWARE: VPP Power -> %d%%", vpp_power)
            controller._service_call_with_retry(
                "number",
                "set_value",
                operation=f"VPP set initial power -> {vpp_power}%",
                entity_id=VPP_POWER_ENTITY,
                value=vpp_power,
            )
            self._last_written_vpp_power = vpp_power
            logger.info(
                "VPP: Initial write — power=%d%% (period %d, intent %s)",
                vpp_power,
                current_period,
                intent,
            )
            return 1, 0
        except Exception as e:
            logger.error("FAILED: VPP initial write: %s", e)
            return 0, 0

    def read_and_initialize_from_hardware(self, controller, current_hour: int) -> None:
        """Read VPP state from hardware and seed internal state."""
        self.current_hour = current_hour
        try:
            rc = controller.get_entity_state_raw(VPP_REMOTE_CONTROL_ENTITY)
            self._vpp_enabled = rc["state"] == VPP_ENABLE if rc else False

            status = controller.get_entity_state_raw(VPP_STATUS_ENTITY)
            self._vpp_status_enabled = status["state"] == VPP_ENABLE if status else False

            ac = controller.get_entity_state_raw(VPP_ALLOW_AC_CHARGING_ENTITY)
            self._ac_charging_enabled = ac["state"] == VPP_ENABLE if ac else False

            power = controller.get_entity_state_raw(VPP_POWER_ENTITY)
            self._last_written_vpp_power = (
                int(float(power["state"])) if power else None
            )

            logger.info(
                "VPP: Initialised from hardware — remote_control=%s "
                "status=%s ac_charging=%s power=%s%%",
                self._vpp_enabled,
                self._vpp_status_enabled,
                self._ac_charging_enabled,
                self._last_written_vpp_power,
            )
        except Exception as e:
            logger.warning("VPP: Could not read hardware state: %s — resetting", e)
            self._vpp_enabled = False
            self._vpp_status_enabled = False
            self._ac_charging_enabled = False
            self._last_written_vpp_power = None

        self._update_vpp_display_state()

    def initialize_hardware(self, controller) -> None:
        """Sync SOC limits — VPP enabled on first write_schedule_to_hardware."""
        self.sync_soc_limits(controller)

    # ── Schedule comparison ──────────────────────────────────────────────────

    def compare_schedules(
        self,
        other_schedule: "SolaxModbusGrowattController",
        from_period: int = 0,
    ) -> tuple[bool, str]:
        """Compare schedules by strategic intent list.

        Two schedules differ when any period at or after from_period has a
        different strategic intent.
        """
        current = self.strategic_intents
        new = other_schedule.strategic_intents

        if not current and not new:
            return False, ""

        if len(current) != len(new):
            return True, f"Intent count differs: {len(current)} vs {len(new)}"

        for period in range(from_period, len(current)):
            if current[period] != new[period]:
                logger.info(
                    "DECISION: Intent differs at period %d — current=%s new=%s",
                    period,
                    current[period],
                    new[period],
                )
                return True, f"Strategic intents differ from period {period}"

        logger.info("DECISION: Schedules match")
        return False, ""

    # ── Display / logging ─────────────────────────────────────────────────────

    def _update_vpp_display_state(self) -> None:
        """Update TOU interval lists for API/display compatibility."""
        # Build display segments from strategic intents for UI consumption
        groups = self.get_detailed_period_groups()
        if not groups:
            self.tou_intervals = []
            self._active_tou_intervals = []
            return

        now = time_utils.now()
        current_p = now.hour * 4 + now.minute // 15
        segments = []
        for group in groups:
            mode = self.INTENT_TO_MODE.get(group["intent"], "load_first")
            is_current = group["start_period"] <= current_p <= group["end_period"]
            segments.append(
                {
                    "segment_id": len(segments) + 1,
                    "start_time": group["start_time"],
                    "end_time": group["end_time"],
                    "batt_mode": mode,
                    "enabled": mode != "load_first",
                    "is_default": mode == "load_first",
                    "is_current": is_current,
                    "strategic_intent": group["intent"],
                }
            )
        self.tou_intervals = segments
        self._active_tou_intervals = segments

    def get_daily_TOU_settings(self) -> list[dict]:
        """Return display segments for API/UI consumption."""
        return [seg.copy() for seg in self.tou_intervals]

    def get_all_tou_segments(self, current_period: int | None = None):
        """Return display segments with is_current flag."""
        return self._update_vpp_display_state() or self.tou_intervals

    def log_current_TOU_schedule(self, header=None) -> None:
        """Log current VPP state."""
        if header:
            logger.info(header)
        logger.info(
            "VPP: remote_control=%s status=%s power=%s%% ac_charging=%s",
            VPP_ENABLE if self._vpp_enabled else VPP_DISABLE,
            VPP_ENABLE if self._vpp_status_enabled else VPP_DISABLE,
            self._last_written_vpp_power,
            VPP_ENABLE if self._ac_charging_enabled else VPP_DISABLE,
        )

    # ── Health check ─────────────────────────────────────────────────────────

    def check_health(self, controller) -> list:
        """Check VPP control entity availability."""
        health_check = perform_health_check(
            component_name="Battery Control",
            description="Controls battery via VPP remote power control",
            is_required=True,
            controller=controller,
            all_methods=[
                "get_charging_power_rate",
                "get_discharging_power_rate",
                "get_charge_stop_soc",
                "get_discharge_stop_soc",
            ],
        )

        vpp_entities = [
            VPP_REMOTE_CONTROL_ENTITY,
            VPP_STATUS_ENTITY,
            VPP_ALLOW_AC_CHARGING_ENTITY,
            VPP_POWER_ENTITY,
            VPP_TIME_ENTITY,
        ]
        for entity_id in vpp_entities:
            try:
                response = controller.get_entity_state_raw(entity_id)
                status = "OK" if response is not None else "ERROR"
                error = None if response is not None else "Entity not found or unavailable"
            except Exception as e:
                status = "ERROR"
                error = str(e)

            health_check["checks"].append(
                {
                    "name": f"VPP Entity: {entity_id}",
                    "key": entity_id,
                    "method_name": None,
                    "entity_id": entity_id,
                    "status": status,
                    "rawValue": None,
                    "displayValue": entity_id,
                    "error": error,
                }
            )

        has_error = any(c["status"] == "ERROR" for c in health_check["checks"])
        if has_error:
            health_check["status"] = "ERROR"

        return [health_check]
