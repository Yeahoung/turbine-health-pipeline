from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import pandas as pd


@dataclass
class DailyMetadataBuilderConfig:
    input_csv: Optional[str] = None
    output_root: str = "out"
    save_outputs: bool = True


class DailyMetadataBuilder:
    """
    Stage 2:
    - Normalize anomaly event table
    - Aggregate daily metadata by turbine and date
    - Save timestamped outputs for traceability
    """

    def __init__(self, config: DailyMetadataBuilderConfig) -> None:
        self.config = config
        self.stage_name = "build_metadata"
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(self.config.output_root) / self.stage_name / self.run_timestamp

    def run(self, events_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Execute the daily metadata stage.

        Args:
            events_df: optional in-memory anomaly events dataframe.
                      If None, input_csv must be provided.

        Returns:
            {
                "events_df": pd.DataFrame,
                "daily_meta_df": pd.DataFrame,
                "run_dir": str,
                "run_timestamp": str,
            }
        """
        if self.config.save_outputs:
            self.run_dir.mkdir(parents=True, exist_ok=True)

        df_events = self._load_events(events_df)
        df_events = self._normalize_event_table(df_events)
        daily_meta_df = self._aggregate_daily_metadata(df_events)

        if self.config.save_outputs:
            self._save_outputs(df_events=df_events, daily_meta_df=daily_meta_df)

        self._print_summary(
            input_events=len(df_events),
            output_rows=len(daily_meta_df),
        )

        return {
            "events_df": df_events,
            "daily_meta_df": daily_meta_df,
            "run_dir": str(self.run_dir),
            "run_timestamp": self.run_timestamp,
        }

    def _load_events(self, events_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        if events_df is not None:
            return events_df.copy()

        if not self.config.input_csv:
            raise ValueError("Either events_df must be provided or input_csv must be set.")

        input_path = Path(self.config.input_csv)
        if not input_path.exists():
            raise FileNotFoundError(f"Cannot find input_csv: {input_path}")

        return pd.read_csv(input_path)

    def _normalize_event_table(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        required_time_cols = ["start_time", "end_time"]
        for col in required_time_cols:
            if col not in out.columns:
                raise ValueError(f"Input events table must contain '{col}' column.")

        if "StationId" not in out.columns:
            raise ValueError("Input events table must contain 'StationId' column.")

        out["start_time"] = pd.to_datetime(out["start_time"], errors="coerce")
        out["end_time"] = pd.to_datetime(out["end_time"], errors="coerce")
        out = out.dropna(subset=["start_time", "end_time"]).copy()

        out["date"] = out["start_time"].dt.date.astype(str)

        if "event_id" not in out.columns:
            out["event_id"] = range(1, len(out) + 1)

        if "alarm_description" not in out.columns:
            out["alarm_description"] = "Unknown alarm"

        if "stopping_alarm" not in out.columns:
            out["stopping_alarm"] = 0

        if "alarm_code_mode" not in out.columns:
            out["alarm_code_mode"] = 0

        if "duration_min" not in out.columns:
            out["duration_min"] = 0

        out["stopping_alarm"] = pd.to_numeric(
            out["stopping_alarm"], errors="coerce"
        ).fillna(0).astype(int)

        out["alarm_code_mode"] = pd.to_numeric(
            out["alarm_code_mode"], errors="coerce"
        ).fillna(0).astype(int)

        out["duration_min"] = pd.to_numeric(
            out["duration_min"], errors="coerce"
        ).fillna(0)

        out["severity_level"] = out.apply(self._severity_from_row, axis=1)
        out["severity_text"] = out["severity_level"].map(self._severity_to_text)

        return out.sort_values(["StationId", "start_time"]).reset_index(drop=True)

    @staticmethod
    def _severity_from_row(row: pd.Series) -> int:
        """
        粗略严重度分级：
        3 = ALARM       停机类
        2 = ATTENTION   非停机但持续较长或有明确 alarm
        1 = INFO        很短的小异常
        0 = NORMAL
        """
        stopping = int(row.get("stopping_alarm", 0))
        duration = float(row.get("duration_min", 0))
        alarm_code = int(row.get("alarm_code_mode", 0))

        if stopping == 1:
            return 3
        if alarm_code != 0 and duration >= 60:
            return 2
        if alarm_code != 0 or duration > 0:
            return 1
        return 0

    @staticmethod
    def _severity_to_text(level: int) -> str:
        return {
            0: "NORMAL",
            1: "INFO",
            2: "ATTENTION",
            3: "ALARM",
        }.get(level, "UNKNOWN")

    @staticmethod
    def _unique_join(series) -> str:
        vals = []
        for x in series:
            if pd.isna(x):
                continue
            x = str(x).strip()
            if x and x not in vals:
                vals.append(x)
        return " | ".join(vals)

    @staticmethod
    def _daily_health_label(row: pd.Series) -> str:
        if row["stopping_event_count"] >= 1:
            return "ALARM"
        if row["event_count"] >= 3:
            return "ATTENTION"
        if row["total_abnormal_minutes"] >= 120:
            return "ATTENTION"
        if row["event_count"] >= 1:
            return "INFO"
        return "NORMAL"

    def _build_summary_hint(self, row: pd.Series) -> str:
        parts = [
            f"{row['date']} turbine {row['StationId']}",
            f"{row['event_count']} abnormal events",
            f"total abnormal duration {int(row['total_abnormal_minutes'])} min",
        ]

        if int(row["stopping_event_count"]) > 0:
            parts.append(f"{row['stopping_event_count']} stopping events")

        if str(row["alarm_codes"]).strip():
            parts.append(f"alarm codes: {row['alarm_codes']}")

        first_time = str(row.get("first_event_time", "")).strip()
        last_time = str(row.get("last_event_time", "")).strip()
        if first_time and last_time and first_time != "nan" and last_time != "nan":
            parts.append(f"active window {first_time} to {last_time}")

        parts.append(f"health label: {row['health_label']}")
        return "; ".join(parts)

    def _aggregate_daily_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        grouped = df.groupby(["StationId", "date"], as_index=False)

        daily = grouped.agg(
            event_count=("event_id", "count"),
            total_abnormal_minutes=("duration_min", "sum"),
            max_single_event_minutes=("duration_min", "max"),
            stopping_event_count=("stopping_alarm", "sum"),
            distinct_alarm_code_count=(
                "alarm_code_mode",
                lambda s: len(set(int(x) for x in s if int(x) != 0)),
            ),
            alarm_codes=(
                "alarm_code_mode",
                lambda s: self._unique_join([int(x) for x in s if int(x) != 0]),
            ),
            alarm_descriptions=("alarm_description", self._unique_join),
            top_severity_level=("severity_level", "max"),
            first_event_dt=("start_time", "min"),
            last_event_dt=("end_time", "max"),
        )

        daily["top_severity"] = daily["top_severity_level"].map(self._severity_to_text)
        daily["first_event_time"] = pd.to_datetime(daily["first_event_dt"]).dt.strftime("%H:%M")
        daily["last_event_time"] = pd.to_datetime(daily["last_event_dt"]).dt.strftime("%H:%M")

        daily["health_label"] = daily.apply(self._daily_health_label, axis=1)
        daily["summary_hint"] = daily.apply(self._build_summary_hint, axis=1)

        daily = daily[
            [
                "StationId",
                "date",
                "event_count",
                "total_abnormal_minutes",
                "max_single_event_minutes",
                "stopping_event_count",
                "distinct_alarm_code_count",
                "alarm_codes",
                "alarm_descriptions",
                "top_severity",
                "first_event_time",
                "last_event_time",
                "health_label",
                "summary_hint",
            ]
        ]

        return daily.sort_values(["StationId", "date"]).reset_index(drop=True)

    def _save_outputs(self, df_events: pd.DataFrame, daily_meta_df: pd.DataFrame) -> None:
        df_events.to_csv(self.run_dir / "normalized_anomaly_events.csv", index=False)
        daily_meta_df.to_csv(
            self.run_dir / "daily_turbine_metadata.csv",
            index=False,
            encoding="utf-8-sig",
        )

        summary = pd.DataFrame(
            [
                {
                    "run_timestamp": self.run_timestamp,
                    "input_events": len(df_events),
                    "output_rows": len(daily_meta_df),
                }
            ]
        )
        summary.to_csv(self.run_dir / "run_summary.csv", index=False)

    def _print_summary(self, input_events: int, output_rows: int) -> None:
        print("=" * 60)
        print(f"[{self.stage_name}] done")
        print("=" * 60)
        print(f"Run timestamp : {self.run_timestamp}")
        print(f"Output dir    : {self.run_dir}")
        print(f"Input events  : {input_events}")
        print(f"Output rows   : {output_rows}")