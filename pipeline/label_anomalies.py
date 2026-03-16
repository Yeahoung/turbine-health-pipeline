from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import pandas as pd


@dataclass
class AnomalyExtractorConfig:
    input_csv: str
    alarm_desc_csv: str
    output_root: str = "out"
    time_gap_minutes: int = 10
    save_outputs: bool = True


class AnomalyExtractor:
    """
    Stage 1:
    - Load raw SCADA data
    - Label anomaly rows
    - Build anomaly events
    - Save timestamped outputs for traceability
    """

    def __init__(self, config: AnomalyExtractorConfig) -> None:
        self.config = config
        self.stage_name = "label_anomalies"
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = (
            Path(self.config.output_root) / self.stage_name / self.run_timestamp
        )

    def run(self) -> Dict[str, Any]:
        """
        Execute the full extraction stage.

        Returns:
            {
                "df_labeled": pd.DataFrame,
                "anomaly_rows": pd.DataFrame,
                "events_df": pd.DataFrame,
                "run_dir": str,
                "run_timestamp": str,
            }
        """
        if self.config.save_outputs:
            self.run_dir.mkdir(parents=True, exist_ok=True)

        alarm_map, stop_map = self._load_alarm_table(self.config.alarm_desc_csv)
        df_raw = self._load_data(self.config.input_csv)

        df_labeled = self._label_anomalies(df_raw, alarm_map, stop_map)
        anomaly_rows = df_labeled[df_labeled["is_anomaly"]].copy()
        events_df = self._build_events(df_labeled)

        if self.config.save_outputs:
            self._save_outputs(
                df_raw=df_raw,
                df_labeled=df_labeled,
                anomaly_rows=anomaly_rows,
                events_df=events_df,
            )

        self._print_summary(
            total_rows=len(df_raw),
            anomaly_rows=len(anomaly_rows),
            events=len(events_df),
        )

        return {
            "df_labeled": df_labeled,
            "anomaly_rows": anomaly_rows,
            "events_df": events_df,
            "run_dir": str(self.run_dir),
            "run_timestamp": self.run_timestamp,
        }

    def _load_alarm_table(self, path: str) -> tuple[dict[int, str], dict[int, int]]:
        alarm_df = pd.read_csv(path)

        if "Alarm Code" not in alarm_df.columns:
            raise ValueError("Alarm description CSV must contain 'Alarm Code' column.")
        if "Description" not in alarm_df.columns:
            raise ValueError("Alarm description CSV must contain 'Description' column.")
        if "Stopping" not in alarm_df.columns:
            alarm_df["Stopping"] = 0

        alarm_df["Alarm Code"] = pd.to_numeric(
            alarm_df["Alarm Code"], errors="coerce"
        ).fillna(0).astype(int)

        alarm_df["Stopping"] = pd.to_numeric(
            alarm_df["Stopping"], errors="coerce"
        ).fillna(0).astype(int)

        alarm_map = dict(zip(alarm_df["Alarm Code"], alarm_df["Description"]))
        stop_map = dict(zip(alarm_df["Alarm Code"], alarm_df["Stopping"]))

        return alarm_map, stop_map

    def _load_data(self, csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)

        if "TimeStamp" not in df.columns:
            raise ValueError("Input CSV must contain 'TimeStamp' column.")
        if "StationId" not in df.columns:
            raise ValueError("Input CSV must contain 'StationId' column.")

        df["TimeStamp"] = pd.to_datetime(df["TimeStamp"], errors="coerce")
        df = df.dropna(subset=["TimeStamp"]).copy()

        df = df.sort_values(["StationId", "TimeStamp"]).reset_index(drop=True)
        return df

    def _label_anomalies(
        self,
        df: pd.DataFrame,
        alarm_map: dict[int, str],
        stop_map: dict[int, int],
    ) -> pd.DataFrame:
        out = df.copy()

        numeric_cols = [
            "wtc_AlarmCde_endvalue",
            "wtc_ScFrsErr_endvalue",
            "wtc_OpCode_endvalue",
            "wtc_ScEnvSto_endvalue",
            "wtc_ScComSto_endvalue",
            "wtc_ScTurSto_endvalue",
            "wtc_ScGrdSto_endvalue",
        ]

        for col in numeric_cols:
            if col not in out.columns:
                out[col] = 0
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)

        out["flag_alarm_code"] = out["wtc_AlarmCde_endvalue"] != 0
        out["flag_frs_error"] = out["wtc_ScFrsErr_endvalue"] != 0
        out["flag_opcode_nonzero"] = out["wtc_OpCode_endvalue"] != 0

        out["flag_any_stop"] = (
            (out["wtc_ScEnvSto_endvalue"] != 0)
            | (out["wtc_ScComSto_endvalue"] != 0)
            | (out["wtc_ScTurSto_endvalue"] != 0)
            | (out["wtc_ScGrdSto_endvalue"] != 0)
        )

        out["is_anomaly"] = (
            out["flag_alarm_code"]
            | out["flag_frs_error"]
            | out["flag_opcode_nonzero"]
            | out["flag_any_stop"]
        )

        out["alarm_code"] = out["wtc_AlarmCde_endvalue"].astype(int)

        out["alarm_description"] = out["alarm_code"].map(
            lambda x: alarm_map.get(x, "Unknown alarm")
        )

        out["is_stopping_alarm"] = out["alarm_code"].map(
            lambda x: int(stop_map.get(x, 0))
        )

        return out

    def _build_events(self, df_labeled: pd.DataFrame) -> pd.DataFrame:
        anomaly_df = df_labeled[df_labeled["is_anomaly"]].copy()

        if anomaly_df.empty:
            return pd.DataFrame(
                columns=[
                    "event_id",
                    "StationId",
                    "start_time",
                    "end_time",
                    "duration_min",
                    "rows",
                    "alarm_code_mode",
                    "alarm_description",
                    "stopping_alarm",
                ]
            )

        anomaly_df = anomaly_df.sort_values(
            ["StationId", "TimeStamp"]
        ).reset_index(drop=True)

        anomaly_df["prev_station"] = anomaly_df["StationId"].shift()
        anomaly_df["prev_time"] = anomaly_df["TimeStamp"].shift()

        gap = (
            anomaly_df["TimeStamp"] - anomaly_df["prev_time"]
        ).dt.total_seconds() / 60.0

        anomaly_df["new_event"] = (
            (anomaly_df["StationId"] != anomaly_df["prev_station"])
            | (gap > self.config.time_gap_minutes)
        )

        anomaly_df["event_id"] = anomaly_df["new_event"].cumsum()

        events = []
        for eid, g in anomaly_df.groupby("event_id"):
            start = g["TimeStamp"].iloc[0]
            end = g["TimeStamp"].iloc[-1]

            alarm_mode = self._safe_mode(g["alarm_code"], default=0)
            desc_mode = self._safe_mode(g["alarm_description"], default="Unknown alarm")

            events.append(
                {
                    "event_id": int(eid),
                    "StationId": int(g["StationId"].iloc[0]),
                    "start_time": start,
                    "end_time": end,
                    "duration_min": int((end - start).total_seconds() / 60),
                    "rows": int(len(g)),
                    "alarm_code_mode": int(alarm_mode),
                    "alarm_description": str(desc_mode),
                    "stopping_alarm": int(g["is_stopping_alarm"].max()),
                }
            )

        return pd.DataFrame(events)

    @staticmethod
    def _safe_mode(series: pd.Series, default):
        mode = series.mode(dropna=True)
        if len(mode) == 0:
            return default
        return mode.iloc[0]

    def _save_outputs(
        self,
        df_raw: pd.DataFrame,
        df_labeled: pd.DataFrame,
        anomaly_rows: pd.DataFrame,
        events_df: pd.DataFrame,
    ) -> None:
        df_raw.to_csv(self.run_dir / "raw_sorted.csv", index=False)
        df_labeled.to_csv(self.run_dir / "all_rows_labeled.csv", index=False)
        anomaly_rows.to_csv(self.run_dir / "anomaly_rows.csv", index=False)
        events_df.to_csv(self.run_dir / "anomaly_events.csv", index=False)

        summary = pd.DataFrame(
            [
                {
                    "run_timestamp": self.run_timestamp,
                    "input_csv": self.config.input_csv,
                    "alarm_desc_csv": self.config.alarm_desc_csv,
                    "time_gap_minutes": self.config.time_gap_minutes,
                    "total_rows": len(df_raw),
                    "anomaly_rows": len(anomaly_rows),
                    "event_count": len(events_df),
                }
            ]
        )
        summary.to_csv(self.run_dir / "run_summary.csv", index=False)

    def _print_summary(self, total_rows: int, anomaly_rows: int, events: int) -> None:
        print("=" * 60)
        print(f"[{self.stage_name}] done")
        print("=" * 60)
        print(f"Run timestamp : {self.run_timestamp}")
        print(f"Output dir    : {self.run_dir}")
        print(f"Rows          : {total_rows}")
        print(f"Anomaly rows  : {anomaly_rows}")
        print(f"Events        : {events}")