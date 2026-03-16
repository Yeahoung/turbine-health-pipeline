from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


DEFAULT_INSTRUCTION = (
    "Read the wind turbine daily health metadata and generate a short "
    "maintenance-oriented summary with a health label and action advice."
)

DEFAULT_OUTPUT_FORMAT = (
    "Write the report strictly in English."
)


@dataclass
class DailyReportGeneratorConfig:
    # Local model directory, e.g. "qwen_0_5_fine"
    local_model_path: str = "qwen_0_5_fine"

    # HF repo id fallback, e.g. "landnesses/qwen_0_5_fine"
    hf_repo_id: Optional[str] = None

    # Optional HF token for private repos
    hf_token: Optional[str] = None

    # Input daily metadata CSV, optional if passing dataframe in run()
    input_csv: Optional[str] = None

    output_root: str = "out"
    save_outputs: bool = True

    instruction: str = DEFAULT_INSTRUCTION
    output_format_instruction: str = DEFAULT_OUTPUT_FORMAT

    max_input_length: int = 2048
    max_new_tokens: int = 160
    do_sample: bool = True
    temperature: float = 0
    top_p: float = 0.9

    force_cpu: bool = False
    trust_remote_code: bool = True


class DailyReportGenerator:
    """
    Stage 3:
    - Load local fine-tuned LLM (or fall back to Hugging Face Hub)
    - Read daily turbine metadata
    - Generate short maintenance-oriented natural language reports
    - Save timestamped outputs
    """

    def __init__(self, config: DailyReportGeneratorConfig) -> None:
        self.config = config
        self.stage_name = "generate_reports"
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(self.config.output_root) / self.stage_name / self.run_timestamp

        self.device = self._resolve_device()
        self.model_source = self._resolve_model_source()

        self.tokenizer = None
        self.model = None

    def run(self, daily_meta_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        if self.config.save_outputs:
            self.run_dir.mkdir(parents=True, exist_ok=True)

        df = self._load_daily_metadata(daily_meta_df)
        df = self._normalize_metadata(df)

        self._load_model()

        generated_rows = []
        total = len(df)

        for idx, row in df.iterrows():
            prompt = self._build_prompt(row)
            report_text = self._generate_text(prompt)
            report_text = self._postprocess_output(report_text)

            generated_rows.append(
                {
                    "StationId": row["StationId"],
                    "date": row["date"],
                    "rule_health_label": row.get("health_label", ""),
                    "summary_hint": row.get("summary_hint", ""),
                    "prompt": prompt,
                    "report_text": report_text,
                }
            )

            print(
                f"[{self.stage_name}] {idx + 1}/{total} | "
                f"StationId={row['StationId']} | date={row['date']}"
            )

        reports_df = pd.DataFrame(generated_rows)

        if self.config.save_outputs:
            self._save_outputs(df, reports_df)

        self._print_summary(
            input_rows=len(df),
            output_rows=len(reports_df),
        )

        return {
            "daily_meta_df": df,
            "reports_df": reports_df,
            "run_dir": str(self.run_dir),
            "run_timestamp": self.run_timestamp,
            "device": self.device,
            "model_source": self.model_source,
        }

    def _resolve_device(self) -> str:
        if self.config.force_cpu:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _resolve_model_source(self) -> str:
        local_path = Path(self.config.local_model_path)

        if local_path.exists() and local_path.is_dir():
            print(f"[{self.stage_name}] using local model: {local_path}")
            return str(local_path)

        if self.config.hf_repo_id:
            print(
                f"[{self.stage_name}] local model not found, "
                f"falling back to HF repo: {self.config.hf_repo_id}"
            )
            return self.config.hf_repo_id

        raise FileNotFoundError(
            "Local model directory not found and hf_repo_id is not set. "
            f"Missing local path: {local_path}"
        )

    def _load_daily_metadata(self, daily_meta_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        if daily_meta_df is not None:
            return daily_meta_df.copy()

        if not self.config.input_csv:
            raise ValueError("Either daily_meta_df must be provided or input_csv must be set.")

        input_path = Path(self.config.input_csv)
        if not input_path.exists():
            raise FileNotFoundError(f"Cannot find input_csv: {input_path}")

        return pd.read_csv(input_path)

    def _normalize_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        required_cols = ["StationId", "date"]
        for col in required_cols:
            if col not in out.columns:
                raise ValueError(f"Daily metadata must contain '{col}' column.")

        optional_defaults = {
            "event_count": 0,
            "total_abnormal_minutes": 0,
            "max_single_event_minutes": 0,
            "stopping_event_count": 0,
            "distinct_alarm_code_count": 0,
            "alarm_codes": "",
            "alarm_descriptions": "",
            "top_severity": "",
            "first_event_time": "",
            "last_event_time": "",
            "health_label": "UNKNOWN",
            "summary_hint": "",
        }

        for col, default in optional_defaults.items():
            if col not in out.columns:
                out[col] = default

        return out.sort_values(["StationId", "date"]).reset_index(drop=True)

    def _load_model(self) -> None:
        print("=" * 60)
        print(f"[{self.stage_name}] loading model")
        print("=" * 60)
        print(f"Model source : {self.model_source}")
        print(f"Device       : {self.device}")

        tokenizer_kwargs = {
            "trust_remote_code": self.config.trust_remote_code,
        }
        model_kwargs = {
            "trust_remote_code": self.config.trust_remote_code,
        }

        # hf_token only matters for HF Hub loading, especially private repos
        if self.config.hf_token:
            tokenizer_kwargs["token"] = self.config.hf_token
            model_kwargs["token"] = self.config.hf_token

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_source,
            **tokenizer_kwargs,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.device == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_source,
                torch_dtype=torch.float16,
                **model_kwargs,
            ).to("cuda")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_source,
                torch_dtype=torch.float32,
                **model_kwargs,
            ).to("cpu")

        self.model.eval()

    def _build_prompt(self, row: pd.Series) -> str:
        metadata_lines = [
            f"StationId: {row['StationId']}",
            f"Date: {row['date']}",
            f"Event count: {row.get('event_count', 0)}",
            f"Total abnormal minutes: {row.get('total_abnormal_minutes', 0)}",
            f"Max single event minutes: {row.get('max_single_event_minutes', 0)}",
            f"Stopping event count: {row.get('stopping_event_count', 0)}",
            f"Distinct alarm code count: {row.get('distinct_alarm_code_count', 0)}",
            f"Alarm codes: {row.get('alarm_codes', '')}",
            f"Alarm descriptions: {row.get('alarm_descriptions', '')}",
            f"Top severity: {row.get('top_severity', '')}",
            f"First event time: {row.get('first_event_time', '')}",
            f"Last event time: {row.get('last_event_time', '')}",
            f"Rule-based health label: {row.get('health_label', '')}",
            f"Summary hint: {row.get('summary_hint', '')}",
        ]
        user_block = "\n".join(metadata_lines)

        # Prefer chat template if available
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [
                {
                    "role": "system",
                    "content": (
                        f"{self.config.instruction}\n\n"
                        f"{self.config.output_format_instruction}"
                    ),
                },
                {
                    "role": "user",
                    "content": user_block,
                },
            ]
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass

        # Plain fallback prompt
        return (
            f"Instruction:\n{self.config.instruction}\n\n"
            f"Output format:\n{self.config.output_format_instruction}\n\n"
            f"Input metadata:\n{user_block}\n\n"
            f"Output:\n"
        )

    def _generate_text(self, prompt: str) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_input_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=self.config.do_sample,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text.strip()

    def _postprocess_output(self, text: str) -> str:
        text = text.strip()

        # Mild cleanup in case the small model rambles
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        # Keep only the first Health label / Summary / Advice trio if present
        health = None
        summary = None
        advice = None

        for line in lines:
            low = line.lower()
            if health is None and low.startswith("health label:"):
                health = line
            elif summary is None and low.startswith("summary:"):
                summary = line
            elif advice is None and low.startswith("advice:"):
                advice = line

        if health or summary or advice:
            result = []
            if health:
                result.append(health)
            if summary:
                result.append(summary)
            if advice:
                result.append(advice)
            return "\n".join(result)

        # fallback: return first few non-empty lines
        return "\n".join(lines[:6])

    def _save_outputs(self, daily_meta_df: pd.DataFrame, reports_df: pd.DataFrame) -> None:
        daily_meta_df.to_csv(
            self.run_dir / "input_daily_metadata.csv",
            index=False,
            encoding="utf-8-sig",
        )

        reports_df.to_csv(
            self.run_dir / "generated_reports.csv",
            index=False,
            encoding="utf-8-sig",
        )

        with open(self.run_dir / "generated_reports.txt", "w", encoding="utf-8") as f:
            for _, row in reports_df.iterrows():
                f.write(f"StationId: {row['StationId']}\n")
                f.write(f"Date: {row['date']}\n")
                f.write("Generated report:\n")
                f.write(f"{row['report_text']}\n")
                f.write("\n" + "=" * 80 + "\n\n")

        summary = pd.DataFrame(
            [
                {
                    "run_timestamp": self.run_timestamp,
                    "model_source": self.model_source,
                    "device": self.device,
                    "input_rows": len(daily_meta_df),
                    "output_rows": len(reports_df),
                }
            ]
        )
        summary.to_csv(self.run_dir / "run_summary.csv", index=False)

    def _print_summary(self, input_rows: int, output_rows: int) -> None:
        print("=" * 60)
        print(f"[{self.stage_name}] done")
        print("=" * 60)
        print(f"Run timestamp : {self.run_timestamp}")
        print(f"Output dir    : {self.run_dir}")
        print(f"Model source  : {self.model_source}")
        print(f"Device        : {self.device}")
        print(f"Input rows    : {input_rows}")
        print(f"Output rows   : {output_rows}")