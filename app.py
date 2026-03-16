import streamlit as st
import pandas as pd
import tempfile
import os
from pipeline.label_anomalies import AnomalyExtractor, AnomalyExtractorConfig
from pipeline.build_metadata import DailyMetadataBuilder, DailyMetadataBuilderConfig
from pipeline.generate_reports import DailyReportGenerator, DailyReportGeneratorConfig

st.set_page_config(page_title="Turbine Health Monitor", page_icon="🌬️", layout="wide")

st.title("🌬️ Wind Turbine Health Monitor")
st.write("A lightweight pipeline for extracting wind turbine SCADA anomalies and generating AI-powered maintenance reports.")

# IMPORTANT NOTE: In Hugging Face spaces demo, users must provide their own alarm dict
st.warning("Please upload both SCADA Data and the Alarm Description Dictionary.")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("1. Upload Data")
    uploaded_scada = st.file_uploader("Upload SCADA Data (CSV)", type="csv")
    uploaded_alarm = st.file_uploader("Upload Alarm Description (CSV)", type="csv")
    
    st.header("2. Pipeline Settings")
    time_gap = st.slider("Anomaly Time Gap (minutes)", 1, 60, 10, help="Time gap to merge continuous abnormal events.")
    temperature = st.slider("LLM Temperature", 0.0, 1.0, 0.2, help="Higher temperature makes output more random.")
    
    run_btn = st.button("🚀 Run Pipeline", type="primary", use_container_width=True)

with col2:
    st.header("3. Pipeline Results")
    
    if run_btn:
        if uploaded_scada is None or uploaded_alarm is None:
            st.error("Please upload BOTH the SCADA CSV data AND the Alarm Description CSV.")
        else:
            with st.spinner("Pipeline running... This might take a minute as the AI generates the report."):
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_scada, \
                     tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_alarm:
                    
                    tmp_scada.write(uploaded_scada.getvalue())
                    tmp_scada_path = tmp_scada.name
                    
                    tmp_alarm.write(uploaded_alarm.getvalue())
                    tmp_alarm_path = tmp_alarm.name
                
                try:
                    # Step 1: Anomaly Extraction
                    st.text("Step 1/3: Extracting Anomalies...")
                    extractor = AnomalyExtractor(
                        AnomalyExtractorConfig(
                            input_csv=tmp_scada_path,
                            alarm_desc_csv=tmp_alarm_path,
                            output_root="out",
                            time_gap_minutes=time_gap,
                            save_outputs=False
                        )
                    )
                    extract_result = extractor.run()
                    events_df = extract_result["events_df"]
                    
                    if events_df.empty:
                        st.success("No anomalies found in the uploaded data! The turbine operated normally.")
                    else:
                        # Step 2: Metadata Builder
                        st.text("Step 2/3: Building Daily Metadata...")
                        metadata_builder = DailyMetadataBuilder(
                            DailyMetadataBuilderConfig(
                                output_root="out",
                                save_outputs=False
                            )
                        )
                        meta_result = metadata_builder.run(events_df=events_df)
                        daily_meta_df = meta_result["daily_meta_df"]
                        
                        # Step 3: Report Generator
                        st.text("Step 3/3: Generating AI Maintenance Reports (Loading Model)...")
                        report_generator = DailyReportGenerator(
                            DailyReportGeneratorConfig(
                                local_model_path="qwen_0_5_fine",  
                                hf_repo_id="LAND223/qwen_0_5_fine_report_generator",  
                                hf_token=None,                          
                                output_root="out",
                                save_outputs=False,
                                force_cpu=True, 
                                temperature=temperature,
                            )
                        )
                        report_result = report_generator.run(daily_meta_df=daily_meta_df)
                        reports_df = report_result["reports_df"]
                        
                        st.success("✨ Pipeline completed successfully!")
                        
                        # Display Results
                        st.subheader("📊 Output Reports")
                        
                        # Merge text reports with the metadata so we have all columns available
                        merged_df = pd.merge(reports_df, daily_meta_df, on=["StationId", "date"], how="left")
                        
                        for idx, row in merged_df.iterrows():
                            # health_label comes from daily_meta_df now
                            h_label = row.get('health_label', 'UNKNOWN')
                            indicator = "🟢" if h_label == 'NORMAL' else "🟡" if h_label in ['INFO', 'ATTENTION'] else "🔴"
                            
                            with st.expander(f"{indicator} Turbine {row['StationId']} - {row['date']} (Status: {h_label})", expanded=True):
                                col_a, col_b = st.columns([1, 1])
                                with col_a:
                                    st.markdown("**📋 AI Maintenance Report:**")
                                    st.info(row['report_text'])
                                with col_b:
                                    st.markdown("**⚙️ Metadata Summary:**")
                                    st.markdown(f"- **Stopping Events (ALARM):** {row['stopping_event_count']}")
                                    st.markdown(f"- **Total Abnormal Events:** {row['event_count']}")
                                    st.markdown(f"- **Total Abnormal Duration:** {row['total_abnormal_minutes']:.1f} minutes")
                                    st.markdown(f"- **Unique Alarm Codes:** `{row['alarm_codes']}`")
                                    st.markdown(f"- **Top Severity:** {row['top_severity']}")
                except Exception as e:
                    st.error(f"Error during execution: {e}")
                finally:
                    if os.path.exists(tmp_scada_path): os.remove(tmp_scada_path)
                    if os.path.exists(tmp_alarm_path): os.remove(tmp_alarm_path)
