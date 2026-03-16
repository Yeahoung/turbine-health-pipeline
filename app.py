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

# Use the default alarm description file in the repository
ALARM_DESC_DEFAULT = "data/raw/Hill_of_Towie_alarms_description.csv"

col1, col2 = st.columns([1, 2])

with col1:
    st.header("1. Upload Data")
    uploaded_scada = st.file_uploader("Upload SCADA Data (CSV)", type="csv")
    
    # We provide the default sample file path as a tip
    st.info("Don't have a file? Try uploading `data/raw/2016_01_01.csv` from this project's folder.")
    
    st.header("2. Pipeline Settings")
    time_gap = st.slider("Anomaly Time Gap (minutes)", 1, 60, 10, help="Time gap to merge continuous abnormal events.")
    temperature = st.slider("LLM Temperature", 0.0, 1.0, 0.2, help="Higher temperature makes output more random, lower makes it more deterministic.")
    
    run_btn = st.button("🚀 Run Pipeline", type="primary", use_container_width=True)

with col2:
    st.header("3. Pipeline Results")
    
    if run_btn:
        if uploaded_scada is None:
            st.warning("Please upload SCADA CSV data first to run the pipeline.")
        else:
            with st.spinner("Pipeline running... This might take a minute as the AI generates the report."):
                # Save uploaded file to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_scada:
                    tmp_scada.write(uploaded_scada.getvalue())
                    tmp_scada_path = tmp_scada.name
                
                try:
                    # Step 1: Anomaly Extraction
                    st.text("Step 1/3: Extracting Anomalies...")
                    extractor = AnomalyExtractor(
                        AnomalyExtractorConfig(
                            input_csv=tmp_scada_path,
                            alarm_desc_csv=ALARM_DESC_DEFAULT,
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
                                force_cpu=True, # Force CPU to match free Hugging Face Spaces environment
                                temperature=temperature,
                            )
                        )
                        report_result = report_generator.run(daily_meta_df=daily_meta_df)
                        reports_df = report_result["reports_df"]
                        
                        st.success("✨ Pipeline completed successfully!")
                        
                        # Display Results
                        st.subheader("📊 Output Reports")
                        for idx, row in reports_df.iterrows():
                            # Determine color based on health label
                            indicator = "🟢" if row['health_label'] == 'HEALTHY' else "🟡" if row['health_label'] == 'ATTENTION' else "🔴"
                            
                            with st.expander(f"{indicator} Turbine {row['StationId']} - {row['date']} (Status: {row['health_label']})", expanded=True):
                                col_a, col_b = st.columns([1, 1])
                                with col_a:
                                    st.markdown("**📋 AI Maintenance Report:**")
                                    st.info(row['report_text'])
                                with col_b:
                                    st.markdown("**⚙️ Metadata Summary:**")
                                    st.markdown(f"- **Alarm Events (Level 3):** {row['severity_3_events']}")
                                    st.markdown(f"- **Attention Events (Level 2):** {row['severity_2_events']}")
                                    st.markdown(f"- **Total Abnormal Duration:** {row['total_abnormal_duration_minutes']:.1f} minutes")
                                    st.markdown(f"- **Unique Alarm Codes:** `{row['unique_alarm_codes']}`")
                    
                except Exception as e:
                    st.error(f"Error during execution: {e}")
                finally:
                    # cleanup
                    if os.path.exists(tmp_scada_path):
                        os.remove(tmp_scada_path)
