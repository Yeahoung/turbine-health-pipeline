from pipeline.label_anomalies import AnomalyExtractor, AnomalyExtractorConfig
from pipeline.build_metadata import DailyMetadataBuilder, DailyMetadataBuilderConfig
from pipeline.generate_reports import DailyReportGenerator, DailyReportGeneratorConfig


def main():
    extractor = AnomalyExtractor(
        AnomalyExtractorConfig(
            input_csv="data/raw/2016_01_01.csv",
            alarm_desc_csv="data/raw/Hill_of_Towie_alarms_description.csv",
            output_root="out",
            time_gap_minutes=10,
            save_outputs=True,
        )
    )
    extract_result = extractor.run()

    metadata_builder = DailyMetadataBuilder(
        
        DailyMetadataBuilderConfig(
            output_root="out",
            save_outputs=True,
        )
    )
    meta_result = metadata_builder.run(events_df=extract_result["events_df"])

    report_generator = DailyReportGenerator(
        DailyReportGeneratorConfig(
            local_model_path="qwen_0_5_fine",       # for local model testing, replace with the actual path to the model
            hf_repo_id="LAND223/qwen_0_5_fine_report_generator",  
            hf_token=None,                          
            output_root="out",
            save_outputs=True,
            force_cpu=False,                        # if True, forces using CPU even if GPU is available (useful for testing)
            max_new_tokens=160,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
        )
    )
    report_result = report_generator.run(
        daily_meta_df=meta_result["daily_meta_df"]
    )

    print(report_result["reports_df"].head())


if __name__ == "__main__":
    main()