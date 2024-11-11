from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    preprocess_analyst_estimates,
    preprocess_earning_calendar,
    preprocess_earnings_calls,
    preprocess_fred_data,
    preprocess_historical_price_full,
    betas_table_model_input,
    preprocess_etf_prices,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_fred_data,
                inputs="fred_data",
                outputs="preprocessed_fred_data",
                name="preprocess_fred_data_node",
            ),
            node(
                func=preprocess_etf_prices,
                inputs="etf_prices",
                outputs="preprocessed_etf_prices",
                name="preprocess_etf_prices_node",
            ),
            node(
                func=preprocess_historical_price_full,
                inputs="historical_price_full",
                outputs="preprocessed_historical_price_full",
                name="preprocess_historical_price_full_node",
            ),
            node(
                func=preprocess_analyst_estimates,
                inputs="analyst_estimates",
                outputs="preprocessed_analyst_estimates",
                name="preprocess_analyst_estimates_node",
            ),
            node(
                func=preprocess_earning_calendar,
                inputs="earning_calendar",
                outputs="preprocessed_earning_calendar",
                name="preprocess_earning_calendar_node",
            ),
            # node(
            #     func=preprocess_earnings_calls,
            #     inputs=[
            #         "earnings_calls_2020",
            #         "earnings_calls_2021",
            #         "earnings_calls_2022",
            #         "earnings_calls_2023",
            #         "earnings_calls_2024",
            #     ],
            #     outputs=[
            #         "preprocessed_earnings_calls",
            #         "preprocessed_earnings_calls_statements",
            #     ],
            #     name="preprocessed_earnings_calls_node",
            # ),
            node(
                func=betas_table_model_input,
                inputs=[
                    "preprocessed_fred_data",
                    "preprocessed_historical_price_full",
                    "preprocessed_etf_prices",
                    "preprocessed_earning_calendar",
                ],
                outputs="betas_table",
                name="betas_table_model_input_node",
            ),
        ]
    )
