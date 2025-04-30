import json
import ollama
import pandas as pd
from typing import List

class StrategyInterpreter:
    def __init__(self, model_name="gemma3"):
        self.model_name = model_name

    def interpret_strategies(self, strategies: List[dict], metrics_df: pd.DataFrame, user_guidance: str = "") -> str:
        """
        Uses a local LLM to provide an interpretation of the best-performing strategy.

        Args:
            strategies (List[dict]): Generated strategy configurations
            metrics_df (pd.DataFrame): Strategy metrics (must include 'strategy_name', 'Sharpe', 'Max_Drawdown', 'Hit_Rate')
            user_guidance (str): Optional original user prompt for context

        Returns:
            str: LLM-generated interpretation of the results (max ~4 sentences, markdown-friendly)
        """
        top_strat = metrics_df.sort_values("Sharpe", ascending=False).head(1)

        if top_strat.empty:
            return "No strategy interpretation could be generated. Please re-run with valid strategies."

        strat_name = top_strat['strategy_name'].values[0]
        sharpe = round(top_strat['Sharpe'].values[0], 2)
        max_dd = round(top_strat['Max_Drawdown'].values[0] * 100, 1)
        hit_rate = round(top_strat['Hit_Rate'].values[0] * 100, 1)

        long_feature = next((s['long_feature'] for s in strategies if s['strategy_name'] == strat_name), None)
        short_feature = next((s['short_feature'] for s in strategies if s['strategy_name'] == strat_name), None)

        prompt = f"""
        You are an expert quantitative strategist. A backtest was run on several AI-generated trading strategies.

        The best performing strategy was:
        - Name: {strat_name}
        - Long Feature: {long_feature}
        - Short Feature: {short_feature}
        - Sharpe Ratio: {sharpe}
        - Max Drawdown: {max_dd}%
        - Hit Rate: {hit_rate}%

        User guidance: "{user_guidance}"

        Provide a concise, professional summary (max 4 sentences) that explains:
        1. Why this strategy likely worked
        2. What market inefficiency it may be exploiting
        3. Any tradeoff observed (risk, signal reliability)
        4. One short suggestion for improvement or next steps

        Format your answer using clear markdown — bold keywords, bulleted structure if useful, and no additional prefix or signature.
        """

        response = ollama.chat(model=self.model_name, messages=[
            {"role": "system", "content": "You are a quant analyst providing insightful strategic analysis."},
            {"role": "user", "content": prompt}
        ])

        return response.message.content.strip()