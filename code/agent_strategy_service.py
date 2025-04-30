import ollama
import json
import re
import pandas as pd

class AgentService:
    def __init__(self, model_name="gemma3"):
        self.model_name = model_name

    def generate_strategies(self, user_guidance: str, kmeans_stats_path: str) -> list:
        """
        Generate trading strategies using a local LLM.

        Args:
            user_guidance (str): Additional prompt text from the user.
            kmeans_stats_path (str): Path to the kmeans_cluster_stats.csv file.

        Returns:
            List[dict]: Generated regime strategies
        """
        # Load and summarize cluster stats
        kmeans_stats = pd.read_csv(kmeans_stats_path)

        # Build the prompt
        prompt = f"""
        You are a financial quant researcher designing trading strategies.

        Here are the provided cluster stats:
        {kmeans_stats.to_markdown()}

        Additional User Guidance:
        {user_guidance}

        Based on this, for each cluster, recommend:
        - One feature to LONG (higher is better)
        - One feature to SHORT (lower is better)
        - A short, descriptive strategy name

        Output STRICTLY as a JSON list in this format:

        [
          {{ "regime": 0, "long_feature": "value_score", "short_feature": "rel_volume", "strategy_name": "Long Value, Short High Volume" }},
          ...
        ]

        Do not include markdown or any explanation. Only output valid JSON.
        """

        # Send to LLM
        response = ollama.chat(model=self.model_name, messages=[
            {"role": "system", "content": "You are a financial quant strategist."},
            {"role": "user", "content": prompt}
        ])

        raw_content = response.message.content

        # Clean response
        cleaned = raw_content.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[len("```json"):].strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

        # Parse JSON
        try:
            strategies = json.loads(cleaned)
        except json.JSONDecodeError as e:
            print("Failed to parse LLM JSON output:\n")
            print(cleaned)
            raise e

        return strategies