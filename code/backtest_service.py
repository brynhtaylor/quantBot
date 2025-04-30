import pandas as pd
import numpy as np
import os

class BacktestService:
    def __init__(self, clustered_path, merged_path, crsp_path, winsorize_limit=0.01, total_capital=1_000_000, basket_size=10):
        self.clustered_path = clustered_path
        self.merged_path = merged_path
        self.crsp_path = crsp_path
        self.winsorize_limit = winsorize_limit
        self.total_capital = total_capital
        self.basket_size = basket_size

        self.df = self._prepare_data()

    def _prepare_data(self):
        clustered = pd.read_csv(self.clustered_path, low_memory=False)
        merged = pd.read_csv(self.merged_path, low_memory=False)

        real_data = merged[['year_month', 'PERMNO', 'RET', 'volume_millions', 'PRC']]
        clustered = clustered.drop(columns=['RET', 'volume_millions', 'PRC'], errors='ignore')

        df = clustered.merge(real_data, on=['year_month', 'PERMNO'], how='left')

        # Winsorize RET
        lower = df['RET'].quantile(self.winsorize_limit)
        upper = df['RET'].quantile(1 - self.winsorize_limit)
        df['RET'] = df['RET'].clip(lower, upper)

        return df

    def _nonlinear_transaction_cost(self, trade_usd, adv_usd):
        pct_adv = trade_usd / adv_usd if adv_usd > 0 else 0
        base_cost = 0.0005
        return base_cost * (1 + (pct_adv * 5))

    def _dynamic_position_size(self, volume_millions, prc):
        adv_usd = volume_millions * 1e6 * abs(prc)
        size = np.minimum(self.total_capital / (2 * self.basket_size), adv_usd * 0.01)
        return size

    def _backtest_single_strategy(self, regime, config):
        df_regime = self.df[self.df['cluster_kmeans'] == regime]
        results = []

        months = sorted(df_regime['year_month'].unique())

        for i in range(len(months) - 1):
            month_t = months[i]
            month_tp1 = months[i + 1]

            group_t = df_regime[df_regime['year_month'] == month_t]
            group_tp1 = df_regime[df_regime['year_month'] == month_tp1]

            if group_t.empty or group_tp1.empty:
                continue

            longs = group_t.nlargest(self.basket_size, config['long_feature'])
            shorts = group_t.nsmallest(self.basket_size, config['short_feature'])

            # Ensure mutual exclusivity
            shorts = shorts[~shorts['PERMNO'].isin(longs['PERMNO'])]

            if longs.empty or shorts.empty:
                continue

            basket_returns = []

            for _, row in pd.concat([longs, shorts]).iterrows():
                next_month = group_tp1[group_tp1['PERMNO'] == row['PERMNO']]
                if next_month.empty:
                    continue

                gross_ret = next_month['RET'].values[0]

                # Skip rows with NaNs or infinite returns
                if pd.isna(gross_ret) or not np.isfinite(gross_ret):
                    continue

                # Never allow a return of -100% or worse
                if gross_ret <= -1:
                    continue  # skip this security

                # Cap monthly returns for sanity (e.g., + 100%)
                gross_ret = np.clip(gross_ret, -1, 1)

                adv_usd = row['volume_millions'] * 1e6 * abs(row['PRC'])
                position_usd = self._dynamic_position_size(row['volume_millions'], row['PRC'])
                cost = self._nonlinear_transaction_cost(position_usd, adv_usd)
                side = 1 if row['PERMNO'] in longs['PERMNO'].values else -1

                basket_returns.append(side * (gross_ret - cost))

            if len(basket_returns) >= 2:
                port_ret = np.mean(basket_returns)
            else:
                port_ret = np.nan

            results.append({'year_month': month_tp1, 'strategy_return': port_ret})

        return pd.DataFrame(results)

    def backtest_strategies(self, strategy_configs: list):
        realistic_results = []
        metrics_summary = []

        for strategy in strategy_configs:
            regime = strategy['regime']
            config = {
                'long_feature': strategy['long_feature'],
                'short_feature': strategy['short_feature'],
                'strategy_name': strategy['strategy_name']
            }

            res = self._backtest_single_strategy(regime, config)
            res['regime'] = regime
            res['strategy_name'] = config['strategy_name']
            res = res.sort_values('year_month')

            res['cumulative_return'] = (1 + res['strategy_return'].fillna(0)).cumprod()
            res['rolling_sharpe'] = res['strategy_return'].rolling(12).mean() / res['strategy_return'].rolling(12).std() * np.sqrt(12)

            cagr, vol, sharpe, max_dd, hit_rate = self._calculate_metrics(res['strategy_return'])
            metrics_summary.append({
                'strategy_name': config['strategy_name'],
                'CAGR': cagr,
                'Volatility': vol,
                'Sharpe': sharpe,
                'Max_Drawdown': max_dd,
                'Hit_Rate': hit_rate
            })

            realistic_results.append(res)

        results_df = pd.concat(realistic_results)
        metrics_df = pd.DataFrame(metrics_summary)
        return results_df, metrics_df

    def _calculate_metrics(self, returns):
        returns = returns.dropna()
        if returns.empty:
            return np.nan, np.nan, np.nan, np.nan, np.nan
        cumulative = (1 + returns).cumprod()
        total_return = cumulative.iloc[-1] - 1
        n_years = len(returns) / 12
        cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else np.nan
        volatility = returns.std() * np.sqrt(12)
        sharpe = returns.mean() / returns.std() * np.sqrt(12) if returns.std() > 0 else np.nan
        max_drawdown = (cumulative / cumulative.cummax() - 1).min()
        hit_rate = (returns > 0).mean()
        return cagr, volatility, sharpe, max_drawdown, hit_rate