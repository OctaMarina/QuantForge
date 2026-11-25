"""Evaluation of meta-model probabilities and of sized trading rules.

* :mod:`.metrics` - per-event log loss, Δ log loss vs a prior with monthly block-bootstrap CI, AUC, rank IC
  within groups, reliability / ECE, a combined probability-quality table;
* :mod:`.economics` - monthly PnL, annualised Sharpe, exposure path, drawdown, paired bootstrap and
  permutation tests over sizing rules, paired Sharpe-difference tables;
* :mod:`.portfolio` - daily mark-to-market PnL on a common calendar, portfolio metrics under a capital budget,
  paired monthly block bootstrap across rules and across trade lists (e.g. bar configurations).

    from quant_forge.evaluation import evaluate_sizing, paired_sharpe_table, delta_log_loss_ci
    ev = evaluate_sizing(pred, sizes)                 # table, monthly, boot, budget
    paired_sharpe_table(ev, [("P", "U"), ("DxP", "D")])
"""

from .economics import (SizingEvaluation, annualized_sharpe, evaluate_sizing, exposure_path, max_drawdown,
                        monthly_pnl, paired_sharpe_table, verdict)
from .metrics import (EPS, auc_safe, delta_log_loss_ci, information_coefficient, log_loss_vector,
                      probability_quality_table, reliability_table)
from .portfolio import (TradeDays, bootstrap_sharpe, daily_close_panel, daily_pnl, expand_trade_days,
                        max_drawdown_daily, month_labels, paired_differences, portfolio_table)

__all__ = ["EPS", "log_loss_vector", "auc_safe", "delta_log_loss_ci", "information_coefficient", "reliability_table",
           "probability_quality_table", "monthly_pnl", "annualized_sharpe", "exposure_path", "max_drawdown",
           "SizingEvaluation", "evaluate_sizing", "paired_sharpe_table", "verdict",
           "daily_close_panel", "TradeDays", "expand_trade_days", "daily_pnl", "portfolio_table", "max_drawdown_daily",
           "bootstrap_sharpe", "paired_differences", "month_labels"]
