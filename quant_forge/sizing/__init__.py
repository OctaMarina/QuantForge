"""Bet sizing (AFML ch. 10) for meta-labeled trades, and a capital-budget replay (needs scipy).

    from quant_forge.sizing import fit_sizing_constants, rule_sizes, normalized_sizes, capital_constrained_sizes

    k = fit_sizing_constants(oof_p, y_train, r_train, vol_train)     # training-only constants
    sizes = rule_sizes(p_test, vol_test, k)                          # {"U", "D", "F_be", "P", ...}
"""

from .bet_sizing import (SIZING_RULES, SizingConstants, afml_bet_size, break_even_win_rate, discretize_size, ecdf,
                         fit_sizing_constants, normalized_sizes, rule_sizes)
from .capital import capital_constrained_sizes

__all__ = ["afml_bet_size", "discretize_size", "ecdf", "break_even_win_rate", "SIZING_RULES", "SizingConstants",
           "fit_sizing_constants", "rule_sizes", "normalized_sizes", "capital_constrained_sizes"]
