"""Dixon-Coles-style low-score-adjusted Poisson outcome baseline POC."""

from __future__ import annotations

import math


def _poisson(mean: float, count: int) -> float:
    return math.exp(-mean) * mean**count / math.factorial(count)


def _tau(home: int, away: int, home_mean: float, away_mean: float, rho: float) -> float:
    if home == 0 and away == 0:
        return 1 - home_mean * away_mean * rho
    if home == 0 and away == 1:
        return 1 + home_mean * rho
    if home == 1 and away == 0:
        return 1 + away_mean * rho
    if home == 1 and away == 1:
        return 1 - rho
    return 1.0


def dixon_coles_outcome_probabilities(
    *,
    home_goal_rate: float,
    away_goal_rate: float,
    home_goals: int,
    away_goals: int,
    minutes_remaining: int | float,
    max_additional_goals: int,
    rho: float,
) -> dict[str, float]:
    numeric = (home_goal_rate, away_goal_rate, float(minutes_remaining), rho)
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("soccer baseline inputs must be finite")
    if home_goal_rate < 0 or away_goal_rate < 0 or minutes_remaining < 0:
        raise ValueError("rates and remaining time must be nonnegative")
    if type(home_goals) is not int or type(away_goals) is not int:
        raise ValueError("current goals must be integers")
    if home_goals < 0 or away_goals < 0:
        raise ValueError("current goals must be nonnegative")
    if type(max_additional_goals) is not int or max_additional_goals < 2:
        raise ValueError("max_additional_goals must be an integer >= 2")
    home_mean = home_goal_rate * float(minutes_remaining) / 90.0
    away_mean = away_goal_rate * float(minutes_remaining) / 90.0
    totals = {"home_win": 0.0, "draw": 0.0, "away_win": 0.0}
    for extra_home in range(max_additional_goals + 1):
        for extra_away in range(max_additional_goals + 1):
            probability = (
                _poisson(home_mean, extra_home)
                * _poisson(away_mean, extra_away)
                * _tau(extra_home, extra_away, home_mean, away_mean, rho)
            )
            if probability < 0:
                raise ValueError("rho produces a negative low-score adjustment")
            final_home = home_goals + extra_home
            final_away = away_goals + extra_away
            key = (
                "home_win"
                if final_home > final_away
                else "away_win"
                if final_home < final_away
                else "draw"
            )
            totals[key] += probability
    mass = sum(totals.values())
    if mass <= 0:
        raise ValueError("truncated outcome mass must be positive")
    return {name: value / mass for name, value in totals.items()}


__all__ = ["dixon_coles_outcome_probabilities"]
