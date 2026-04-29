from __future__ import annotations

from quant_for_fun.backtest.vectorized import backtest_long_flat
from quant_for_fun.data.synthetic import make_price_series
from quant_for_fun.features.technical import add_forward_return_label, add_technical_features
from quant_for_fun.models.baseline import train_random_forest_classifier


def main() -> None:
    prices = make_price_series(n_days=1_250)
    dataset = add_forward_return_label(add_technical_features(prices), horizon=5)
    result = train_random_forest_classifier(dataset)
    _, backtest_metrics = backtest_long_flat(result.predictions)

    print("Train metrics")
    _print_metrics(result.train_metrics)
    print("\nTest metrics")
    _print_metrics(result.test_metrics)
    print("\nBacktest metrics")
    _print_metrics(backtest_metrics)


def _print_metrics(metrics: dict[str, float]) -> None:
    for key, value in metrics.items():
        print(f"{key:>18}: {value: .4f}")


if __name__ == "__main__":
    main()
