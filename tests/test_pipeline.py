from quant_for_fun.backtest.vectorized import backtest_long_flat
from quant_for_fun.data.synthetic import make_price_series
from quant_for_fun.features.technical import add_forward_return_label, add_technical_features
from quant_for_fun.models.baseline import train_random_forest_classifier


def test_baseline_pipeline_runs() -> None:
    prices = make_price_series(n_days=400)
    dataset = add_forward_return_label(add_technical_features(prices), horizon=5)
    result = train_random_forest_classifier(dataset)
    backtest, metrics = backtest_long_flat(result.predictions)

    assert not result.predictions.empty
    assert not backtest.empty
    assert "sharpe" in metrics
