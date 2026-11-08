import numpy as np
import pytest
from sklearn.model_selection import StratifiedGroupKFold
from ventureguard.models import TorchResidualClassifier, make_model
from ventureguard.data import load_data, CORE
from ventureguard.training import metrics


def test_top_decile_constant_predictor_is_independent_of_row_order():
    y = np.array([0] * 90 + [1] * 10)
    expected = metrics(y, np.full(100, .1))
    assert expected["recall_at_10pct"] == pytest.approx(.1)
    assert expected["precision_at_10pct"] == pytest.approx(.1)
    reverse = metrics(y[::-1], np.full(100, .1))
    assert reverse["recall_at_10pct"] == pytest.approx(expected["recall_at_10pct"])


def test_source_contract_and_group_separation():
    x, y, groups, audit = load_data(write_audit=False)
    assert x.shape == (5910, 64)
    assert y.sum() == 410
    assert set(CORE) <= set(x.columns)
    assert len(audit["source_sha256"]) == 64
    for train, test in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20261108).split(x, y, groups):
        assert not set(groups[train]) & set(groups[test])


def test_transformations_do_not_fit_on_validation_rows():
    x = np.array([[1., 2.], [2., 3.], [3., 4.], [4., np.nan], [5., 6.], [6., 7.]])
    y = np.array([0, 1, 0, 1, 0, 1])
    pipeline = make_model("logistic", {"C": .1, "balanced": False}, 42)
    pipeline.fit(x, y)
    before = pipeline["impute"].statistics_.copy()
    pipeline.predict_proba(np.array([[1e20, np.nan]]))
    np.testing.assert_array_equal(before, pipeline["impute"].statistics_)


def test_numpy_export_matches_pytorch_and_joblib(tmp_path):
    pytest.importorskip("torch")
    import joblib
    rng = np.random.default_rng(73)
    x = rng.normal(size=(140, 7)).astype(np.float32)
    y = (x[:, 0] + x[:, 1] > 0).astype(int)
    model = TorchResidualClassifier(max_epochs=4, patience=2, random_state=73).fit(x, y)
    assert model.export_max_error_ < 2e-5
    before = model.predict_proba(x)
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    np.testing.assert_allclose(joblib.load(path).predict_proba(x), before)
    assert np.isfinite(before).all()
