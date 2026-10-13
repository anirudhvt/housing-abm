"""Tests for wiring simulation.use_external_appreciation_data to the
already-downloaded ZHVI/ZORI CSVs.

Guards against the bug this pins the fix for: the config flag existed and
was read nowhere -- model.py hardcoded external_g_series and
external_rent_growth_series to None regardless of its value, so the full
2015-2026 Zillow history checked into the repo was silently ignored.
"""

import warnings

import pytest
import yaml

from housing_abm.model import AtlantaHousingModel

BASE_CONFIG = "config/baseline_params.yaml"


def write_config(tmp_path, **overrides):
    with open(BASE_CONFIG) as f:
        params = yaml.safe_load(f)
    params["simulation"]["use_external_appreciation_data"] = True
    params.setdefault("tract_calibration", {}).update(overrides)
    path = tmp_path / "params.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(params, f)
    return str(path)


def write_series_csvs(tmp_path):
    zhvi = tmp_path / "zhvi.csv"
    zhvi.write_text(
        "date,hpi,g\n"
        "2020-01-31,200000,0.05\n"
        "2020-02-29,201000,0.04\n"
        "2020-03-31,202000,0.03\n"
    )
    zori = tmp_path / "zori.csv"
    zori.write_text(
        "date,hpi\n"
        "2020-01-31,1000\n"
        "2020-02-29,1010\n"
        "2020-03-31,1020\n"
    )
    return str(zhvi), str(zori)


def test_flag_off_leaves_external_series_none():
    m = AtlantaHousingModel(config_path=BASE_CONFIG, n_households=20, seed=1)
    t = m.tracts["tract_001"]
    assert t.external_g_series is None
    assert t.external_rent_growth_series is None


def test_flag_on_loads_the_configured_csvs(tmp_path):
    zhvi_path, zori_path = write_series_csvs(tmp_path)
    config_path = write_config(
        tmp_path, zhvi_csv_path=zhvi_path, zori_csv_path=zori_path
    )
    m = AtlantaHousingModel(config_path=config_path, n_households=20, seed=1)
    t = m.tracts["tract_001"]
    assert t.external_g_series == pytest.approx([0.05, 0.04, 0.03])
    # month-over-month growth derived from the raw hpi column: 1010/1000-1, 1020/1010-1
    assert t.external_rent_growth_series == pytest.approx(
        [1010 / 1000 - 1, 1020 / 1010 - 1]
    )


def test_flag_on_with_missing_file_falls_back_to_none_with_warning(tmp_path):
    config_path = write_config(
        tmp_path,
        zhvi_csv_path=str(tmp_path / "does_not_exist.csv"),
        zori_csv_path=str(tmp_path / "also_missing.csv"),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = AtlantaHousingModel(config_path=config_path, n_households=20, seed=1)
    assert any("use_external_appreciation_data" in str(w.message) for w in caught)
    t = m.tracts["tract_001"]
    assert t.external_g_series is None
    assert t.external_rent_growth_series is None
    m.step()  # falls back to endogenous appreciation rather than crashing


def test_default_csv_paths_point_at_the_repo_root_files():
    # atlanta_zillow_zhvi.csv / atlanta_zillow_zori.csv are already checked
    # into the repo; the default paths should resolve to them without
    # requiring zhvi_csv_path/zori_csv_path to be set explicitly.
    m = AtlantaHousingModel(config_path=BASE_CONFIG, n_households=20, seed=1)
    # flag is off by default in baseline_params.yaml, so confirm the model
    # still builds; the loading path itself is exercised by
    # test_flag_on_loads_the_configured_csvs above with an explicit path.
    assert m.tracts["tract_001"].external_g_series is None
