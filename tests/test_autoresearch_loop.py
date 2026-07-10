"""Offline unit tests for the torch-free autoresearch loop runner."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "autoresearch_loop", REPO_ROOT / "autoresearch" / "loop.py"
)
assert _SPEC is not None and _SPEC.loader is not None
loop = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(loop)


class TestDecide:
    def test_first_finite_result_becomes_baseline(self) -> None:
        assert loop.decide(0.01, None) is True

    def test_nan_candidate_is_rejected(self) -> None:
        assert loop.decide(float("nan"), None) is False
        assert loop.decide(float("nan"), 0.05) is False

    def test_none_candidate_is_rejected(self) -> None:
        assert loop.decide(None, 0.05) is False

    def test_must_beat_best_by_epsilon(self) -> None:
        # Clearly inside / clearly outside the band; no exact-FP-bound asserts.
        assert loop.decide(0.0649, 0.05, epsilon=0.02) is False
        assert loop.decide(0.0751, 0.05, epsilon=0.02) is True

    def test_default_epsilon_is_applied(self) -> None:
        assert loop.decide(0.055, 0.05) is False


class TestParsePorcelain:
    def test_modified_and_untracked_are_split(self) -> None:
        text = " M autoresearch/train_experiment.py\n?? autoresearch/runs/s1/.hypothesis\n"
        modified, untracked = loop.parse_porcelain(text)
        assert modified == ["autoresearch/train_experiment.py"]
        assert untracked == ["autoresearch/runs/s1/.hypothesis"]

    def test_empty_status_is_clean(self) -> None:
        assert loop.parse_porcelain("") == ([], [])

    def test_staged_and_renamed_count_as_modified(self) -> None:
        text = "M  a.py\nR  old.py -> new.py\n"
        modified, _ = loop.parse_porcelain(text)
        assert "a.py" in modified
        assert "new.py" in modified


class TestDiffIsValid:
    SESSION = "autoresearch/runs/s1"

    def test_exactly_the_mutable_file_is_valid(self) -> None:
        assert loop.diff_is_valid([loop.MUTABLE_FILE], [], self.SESSION) is True

    def test_no_edit_is_invalid(self) -> None:
        assert loop.diff_is_valid([], [], self.SESSION) is False

    def test_touching_other_tracked_files_is_invalid(self) -> None:
        assert (
            loop.diff_is_valid([loop.MUTABLE_FILE, "src/ophir/safety.py"], [], self.SESSION)
            is False
        )

    def test_untracked_inside_session_dir_is_allowed(self) -> None:
        assert (
            loop.diff_is_valid([loop.MUTABLE_FILE], [f"{self.SESSION}/.hypothesis"], self.SESSION)
            is True
        )

    def test_untracked_outside_session_dir_is_invalid(self) -> None:
        assert loop.diff_is_valid([loop.MUTABLE_FILE], ["evil.py"], self.SESSION) is False


class TestValidateExperimentSource:
    def test_valid_source_passes(self) -> None:
        text = f"import os\n{loop.SEALED_IMPORT_LINE}\nx = 128\nhidden = 2048\n"
        assert loop.validate_experiment_source(text) is None

    def test_missing_sealed_import_is_rejected(self) -> None:
        reason = loop.validate_experiment_source("x = 1\n")
        assert reason is not None and "sealed import" in reason

    def test_year_literal_is_rejected(self) -> None:
        text = f"{loop.SEALED_IMPORT_LINE}\nval_max_year = 2025\n"
        reason = loop.validate_experiment_source(text)
        assert reason is not None and "year literal" in reason

    def test_non_year_numbers_are_fine(self) -> None:
        text = f"{loop.SEALED_IMPORT_LINE}\nemb = 2048\nlr = 0.0002\nn = 10000\n"
        assert loop.validate_experiment_source(text) is None

    def test_real_train_experiment_file_validates(self) -> None:
        # The committed mutable file passes years as sealed Names / None kwargs.
        text = (REPO_ROOT / "autoresearch" / "train_experiment.py").read_text()
        assert loop.validate_experiment_source(text) is None

    def test_rebinding_sealed_name_is_rejected(self) -> None:
        text = f"{loop.SEALED_IMPORT_LINE}\nTRAIN_MAX_YEAR = ACCEPT_VAL_MAX_YEAR\n"
        reason = loop.validate_experiment_source(text)
        assert reason is not None and "TRAIN_MAX_YEAR" in reason and "rebound" in reason

    def test_rebinding_sealed_name_in_tuple_target_is_rejected(self) -> None:
        text = f"{loop.SEALED_IMPORT_LINE}\na, NUM_WORKERS = 1, 2\n"
        reason = loop.validate_experiment_source(text)
        assert reason is not None and "NUM_WORKERS" in reason

    def test_annassign_rebinding_sealed_name_is_rejected(self) -> None:
        text = f"{loop.SEALED_IMPORT_LINE}\nNUM_WORKERS: int = 8\n"
        reason = loop.validate_experiment_source(text)
        assert reason is not None and "NUM_WORKERS" in reason

    def test_year_kwarg_with_arithmetic_is_rejected(self) -> None:
        text = f"{loop.SEALED_IMPORT_LINE}\nf(val_min_year=ACCEPT_VAL_MAX_YEAR + 1)\n"
        reason = loop.validate_experiment_source(text)
        assert reason is not None and "val_min_year" in reason

    def test_year_kwarg_with_non_sealed_name_is_rejected(self) -> None:
        text = f"{loop.SEALED_IMPORT_LINE}\nf(max_year=foo)\n"
        reason = loop.validate_experiment_source(text)
        assert reason is not None and "max_year" in reason

    def test_year_kwarg_with_sealed_name_or_none_is_accepted(self) -> None:
        text = (
            f"{loop.SEALED_IMPORT_LINE}\n"
            "f(train_max_year=TRAIN_MAX_YEAR, val_min_year=ACCEPT_VAL_MIN_YEAR, "
            "val_max_year=None)\n"
        )
        assert loop.validate_experiment_source(text) is None

    def test_unparseable_file_is_rejected(self) -> None:
        text = f"{loop.SEALED_IMPORT_LINE}\ndef (:\n"
        assert loop.validate_experiment_source(text) == "unparseable"

    def test_for_loop_rebinding_sealed_name_is_rejected(self) -> None:
        text = f"{loop.SEALED_IMPORT_LINE}\nfor TRAIN_MAX_YEAR in range(3):\n    pass\n"
        reason = loop.validate_experiment_source(text)
        assert reason is not None and "TRAIN_MAX_YEAR" in reason and "rebound" in reason

    def test_walrus_rebinding_sealed_name_is_rejected(self) -> None:
        text = f"{loop.SEALED_IMPORT_LINE}\nx = (NUM_WORKERS := 8)\n"
        reason = loop.validate_experiment_source(text)
        assert reason is not None and "NUM_WORKERS" in reason and "rebound" in reason

    def test_import_as_rebinding_sealed_name_is_rejected(self) -> None:
        text = f"{loop.SEALED_IMPORT_LINE}\nimport os as TRAIN_MAX_YEAR\n"
        reason = loop.validate_experiment_source(text)
        assert reason is not None and "TRAIN_MAX_YEAR" in reason and "rebound" in reason

    def test_from_import_as_rebinding_sealed_name_is_rejected(self) -> None:
        text = f"{loop.SEALED_IMPORT_LINE}\nfrom x import y as NUM_WORKERS\n"
        reason = loop.validate_experiment_source(text)
        assert reason is not None and "NUM_WORKERS" in reason and "rebound" in reason

    def test_function_param_shadowing_sealed_name_is_rejected(self) -> None:
        text = f"{loop.SEALED_IMPORT_LINE}\ndef f(ACCEPT_VAL_MIN_YEAR):\n    return 1\n"
        reason = loop.validate_experiment_source(text)
        assert reason is not None and "ACCEPT_VAL_MIN_YEAR" in reason and "rebound" in reason

    def test_dict_splat_year_key_with_arithmetic_is_rejected(self) -> None:
        text = f'{loop.SEALED_IMPORT_LINE}\nf(**{{"train_max_year": ACCEPT_VAL_MIN_YEAR + 5}})\n'
        reason = loop.validate_experiment_source(text)
        assert reason is not None and "train_max_year" in reason

    def test_dict_splat_year_key_with_sealed_name_is_accepted(self) -> None:
        text = f'{loop.SEALED_IMPORT_LINE}\nf(**{{"val_min_year": ACCEPT_VAL_MIN_YEAR}})\n'
        assert loop.validate_experiment_source(text) is None

    def test_dict_splat_non_literal_key_is_rejected(self) -> None:
        text = f"{loop.SEALED_IMPORT_LINE}\nf(**{{k: 1}})\n"
        reason = loop.validate_experiment_source(text)
        assert reason is not None and "unverifiable splat key" in reason

    def test_splat_of_unresolvable_expression_is_rejected(self) -> None:
        text = f"{loop.SEALED_IMPORT_LINE}\nf(**get_kwargs())\n"
        reason = loop.validate_experiment_source(text)
        assert reason is not None and "unverifiable splat" in reason

    def test_splat_of_unresolvable_name_is_rejected(self) -> None:
        text = f"{loop.SEALED_IMPORT_LINE}\nkw = build()\nf(**kw)\n"
        reason = loop.validate_experiment_source(text)
        assert reason is not None and "unverifiable splat" in reason

    def test_splat_of_compliant_dict_literal_name_is_accepted(self) -> None:
        # The baseline MODEL_KWARGS pattern: module-level annotated dict literal.
        text = (
            f"{loop.SEALED_IMPORT_LINE}\n"
            'MODEL_KWARGS: dict[str, object] = {"emb_dim": 128, "lr": 0.0002}\n'
            "f(max_steps=1, **MODEL_KWARGS)\n"
        )
        assert loop.validate_experiment_source(text) is None

    def test_splat_of_dict_name_with_year_key_arithmetic_is_rejected(self) -> None:
        text = (
            f"{loop.SEALED_IMPORT_LINE}\n"
            'KW = {"val_max_year": ACCEPT_VAL_MAX_YEAR + 1}\n'
            "f(**KW)\n"
        )
        reason = loop.validate_experiment_source(text)
        assert reason is not None and "val_max_year" in reason

    def test_match_case_capture_of_sealed_name_is_rejected(self) -> None:
        # Reviewer exploit: capture a smuggled value into a sealed name via a
        # match-case pattern, then pass it as a *_year keyword.
        text = (
            f"{loop.SEALED_IMPORT_LINE}\n"
            "match compute():\n"
            "    case TRAIN_MAX_YEAR:\n"
            "        f(train_max_year=TRAIN_MAX_YEAR)\n"
        )
        reason = loop.validate_experiment_source(text)
        assert reason is not None and "TRAIN_MAX_YEAR" in reason and "rebound" in reason

    def test_match_star_capture_of_sealed_name_is_rejected(self) -> None:
        text = (
            f"{loop.SEALED_IMPORT_LINE}\nmatch xs:\n    case [first, *NUM_WORKERS]:\n        pass\n"
        )
        reason = loop.validate_experiment_source(text)
        assert reason is not None and "NUM_WORKERS" in reason and "rebound" in reason

    def test_match_mapping_rest_capture_of_sealed_name_is_rejected(self) -> None:
        text = (
            f"{loop.SEALED_IMPORT_LINE}\n"
            "match d:\n"
            '    case {"a": v, **ACCEPT_VAL_MIN_YEAR}:\n'
            "        pass\n"
        )
        reason = loop.validate_experiment_source(text)
        assert reason is not None and "ACCEPT_VAL_MIN_YEAR" in reason and "rebound" in reason

    def test_stockhandler_import_is_rejected(self) -> None:
        text = f"{loop.SEALED_IMPORT_LINE}\nfrom ophir.ticker import StockHandler\n"
        reason = loop.validate_experiment_source(text)
        assert reason is not None and "StockHandler" in reason

    def test_stockhandler_attribute_use_is_rejected(self) -> None:
        text = (
            f"{loop.SEALED_IMPORT_LINE}\nimport ophir.ticker as ticker\nh = ticker.StockHandler\n"
        )
        reason = loop.validate_experiment_source(text)
        assert reason is not None and "StockHandler" in reason

    def test_stockhandler_import_with_alias_is_rejected(self) -> None:
        text = f"{loop.SEALED_IMPORT_LINE}\nfrom ophir.ticker import StockHandler as SH\n"
        reason = loop.validate_experiment_source(text)
        assert reason is not None and "StockHandler" in reason


class TestMetricsAndResults:
    def test_parse_metrics_reads_named_keys_only(self, tmp_path: Path) -> None:
        p = tmp_path / "metrics.json"
        p.write_text('{"rank_ic_near": 0.061, "h1": 0.09, "h5": 0.05, "n": 1000, "junk": 9}')
        metrics = loop.parse_metrics(str(p))
        assert metrics["rank_ic_near"] == 0.061
        assert "junk" not in metrics

    def test_parse_metrics_tolerates_nan(self, tmp_path: Path) -> None:
        p = tmp_path / "metrics.json"
        p.write_text('{"rank_ic_near": NaN}')
        assert math.isnan(loop.parse_metrics(str(p))["rank_ic_near"])

    def test_result_row_is_tab_separated_and_sanitized(self) -> None:
        row = loop.format_result_row(
            iteration=3,
            utc="2026-07-07T05:00:00Z",
            hypothesis="try\tranking\nloss",
            status="keep",
            rank_ic_near=0.061,
            h1=0.09,
            h5=0.05,
            wall_s=412.0,
            commit="abc1234",
            seed_ics="0.06100|0.05900|0.06300",
        )
        cells = row.split("\t")
        assert len(cells) == len(loop.RESULTS_HEADER.split("\t"))
        assert cells[2] == "try ranking loss"
        assert cells[3] == "keep"
        assert cells[-1] == "0.06100|0.05900|0.06300"

    def test_result_row_seed_ics_defaults_empty(self) -> None:
        row = loop.format_result_row(
            iteration=0,
            utc="2026-07-07T05:00:00Z",
            hypothesis="baseline",
            status="keep",
            rank_ic_near=0.06,
            h1=None,
            h5=None,
            wall_s=1.0,
            commit="abc1234",
        )
        cells = row.split("\t")
        assert len(cells) == len(loop.RESULTS_HEADER.split("\t"))
        assert cells[-1] == ""

    def test_append_result_writes_header_once(self, tmp_path: Path) -> None:
        tsv = tmp_path / "results.tsv"
        loop.append_result(str(tsv), "1\trow")
        loop.append_result(str(tsv), "2\trow")
        lines = tsv.read_text().splitlines()
        assert lines[0] == loop.RESULTS_HEADER
        assert lines[1:] == ["1\trow", "2\trow"]


class FakeRunner:
    """Scripted subprocess stand-in: maps a command marker to (rc, output).

    ``effects`` maps a marker to a callable invoked with the command before
    the scripted result is returned — it materializes what the real
    subprocess would have written (checkpoints, metrics files).
    """

    def __init__(
        self,
        script: dict[str, tuple[int, str]],
        effects: dict[str, Callable[[list[str]], None]] | None = None,
    ) -> None:
        self.script = script
        self.effects = effects or {}
        self.calls: list[list[str]] = []

    def __call__(
        self,
        cmd: list[str],
        *,
        cwd: str,
        timeout: float | None = None,
        input_text: str | None = None,
    ) -> tuple[int, str]:
        self.calls.append(cmd)
        joined = " ".join(cmd)
        for marker, effect in self.effects.items():
            if marker in joined:
                effect(cmd)
        for marker, result in self.script.items():
            if marker in joined:
                return result
        return (0, "")

    def commands(self, marker: str) -> list[list[str]]:
        return [c for c in self.calls if marker in " ".join(c)]


BASE_SHA = "base0000"


def _make_session(tmp_path: Path) -> str:
    """Bare session dir; train/eval artifacts come from FakeRunner effects."""
    session_dir = tmp_path / "runs" / "s1"
    session_dir.mkdir(parents=True)
    (session_dir / ".hypothesis").write_text("wider near-band loss weighting")
    return str(session_dir)


def _train_eval_effects(
    metrics_per_seed: dict[str, str], skip_ckpt_for: str | None = None
) -> dict[str, Callable[[list[str]], None]]:
    """Effects that write what real train/eval subprocesses would write.

    ``metrics_per_seed`` keys are seed dir names (``seed-0`` ...);
    ``skip_ckpt_for`` simulates a training run that produced no checkpoint.
    """

    def _train(cmd: list[str]) -> None:
        out_dir = Path(cmd[cmd.index("--out-dir") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        if out_dir.name != skip_ckpt_for:
            (out_dir / "best-step=1.ckpt").write_text("stub")

    def _eval(cmd: list[str]) -> None:
        out = Path(cmd[cmd.index("--out") + 1])
        out.write_text(metrics_per_seed[out.parent.name])

    return {"train_experiment.py --max-steps": _train, "eval_harness.py": _eval}


def _same_metrics(payload: str) -> dict[str, str]:
    return {f"seed-{seed}": payload for seed in loop.SEEDS}


def _experiment_file_ok(monkeypatch, tmp_path: Path) -> None:
    exp = tmp_path / "train_experiment.py"
    exp.write_text(f"{loop.SEALED_IMPORT_LINE}\n")
    monkeypatch.setattr(loop, "MUTABLE_PATH", str(exp))


class TestRunIteration:
    def _propose_runner(
        self, effects: dict[str, Callable[[list[str]], None]] | None = None
    ) -> FakeRunner:
        return FakeRunner({"status --porcelain": (0, f" M {loop.MUTABLE_FILE}\n")}, effects)

    def test_keep_flow_commits_and_never_resets(self, tmp_path, monkeypatch) -> None:
        session_dir = _make_session(tmp_path)
        _experiment_file_ok(monkeypatch, tmp_path)
        runner = self._propose_runner(
            _train_eval_effects(
                {
                    "seed-0": '{"rank_ic_near": 0.30, "h1": 0.2, "h5": 0.1}',
                    "seed-1": '{"rank_ic_near": 0.24, "h1": 0.1, "h5": 0.0}',
                    "seed-2": '{"rank_ic_near": 0.24, "h1": 0.0, "h5": 0.2}',
                }
            )
        )
        result = loop.run_iteration(
            1, session_dir, 0.03, BASE_SHA, propose=True, epsilon=0.02, runner=runner
        )
        assert result.status == "keep"
        assert result.rank_ic_near == pytest.approx(0.26)  # mean over the 3 seeds
        assert result.h1 == pytest.approx(0.1)
        assert result.seed_ics == "0.30000|0.24000|0.24000"
        train_cmds = runner.commands("train_experiment.py --max-steps")
        seeds = [c[c.index("--seed") + 1] for c in train_cmds]
        assert seeds == [str(s) for s in loop.SEEDS]  # one train per seed, in order
        commit_cmds = runner.commands("git commit")
        assert commit_cmds and "--no-verify" in commit_cmds[0]
        assert not runner.commands("reset --hard")

    def test_discard_flow_resets_to_base_sha(self, tmp_path, monkeypatch) -> None:
        session_dir = _make_session(tmp_path)
        _experiment_file_ok(monkeypatch, tmp_path)
        runner = self._propose_runner(_train_eval_effects(_same_metrics('{"rank_ic_near": 0.031}')))
        result = loop.run_iteration(
            1, session_dir, 0.03, BASE_SHA, propose=True, epsilon=0.02, runner=runner
        )
        assert result.status == "discard"
        assert result.rank_ic_near == pytest.approx(0.031)
        resets = runner.commands("reset --hard")
        assert resets and resets[0][-1] == BASE_SHA

    def test_missing_seed_checkpoint_is_crash(self, tmp_path, monkeypatch) -> None:
        session_dir = _make_session(tmp_path)
        _experiment_file_ok(monkeypatch, tmp_path)
        second = f"seed-{loop.SEEDS[1]}"
        runner = self._propose_runner(
            _train_eval_effects(_same_metrics('{"rank_ic_near": 0.30}'), skip_ckpt_for=second)
        )
        result = loop.run_iteration(
            1, session_dir, None, BASE_SHA, propose=True, epsilon=0.02, runner=runner
        )
        assert result.status == "crash"
        assert runner.commands("reset --hard")

    def test_stale_seed_artifacts_are_purged_before_training(self, tmp_path, monkeypatch) -> None:
        # Regression (2026-07-09): a restarted session reuses iteration dirs;
        # a dead attempt's checkpoint sorted after the fresh one and got
        # scored. The purge must remove it before training.
        session_dir = _make_session(tmp_path)
        stale_dir = Path(session_dir) / "iter-001" / f"seed-{loop.SEEDS[0]}"
        stale_dir.mkdir(parents=True)
        stale_ckpt = stale_dir / "best-step=9999-val_rank_ic_near=0.99999.ckpt"
        stale_ckpt.write_text("stale")
        (stale_dir / "metrics.json").write_text('{"rank_ic_near": 0.99}')
        _experiment_file_ok(monkeypatch, tmp_path)
        runner = self._propose_runner(_train_eval_effects(_same_metrics('{"rank_ic_near": 0.05}')))
        result = loop.run_iteration(
            1, session_dir, None, BASE_SHA, propose=True, epsilon=0.02, runner=runner
        )
        assert result.rank_ic_near == pytest.approx(0.05)  # not the stale 0.99
        assert not stale_ckpt.exists()
        scored = [c[c.index("--ckpt") + 1] for c in runner.commands("eval_harness.py")]
        assert scored and all("9999" not in p for p in scored)

    def test_concurrent_seeds_train_via_runner_many(self, tmp_path, monkeypatch) -> None:
        session_dir = _make_session(tmp_path)
        _experiment_file_ok(monkeypatch, tmp_path)
        effects = _train_eval_effects(_same_metrics('{"rank_ic_near": 0.30}'))
        captured: dict[str, object] = {}

        def fake_many(
            cmds: list[list[str]],
            *,
            cwd: str,
            timeout: float | None = None,
            log_paths: list[str] | None = None,
        ) -> list[int]:
            captured["cmds"] = cmds
            captured["logs"] = log_paths
            captured["timeout"] = timeout
            for cmd in cmds:  # materialize checkpoints like real training would
                effects["train_experiment.py --max-steps"](cmd)
            return [0] * len(cmds)

        runner = self._propose_runner(effects)
        result = loop.run_iteration(
            1,
            session_dir,
            None,
            BASE_SHA,
            propose=True,
            epsilon=0.02,
            runner=runner,
            concurrent=3,
            runner_many=fake_many,
        )
        assert result.status == "keep"
        cmds = captured["cmds"]
        assert cmds is not None
        seeds = [c[c.index("--seed") + 1] for c in cmds]
        assert seeds == [str(s) for s in loop.SEEDS]  # all seeds in one batch
        logs = captured["logs"]
        assert logs is not None and all(log.endswith("train.log") for log in logs)
        # budget = total GPU-seconds per trial, not per-seed wall under contention
        assert captured["timeout"] == loop.TRAIN_TIMEOUT_S * len(loop.SEEDS)
        # trainings did NOT go through the sequential runner
        assert not runner.commands("train_experiment.py --max-steps")
        # evals still did, one per seed
        assert len(runner.commands("eval_harness.py")) == len(loop.SEEDS)

    def test_concurrent_seed_timeout_is_crash(self, tmp_path, monkeypatch) -> None:
        session_dir = _make_session(tmp_path)
        _experiment_file_ok(monkeypatch, tmp_path)
        effects = _train_eval_effects(_same_metrics('{"rank_ic_near": 0.30}'))

        def fake_many(
            cmds: list[list[str]],
            *,
            cwd: str,
            timeout: float | None = None,
            log_paths: list[str] | None = None,
        ) -> list[int]:
            for cmd in cmds:
                effects["train_experiment.py --max-steps"](cmd)
            return [0, -1, 0]  # middle seed timed out

        runner = self._propose_runner(effects)
        result = loop.run_iteration(
            1,
            session_dir,
            None,
            BASE_SHA,
            propose=True,
            epsilon=0.02,
            runner=runner,
            concurrent=3,
            runner_many=fake_many,
        )
        assert result.status == "crash"
        assert runner.commands("reset --hard")

    def test_invalid_diff_never_trains(self, tmp_path, monkeypatch) -> None:
        session_dir = _make_session(tmp_path)
        _experiment_file_ok(monkeypatch, tmp_path)
        runner = FakeRunner({"status --porcelain": (0, " M src/ophir/safety.py\n")})
        result = loop.run_iteration(
            1, session_dir, None, BASE_SHA, propose=True, epsilon=0.02, runner=runner
        )
        assert result.status == "invalid"
        assert not runner.commands("train_experiment.py --max-steps")
        assert runner.commands("reset --hard")

    def test_failed_commit_is_invalid_and_never_trains(self, tmp_path, monkeypatch) -> None:
        session_dir = _make_session(tmp_path)
        _experiment_file_ok(monkeypatch, tmp_path)
        runner = FakeRunner(
            {
                "status --porcelain": (0, f" M {loop.MUTABLE_FILE}\n"),
                "git commit": (1, "hook rejected"),
            }
        )
        result = loop.run_iteration(
            1, session_dir, None, BASE_SHA, propose=True, epsilon=0.02, runner=runner
        )
        assert result.status == "invalid"
        assert not runner.commands("train_experiment.py --max-steps")

    def test_proposer_failure_is_its_own_status(self, tmp_path, monkeypatch) -> None:
        session_dir = _make_session(tmp_path)
        _experiment_file_ok(monkeypatch, tmp_path)
        runner = FakeRunner({"claude": (1, "not logged in")})
        result = loop.run_iteration(
            1, session_dir, None, BASE_SHA, propose=True, epsilon=0.02, runner=runner
        )
        assert result.status == "proposer-fail"
        assert not runner.commands("train_experiment.py --max-steps")

    def test_train_timeout_is_crash_and_resets(self, tmp_path, monkeypatch) -> None:
        session_dir = _make_session(tmp_path)
        _experiment_file_ok(monkeypatch, tmp_path)
        runner = FakeRunner(
            {
                "status --porcelain": (0, f" M {loop.MUTABLE_FILE}\n"),
                "train_experiment.py": (-1, "TIMEOUT"),
            }
        )
        result = loop.run_iteration(
            1, session_dir, None, BASE_SHA, propose=True, epsilon=0.02, runner=runner
        )
        assert result.status == "crash"
        assert runner.commands("reset --hard")

    def test_baseline_iteration_skips_proposal_and_never_commits(
        self, tmp_path, monkeypatch
    ) -> None:
        session_dir = _make_session(tmp_path)
        _experiment_file_ok(monkeypatch, tmp_path)
        runner = FakeRunner({}, _train_eval_effects(_same_metrics('{"rank_ic_near": 0.06}')))
        result = loop.run_iteration(
            0, session_dir, None, BASE_SHA, propose=False, epsilon=0.02, runner=runner
        )
        assert result.status == "keep"
        assert result.hypothesis == "baseline"
        assert not runner.commands("claude")
        assert not runner.commands("commit")
        assert not runner.commands("reset --hard")


class TestPromptAndPins:
    def test_prompt_carries_contract_and_context(self) -> None:
        prompt = loop.build_prompt("PROGRAM", "iter\t...", "abc fix loss", "/tmp/s/.hypothesis")
        assert "PROGRAM" in prompt
        assert loop.MUTABLE_FILE in prompt
        assert ".hypothesis" in prompt
        assert "one" in prompt.lower()

    def test_pins_detect_tampering(self, tmp_path, monkeypatch) -> None:
        target = tmp_path / "eval_harness.py"
        target.write_text("original")
        monkeypatch.setattr(loop, "PINNED_FILES", (str(target),))
        pins = loop.pin_hashes()
        assert loop.check_pins(pins) == []
        target.write_text("tampered")
        assert loop.check_pins(pins) == [str(target)]

    def test_absent_pin_flips_when_file_appears(self, tmp_path) -> None:
        missing = tmp_path / ".claude" / "settings.json"
        pins = loop.pin_hashes((str(missing),), allow_absent=True)
        assert pins[str(missing)] == loop.ABSENT_PIN
        assert loop.check_pins(pins) == []
        missing.parent.mkdir(parents=True)
        missing.write_text("{}")
        assert loop.check_pins(pins) == [str(missing)]

    def test_content_pin_flips_when_file_changes(self, tmp_path) -> None:
        present = tmp_path / "CLAUDE.md"
        present.write_text("rules")
        pins = loop.pin_hashes((str(present),), allow_absent=True)
        assert loop.check_pins(pins) == []
        present.write_text("tampered")
        assert loop.check_pins(pins) == [str(present)]

    def test_unchanged_absent_pin_passes(self, tmp_path) -> None:
        missing = tmp_path / "AGENTS.md"
        pins = loop.pin_hashes((str(missing),), allow_absent=True)
        assert loop.check_pins(pins) == []

    def test_content_pin_flips_when_file_disappears(self, tmp_path) -> None:
        present = tmp_path / "CLAUDE.md"
        present.write_text("rules")
        pins = loop.pin_hashes((str(present),), allow_absent=True)
        present.unlink()
        assert loop.check_pins(pins) == [str(present)]


class TestGitHooksIsolation:
    def test_git_carries_hookspath_when_set(self, monkeypatch) -> None:
        monkeypatch.setattr(loop, "HOOKS_PATH", "/session/hooks")
        runner = FakeRunner({})
        loop._git(runner, ["status", "--porcelain"])
        cmd = runner.calls[0]
        assert cmd[:4] == ["git", "-c", "core.hooksPath=/session/hooks", "status"]

    def test_git_is_plain_when_hookspath_unset(self, monkeypatch) -> None:
        monkeypatch.setattr(loop, "HOOKS_PATH", None)
        runner = FakeRunner({})
        loop._git(runner, ["log"])
        assert runner.calls[0] == ["git", "log"]

    def test_head_sha_carries_hookspath_when_set(self, monkeypatch) -> None:
        monkeypatch.setattr(loop, "HOOKS_PATH", "/session/hooks")
        calls: list[list[str]] = []

        def fake_run(cmd: list[str], *, cwd: str, **_: object) -> tuple[int, str]:
            calls.append(cmd)
            return (0, "sha1234\n")

        monkeypatch.setattr(loop, "run", fake_run)
        assert loop._head_sha() == "sha1234"
        assert calls[0][:3] == ["git", "-c", "core.hooksPath=/session/hooks"]
        assert "rev-parse" in calls[0]


class TestCrossSessionRecovery:
    def test_recovers_all_sessions_and_appends_rows(self, tmp_path, monkeypatch) -> None:
        harness = tmp_path / "autoresearch"
        for name, sha in (("s1", "aaaa111"), ("s2", "bbbb222")):
            sdir = harness / "runs" / name
            sdir.mkdir(parents=True)
            (sdir / ".in-flight").write_text(sha)
        monkeypatch.setattr(loop, "HARNESS_DIR", str(harness))
        monkeypatch.setattr(loop, "HOOKS_PATH", None)
        calls: list[list[str]] = []

        def fake_run(cmd: list[str], *, cwd: str, **_: object) -> tuple[int, str]:
            calls.append(cmd)
            if cmd[:2] == ["git", "rev-parse"]:
                return (0, "post9999\n")
            return (0, "")

        monkeypatch.setattr(loop, "run", fake_run)
        loop._recover_all_in_flight()

        for name in ("s1", "s2"):
            sdir = harness / "runs" / name
            assert not (sdir / ".in-flight").exists()
            rows = (sdir / "results.tsv").read_text().splitlines()
            assert rows[0] == loop.RESULTS_HEADER
            assert "runner-died" in rows[1]
            assert "(recovered interrupted iteration)" in rows[1]
            header = loop.RESULTS_HEADER.split("\t")
            assert rows[1].split("\t")[header.index("commit")] == "post9999"[:7]
        reset_shas = [c[-1] for c in calls if c[:3] == ["git", "reset", "--hard"]]
        assert "aaaa111" in reset_shas and "bbbb222" in reset_shas


class TestMainTrackedResultsGuard:
    def _run_main(self, tmp_path, monkeypatch, ls_files_rc: int) -> int:
        harness = tmp_path / "autoresearch"
        (harness / "runs").mkdir(parents=True)
        monkeypatch.setattr(loop, "HARNESS_DIR", str(harness))
        # main() mutates the module-level HOOKS_PATH; monkeypatch restores it.
        monkeypatch.setattr(loop, "HOOKS_PATH", None)

        # main() threads -c core.hooksPath into every git call, so match on the
        # subcommand rather than a fixed prefix.
        def fake_run(cmd: list[str], *, cwd: str, **_: object) -> tuple[int, str]:
            if "ls-files" in cmd:
                return (ls_files_rc, "")
            if "rev-parse" in cmd:
                return (0, "sha1234\n")
            return (0, "")

        monkeypatch.setattr(loop, "run", fake_run)
        # Baseline iteration returns keep with an in-band metric so a proceeding
        # run terminates cleanly at max-iters=1.
        monkeypatch.setattr(
            loop,
            "run_iteration",
            lambda *a, **k: loop.IterationResult("keep", 0.06, 0.1, 0.05, "baseline", 1.0),
        )
        return loop.main(["--session", "guard", "--max-iters", "1"])

    def test_tracked_results_aborts(self, tmp_path, monkeypatch) -> None:
        assert self._run_main(tmp_path, monkeypatch, ls_files_rc=0) == 4

    def test_untracked_results_proceeds(self, tmp_path, monkeypatch) -> None:
        assert self._run_main(tmp_path, monkeypatch, ls_files_rc=1) == 0
