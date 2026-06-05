"""Tests for the CLI utilities and argument parser in folktexts/cli/."""

from __future__ import annotations

import json

import pytest

from folktexts.cli._utils import cmd_line_args_to_kwargs, get_or_create_results_dir
from folktexts.cli.run_acs_benchmark import _loggable_args, setup_arg_parser
from folktexts.prompting import PROMPT_DEFAULT

# ----------------------------------------------------------------------
# get_or_create_results_dir
# ----------------------------------------------------------------------


class TestGetOrCreateResultsDir:
    def test_returns_path_with_model_and_task(self, tmp_path):
        result = get_or_create_results_dir("gpt2", "ACSIncome", tmp_path)
        assert "gpt2" in result.as_posix()
        assert "ACSIncome" in result.as_posix()

    def test_creates_directory(self, tmp_path):
        result = get_or_create_results_dir("gpt2", "ACSIncome", tmp_path)
        assert result.exists()
        assert result.is_dir()

    def test_idempotent(self, tmp_path):
        r1 = get_or_create_results_dir("gpt2", "ACSIncome", tmp_path)
        r2 = get_or_create_results_dir("gpt2", "ACSIncome", tmp_path)
        assert r1 == r2
        assert r2.exists()


# ----------------------------------------------------------------------
# cmd_line_args_to_kwargs
# ----------------------------------------------------------------------


class TestCmdLineArgsToKwargs:
    def test_string_value(self):
        result = cmd_line_args_to_kwargs(["--key=value"])
        assert result == {"key": "value"}

    def test_int_value(self):
        result = cmd_line_args_to_kwargs(["--count=42"])
        assert result == {"count": 42}
        assert isinstance(result["count"], int)

    def test_float_value(self):
        result = cmd_line_args_to_kwargs(["--ratio=0.5"])
        assert result == {"ratio": 0.5}
        assert isinstance(result["ratio"], float)

    def test_bool_true(self):
        result = cmd_line_args_to_kwargs(["--flag=true"])
        assert result == {"flag": True}
        assert isinstance(result["flag"], bool)

    def test_bool_false(self):
        result = cmd_line_args_to_kwargs(["--flag=false"])
        assert result == {"flag": False}
        assert isinstance(result["flag"], bool)

    def test_bool_case_insensitive(self):
        assert cmd_line_args_to_kwargs(["--x=True"])["x"] is True
        assert cmd_line_args_to_kwargs(["--x=FALSE"])["x"] is False

    def test_flag_style_becomes_true(self):
        result = cmd_line_args_to_kwargs(["--verbose"])
        assert result == {"verbose": True}

    def test_hyphen_to_underscore_in_key(self):
        result = cmd_line_args_to_kwargs(["--my-key=1"])
        assert "my_key" in result

    def test_leading_dashes_stripped(self):
        result = cmd_line_args_to_kwargs(["--key=val"])
        assert "key" in result
        assert "--key" not in result

    def test_multiple_args(self):
        result = cmd_line_args_to_kwargs(["--a=1", "--b=hello", "--flag"])
        assert result == {"a": 1, "b": "hello", "flag": True}

    def test_empty_list(self):
        assert cmd_line_args_to_kwargs([]) == {}


# ----------------------------------------------------------------------
# setup_arg_parser
# ----------------------------------------------------------------------


class TestSetupArgParser:
    def _parse(self, args: list[str]):
        return setup_arg_parser().parse_args(args)

    def _required(self):
        return [
            "--model",
            "gpt2",
            "--results-dir",
            "/tmp/results",
            "--data-dir",
            "/tmp/data",
        ]

    def test_required_args_parsed(self):
        args = self._parse(self._required())
        assert args.model == "gpt2"
        assert args.results_dir == "/tmp/results"
        assert args.data_dir == "/tmp/data"

    def test_default_task(self):
        args = self._parse(self._required())
        assert args.task == "ACSIncome"

    def test_default_batch_size(self):
        args = self._parse(self._required())
        assert args.batch_size == 16

    def test_default_context_size(self):
        args = self._parse(self._required())
        assert args.context_size == 600

    def test_default_seed(self):
        args = self._parse(self._required())
        assert args.seed == 42

    def test_missing_required_arg_exits(self):
        with pytest.raises(SystemExit):
            setup_arg_parser().parse_args(["--model", "gpt2"])

    def test_boolean_flag_use_web_api(self):
        args = self._parse(self._required() + ["--use-web-api-model"])
        assert args.use_web_api_model is True

    def test_boolean_flag_dont_correct_order_bias(self):
        args = self._parse(self._required() + ["--dont-correct-order-bias"])
        assert args.dont_correct_order_bias is True

    def test_boolean_flag_numeric_risk_prompting(self):
        args = self._parse(self._required() + ["--numeric-risk-prompting"])
        assert args.numeric_risk_prompting is True

    def test_few_shot_arg(self):
        args = self._parse(self._required() + ["--few-shot", "3"])
        assert args.few_shot == 3

    def test_variation_parsed_as_dict(self):
        args = self._parse(
            self._required() + ["--variation", "format=bullet;connector=is"]
        )
        assert args.variation.get("format") == "bullet"
        assert args.variation.get("connector") == "is"

    def test_variation_bool_false_is_false(self):
        """ParseDict regression: 'False'/'false' used to coerce to Python True."""
        args = self._parse(self._required() + ["--variation", "show_question=False"])
        assert args.variation["show_question"] is False
        args2 = self._parse(self._required() + ["--variation", "show_question=true"])
        assert args2.variation["show_question"] is True

    def test_variation_space_separated_pairs(self):
        """ParseDict regression: only values[0] was read, dropping later pairs."""
        args = self._parse(
            self._required() + ["--variation", "format=bullet", "connector=is"]
        )
        assert args.variation.get("format") == "bullet"
        assert args.variation.get("connector") == "is"

    def test_variation_empty_is_empty_dict(self):
        """ParseDict regression: an argless --variation raised IndexError on values[0]."""
        args = self._parse(self._required() + ["--variation"])
        assert args.variation == {}

    def test_variation_numeric_coercion(self):
        args = self._parse(self._required() + ["--variation", "a=3;b=0.5;c=is:"])
        assert args.variation["a"] == 3 and isinstance(args.variation["a"], int)
        assert args.variation["b"] == 0.5 and isinstance(args.variation["b"], float)
        assert args.variation["c"] == "is:"

    def test_subsampling_arg(self):
        args = self._parse(self._required() + ["--subsampling", "0.1"])
        assert args.subsampling == pytest.approx(0.1)

    def test_logger_level_default(self):
        args = self._parse(self._required())
        assert args.logger_level == "WARNING"

    def test_logger_level_choices(self):
        for level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            args = self._parse(self._required() + ["--logger-level", level])
            assert args.logger_level == level

    def test_default_args_are_json_serializable(self):
        """B1 regression: --chat-prompt/--system-prompt default to the PROMPT_DEFAULT
        sentinel, so main()'s ``json.dumps(vars(args))`` crashed on every default run.
        ``_loggable_args`` must resolve the sentinel so the args-logging step succeeds."""
        args = self._parse(self._required())
        assert args.chat_prompt is PROMPT_DEFAULT
        assert args.system_prompt is PROMPT_DEFAULT

        # The raw namespace is not serializable (the original bug)...
        with pytest.raises(TypeError):
            json.dumps(vars(args))

        # ...but the dict used for logging is, with the sentinel shown as "default".
        loggable = _loggable_args(args)
        json.dumps(loggable)  # must not raise
        assert loggable["chat_prompt"] == "default"
        assert loggable["system_prompt"] == "default"
