"""Tests for the prompt variation framework in folktexts/prompting.py."""

from __future__ import annotations

import re

import pytest
from folktexts.prompting import (
    FeatureItem,
    FewShotConfig,
    PromptBuilder,
    PromptConfig,
    VaryConnector,
    VaryFeatureOrder,
    VaryFormat,
    VaryOrder,
    VaryPrefix,
    VarySuffix,
    VarySystemPrompt,
    VaryValueMap,
    encode_row_prompt,
    encode_row_prompt_few_shot,
)


def _make_items(task, row) -> list[FeatureItem]:
    """Create FeatureItems from a task and row (pre-VaryValueMap)."""
    return [
        FeatureItem(col=col, label=task.cols_to_text[col].short_description, raw_value=row[col])
        for col in task.features
        if col in row.index
    ]


class TestVaryValueMap:
    def test_original_returns_strings(self, acs_income_task, acs_row):
        items = _make_items(acs_income_task, acs_row)
        result = VaryValueMap(cols_to_text=acs_income_task.cols_to_text)(items)
        for item in result:
            assert isinstance(item.text_value, str), f"Expected str for col={item.col!r}, got {type(item.text_value)}"

    def test_original_age_exact(self, acs_income_task, acs_row):
        items = _make_items(acs_income_task, acs_row)
        result = VaryValueMap(cols_to_text=acs_income_task.cols_to_text)(items)
        agep = next(i for i in result if i.col == "AGEP")
        assert "years old" in agep.text_value
        assert "-" not in agep.text_value.split("years")[0]

    def test_original_wkhp_exact(self, acs_income_task, acs_row):
        items = _make_items(acs_income_task, acs_row)
        result = VaryValueMap(cols_to_text=acs_income_task.cols_to_text)(items)
        wkhp = next(i for i in result if i.col == "WKHP")
        assert "hours" in wkhp.text_value
        assert "-" not in wkhp.text_value.split("hours")[0]

    def test_low_returns_strings(self, acs_income_task, acs_row):
        from folktexts.acs.acs_columns_simplified import simplified_value_maps

        items = _make_items(acs_income_task, acs_row)
        vm = VaryValueMap.with_low_granularity(acs_income_task.cols_to_text, simplified_value_maps)
        result = vm(items)
        for item in result:
            assert isinstance(item.text_value, str), (
                f"Expected str for col={item.col!r} with low granularity, got {type(item.text_value)}"
            )

    def test_low_age_is_range(self, acs_income_task, acs_row):
        from folktexts.acs.acs_columns_simplified import simplified_value_maps

        items = _make_items(acs_income_task, acs_row)
        vm = VaryValueMap.with_low_granularity(acs_income_task.cols_to_text, simplified_value_maps)
        result = vm(items)
        agep = next(i for i in result if i.col == "AGEP")
        assert "years old" in agep.text_value
        age_part = agep.text_value.split("years")[0].strip()
        is_range = "-" in age_part
        is_edge = age_part.startswith("Less than") or age_part.endswith("or more")
        assert is_range or is_edge, f"Expected age range, got {agep.text_value!r}"

    def test_low_wkhp_is_range(self, acs_income_task, acs_row):
        from folktexts.acs.acs_columns_simplified import simplified_value_maps

        items = _make_items(acs_income_task, acs_row)
        vm = VaryValueMap.with_low_granularity(acs_income_task.cols_to_text, simplified_value_maps)
        result = vm(items)
        wkhp = next(i for i in result if i.col == "WKHP")
        is_range = "-" in wkhp.text_value.split("hours")[0]
        is_edge = wkhp.text_value.startswith("more than") or wkhp.text_value.startswith("N/A")
        assert is_range or is_edge, f"Expected hours range, got {wkhp.text_value!r}"

    def test_with_low_granularity_does_not_mutate_task(self, acs_income_task, acs_row):
        from folktexts.acs.acs_columns_simplified import simplified_value_maps

        original_map = acs_income_task.cols_to_text["AGEP"]._value_map
        VaryValueMap.with_low_granularity(acs_income_task.cols_to_text, simplified_value_maps)
        assert acs_income_task.cols_to_text["AGEP"]._value_map is original_map


class TestVaryOrder:
    def test_reversed(self, acs_income_task, acs_row):
        features = acs_income_task.features
        reversed_order = list(reversed(features))
        items = _make_items(acs_income_task, acs_row)
        result = VaryOrder(order=reversed_order)(items)
        assert [i.col for i in result] == reversed_order

    def test_none_leaves_order_unchanged(self, acs_income_task, acs_row):
        items = _make_items(acs_income_task, acs_row)
        result = VaryOrder(order=None)(items)
        assert [i.col for i in result] == [i.col for i in items]

    def test_alias(self):
        assert VaryFeatureOrder is VaryOrder


class TestVaryConnector:
    @pytest.mark.parametrize(
        "connector,expected_sep",
        [
            ("is", " is "),
            ("=", " = "),
            (":", ": "),
        ],
    )
    def test_connector(self, acs_income_task, acs_row, connector, expected_sep):
        items = _make_items(acs_income_task, acs_row)
        items = VaryValueMap(cols_to_text=acs_income_task.cols_to_text)(items)
        result = VaryConnector(connector=connector)(items)
        for item in result:
            assert expected_sep in item.connected, (
                f"connector={connector!r}: separator {expected_sep!r} not found in {item.connected!r}"
            )


class TestVaryFormat:
    @pytest.mark.parametrize(
        "fmt,expected_start",
        [
            ("bullet", "- "),
            ("comma", None),
            ("text", "The "),
            ("textbullet", "- The "),
        ],
    )
    def test_format(self, acs_income_task, acs_row, fmt, expected_start):
        items = _make_items(acs_income_task, acs_row)
        items = VaryValueMap(cols_to_text=acs_income_task.cols_to_text)(items)
        items = VaryConnector(connector="is")(items)
        result = VaryFormat(format=fmt)(items)
        assert isinstance(result, str)
        if expected_start:
            assert result.startswith(expected_start), (
                f"format={fmt!r}: expected start {expected_start!r}, got beginning {result[:20]!r}"
            )
        assert result[-1] not in {",", " ", "\n"}, f"format={fmt!r}: result ends with trailing {result[-1]!r}"

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError):
            VaryFormat(format="invalid_format")


class TestVarySystemPrompt:
    def test_returns_system_prompt_string(self):
        sp = "You are a helpful assistant."
        vsp = VarySystemPrompt(system_prompt=sp)
        assert vsp() == sp

    def test_default_derives_from_qa_type(self, acs_income_task):
        """Default config derives system_prompt from question type; pass None to suppress."""
        import dataclasses

        from folktexts.qa_interface import DirectNumericQA, MultipleChoiceQA

        config = PromptConfig.default(acs_income_task)
        if isinstance(acs_income_task.question, MultipleChoiceQA):
            assert config.system_prompt is not None
            assert "multiple-choice" in config.system_prompt().lower()

        numeric_qa = DirectNumericQA(column="PINCP", text="What is this person's estimated yearly income?")
        numeric_task = dataclasses.replace(acs_income_task, direct_numeric_qa=numeric_qa)
        numeric_task.set_question(numeric_task.direct_numeric_qa)
        numeric_config = PromptConfig.default(numeric_task)
        assert numeric_config.system_prompt is not None
        sp_text = numeric_config.system_prompt().lower()
        assert "numeric" in sp_text or "probabilit" in sp_text

    def test_suppress_default_system_prompt(self, acs_income_task):
        config_no_sp = PromptConfig.from_dict({}, task=acs_income_task, system_prompt=None)
        assert config_no_sp.system_prompt is None

    def test_explicit_system_prompt_overrides_default(self, acs_income_task):
        sp = "System instruction."
        config = PromptConfig.from_dict({}, task=acs_income_task, system_prompt=sp)
        assert config.system_prompt() == sp


class TestVaryPrefix:
    def test_contains_task_description(self):
        desc = "Custom task description.\n"
        vp = VaryPrefix(task_description=desc, add_task_description=True)
        result = vp()
        assert desc in result

    def test_no_task_description(self):
        vp = VaryPrefix(task_description="Some desc.\n", add_task_description=False)
        result = vp()
        assert "Information:" in result
        assert "Some desc." not in result

    def test_custom_prefix_appended(self):
        vp = VaryPrefix(task_description="Desc.\n", add_task_description=True, custom_prefix="Extra context.")
        result = vp()
        assert "Extra context." in result
        assert "Desc." in result


class TestVarySuffix:
    def test_contains_question_text(self, acs_income_task):
        vs = VarySuffix(question=acs_income_task.question, show_question=True)
        result = vs()
        assert acs_income_task.question.get_question_prompt() in result

    def test_show_question_false_uses_answer_prefix(self, acs_income_task):
        vs = VarySuffix(question=acs_income_task.question, show_question=False)
        result = vs()
        assert acs_income_task.question.get_answer_prefix() in result
        assert acs_income_task.question.get_question_prompt() not in result

    def test_show_label(self, acs_income_task):
        vs = VarySuffix(question=acs_income_task.question, show_question=False, show_label=True, label="A")
        result = vs()
        assert " A" in result

    def test_show_label_without_label_raises(self, acs_income_task):
        with pytest.raises(ValueError, match="show_label=True requires label"):
            VarySuffix(question=acs_income_task.question, show_label=True, label=None)


class TestFewShotConfig:
    def test_valid_construction(self):
        cfg = FewShotConfig(n_shots=3)
        assert cfg.n_shots == 3
        assert cfg.compose == "random"

    def test_n_shots_zero_raises(self):
        with pytest.raises(ValueError, match="n_shots must be >= 1"):
            FewShotConfig(n_shots=0)

    def test_n_shots_negative_raises(self):
        with pytest.raises(ValueError, match="n_shots must be >= 1"):
            FewShotConfig(n_shots=-1)

    def test_example_order_string_parsed(self):
        cfg = FewShotConfig(n_shots=3, example_order="2,0,1")
        assert cfg.example_order == (2, 0, 1)

    def test_example_order_invalid_permutation_raises(self):
        with pytest.raises(ValueError, match="permutation"):
            FewShotConfig(n_shots=3, example_order=[0, 1, 3])

    def test_example_order_wrong_length_raises(self):
        with pytest.raises(ValueError, match="permutation"):
            FewShotConfig(n_shots=3, example_order=[0, 1])

    def test_compose_list_converted_to_tuple(self):
        cfg = FewShotConfig(n_shots=3, compose=[2, 1])
        assert isinstance(cfg.compose, tuple)
        assert cfg.compose == (2, 1)

    def test_compose_string_with_commas_converted_to_tuple(self):
        cfg = FewShotConfig(n_shots=3, compose="2,1")
        assert isinstance(cfg.compose, tuple)
        assert cfg.compose == (2, 1)

    def test_compose_tuple_wrong_sum_raises(self):
        with pytest.raises(ValueError, match="sum"):
            FewShotConfig(n_shots=3, compose=[1, 1])

    def test_compose_tuple_negative_count_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            FewShotConfig(n_shots=3, compose=[-1, 4])

    def test_compose_invalid_string_raises(self):
        with pytest.raises(ValueError, match="compose"):
            FewShotConfig(n_shots=2, compose="circular")

    def test_compose_balanced_accepted(self):
        cfg = FewShotConfig(n_shots=2, compose="balanced")
        assert cfg.compose == "balanced"


class TestPromptBuilder:
    def test_build_returns_nonempty_string(self, acs_income_task, acs_row):
        config = PromptConfig.default(acs_income_task)
        prompt = PromptBuilder(acs_income_task).build(acs_row, config)
        assert isinstance(prompt, str) and len(prompt) > 0

    def test_build_contains_question(self, acs_income_task, acs_row):
        config = PromptConfig.default(acs_income_task)
        prompt = PromptBuilder(acs_income_task).build(acs_row, config)
        assert acs_income_task.question.get_question_prompt() in prompt

    def test_build_contains_task_description(self, acs_income_task, acs_row):
        config = PromptConfig.default(acs_income_task)
        prompt = PromptBuilder(acs_income_task).build(acs_row, config)
        assert "survey" in prompt.lower()


class TestEncodeRowPrompt:
    def test_returns_nonempty_string(self, acs_income_task, acs_row):
        prompt = encode_row_prompt(acs_row, task=acs_income_task)
        print(f"\n--- default ---\n{prompt}")
        assert isinstance(prompt, str) and len(prompt) > 0

    def test_contains_question(self, acs_income_task, acs_row):
        prompt = encode_row_prompt(acs_row, task=acs_income_task)
        assert acs_income_task.question.get_question_prompt() in prompt

    def test_contains_task_description(self, acs_income_task, acs_row):
        prompt = encode_row_prompt(acs_row, task=acs_income_task)
        assert "survey" in prompt.lower()

    def test_different_formats_produce_different_prompts(self, acs_income_task, acs_row):
        prompt_bullet = encode_row_prompt(
            acs_row,
            task=acs_income_task,
            prompt_config=PromptConfig.from_dict({"format": "bullet"}, task=acs_income_task),
        )
        prompt_comma = encode_row_prompt(
            acs_row,
            task=acs_income_task,
            prompt_config=PromptConfig.from_dict({"format": "comma"}, task=acs_income_task),
        )
        print(f"\n--- bullet ---\n{prompt_bullet}")
        print(f"\n--- comma ---\n{prompt_comma}")
        assert prompt_bullet != prompt_comma

    @pytest.mark.parametrize("connector", ["is", "=", ":"])
    def test_connector_variation(self, acs_income_task, acs_row, connector):
        prompt = encode_row_prompt(
            acs_row,
            task=acs_income_task,
            prompt_config=PromptConfig.from_dict({"connector": connector}, task=acs_income_task),
        )
        print(f"\n--- connector={connector!r} ---\n{prompt}")
        assert isinstance(prompt, str) and len(prompt) > 0

    def test_low_granularity_variation(self, acs_income_task, acs_row):
        prompt_orig = encode_row_prompt(
            acs_row,
            task=acs_income_task,
            prompt_config=PromptConfig.from_dict({"granularity": "original"}, task=acs_income_task),
        )
        prompt_low = encode_row_prompt(
            acs_row,
            task=acs_income_task,
            prompt_config=PromptConfig.from_dict({"granularity": "low"}, task=acs_income_task),
        )
        print(f"\n--- granularity=original ---\n{prompt_orig}")
        print(f"\n--- granularity=low ---\n{prompt_low}")
        assert isinstance(prompt_low, str) and len(prompt_low) > 0
        assert prompt_orig != prompt_low

    def test_order_variation(self, acs_income_task, acs_row):
        features = acs_income_task.features
        reversed_order = list(reversed(features))
        prompt_default = encode_row_prompt(acs_row, task=acs_income_task)
        prompt_reversed = encode_row_prompt(
            acs_row,
            task=acs_income_task,
            prompt_config=PromptConfig.from_dict({"order": reversed_order}, task=acs_income_task),
        )
        print(f"\n--- order=default ---\n{prompt_default}")
        print(f"\n--- order=reversed ---\n{prompt_reversed}")
        assert prompt_default != prompt_reversed

    def test_custom_prompt_prefix(self, acs_income_task, acs_row):
        prompt = encode_row_prompt(
            acs_row,
            task=acs_income_task,
            prompt_config=PromptConfig.from_dict({"custom_prompt_prefix": "Extra context here."}, task=acs_income_task),
        )
        print(f"\n--- custom_prompt_prefix ---\n{prompt}")
        assert "Extra context here." in prompt

    def test_custom_prompt_suffix(self, acs_income_task, acs_row):
        prompt = encode_row_prompt(
            acs_row,
            task=acs_income_task,
            prompt_config=PromptConfig.from_dict({"custom_prompt_suffix": " [end]"}, task=acs_income_task),
        )
        print(f"\n--- custom_prompt_suffix ---\n{prompt}")
        assert prompt.endswith(" [end]")

    def test_unknown_variation_key_raises(self, acs_income_task):
        with pytest.raises(ValueError, match="Unknown prompt_variation keys"):
            PromptConfig.from_dict({"nonexistent_key": "value"}, task=acs_income_task)


class TestEncodeRowPromptFewShot:
    @pytest.mark.parametrize("composition", ["random", "balanced"])
    def test_returns_string(self, acs_income_task, acs_income_dataset, acs_row, composition):
        prompt = encode_row_prompt_few_shot(
            acs_row,
            task=acs_income_task,
            dataset=acs_income_dataset,
            n_shots=2,
            reuse_examples=True,
            compose_few_shot_examples=composition,
        )
        print(f"\n--- few-shot (2 shots, composition={composition!r}) ---\n{prompt}")
        assert isinstance(prompt, str) and len(prompt) > 0

    def test_balanced_examples_contain_both_labels(self, acs_income_task, acs_income_dataset, acs_row):
        n_shots = 2
        prompt = encode_row_prompt_few_shot(
            acs_row,
            task=acs_income_task,
            dataset=acs_income_dataset,
            n_shots=n_shots,
            reuse_examples=True,
            compose_few_shot_examples="balanced",
        )
        print(f"\n--- few-shot balanced ---\n{prompt}")
        answer_prefix = acs_income_task.question.get_answer_prefix()
        answers = re.findall(rf"{re.escape(answer_prefix)}\s*(\w+)", prompt)
        assert len(answers) == n_shots, f"Expected {n_shots} answers, got {answers}"
        assert len(set(answers)) == 2, f"Expected both labels in balanced examples, got {set(answers)}"

    def test_question_appears_once_at_end(self, acs_income_task, acs_income_dataset, acs_row):
        prompt = encode_row_prompt_few_shot(
            acs_row,
            task=acs_income_task,
            dataset=acs_income_dataset,
            n_shots=2,
            reuse_examples=False,
        )
        question_text = acs_income_task.question.get_question_prompt()
        assert prompt.count(question_text) == 1
        assert prompt.endswith(question_text)


class TestOrderBiasCorrection:
    """Verify that a permuted question reaches the rendered prompt even when
    a pre-built PromptConfig is provided (regression test for the bug where
    `question` was silently dropped when `prompt_config` was not None)."""

    def _get_permuted_question(self, task):
        from folktexts.qa_interface import MultipleChoiceQA

        q = task.question
        assert isinstance(q, MultipleChoiceQA), "order-bias correction only applies to MultipleChoiceQA"
        permutations = list(MultipleChoiceQA.create_answer_keys_permutations(q))
        # Return the permutation whose choices differ from the original
        for perm in permutations:
            if perm.choices != q.choices:
                return perm
        pytest.skip("Only one permutation exists for this question; cannot test reordering.")

    def test_zero_shot_question_override_with_prompt_config(self, acs_income_task, acs_row):
        config = PromptConfig.from_dict({}, task=acs_income_task)
        permuted_q = self._get_permuted_question(acs_income_task)

        prompt_default = encode_row_prompt(acs_row, task=acs_income_task, prompt_config=config)
        prompt_permuted = encode_row_prompt(acs_row, task=acs_income_task, prompt_config=config, question=permuted_q)

        print(f"\n--- zero-shot default question ---\n{acs_income_task.question.get_question_prompt()}")
        print(f"\n--- zero-shot permuted question ---\n{permuted_q.get_question_prompt()}")

        assert prompt_default != prompt_permuted, (
            "Permuted question had no effect on the prompt — question override was silently dropped"
        )
        assert permuted_q.get_question_prompt() in prompt_permuted

    def test_few_shot_question_override_with_prompt_config(self, acs_income_task, acs_income_dataset, acs_row):
        config = PromptConfig.from_dict({}, task=acs_income_task)
        permuted_q = self._get_permuted_question(acs_income_task)

        prompt_default = encode_row_prompt_few_shot(
            acs_row,
            task=acs_income_task,
            dataset=acs_income_dataset,
            prompt_config=config,
            few_shot_config=FewShotConfig(n_shots=2, reuse_examples=True),
        )
        prompt_permuted = encode_row_prompt_few_shot(
            acs_row,
            task=acs_income_task,
            dataset=acs_income_dataset,
            prompt_config=config,
            question=permuted_q,
            few_shot_config=FewShotConfig(n_shots=2, reuse_examples=True),
        )

        print(f"\n--- few-shot default question ---\n{acs_income_task.question.get_question_prompt()}")
        print(f"\n--- few-shot permuted question ---\n{permuted_q.get_question_prompt()}")

        assert prompt_default != prompt_permuted, (
            "Permuted question had no effect on few-shot prompt — question override was silently dropped"
        )
        assert permuted_q.get_question_prompt() in prompt_permuted
