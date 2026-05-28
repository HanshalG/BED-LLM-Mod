from core.experiment import required_model_roles_for_config
from helpers import Config, ModelPair, ModelSpec, build_models


def test_build_models_keys_by_full_model_spec():
    qwen_plain = ModelSpec(model="Qwen/Qwen3.5-4B", thinking=False)
    qwen_thinking = ModelSpec(model="Qwen/Qwen3.5-4B", thinking=True)
    gemma_thinking = ModelSpec(model="google/gemma-4-E4B-it", thinking=True)
    qwen25_logprobs = ModelSpec(model="Qwen/Qwen2.5-7B-Instruct", thinking=False, use_logprobs=True)

    model_pairs = [
        ModelPair(questioner=qwen_plain, answerer=gemma_thinking),
        ModelPair(questioner=qwen_thinking, answerer=gemma_thinking),
        ModelPair(questioner=qwen25_logprobs, answerer=qwen_plain),
    ]

    calls = []

    def fake_build_model_adapter(spec):
        calls.append(spec)
        return f"adapter:{spec}"

    models = build_models(model_pairs, fake_build_model_adapter)

    assert len(models) == 4
    assert set(calls) == {qwen_plain, qwen_thinking, gemma_thinking, qwen25_logprobs}
    assert models[qwen_plain] != models[qwen_thinking]


def test_build_models_can_limit_to_environment_required_roles():
    questioner = ModelSpec(model="questioner")
    answerer = ModelSpec(model="answerer")
    pair = ModelPair(questioner=questioner, answerer=answerer)
    calls = []

    def fake_build_model_adapter(spec):
        calls.append(spec)
        return f"adapter:{spec.model}"

    models = build_models([pair], fake_build_model_adapter, roles=("questioner",))

    assert calls == [questioner]
    assert set(models) == {questioner}


def test_location_finding_declares_questioner_only_model_role():
    config = Config(task="location_finding")

    assert required_model_roles_for_config(config) == ("questioner",)
