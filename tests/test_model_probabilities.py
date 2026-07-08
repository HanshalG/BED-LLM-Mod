import sys
import types
import importlib.util
import json
import math
from enum import Enum
from pathlib import Path

import pytest

fake_torch_module = types.ModuleType("torch")
fake_torch_module.float16 = object()
fake_torch_module.float32 = object()
fake_torch_module.bfloat16 = object()


class _FakeCuda:
    @staticmethod
    def is_available():
        return True

    @staticmethod
    def device_count():
        return 1


fake_torch_module.cuda = _FakeCuda()
sys.modules.setdefault("torch", fake_torch_module)

fake_wandb_module = types.ModuleType("wandb")
fake_wandb_module.log = lambda *args, **kwargs: None
sys.modules.setdefault("wandb", fake_wandb_module)

fake_transformers_module = types.ModuleType("transformers")


class _FakeAutoTokenizer:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise AssertionError("Tokenizer construction is not expected in these tests")


class _FakeAutoModelForCausalLM:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise AssertionError("Model construction is not expected in these tests")


fake_transformers_module.AutoTokenizer = _FakeAutoTokenizer
sys.modules.setdefault("transformers", fake_transformers_module)

fake_vllm_module = types.ModuleType("vllm")


class _FakeLLM:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def generate(self, prompts, sampling_params):
        raise AssertionError("LLM generation is not expected in these tests")


class _FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


fake_vllm_module.LLM = _FakeLLM
fake_vllm_module.SamplingParams = _FakeSamplingParams
sys.modules.setdefault("vllm", fake_vllm_module)

fake_openai_harmony_module = types.ModuleType("openai_harmony")


class _FakeHarmonyRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    DEVELOPER = "developer"
    TOOL = "tool"


class _FakeHarmonyEncodingName(Enum):
    HARMONY_GPT_OSS = "harmony_gpt_oss"


class _FakeHarmonyContent:
    def __init__(self, text: str):
        self.text = text


class _FakeHarmonyMessage:
    def __init__(self, role, content):
        self.author = types.SimpleNamespace(role=role, name=None)
        self.content = [_FakeHarmonyContent(content)]
        self.channel = None
        self.recipient = None
        self.content_type = None

    @classmethod
    def from_role_and_content(cls, role, content):
        return cls(role, content)

    def with_channel(self, channel):
        self.channel = channel
        return self

    def with_recipient(self, recipient):
        self.recipient = recipient
        return self

    def with_content_type(self, content_type):
        self.content_type = content_type
        return self


class _FakeHarmonyConversation:
    def __init__(self, messages):
        self.messages = messages

    @classmethod
    def from_messages(cls, messages):
        return cls(messages)


class _FakeHarmonyEncoding:
    def __init__(self):
        self.render_calls = []
        self.parse_calls = []
        self.messages_to_return = []
    def render_conversation_for_completion(self, conversation, next_turn_role, config=None):
        self.render_calls.append((conversation, next_turn_role, config))
        return [11, 22, 33]

    def parse_messages_from_completion_tokens(self, tokens, role=None, strict=True):
        self.parse_calls.append((list(tokens), role, strict))
        return self.messages_to_return

    def decode_utf8(self, tokens):
        return self.decoded_text

    def stop_tokens_for_assistant_actions(self):
        return [200002, 200012]


def _fake_load_harmony_encoding(name):
    encoding = _FakeHarmonyEncoding()
    _fake_load_harmony_encoding.calls.append((name, encoding))
    return encoding


_fake_load_harmony_encoding.calls = []

fake_openai_harmony_module.Conversation = _FakeHarmonyConversation
fake_openai_harmony_module.HarmonyEncodingName = _FakeHarmonyEncodingName
fake_openai_harmony_module.Message = _FakeHarmonyMessage
fake_openai_harmony_module.Role = _FakeHarmonyRole
fake_openai_harmony_module.load_harmony_encoding = _fake_load_harmony_encoding
sys.modules["openai_harmony"] = fake_openai_harmony_module

ROOT = Path(__file__).resolve().parents[1]
helpers_spec = importlib.util.spec_from_file_location("real_helpers_module", ROOT / "helpers.py")
helpers = importlib.util.module_from_spec(helpers_spec)
assert helpers_spec.loader is not None
sys.modules[helpers_spec.name] = helpers
helpers_spec.loader.exec_module(helpers)

import environments.animals.prompts as prompts_module

spec = importlib.util.spec_from_file_location("real_model_module", ROOT / "model.py")
model = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = model
spec.loader.exec_module(model)


def _make_runtime_config() -> helpers.Config:
    return helpers.Config()


def _likelihood_conversation(question: str) -> list[dict[str, str]]:
    return prompts_module.answer_likelihood_messages("Wolverine", question, ["Yes", "No"])


class StubProbabilityModel(model.Model):
    def __init__(self, completions_by_question: dict[str, str | list[str]], fallback_to_uniform: bool = False):
        self.completions_by_question = completions_by_question
        self.fallback_to_uniform = fallback_to_uniform
        self.block_questions: list[list[str]] = []
        self.temperatures: list[float] = []
        self.max_new_tokens: list[int] = []

    def _chat_complete_messages_batched(self, batch_messages, temperature, block_size, max_new_tokens):
        self.temperatures.append(temperature)
        self.max_new_tokens.append(max_new_tokens)
        completions: list[str] = []

        for start_idx in range(0, len(batch_messages), block_size):
            block_messages = batch_messages[start_idx:start_idx + block_size]
            block_questions: list[str] = []

            for messages in block_messages:
                last_user_message = messages[-1]["content"]
                prefix = "Question:\n"
                suffix = "\n\nAllowed answer labels:"
                question_start = last_user_message.find(prefix)
                question_end = last_user_message.find(suffix)
                assert question_start >= 0
                assert question_end > question_start
                question = last_user_message[question_start + len(prefix):question_end]
                assert (
                    "Return exactly one JSON object using exactly these keys" in last_user_message
                )
                block_questions.append(question)
                question_completion = self.completions_by_question[question]
                if isinstance(question_completion, list):
                    if len(question_completion) > 1:
                        completions.append(question_completion.pop(0))
                    else:
                        completions.append(question_completion[0])
                else:
                    completions.append(question_completion)

            self.block_questions.append(block_questions)

        return completions

    def chat_complete(self, messages: list[dict[str, str]], temperature: float, num_responses: int = 1) -> list[str]:
        raise AssertionError("chat_complete should not be used in these tests")

    def chat_complete_messages_batched(self, batch_messages: list[list[dict[str, str]]], temperature: float,
                                       block_size: int, max_new_tokens: int = 8192) -> list[str]:
        return self._chat_complete_messages_batched(
            batch_messages,
            temperature,
            block_size,
            max_new_tokens,
        )

    def chat_probabilities_messages_batched(self, messages: list[list[dict[str, str]]], responses: list[str],
                                            temperature: float, block_size: int) -> list[dict[str, float]]:
        return helpers._probability_results_from_messages(
            messages,
            responses,
            block_size,
            temperature,
            self._chat_complete_messages_batched,
            fallback_to_uniform=self.fallback_to_uniform,
        )


def test_normalize_probability_response_preserves_valid_json():
    probabilities = helpers._normalize_probability_response(
        '{"Yes": 0.8, "No": 0.2}',
        ["Yes", "No"],
    )

    assert probabilities == pytest.approx({"Yes": 0.8, "No": 0.2})


def test_normalize_probability_response_normalizes_weights_and_ignores_extra_keys():
    probabilities = helpers._normalize_probability_response(
        '```json\n{"Yes": "2", "No": 1, "Maybe": 99}\n```',
        ["Yes", "No"],
    )

    assert probabilities == pytest.approx({"Yes": 2 / 3, "No": 1 / 3})


def test_normalize_probability_response_handles_missing_keys():
    probabilities = helpers._normalize_probability_response(
        '{"Yes": 4, "Maybe": 1}',
        ["Yes", "No"],
    )

    assert probabilities == pytest.approx({"Yes": 1.0, "No": 0.0})


def test_normalize_probability_response_extracts_first_balanced_json_object():
    probabilities = helpers._normalize_probability_response(
        'analysis {"Yes": 7, "No": 3}\n\nextra trailing text',
        ["Yes", "No"],
    )

    assert probabilities == pytest.approx({"Yes": 0.7, "No": 0.3})


@pytest.mark.parametrize(
    "payload",
    [
        "not valid json",
        'analysis {"Yes": 1, "No": 0',
        '{"Yes": "high", "No": "low"}',
        '{"Yes": 0, "No": 0}',
    ],
)
def test_normalize_probability_response_raises_for_invalid_payloads(payload):
    with pytest.raises(ValueError):
        helpers._normalize_probability_response(payload, ["Yes", "No"])


def test_answer_likelihood_prompt_construction_uses_dedicated_likelihood_role():
    messages = prompts_module.answer_likelihood_messages(
        "Wolverine",
        "Is it native to North America?",
        ["Yes", "No"],
    )

    assert messages[0]["role"] == "system"
    assert "You estimate answer likelihoods" in messages[0]["content"]
    assert "reply exactly" not in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "Hypothesized target animal:\nWolverine" in messages[1]["content"]
    assert "Question:\nIs it native to North America?" in messages[1]["content"]
    assert 'Allowed answer labels:\n["Yes", "No"]' in messages[1]["content"]
    assert "values must sum to 1" in messages[1]["content"]


def test_probability_results_accepts_dedicated_likelihood_messages():
    # The validation that was formerly in _build_probability_messages is now inlined
    # into _probability_results_from_messages.  Dedicated answer_likelihood_messages
    # must be passed through unchanged to the batched completion callable.
    messages = prompts_module.answer_likelihood_messages(
        "Wolverine",
        "Is it native to North America?",
        ["Yes", "No"],
    )

    captured: list[list[dict[str, str]]] = []

    def fake_complete_messages_batched(batch_messages, *, temperature, block_size, max_new_tokens):
        captured.append([list(m) for m in batch_messages])
        # Return a valid JSON completion for each prompt so parsing succeeds.
        return ['{"Yes": 0.7, "No": 0.3}'] * len(batch_messages)

    results = helpers._probability_results_from_messages(
        [messages],
        ["Yes", "No"],
        block_size=4,
        temperature=0.0,
        complete_messages_batched=fake_complete_messages_batched,
    )

    assert len(results) == 1
    assert set(results[0].keys()) == {"Yes", "No"}
    # The exact same messages should have been forwarded (modulo shallow-copy).
    assert captured[0][0] == messages
    # And there should be no probability-prompt augmentation.
    assert "Estimate the relative probability of each possible answer" not in captured[0][0][-1]["content"]


def test_probability_results_rejects_generic_answerer_messages():
    messages = [
        prompts_module.answer_question_yesno_system_prompt("Wolverine"),
        {"role": "user", "content": "Is it native to North America?"},
    ]

    def fake_complete_messages_batched(batch_messages, *, temperature, block_size, max_new_tokens):
        raise AssertionError("Completion should not be called when validation fails")

    with pytest.raises(ValueError, match="answer_likelihood_messages"):
        helpers._probability_results_from_messages(
            [messages],
            ["Yes", "No"],
            block_size=4,
            temperature=0.0,
            complete_messages_batched=fake_complete_messages_batched,
        )


def test_chat_probabilities_messages_batched_preserves_order_across_blocks_and_ignores_temperature():
    stub_model = StubProbabilityModel(
        {
            "Question A?": '{"Yes": 1, "No": 0}',
            "Question B?": '{"Yes": 0, "No": 5}',
            "Question C?": '{"Yes": 3, "No": 1}',
        }
    )
    conversations = [
        _likelihood_conversation("Question A?"),
        _likelihood_conversation("Question B?"),
        _likelihood_conversation("Question C?"),
    ]

    low_temp = stub_model.chat_probabilities_messages_batched(
        conversations,
        ["Yes", "No"],
        temperature=0.1,
        block_size=2,
    )
    high_temp = stub_model.chat_probabilities_messages_batched(
        conversations,
        ["Yes", "No"],
        temperature=1.7,
        block_size=2,
    )

    expected = [
        {"Yes": 1.0, "No": 0.0},
        {"Yes": 0.0, "No": 1.0},
        {"Yes": 0.75, "No": 0.25},
    ]

    assert low_temp == pytest.approx(expected)
    assert high_temp == pytest.approx(expected)
    assert stub_model.temperatures == [0.1, 1.7]
    assert stub_model.max_new_tokens == [None, None]
    assert stub_model.block_questions == [
        ["Question A?", "Question B?"],
        ["Question C?"],
        ["Question A?", "Question B?"],
        ["Question C?"],
    ]


def test_chat_probabilities_messages_batched_retries_only_failed_items_and_preserves_order():
    stub_model = StubProbabilityModel(
        {
            "Question A?": ['analysis {"Yes": 1, "No": 0}'],
            "Question B?": ["not valid json", '{"Yes": 0, "No": 4}'],
            "Question C?": ['{"Yes": 1, "No": 3}'],
        }
    )
    conversations = [
        _likelihood_conversation("Question A?"),
        _likelihood_conversation("Question B?"),
        _likelihood_conversation("Question C?"),
    ]

    probabilities = stub_model.chat_probabilities_messages_batched(
        conversations,
        ["Yes", "No"],
        temperature=0.25,
        block_size=2,
    )

    assert probabilities == pytest.approx([
        {"Yes": 1.0, "No": 0.0},
        {"Yes": 0.0, "No": 1.0},
        {"Yes": 0.25, "No": 0.75},
    ])
    assert stub_model.block_questions == [
        ["Question A?", "Question B?"],
        ["Question C?"],
        ["Question B?"],
    ]


def test_chat_probabilities_messages_batched_raises_after_exhausting_retries_by_default():
    stub_model = StubProbabilityModel(
        {
            "Question A?": ["still bad", "still bad", "still bad"],
        }
    )

    with pytest.raises(ValueError, match="Failed to parse probability JSON after 3 attempts"):
        stub_model.chat_probabilities_messages_batched(
            [_likelihood_conversation("Question A?")],
            ["Yes", "No"],
            temperature=0.4,
            block_size=1,
        )


def test_chat_probabilities_messages_batched_can_fall_back_to_uniform_when_enabled():
    stub_model = StubProbabilityModel(
        {
            "Question A?": ["still bad", "still bad", "still bad"],
        },
        fallback_to_uniform=True,
    )

    probabilities = stub_model.chat_probabilities_messages_batched(
        [_likelihood_conversation("Question A?")],
        ["Yes", "No"],
        temperature=0.4,
        block_size=1,
    )

    assert probabilities == pytest.approx([
        {"Yes": 0.5, "No": 0.5},
    ])


def test_chat_probabilities_messages_batched_defaults_to_uniform_fallback_in_runtime_config():
    class _StubAdapter:
        def __init__(self):
            self.use_logprobs = False
            self.config = _make_runtime_config()

        def _chat_complete_messages_batched(self, batch_messages, temperature, block_size, max_new_tokens):
            return ["still bad"] * len(batch_messages)

    adapter = _StubAdapter()

    probabilities = model.BaseVLLMAdapter.chat_probabilities_messages_batched(
        adapter,
        [_likelihood_conversation("Question A?")],
        ["Yes", "No"],
        temperature=0.4,
        block_size=1,
    )

    assert probabilities == pytest.approx([
        {"Yes": 0.5, "No": 0.5},
    ])


def test_chat_complete_messages_batched_delegates_to_internal_batcher():
    class _StubAdapter:
        def __init__(self):
            self.calls = []

        def _chat_complete_messages_batched(self, batch_messages, temperature, block_size, max_new_tokens):
            self.calls.append(
                {
                    "batch_messages": batch_messages,
                    "temperature": temperature,
                    "block_size": block_size,
                    "max_new_tokens": max_new_tokens,
                }
            )
            return ["A", "B"]

    adapter = _StubAdapter()
    batch_messages = [
        [{"role": "user", "content": "Question A?"}],
        [{"role": "user", "content": "Question B?"}],
    ]

    completions = model.BaseVLLMAdapter.chat_complete_messages_batched(
        adapter,
        batch_messages,
        temperature=0.6,
        block_size=4,
        max_new_tokens=32,
    )

    assert completions == ["A", "B"]
    assert adapter.calls == [
        {
            "batch_messages": batch_messages,
            "temperature": 0.6,
            "block_size": 4,
            "max_new_tokens": 32,
        }
    ]


class _TokenizerRecorder:
    def __init__(self, prompt: str = "prompt"):
        self.prompt = prompt
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return self.prompt


class _TokenizerWithIds(_TokenizerRecorder):
    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return messages[-1]["content"]

    def __call__(self, texts, add_special_tokens=False):
        if isinstance(texts, str):
            return types.SimpleNamespace(input_ids=[ord(char) for char in texts])
        return types.SimpleNamespace(
            input_ids=[[ord(char) for char in text] for text in texts]
        )


class _FakePromptLogprob:
    def __init__(self, logprob: float):
        self.logprob = logprob


def test_build_model_adapter_returns_harmony_adapter_for_gpt_oss():
    adapter = model.build_model_adapter(
        helpers.ModelSpec(model="openai/gpt-oss-20b", reasoning_effort="medium"),
        config=_make_runtime_config(),
        tensor_parallel_size=1,
    )

    prompt = adapter._messages_to_prompt([{"role": "user", "content": "Hi"}])

    assert isinstance(adapter, model.HarmonyVLLMAdapter)
    assert adapter.llm.kwargs["dtype"] == "bfloat16"
    assert adapter.tokenizer is None
    assert _fake_load_harmony_encoding.calls[-1][0] == _FakeHarmonyEncodingName.HARMONY_GPT_OSS
    assert prompt == {"prompt_token_ids": [11, 22, 33]}
    harmony_conversation, next_turn_role, _ = adapter._harmony_encoding.render_calls[-1]
    assert [message.author.role for message in harmony_conversation.messages] == [
        _FakeHarmonyRole.SYSTEM,
        _FakeHarmonyRole.USER,
    ]
    assert harmony_conversation.messages[0].content[0].text.startswith("You are ChatGPT")
    assert "Reasoning: medium" in harmony_conversation.messages[0].content[0].text
    assert harmony_conversation.messages[1].content[0].text == "Hi"
    assert next_turn_role == _FakeHarmonyRole.ASSISTANT


def test_harmony_prompt_remaps_local_system_to_developer_and_assistant_to_final():
    adapter = model.HarmonyVLLMAdapter(
        spec=helpers.ModelSpec(model="openai/gpt-oss-20b", reasoning_effort="low"),
        config=_make_runtime_config(),
        tensor_parallel_size=1,
    )

    adapter._messages_to_prompt([
        {"role": "system", "content": "Follow the user instructions exactly"},
        {"role": "assistant", "content": "Previous answer"},
        {"role": "user", "content": "Next turn"},
    ])

    harmony_conversation, _, _ = adapter._harmony_encoding.render_calls[-1]

    assert [message.author.role for message in harmony_conversation.messages] == [
        _FakeHarmonyRole.SYSTEM,
        _FakeHarmonyRole.DEVELOPER,
        _FakeHarmonyRole.ASSISTANT,
        _FakeHarmonyRole.USER,
    ]
    assert harmony_conversation.messages[1].content[0].text == "# Instructions\n\nFollow the user instructions exactly"
    assert harmony_conversation.messages[2].channel == "final"
    assert harmony_conversation.messages[2].content[0].text == "Previous answer"


def test_base_vllm_adapter_uses_chat_template_for_non_harmony_models(monkeypatch):
    tokenizer = _TokenizerRecorder(prompt="plain prompt")
    tokenizer_calls = []

    def fake_from_pretrained(model_name, **kwargs):
        tokenizer_calls.append((model_name, kwargs))
        return tokenizer

    monkeypatch.setattr(model.AutoTokenizer, "from_pretrained", fake_from_pretrained)

    adapter = model.BaseVLLMAdapter(
        spec=helpers.ModelSpec(model="meta/llama-3"),
        config=_make_runtime_config(),
        tensor_parallel_size=1,
    )

    prompt = adapter._messages_to_prompt([{"role": "user", "content": "Hi"}])

    assert prompt == "plain prompt"
    assert tokenizer_calls == [
        (
            "meta/llama-3",
            {},
        )
    ]
    assert tokenizer.calls == [
        (
            [{"role": "user", "content": "Hi"}],
            {"tokenize": False, "add_generation_prompt": True},
        )
    ]


def test_normalize_model_spec_allows_logprobs_for_qwen25_only():
    supported = helpers._normalize_model_spec(
        {"model": "Qwen/Qwen2.5-7B-Instruct", "use_logprobs": True},
        "model_pairs[0].questioner",
    )

    assert supported == helpers.ModelSpec(
        model="Qwen/Qwen2.5-7B-Instruct",
        thinking=False,
        use_logprobs=True,
    )

    with pytest.raises(ValueError, match="use_logprobs is only supported for Qwen2.5 models"):
        helpers._normalize_model_spec(
            {"model": "Qwen/Qwen3.5-4B", "use_logprobs": True},
            "model_pairs[0].questioner",
        )


def test_base_vllm_adapter_uses_prompt_logprobs_when_enabled():
    adapter = model.BaseVLLMAdapter.__new__(model.BaseVLLMAdapter)
    adapter.model_name = "Qwen/Qwen2.5-7B-Instruct"
    adapter.use_logprobs = True
    adapter.tokenizer = _TokenizerWithIds()

    captured_sampling_params = []
    response_log_scores = {
        ("Question A?", "Yes"): [-0.05, -0.05, -0.05],
        ("Question A?", "No"): [-0.9, -0.9],
        ("Question B?", "Yes"): [-0.8, -0.8, -0.8],
        ("Question B?", "No"): [-0.1, -0.1],
    }

    def fake_generate(prompts, sampling_params):
        captured_sampling_params.append(sampling_params)
        outputs = []
        for prompt in prompts:
            if prompt.endswith("Yes"):
                question = prompt[:-3]
                response = "Yes"
            else:
                question = prompt[:-2]
                response = "No"

            response_token_ids = [ord(char) for char in response]
            prompt_logprobs = [None] * len(prompt)
            for offset, (token_id, logprob) in enumerate(
                zip(response_token_ids, response_log_scores[(question, response)]),
                start=len(question),
            ):
                prompt_logprobs[offset] = {token_id: _FakePromptLogprob(logprob)}

            outputs.append(types.SimpleNamespace(prompt_logprobs=prompt_logprobs))

        return outputs

    adapter.llm = types.SimpleNamespace(generate=fake_generate)

    probabilities = adapter.chat_probabilities_messages_batched(
        [
            [{"role": "user", "content": "Question A?"}],
            [{"role": "user", "content": "Question B?"}],
        ],
        ["Yes", "No"],
        temperature=0.5,
        block_size=1,
    )

    expected_question_a_no = math.exp(-3.3) / (1 + math.exp(-3.3))
    expected_question_b_yes = math.exp(-4.4) / (1 + math.exp(-4.4))
    assert probabilities[0]["Yes"] == pytest.approx(1 - expected_question_a_no)
    assert probabilities[0]["No"] == pytest.approx(expected_question_a_no)
    assert probabilities[1]["Yes"] == pytest.approx(expected_question_b_yes)
    assert probabilities[1]["No"] == pytest.approx(1 - expected_question_b_yes)
    assert [params.kwargs for params in captured_sampling_params] == [
        {"temperature": 0.0, "max_tokens": 1, "prompt_logprobs": 1},
        {"temperature": 0.0, "max_tokens": 1, "prompt_logprobs": 1},
        {"temperature": 0.0, "max_tokens": 1, "prompt_logprobs": 1},
        {"temperature": 0.0, "max_tokens": 1, "prompt_logprobs": 1},
    ]


def test_normalize_completion_output_extracts_final_harmony_message():
    adapter = model.HarmonyVLLMAdapter(
        spec=helpers.ModelSpec(model="openai/gpt-oss-20b", reasoning_effort="low"),
        config=_make_runtime_config(),
        tensor_parallel_size=1,
    )
    adapter._harmony_encoding.messages_to_return = [
        _FakeHarmonyMessage(_FakeHarmonyRole.ASSISTANT, "scratch").with_channel("analysis"),
        _FakeHarmonyMessage(_FakeHarmonyRole.ASSISTANT, '{"Yes": 1}').with_channel("final"),
    ]
    output = types.SimpleNamespace(text="raw text", token_ids=[101, 102, 103])

    assert adapter._normalize_completion_output(output) == '{"Yes": 1}'
    assert adapter._harmony_encoding.parse_calls[-1] == (
        [101, 102, 103],
        _FakeHarmonyRole.ASSISTANT,
        True,
    )


def test_normalize_completion_output_requires_final_harmony_message():
    adapter = model.HarmonyVLLMAdapter(
        spec=helpers.ModelSpec(model="openai/gpt-oss-20b", reasoning_effort="low"),
        config=_make_runtime_config(),
        tensor_parallel_size=1,
    )
    adapter._harmony_encoding.messages_to_return = [
        _FakeHarmonyMessage(_FakeHarmonyRole.ASSISTANT, "scratch").with_channel("analysis"),
    ]

    output = types.SimpleNamespace(text="plain fallback", token_ids=[1, 2, 3])

    with pytest.raises(ValueError, match="final harmony message"):
        adapter._normalize_completion_output(output)


def test_normalize_completion_output_requires_final_harmony_channel():
    adapter = model.HarmonyVLLMAdapter(
        spec=helpers.ModelSpec(model="openai/gpt-oss-20b", reasoning_effort="low"),
        config=_make_runtime_config(),
        tensor_parallel_size=1,
    )
    adapter._harmony_encoding.messages_to_return = [
        _FakeHarmonyMessage(_FakeHarmonyRole.ASSISTANT, "plain fallback"),
    ]

    output = types.SimpleNamespace(text="ignored", token_ids=[1, 2, 3])

    with pytest.raises(ValueError, match="final harmony message"):
        adapter._normalize_completion_output(output)


def test_normalize_completion_output_requires_harmony_token_ids():
    adapter = model.HarmonyVLLMAdapter(
        spec=helpers.ModelSpec(model="openai/gpt-oss-20b", reasoning_effort="low"),
        config=_make_runtime_config(),
        tensor_parallel_size=1,
    )

    with pytest.raises(ValueError, match="missing token_ids"):
        adapter._normalize_completion_output(types.SimpleNamespace(text="plain fallback", token_ids=[]))


def test_chat_complete_uses_harmony_stop_tokens_and_parsed_final_output():
    adapter = model.HarmonyVLLMAdapter(
        spec=helpers.ModelSpec(model="openai/gpt-oss-20b", reasoning_effort="low"),
        config=_make_runtime_config(),
        tensor_parallel_size=1,
    )
    adapter._harmony_encoding.messages_to_return = [
        _FakeHarmonyMessage(_FakeHarmonyRole.ASSISTANT, "result").with_channel("final"),
    ]

    captured = {}

    def fake_generate(prompts, sampling_params):
        captured["prompts"] = prompts
        captured["sampling_params"] = sampling_params
        return [
            types.SimpleNamespace(
                outputs=[types.SimpleNamespace(text="ignored", token_ids=[9, 8, 7])],
                prompt_token_ids=[11, 22, 33],
            )
        ]

    adapter.llm.generate = fake_generate

    completions = adapter.chat_complete(
        [{"role": "user", "content": "Say hi"}],
        temperature=0.3,
    )

    assert completions == ["result"]
    assert captured["prompts"] == [{"prompt_token_ids": [11, 22, 33]}]
    assert captured["sampling_params"].kwargs["stop_token_ids"] == [200002, 200012]
    assert captured["sampling_params"].kwargs["skip_special_tokens"] is False


def test_chat_complete_logs_token_usage_jsonl(tmp_path):
    adapter = model.BaseVLLMAdapter.__new__(model.BaseVLLMAdapter)
    adapter.model_name = "test/model"
    adapter.config = helpers.Config(
        log_path=tmp_path / "run.log",
    )
    adapter.config.location_max_new_tokens = 16
    adapter.tokenizer = _TokenizerWithIds()
    adapter.max_model_len = 1000
    adapter.use_logprobs = False

    def fake_generate(prompts, sampling_params):
        assert prompts == ["Prompt"]
        assert sampling_params.kwargs["max_tokens"] == 16
        return [
            types.SimpleNamespace(
                outputs=[
                    types.SimpleNamespace(
                        text=" done",
                        token_ids=[1, 2, 3],
                        finish_reason="stop",
                    )
                ],
                prompt_token_ids=[11, 22],
            )
        ]

    adapter.llm = types.SimpleNamespace(generate=fake_generate)

    completions = adapter.chat_complete(
        [{"role": "user", "content": "Prompt"}],
        temperature=0.2,
    )

    assert completions == ["done"]
    records = [
        json.loads(line)
        for line in adapter.config.log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert records == [
        {
            "event": "llm_token_usage",
            "model": "test/model",
            "call_type": "chat",
            "prompt_tokens": 2,
            "completion_tokens": 3,
            "total_tokens": 5,
            "max_new_tokens": 16,
            "temperature": 0.2,
            "response_index": 0,
            "finish_reason": "stop",
        }
    ]


def test_chat_complete_messages_batched_logs_token_usage_jsonl(tmp_path):
    adapter = model.BaseVLLMAdapter.__new__(model.BaseVLLMAdapter)
    adapter.model_name = "test/model"
    adapter.config = helpers.Config(
        log_path=tmp_path / "run.log",
    )
    adapter.config.location_max_new_tokens = 16
    adapter.tokenizer = _TokenizerWithIds()
    adapter.max_model_len = 1000
    adapter.use_logprobs = False

    def fake_generate(prompts, sampling_params):
        assert prompts == ["Prompt A", "Prompt B"]
        assert sampling_params.kwargs["max_tokens"] == 12
        return [
            types.SimpleNamespace(
                outputs=[types.SimpleNamespace(text=" A", token_ids=[1])],
                prompt_token_ids=[11, 22, 33],
            ),
            types.SimpleNamespace(
                outputs=[types.SimpleNamespace(text=" B", token_ids=[4, 5])],
                prompt_token_ids=[44],
            ),
        ]

    adapter.llm = types.SimpleNamespace(generate=fake_generate)

    completions = adapter.chat_complete_messages_batched(
        [
            [{"role": "user", "content": "Prompt A"}],
            [{"role": "user", "content": "Prompt B"}],
        ],
        temperature=0.4,
        block_size=2,
        max_new_tokens=12,
    )

    assert completions == ["A", "B"]
    records = [
        json.loads(line)
        for line in adapter.config.log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert records == [
        {
            "event": "llm_token_usage",
            "model": "test/model",
            "call_type": "batched_chat",
            "prompt_tokens": 3,
            "completion_tokens": 1,
            "total_tokens": 4,
            "max_new_tokens": 12,
            "temperature": 0.4,
            "block_index": 0,
        },
        {
            "event": "llm_token_usage",
            "model": "test/model",
            "call_type": "batched_chat",
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "total_tokens": 3,
            "max_new_tokens": 12,
            "temperature": 0.4,
            "block_index": 1,
        },
    ]


def test_normalize_completion_output_falls_back_to_plain_text_for_base_adapter():
    adapter = model.BaseVLLMAdapter.__new__(model.BaseVLLMAdapter)

    assert adapter._normalize_completion_output(types.SimpleNamespace(text="  plain text")) == "plain text"


def test_qwen_adapter_passes_enable_thinking_to_chat_template(monkeypatch):
    tokenizer = _TokenizerRecorder(prompt="qwen prompt")
    tokenizer_calls = []

    def fake_from_pretrained(*args, **kwargs):
        tokenizer_calls.append((args, kwargs))
        return tokenizer

    monkeypatch.setattr(model.AutoTokenizer, "from_pretrained", fake_from_pretrained)

    adapter = model.QwenVLLMAdapter(
        spec=helpers.ModelSpec(model="Qwen/Qwen3.5-4B", thinking=True),
        config=_make_runtime_config(),
        tensor_parallel_size=1,
    )

    prompt = adapter._messages_to_prompt([{"role": "user", "content": "Hi"}])

    assert prompt == "qwen prompt"
    assert tokenizer_calls == [(("Qwen/Qwen3.5-4B",), {})]
    assert tokenizer.calls[-1] == (
        [{"role": "user", "content": "Hi"}],
        {"tokenize": False, "add_generation_prompt": True, "enable_thinking": True},
    )


def test_qwen_adapter_strips_think_tags_from_output():
    adapter = model.QwenVLLMAdapter.__new__(model.QwenVLLMAdapter)
    adapter.model_name = "Qwen/Qwen3.5-4B"
    adapter.thinking = True

    output = types.SimpleNamespace(text="Thinking Process:\n\ndraft\n</think>\n\n{\"Yes\": 1}")

    assert adapter._normalize_completion_output(output) == '{"Yes": 1}'


def test_qwen_adapter_returns_raw_text_when_only_reasoning_is_present():
    # Reasoning-only output (no </think> separator) is now returned verbatim so the
    # downstream JSON parser/fallback can handle it.  This used to raise, but the
    # adapter intentionally became fail-soft.
    adapter = model.QwenVLLMAdapter.__new__(model.QwenVLLMAdapter)
    adapter.model_name = "Qwen/Qwen3.5-4B"
    adapter.thinking = True

    result = adapter._normalize_completion_output(types.SimpleNamespace(text="Thinking Process:\n\ndraft"))
    assert result == "Thinking Process:\n\ndraft"


def _thinking_adapter(adapter_cls, *, max_model_len=10_000, thinking_budget=3, final_budget=2):
    adapter = adapter_cls.__new__(adapter_cls)
    adapter.model_name = "Qwen/Qwen3.5-4B" if adapter_cls is model.QwenVLLMAdapter else "google/gemma-4-E4B-it"
    adapter.config = _make_runtime_config()
    adapter.tokenizer = _TokenizerWithIds()
    adapter.max_model_len = max_model_len
    adapter.thinking = True
    adapter.thinking_max_new_tokens = thinking_budget
    adapter.thinking_final_max_new_tokens = final_budget
    adapter.use_logprobs = False
    return adapter


def test_qwen_forced_thinking_exit_closes_think_channel_and_returns_final_answer():
    adapter = _thinking_adapter(model.QwenVLLMAdapter, thinking_budget=3, final_budget=2)
    calls = []

    def fake_generate(prompts, sampling_params):
        calls.append((prompts, sampling_params))
        if len(calls) == 1:
            return [
                types.SimpleNamespace(
                    outputs=[types.SimpleNamespace(text="draft reasoning", token_ids=[1, 2, 3])],
                    prompt_token_ids=[11, 22],
                )
            ]
        return [
            types.SimpleNamespace(
                outputs=[types.SimpleNamespace(text="Final Answer: {\"Yes\": 1}", token_ids=[4])],
                prompt_token_ids=[33, 44],
            )
        ]

    adapter.llm = types.SimpleNamespace(generate=fake_generate)

    completions = adapter.chat_complete([{"role": "user", "content": "Prompt"}], temperature=0.0)

    assert completions == ['{"Yes": 1}']
    assert calls[0][1].kwargs["max_tokens"] == 3
    assert calls[1][1].kwargs["max_tokens"] == 2
    assert calls[1][0] == ["Promptdraft reasoning\n</think>\nFinal Answer:"]


def test_qwen_forced_thinking_exit_preserves_batched_order():
    adapter = _thinking_adapter(model.QwenVLLMAdapter, thinking_budget=3, final_budget=2)
    calls = []

    def fake_generate(prompts, sampling_params):
        calls.append((prompts, sampling_params))
        if len(calls) == 1:
            return [
                types.SimpleNamespace(
                    outputs=[types.SimpleNamespace(text="<think>a</think>\nA", token_ids=[1])],
                    prompt_token_ids=[11],
                ),
                types.SimpleNamespace(
                    outputs=[types.SimpleNamespace(text="draft", token_ids=[1, 2, 3])],
                    prompt_token_ids=[22],
                ),
                types.SimpleNamespace(
                    outputs=[types.SimpleNamespace(text="short draft", token_ids=[1])],
                    prompt_token_ids=[33],
                ),
            ]
        return [
            types.SimpleNamespace(
                outputs=[types.SimpleNamespace(text="Final Answer: B", token_ids=[4])],
                prompt_token_ids=[44],
            ),
            types.SimpleNamespace(
                outputs=[types.SimpleNamespace(text="Final Answer: C", token_ids=[5])],
                prompt_token_ids=[55],
            )
        ]

    adapter.llm = types.SimpleNamespace(generate=fake_generate)

    completions = adapter.chat_complete_messages_batched(
        [
            [{"role": "user", "content": "Prompt A"}],
            [{"role": "user", "content": "Prompt B"}],
            [{"role": "user", "content": "Prompt C"}],
        ],
        temperature=0.0,
        block_size=3,
        max_new_tokens=99,
    )

    assert completions == ["A", "B", "C"]
    assert calls[1][0] == [
        "Prompt Bdraft\n</think>\nFinal Answer:",
        "Prompt Cshort draft\n</think>\nFinal Answer:",
    ]


def test_qwen_forced_thinking_exit_runs_for_short_thought_only_output():
    adapter = _thinking_adapter(model.QwenVLLMAdapter, thinking_budget=10, final_budget=2)
    calls = []

    def fake_generate(prompts, sampling_params):
        calls.append((prompts, sampling_params))
        if len(calls) == 1:
            return [
                types.SimpleNamespace(
                    outputs=[types.SimpleNamespace(text="short draft", token_ids=[1])],
                    prompt_token_ids=[11],
                )
            ]
        return [
            types.SimpleNamespace(
                outputs=[types.SimpleNamespace(text="Final Answer: done", token_ids=[2])],
                prompt_token_ids=[22],
            )
        ]

    adapter.llm = types.SimpleNamespace(generate=fake_generate)

    assert adapter.chat_complete([{"role": "user", "content": "Prompt"}], temperature=0.0) == ["done"]
    assert calls[0][1].kwargs["max_tokens"] == 10
    assert calls[1][0] == ["Promptshort draft\n</think>\nFinal Answer:"]


def test_qwen_forced_thinking_exit_reserves_final_context_when_clamping():
    adapter = _thinking_adapter(model.QwenVLLMAdapter, max_model_len=50, thinking_budget=30, final_budget=5)
    prompt = "Prompt"
    expected_first_budget = adapter.max_model_len - len(prompt) - adapter._forced_final_reserve_tokens()
    calls = []

    def fake_generate(prompts, sampling_params):
        calls.append((prompts, sampling_params))
        if len(calls) == 1:
            return [
                types.SimpleNamespace(
                    outputs=[types.SimpleNamespace(text="x" * expected_first_budget,
                                                   token_ids=list(range(expected_first_budget)))],
                    prompt_token_ids=[11],
                )
            ]
        return [
            types.SimpleNamespace(
                outputs=[types.SimpleNamespace(text="Final Answer: done", token_ids=[99])],
                prompt_token_ids=[22],
            )
        ]

    adapter.llm = types.SimpleNamespace(generate=fake_generate)

    assert adapter.chat_complete([{"role": "user", "content": prompt}], temperature=0.0) == ["done"]
    assert calls[0][1].kwargs["max_tokens"] == expected_first_budget
    assert calls[1][1].kwargs["max_tokens"] == 5


def test_gemma_adapter_passes_enable_thinking_to_chat_template(monkeypatch):
    tokenizer = _TokenizerRecorder(prompt="gemma prompt")
    tokenizer_calls = []

    def fake_from_pretrained(*args, **kwargs):
        tokenizer_calls.append((args, kwargs))
        return tokenizer

    monkeypatch.setattr(model.AutoTokenizer, "from_pretrained", fake_from_pretrained)

    adapter = model.GemmaVLLMAdapter(
        spec=helpers.ModelSpec(model="google/gemma-4-E4B-it", thinking=True),
        config=_make_runtime_config(),
        tensor_parallel_size=1,
    )

    prompt = adapter._messages_to_prompt([
        {"role": "system", "content": "Follow directions"},
        {"role": "user", "content": "Hi"},
    ])

    assert prompt == "gemma prompt"
    # GemmaVLLMAdapter no longer passes limit_mm_per_prompt to AutoTokenizer.
    assert tokenizer_calls == [
        (
            ("google/gemma-4-E4B-it",),
            {"trust_remote_code": True},
        )
    ]
    assert tokenizer.calls[-1] == (
        [
            {"role": "system", "content": "Follow directions"},
            {"role": "user", "content": "Hi"},
        ],
        {"tokenize": False, "add_generation_prompt": True, "enable_thinking": True},
    )


def test_gemma_adapter_strips_thought_channel_from_output():
    adapter = model.GemmaVLLMAdapter.__new__(model.GemmaVLLMAdapter)
    adapter.model_name = "google/gemma-4-E4B-it"
    adapter.thinking = True
    parse_calls = []
    adapter.tokenizer = types.SimpleNamespace(
        parse_response=lambda payload: parse_calls.append(payload) or {"content": '{"Yes": 1}'}
    )

    output = types.SimpleNamespace(text="ignored", token_ids=[101, 102, 103])

    assert adapter._normalize_completion_output(output) == '{"Yes": 1}'
    assert parse_calls == [[101, 102, 103]]


def test_gemma_adapter_raises_when_parse_response_is_missing():
    adapter = model.GemmaVLLMAdapter.__new__(model.GemmaVLLMAdapter)
    adapter.model_name = "google/gemma-4-E4B-it"
    adapter.thinking = True
    adapter.tokenizer = types.SimpleNamespace()

    with pytest.raises(ValueError, match="parse_response"):
        adapter._normalize_completion_output(types.SimpleNamespace(text="raw", token_ids=[1]))


def test_gemma_adapter_falls_back_to_output_text_when_parse_response_has_no_content():
    # When parse_response returns only a thinking channel, the adapter now falls back
    # to the raw output.text so downstream parsers can recover.  This used to raise.
    adapter = model.GemmaVLLMAdapter.__new__(model.GemmaVLLMAdapter)
    adapter.model_name = "google/gemma-4-E4B-it"
    adapter.thinking = True
    adapter.tokenizer = types.SimpleNamespace(parse_response=lambda payload: {"thinking": "draft"})

    result = adapter._normalize_completion_output(types.SimpleNamespace(text="raw", token_ids=[1]))
    assert result == "raw"


def test_gemma_forced_thinking_exit_uses_channel_closer():
    adapter = _thinking_adapter(model.GemmaVLLMAdapter, thinking_budget=3, final_budget=2)
    adapter.tokenizer.parse_response = lambda payload: (
        {"thinking": "draft"} if payload == [1, 2, 3] else {"content": "already final"}
    )
    calls = []

    def fake_generate(prompts, sampling_params):
        calls.append((prompts, sampling_params))
        if len(calls) == 1:
            return [
                types.SimpleNamespace(
                    outputs=[types.SimpleNamespace(text="raw thought", token_ids=[1, 2, 3])],
                    prompt_token_ids=[11],
                )
            ]
        return [
            types.SimpleNamespace(
                outputs=[types.SimpleNamespace(text="Final Answer: [1, 2]", token_ids=[9])],
                prompt_token_ids=[22],
            )
        ]

    adapter.llm = types.SimpleNamespace(generate=fake_generate)

    completions = adapter.chat_complete([{"role": "user", "content": "Prompt"}], temperature=0.0)

    assert completions == ["[1, 2]"]
    assert calls[0][1].kwargs["max_tokens"] == 3
    assert calls[1][1].kwargs["max_tokens"] == 2
    assert calls[1][0] == ["Promptraw thought\n<channel|>\nFinal Answer:"]


def test_gemma_forced_thinking_exit_runs_for_short_thought_only_output():
    adapter = _thinking_adapter(model.GemmaVLLMAdapter, thinking_budget=10, final_budget=2)
    adapter.tokenizer.parse_response = lambda payload: (
        {"thinking": "draft"} if payload == [1] else {"content": "already final"}
    )
    calls = []

    def fake_generate(prompts, sampling_params):
        calls.append((prompts, sampling_params))
        if len(calls) == 1:
            return [
                types.SimpleNamespace(
                    outputs=[types.SimpleNamespace(text="<|channel>thought\ndraft", token_ids=[1])],
                    prompt_token_ids=[11],
                )
            ]
        return [
            types.SimpleNamespace(
                outputs=[types.SimpleNamespace(text="Final Answer: {\"location\": [0, 0]}", token_ids=[9])],
                prompt_token_ids=[22],
            )
        ]

    adapter.llm = types.SimpleNamespace(generate=fake_generate)

    completions = adapter.chat_complete([{"role": "user", "content": "Prompt"}], temperature=0.0)

    assert completions == ['{"location": [0, 0]}']
    assert calls[0][1].kwargs["max_tokens"] == 10
    assert calls[1][1].kwargs["max_tokens"] == 2
    assert calls[1][0] == ["Prompt<|channel>thought\ndraft\n<channel|>\nFinal Answer:"]


def test_gemma_forced_thinking_exit_rejects_content_that_is_still_thought_channel():
    adapter = _thinking_adapter(model.GemmaVLLMAdapter, thinking_budget=10, final_budget=2)
    adapter.tokenizer.parse_response = lambda payload: (
        {"content": "<|channel>thought\nnot final\n<channel|>"} if payload == [1]
        else {"content": "already final"}
    )
    calls = []

    def fake_generate(prompts, sampling_params):
        calls.append((prompts, sampling_params))
        if len(calls) == 1:
            return [
                types.SimpleNamespace(
                    outputs=[types.SimpleNamespace(
                        text="<|channel>thought\nnot final\n<channel|>",
                        token_ids=[1],
                    )],
                    prompt_token_ids=[11],
                )
            ]
        return [
            types.SimpleNamespace(
                outputs=[types.SimpleNamespace(text="Final Answer: {\"ok\": true}", token_ids=[9])],
                prompt_token_ids=[22],
            )
        ]

    adapter.llm = types.SimpleNamespace(generate=fake_generate)

    completions = adapter.chat_complete([{"role": "user", "content": "Prompt"}], temperature=0.0)

    assert completions == ['{"ok": true}']
    assert calls[1][0] == ["Prompt<|channel>thought\nnot final\n<channel|>\nFinal Answer:"]


def test_gemma_forced_thinking_exit_keeps_raw_text_when_final_continuation_is_empty():
    adapter = _thinking_adapter(model.GemmaVLLMAdapter, thinking_budget=3, final_budget=2)
    adapter.tokenizer.parse_response = lambda payload: {"thinking": "draft"}
    calls = []

    def fake_generate(prompts, sampling_params):
        calls.append((prompts, sampling_params))
        if len(calls) == 1:
            return [
                types.SimpleNamespace(
                    outputs=[types.SimpleNamespace(text="raw thought", token_ids=[1, 2, 3])],
                    prompt_token_ids=[11],
                )
            ]
        return [
            types.SimpleNamespace(
                outputs=[types.SimpleNamespace(text="Final Answer:   ", token_ids=[9])],
                prompt_token_ids=[22],
            )
        ]

    adapter.llm = types.SimpleNamespace(generate=fake_generate)

    assert adapter.chat_complete([{"role": "user", "content": "Prompt"}], temperature=0.0) == ["raw thought"]
    assert len(calls) == 2


def test_gemma_content_output_does_not_force_continuation():
    adapter = _thinking_adapter(model.GemmaVLLMAdapter, thinking_budget=3, final_budget=2)
    adapter.tokenizer.parse_response = lambda payload: {"content": "Yes"}
    calls = []

    def fake_generate(prompts, sampling_params):
        calls.append((prompts, sampling_params))
        return [
            types.SimpleNamespace(
                outputs=[types.SimpleNamespace(text="ignored", token_ids=[1, 2, 3])],
                prompt_token_ids=[11],
            )
        ]

    adapter.llm = types.SimpleNamespace(generate=fake_generate)

    assert adapter.chat_complete([{"role": "user", "content": "Prompt"}], temperature=0.0) == ["Yes"]
    assert len(calls) == 1


def test_gemma_adapter_requires_token_ids_for_parse_response():
    adapter = model.GemmaVLLMAdapter.__new__(model.GemmaVLLMAdapter)
    adapter.model_name = "google/gemma-4-E4B-it"
    adapter.thinking = True
    adapter.tokenizer = types.SimpleNamespace(parse_response=lambda payload: {"content": "Yes"})

    with pytest.raises(ValueError, match="missing token_ids"):
        adapter._normalize_completion_output(types.SimpleNamespace(text="raw", token_ids=[]))


def test_build_model_adapter_returns_family_specific_adapter(monkeypatch):
    monkeypatch.setattr(model.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: _TokenizerRecorder())

    assert isinstance(
        model.build_model_adapter(
            helpers.ModelSpec(model="Qwen/Qwen3.5-4B", thinking=False),
            config=_make_runtime_config(),
            tensor_parallel_size=1,
        ),
        model.QwenVLLMAdapter,
    )
    assert isinstance(
        model.build_model_adapter(
            helpers.ModelSpec(model="google/gemma-4-E4B-it", thinking=False),
            config=_make_runtime_config(),
            tensor_parallel_size=1,
        ),
        model.GemmaVLLMAdapter,
    )
    assert isinstance(
        model.build_model_adapter(
            helpers.ModelSpec(model="meta/llama-3"),
            config=_make_runtime_config(),
            tensor_parallel_size=1,
        ),
        model.BaseVLLMAdapter,
    )


def test_build_model_adapter_routes_qwen25_to_base_adapter(monkeypatch):
    monkeypatch.setattr(model.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: _TokenizerRecorder())

    adapter = model.build_model_adapter(
        helpers.ModelSpec(model="Qwen/Qwen2.5-0.5B-Instruct"),
        config=_make_runtime_config(),
        tensor_parallel_size=1,
    )

    assert isinstance(adapter, model.BaseVLLMAdapter)
    assert not isinstance(adapter, model.QwenVLLMAdapter)


def test_build_model_adapter_passes_explicit_tensor_parallel_size_through(monkeypatch):
    monkeypatch.setattr(model.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: _TokenizerRecorder())

    adapter = model.build_model_adapter(
        helpers.ModelSpec(model="Qwen/Qwen3.5-4B", thinking=False),
        config=_make_runtime_config(),
        tensor_parallel_size=2,
    )

    assert adapter.llm.kwargs["tensor_parallel_size"] == 2
