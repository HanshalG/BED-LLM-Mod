import sys
import types
from enum import Enum
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Optional heavy-dependency shims
#
# Many tests in this repo exercise the LLM adapter / model loading code by
# importing ``model.py``, which transitively imports ``torch``, ``vllm``,
# ``transformers`` and ``openai_harmony``.  Those packages are heavyweight and
# are often not installed in unit-test environments.  Because each ``setdefault``
# below is a no-op when the real package is already present, these shims are
# safe to install unconditionally: real GPU environments still get the real
# packages, and CPU-only environments get just enough fake surface for module
# import to succeed.
# ---------------------------------------------------------------------------

# torch
if "torch" not in sys.modules:
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
    sys.modules["torch"] = fake_torch_module


# wandb
if "wandb" not in sys.modules:
    fake_wandb_module = types.ModuleType("wandb")
    fake_wandb_module.log = lambda *args, **kwargs: None
    fake_wandb_module.init = lambda *args, **kwargs: None
    sys.modules["wandb"] = fake_wandb_module


# transformers
if "transformers" not in sys.modules:
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
    fake_transformers_module.AutoModelForCausalLM = _FakeAutoModelForCausalLM
    sys.modules["transformers"] = fake_transformers_module


# vllm
if "vllm" not in sys.modules:
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
    sys.modules["vllm"] = fake_vllm_module


# openai_harmony
if "openai_harmony" not in sys.modules:
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

    def _fake_load_harmony_encoding(name):
        return types.SimpleNamespace(
            encode=lambda text, allowed_special=None: [0],
            decode=lambda tokens: "",
            render_conversation_for_completion=lambda *args, **kwargs: [],
            parse_messages_from_completion_tokens=lambda *args, **kwargs: [],
        )

    fake_openai_harmony_module.Role = _FakeHarmonyRole
    fake_openai_harmony_module.HarmonyEncodingName = _FakeHarmonyEncodingName
    fake_openai_harmony_module.Message = _FakeHarmonyMessage
    fake_openai_harmony_module.Conversation = type(
        "Conversation",
        (),
        {"from_messages": classmethod(lambda cls, messages: types.SimpleNamespace(messages=list(messages)))},
    )
    fake_openai_harmony_module.SystemContent = type(
        "SystemContent",
        (),
        {"new": staticmethod(lambda: types.SimpleNamespace(with_reasoning_effort=lambda effort: types.SimpleNamespace(with_conversation_start_date=lambda *_args, **_kwargs: object())))},
    )
    fake_openai_harmony_module.ReasoningEffort = Enum(
        "ReasoningEffort",
        {"LOW": "low", "MEDIUM": "medium", "HIGH": "high"},
    )
    fake_openai_harmony_module.load_harmony_encoding = _fake_load_harmony_encoding
    sys.modules["openai_harmony"] = fake_openai_harmony_module
