import BasicImage from '../../public/img/frontpage/blockchain-1.png'
import BasicImage1 from '../../public/img/frontpage/blockchain-2.png'

export const modulesList = [
  {
    image_url: BasicImage,
    address: "158.247.70.45:8888",
    functions: ["generate"],
    attributes: [],
    name: "model.openai",
    path: "model.openai",
    chash: "f789f265b6aa88ea597c4f7ea608fc508a56ca7bd8cb766cbf03c250dff970d2",
    hash: "e39d8385c6717b89cf3e35600f4643683afea6ad60efca3fc7ccbc2079e58ad2",
    signature: "d047efac1b7320c9428c51f564a35a029ca687b6cc0b9e7301d4115cf6383755547e084d69aa7024cdbeaef24596be6c7d3c8324008bdfe0ea6183e6c403ec8e",
    ss58_address: "5HarzAYD37Sp3vJs385CLvhDPN52Cb1Q352yxZnDZchznPaS",
    dislike: 0,
    like: 55,
    heart: 45,
    schema: {
      generate: {
        input: {
          prompt: "str",
          model: "str",
          presence_penalty: "float",
          frequency_penalty: "float",
          temperature: "float",
          max_tokens: "int",
          top_p: "float",
          choice_idx: "int",
          api_key: "str",
          retry: "bool",
          role: "str",
          history: "list",
        },
        default: {
          prompt: "sup?",
          model: "gpt-3.5-turbo",
          presence_penalty: 0.0,
          frequency_penalty: 0.0,
          temperature: 0.9,
          max_tokens: 100,
          top_p: 1,
          choice_idx: 0,
          api_key: null,
          retry: true,
          role: "user",
          history: null,
          kwargs: null,
        },
        output: {},
        type: "self",
      },
    },
  },
  {
    image_url: BasicImage1,
    address: "158.247.70.45:8888",
    functions: ["put", "get_hash", "get"],
    attributes: [],
    name: "storage",
    path: "storage",
    description: "not available",
    chash: "45235f1066ab82648abf3da9adccbedebf628d7eff9de1da44a1372556468a3c",
    hash: "da6a2f603ab3365b779e98a67dc84ce0afd3d69f6b336e698f68a50cdd920c4e",
    signature: "8c154638f98b3037c720e56281bac5dd0f2141f5ad189588befc3030d6b4b431b164f97365145c2b3c6dde6706b1552115b7759cd06c0a2aecb14cfbf4c6a78a",
    ss58_address: "5H6P4f9VoFLSsuPnq6KtEbG9VTEP2oTddXEHaJ4BDz9GXUgi",
    pub_key: '0x06739d386D215deda42Fc57b595c0e3FA3715687',
    schema: {
      put: {
        input: { v: "dict", encrypt: "bool", replicas: "int", k: "str" },
        default: { k: null, v: null, encrypt: false, replicas: 1 },
        output: {},
        type: "self",
      },
      get_hash: {
        input: { k: "str", seed: "int", seed_sep: "str" },
        default: { k: null, seed: null, seed_sep: "<SEED>" },
        output: "str", 
        type: "self",
      },
      get: {
        input: { deserialize: "bool", k: "NA" },
        default: { k: null, deserialize: true },
        output: "any",
        type: "self",
      },
    },
  },
  {
    name: "demo_c",
    description: "demo_c",
    address: "158.247.70.45:8888"
  },
  {
    name: "api",
    description: "api",
    address: "158.247.70.45:8888"
  },
  {
    name: "websocket",
    description: "websocket",
    address: "158.247.70.45:8888"
  },
  {
    name: "namespace",
    description: "namespace",
    address: "158.247.70.45:8888"
  },
  {
    name: "text2image",
    description: "text2image",
    address: "158.247.70.45:8888"
  },
  {
    name: "loop",
    description: "loop",
    address: "158.247.70.45:8888"
  },
  {
    name: "logger",
    description: "logger",
    address: "158.247.70.45:8888"
  },
  {
    name: "search",
    description: "search",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.trl.core",
    description: "archive.trl.core",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.trl.models.modeling_value_head",
    description: "archive.trl.models.modeling_value_head",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.trl.models.modeling_base",
    description: "archive.trl.models.modeling_base",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.trl.trainer",
    description: "archive.trl.trainer",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.trl.trainer.base",
    description: "archive.trl.trainer.base",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.trl.trainer.ppo_config",
    description: "archive.trl.trainer.ppo_config",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.logger",
    description: "archive.logger",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.tokenizer_map",
    description: "archive.tokenizer_map",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.bot",
    description: "archive.bot",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.ray.queue",
    description: "archive.ray.queue",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.ray.client",
    description: "archive.ray.client",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.ray.server.redis",
    description: "archive.ray.server.redis",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.ray.server.queue",
    description: "archive.ray.server.queue",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.ray.server.object",
    description: "archive.ray.server.object",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.aggregator.mean",
    description: "archive.aggregator.mean",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.aggregator.sum",
    description: "archive.aggregator.sum",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.agent.judge",
    description: "archive.agent.judge",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.agent.chat",
    description: "archive.agent.chat",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.agent.knowledge_graph",
    description: "archive.agent.knowledge_graph",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.tokenizer",
    description: "archive.tokenizer",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.asyncio.task_manager",
    description: "archive.asyncio.task_manager",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.asyncio.queue_server",
    description: "archive.asyncio.queue_server",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.fastchat",
    description: "archive.fastchat",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.fastchat.conversation",
    description: "archive.fastchat.conversation",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.fastchat.utils",
    description: "archive.fastchat.utils",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.fastchat.serve.controller",
    description: "archive.fastchat.serve.controller",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.fastchat.serve.gradio_patch",
    description: "archive.fastchat.serve.gradio_patch",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.fastchat.serve.compression",
    description: "archive.fastchat.serve.compression",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.fastchat.serve.cli",
    description: "archive.fastchat.serve.cli",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.fastchat.serve.inference",
    description: "archive.fastchat.serve.inference",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.fastchat.train",
    description: "archive.fastchat.train",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.pipeline",
    description: "archive.pipeline",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.pool",
    description: "archive.pool",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.litgpt",
    description: "archive.litgpt",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.litgpt.tests.test_utils",
    description: "archive.litgpt.tests.test_utils",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.litgpt.tests.test_lora",
    description: "archive.litgpt.tests.test_lora",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.litgpt.tests.test_packed_dataset",
    description: "archive.litgpt.tests.test_packed_dataset",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.litgpt.quantize.bnb",
    description: "archive.litgpt.quantize.bnb",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.litgpt.quantize.gptq",
    description: "archive.litgpt.quantize.gptq",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.litgpt.lit_gpt.adapter",
    description: "archive.litgpt.lit_gpt.adapter",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.litgpt.lit_gpt.utils",
    description: "archive.litgpt.lit_gpt.utils",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.litgpt.lit_gpt.model",
    description: "archive.litgpt.lit_gpt.model",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.litgpt.lit_gpt.speed_monitor",
    description: "archive.litgpt.lit_gpt.speed_monitor",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.litgpt.lit_gpt.rmsnorm",
    description: "archive.litgpt.lit_gpt.rmsnorm",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.litgpt.lit_gpt.packed_dataset",
    description: "archive.litgpt.lit_gpt.packed_dataset",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.litgpt.lit_gpt.lora",
    description: "archive.litgpt.lit_gpt.lora",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.litgpt.pretrain.openwebtext_trainer",
    description: "archive.litgpt.pretrain.openwebtext_trainer",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.dreambooth",
    description: "archive.dreambooth",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.dreambooth.dataset",
    description: "archive.dreambooth.dataset",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.dreambooth.lora",
    description: "archive.dreambooth.lora",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.threading.thread_queue",
    description: "archive.threading.thread_queue",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.threading.pool",
    description: "archive.threading.pool",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.threading.custom_thread",
    description: "archive.threading.custom_thread",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.threading.thread_types.asyncio",
    description: "archive.threading.thread_types.asyncio",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.threading.thread_types.producer",
    description: "archive.threading.thread_types.producer",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.streamlit",
    description: "archive.streamlit",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.tool.copy",
    description: "archive.tool.copy",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.andromeda",
    description: "archive.andromeda",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.andromeda.testing.main",
    description: "archive.andromeda.testing.main",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.andromeda.Andromeda.model",
    description: "archive.andromeda.Andromeda.model",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.andromeda.Andromeda.old.sophia",
    description: "archive.andromeda.Andromeda.old.sophia",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.andromeda.Andromeda.utils.stable_adamw",
    description: "archive.andromeda.Andromeda.utils.stable_adamw",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.andromeda.Andromeda.utils",
    description: "archive.andromeda.Andromeda.utils",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.andromeda.Andromeda.optimus_prime.xl_autoregressive_wrapper",
    description: "archive.andromeda.Andromeda.optimus_prime.xl_autoregressive_wrapper",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.andromeda.Andromeda.optimus_prime.nonautoregressive_wrapper",
    description: "archive.andromeda.Andromeda.optimus_prime.nonautoregressive_wrapper",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.andromeda.Andromeda.optimus_prime.attend",
    description: "archive.andromeda.Andromeda.optimus_prime.attend",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.andromeda.Andromeda.optimus_prime.autoregressive_wrapper",
    description: "archive.andromeda.Andromeda.optimus_prime.autoregressive_wrapper",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.andromeda.Andromeda.optimus_prime.continuous_autoregressive_wrapper",
    description: "archive.andromeda.Andromeda.optimus_prime.continuous_autoregressive_wrapper",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.andromeda.Andromeda.optimus_prime.x_transformers",
    description: "archive.andromeda.Andromeda.optimus_prime.x_transformers",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.audiocraft",
    description: "archive.audiocraft",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.audiocraft.seanet",
    description: "archive.audiocraft.seanet",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.audiocraft.codebooks_patterns",
    description: "archive.audiocraft.codebooks_patterns",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.audiocraft.conv",
    description: "archive.audiocraft.conv",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.audiocraft.transformer",
    description: "archive.audiocraft.transformer",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.audiocraft.streaming",
    description: "archive.audiocraft.streaming",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.audiocraft.rope",
    description: "archive.audiocraft.rope",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.audiocraft.conditioners",
    description: "archive.audiocraft.conditioners",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.audiocraft.lstm",
    description: "archive.audiocraft.lstm",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.audiocraft.activations",
    description: "archive.audiocraft.activations",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.audiocraft.quantization.core_vq",
    description: "archive.audiocraft.quantization.core_vq",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.audiocraft.quantization.base",
    description: "archive.audiocraft.quantization.base",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.audiocraft.quantization.vq",
    description: "archive.audiocraft.quantization.vq",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.audiocraft.utils",
    description: "archive.audiocraft.utils",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.audiocraft.models.encodec",
    description: "archive.audiocraft.models.encodec",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.audiocraft.models.lm",
    description: "archive.audiocraft.models.lm",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.insurance",
    description: "archive.insurance",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.storage",
    description: "archive.storage",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.subprocess",
    description: "archive.subprocess",
    address: "158.247.70.45:8888"
  },
  {
    name: "archive.gpu_manager",
    description: "archive.gpu_manager",
    address: "158.247.70.45:8888"
  },
  {
    name: "frontend",
    description: "frontend",
    address: "158.247.70.45:8888"
  },
  {
    name: "ssl",
    description: "ssl",
    address: "158.247.70.45:8888"
  },
  {
    name: "cli",
    description: "cli",
    address: "158.247.70.45:8888"
  },
  {
    name: "tokenomics",
    description: "tokenomics",
    address: "158.247.70.45:8888"
  },
  {
    name: "socket",
    description: "socket",
    address: "158.247.70.45:8888"
  },
  {
    name: "otc",
    description: "otc",
    address: "158.247.70.45:8888"
  },
  {
    name: "ssh",
    description: "ssh",
    address: "158.247.70.45:8888"
  },
  {
    name: "serializer",
    description: "serializer",
    address: "158.247.70.45:8888"
  },
  {
    name: "stability",
    description: "stability",
    address: "158.247.70.45:8888"
  },
  {
    name: "key",
    description: "key",
    address: "158.247.70.45:8888"
  },
  {
    name: "key.aes",
    description: "key.aes",
    address: "158.247.70.45:8888"
  },
  {
    name: "key.dashboard",
    description: "key.dashboard",
    address: "158.247.70.45:8888"
  },
  {
    name: "key.evm",
    description: "key.evm",
    address: "158.247.70.45:8888"
  },
  {
    name: "explorer",
    description: "explorer",
    address: "158.247.70.45:8888"
  },
  {
    name: "process.pipe",
    description: "process.pipe",
    address: "158.247.70.45:8888"
  },
  {
    name: "process.utils",
    description: "process.utils",
    address: "158.247.70.45:8888"
  },
  {
    name: "process.pool",
    description: "process.pool",
    address: "158.247.70.45:8888"
  },
  {
    name: "process",
    description: "process",
    address: "158.247.70.45:8888"
  },
  {
    name: "comfyui_lcm",
    description: "comfyui_lcm",
    address: "158.247.70.45:8888"
  },
  {
    name: "comfyui_lcm.lcm.lcm_scheduler",
    description: "comfyui_lcm.lcm.lcm_scheduler",
    address: "158.247.70.45:8888"
  },
  {
    name: "comfyui_lcm.lcm.lcm_i2i_pipeline",
    description: "comfyui_lcm.lcm.lcm_i2i_pipeline",
    address: "158.247.70.45:8888"
  },
  {
    name: "comfyui_lcm.lcm.lcm_pipeline",
    description: "comfyui_lcm.lcm.lcm_pipeline",
    address: "158.247.70.45:8888"
  },
  {
    name: "demo",
    description: "demo",
    address: "158.247.70.45:8888"
  },
  {
    name: "finetune",
    description: "finetune",
    address: "158.247.70.45:8888"
  },
  {
    name: "finetune.data",
    description: "finetune.data",
    address: "158.247.70.45:8888"
  },
  {
    name: "chat",
    description: "chat",
    address: "158.247.70.45:8888"
  },
  {
    name: "network",
    description: "network",
    address: "158.247.70.45:8888"
  },
  {
    name: "subspace.test",
    description: "subspace.test",
    address: "158.247.70.45:8888"
  },
  {
    name: "subspace.errors",
    description: "subspace.errors",
    address: "158.247.70.45:8888"
  },
  {
    name: "subspace.tokenomics",
    description: "subspace.tokenomics",
    address: "158.247.70.45:8888"
  },
  {
    name: "subspace",
    description: "subspace",
    address: "158.247.70.45:8888"
  },
  {
    name: "subspace.voting",
    description: "subspace.voting",
    address: "158.247.70.45:8888"
  },
  {
    name: "subspace.chain",
    description: "subspace.chain",
    address: "158.247.70.45:8888"
  },
  {
    name: "subspace.dashboard",
    description: "subspace.dashboard",
    address: "158.247.70.45:8888"
  },
  {
    name: "subspace.telemetry",
    description: "subspace.telemetry",
    address: "158.247.70.45:8888"
  },
  {
    name: "metamask",
    description: "metamask",
    address: "158.247.70.45:8888"
  },
  {
    name: "metamask.components.metamask",
    description: "metamask.components.metamask",
    address: "158.247.70.45:8888"
  },
  {
    name: "tree",
    description: "tree",
    address: "158.247.70.45:8888"
  },
  {
    name: "gpt.openopenai",
    description: "gpt.openopenai",
    address: "158.247.70.45:8888"
  },
  {
    name: "gpt.openai",
    description: "gpt.openai",
    address: "158.247.70.45:8888"
  },
  {
    name: "crypto",
    description: "crypto",
    address: "158.247.70.45:8888"
  },
  {
    name: "crypto.hash",
    description: "crypto.hash",
    address: "158.247.70.45:8888"
  },
  {
    name: "crypto.key.aes",
    description: "crypto.key.aes",
    address: "158.247.70.45:8888"
  },
  {
    name: "pip",
    description: "pip",
    address: "158.247.70.45:8888"
  },
  {
    name: "base",
    description: "base",
    address: "158.247.70.45:8888"
  },
  {
    name: "docker",
    description: "docker",
    address: "158.247.70.45:8888"
  },
  {
    name: "ray",
    description: "ray",
    address: "158.247.70.45:8888"
  },
  {
    name: "ray.queue",
    description: "ray.queue",
    address: "158.247.70.45:8888"
  },
  {
    name: "ray.client",
    description: "ray.client",
    address: "158.247.70.45:8888"
  },
  {
    name: "ray.server.redis",
    description: "ray.server.redis",
    address: "158.247.70.45:8888"
  },
  {
    name: "ray.server.queue",
    description: "ray.server.queue",
    address: "158.247.70.45:8888"
  },
  {
    name: "ray.server.object",
    description: "ray.server.object",
    address: "158.247.70.45:8888"
  },
  {
    name: "playground",
    description: "playground",
    address: "158.247.70.45:8888"
  },
  {
    name: "web",
    description: "web",
    address: "158.247.70.45:8888"
  },
  {
    name: "web.selenium",
    description: "web.selenium",
    address: "158.247.70.45:8888"
  },
  {
    name: "queue",
    description: "queue",
    address: "158.247.70.45:8888"
  },
  {
    name: "web3.utils",
    description: "web3.utils",
    address: "158.247.70.45:8888"
  },
  {
    name: "web3.substrate.contract",
    description: "web3.substrate.contract",
    address: "158.247.70.45:8888"
  },
  {
    name: "web3.substrate.network",
    description: "web3.substrate.network",
    address: "158.247.70.45:8888"
  },
  {
    name: "web3.substrate.account",
    description: "web3.substrate.account",
    address: "158.247.70.45:8888"
  },
  {
    name: "web3.graph",
    description: "web3.graph",
    address: "158.247.70.45:8888"
  },
  {
    name: "web3.evm",
    description: "web3.evm",
    address: "158.247.70.45:8888"
  },
  {
    name: "web3.evm.contract",
    description: "web3.evm.contract",
    address: "158.247.70.45:8888"
  },
  {
    name: "web3.evm.key",
    description: "web3.evm.key",
    address: "158.247.70.45:8888"
  },
  {
    name: "web3.evm.network",
    description: "web3.evm.network",
    address: "158.247.70.45:8888"
  },
  {
    name: "vali",
    description: "vali",
    address: "158.247.70.45:8888"
  },
  {
    name: "vali.replica",
    description: "vali.replica",
    address: "158.247.70.45:8888"
  },
  {
    name: "vali.bt",
    description: "vali.bt",
    address: "158.247.70.45:8888"
  },
  {
    name: "vali.text",
    description: "vali.text",
    address: "158.247.70.45:8888"
  },
  {
    name: "vali.text.mmlu",
    description: "vali.text.mmlu",
    address: "158.247.70.45:8888"
  },
  {
    name: "vali.text.truthqa",
    description: "vali.text.truthqa",
    address: "158.247.70.45:8888"
  },
  {
    name: "vali.text.realfake",
    description: "vali.text.realfake",
    address: "158.247.70.45:8888"
  },
  {
    name: "vali.text.math",
    description: "vali.text.math",
    address: "158.247.70.45:8888"
  },
  {
    name: "vali.parity",
    description: "vali.parity",
    address: "158.247.70.45:8888"
  },
  {
    name: "captcha",
    description: "captcha",
    address: "158.247.70.45:8888"
  },
  {
    name: "agent",
    description: "agent",
    address: "158.247.70.45:8888"
  },
  {
    name: "agent.factory",
    description: "agent.factory",
    address: "158.247.70.45:8888"
  },
  {
    name: "agent.sumarizer",
    description: "agent.sumarizer",
    address: "158.247.70.45:8888"
  },
  {
    name: "agent.data",
    description: "agent.data",
    address: "158.247.70.45:8888"
  },
  {
    name: "agent.maker",
    description: "agent.maker",
    address: "158.247.70.45:8888"
  },
  {
    name: "tokenizer",
    description: "tokenizer",
    address: "158.247.70.45:8888"
  },
  {
    name: "wallet",
    description: "wallet",
    address: "158.247.70.45:8888"
  },
  {
    name: "asyncio",
    description: "asyncio",
    address: "158.247.70.45:8888"
  },
  {
    name: "asyncio.task_manager",
    description: "asyncio.task_manager",
    address: "158.247.70.45:8888"
  },
  {
    name: "asyncio.queue_server",
    description: "asyncio.queue_server",
    address: "158.247.70.45:8888"
  },
  {
    name: "git",
    description: "git",
    address: "158.247.70.45:8888"
  },
  {
    name: "ansible",
    description: "ansible",
    address: "158.247.70.45:8888"
  },
  {
    name: "peer",
    description: "peer",
    address: "158.247.70.45:8888"
  },
  {
    name: "discord",
    description: "discord",
    address: "158.247.70.45:8888"
  },
  {
    name: "almeche",
    description: "almeche",
    address: "158.247.70.45:8888"
  },
  {
    name: "pipeline",
    description: "pipeline",
    address: "158.247.70.45:8888"
  },
  {
    name: "data",
    description: "data",
    address: "158.247.70.45:8888"
  },
  {
    name: "data.image.globe",
    description: "data.image.globe",
    address: "158.247.70.45:8888"
  },
  {
    name: "data.text.squad",
    description: "data.text.squad",
    address: "158.247.70.45:8888"
  },
  {
    name: "data.text.truthqa",
    description: "data.text.truthqa",
    address: "158.247.70.45:8888"
  },
  {
    name: "data.text.bt.prompt",
    description: "data.text.bt.prompt",
    address: "158.247.70.45:8888"
  },
  {
    name: "data.text.bt.pile",
    description: "data.text.bt.pile",
    address: "158.247.70.45:8888"
  },
  {
    name: "data.text.code",
    description: "data.text.code",
    address: "158.247.70.45:8888"
  },
  {
    name: "data.text.realfake",
    description: "data.text.realfake",
    address: "158.247.70.45:8888"
  },
  {
    name: "data.text.folder",
    description: "data.text.folder",
    address: "158.247.70.45:8888"
  },
  {
    name: "data.text.bittensor",
    description: "data.text.bittensor",
    address: "158.247.70.45:8888"
  },
  {
    name: "data.text.math",
    description: "data.text.math",
    address: "158.247.70.45:8888"
  },
  {
    name: "data.text.pile",
    description: "data.text.pile",
    address: "158.247.70.45:8888"
  },
  {
    name: "data.diffusion.dream.dream_dataset",
    description: "data.diffusion.dream.dream_dataset",
    address: "158.247.70.45:8888"
  },
  {
    name: "data.diffusion.dream.prompt_dataset",
    description: "data.diffusion.dream.prompt_dataset",
    address: "158.247.70.45:8888"
  },
  {
    name: "data.hf",
    description: "data.hf",
    address: "158.247.70.45:8888"
  },
  {
    name: "client",
    description: "client",
    address: "158.247.70.45:8888"
  },
  {
    name: "client.s3",
    description: "client.s3",
    address: "158.247.70.45:8888"
  },
  {
    name: "client.http",
    description: "client.http",
    address: "158.247.70.45:8888"
  },
  {
    name: "client.ray",
    description: "client.ray",
    address: "158.247.70.45:8888"
  },
  {
    name: "client.pool",
    description: "client.pool",
    address: "158.247.70.45:8888"
  },
  {
    name: "client.rest",
    description: "client.rest",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor",
    description: "bittensor",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.bittensor_dashboard",
    description: "bittensor.bittensor_dashboard",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.relayer",
    description: "bittensor.relayer",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.receptor.receptor_pool_impl",
    description: "bittensor.receptor.receptor_pool_impl",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.receptor.receptor_impl",
    description: "bittensor.receptor.receptor_impl",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.example",
    description: "bittensor.example",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.compute",
    description: "bittensor.compute",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.compute.protocol",
    description: "bittensor.compute.protocol",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.subtensor.errors",
    description: "bittensor.subtensor.errors",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.subtensor.subtensor_mock",
    description: "bittensor.subtensor.subtensor_mock",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.subtensor.chain_data",
    description: "bittensor.subtensor.chain_data",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.wallet.wallet_mock",
    description: "bittensor.wallet.wallet_mock",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.wallet.wallet_impl",
    description: "bittensor.wallet.wallet_impl",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.utils.stats",
    description: "bittensor.utils.stats",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.utils",
    description: "bittensor.utils",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.utils.registratrion_old",
    description: "bittensor.utils.registratrion_old",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.utils.networking",
    description: "bittensor.utils.networking",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.utils.registration",
    description: "bittensor.utils.registration",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.keyfile.keyfile_impl",
    description: "bittensor.keyfile.keyfile_impl",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.keyfile.__init__",
    description: "bittensor.keyfile.__init__",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.gooseai.neuron",
    description: "bittensor.neurons.text.prompting.miners.gooseai.neuron",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.gpt4all.neuron",
    description: "bittensor.neurons.text.prompting.miners.gpt4all.neuron",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.cerebras.neuron",
    description: "bittensor.neurons.text.prompting.miners.cerebras.neuron",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.self_hosted.neuron",
    description: "bittensor.neurons.text.prompting.miners.self_hosted.neuron",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.self_hosted.coati.dataset",
    description: "bittensor.neurons.text.prompting.miners.self_hosted.coati.dataset",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.self_hosted.coati.replay_buffer.naive",
    description: "bittensor.neurons.text.prompting.miners.self_hosted.coati.replay_buffer.naive",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.self_hosted.coati.replay_buffer.base",
    description: "bittensor.neurons.text.prompting.miners.self_hosted.coati.replay_buffer.base",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.self_hosted.coati.utils",
    description: "bittensor.neurons.text.prompting.miners.self_hosted.coati.utils",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.self_hosted.coati.experience_maker.naive",
    description: "bittensor.neurons.text.prompting.miners.self_hosted.coati.experience_maker.naive",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.self_hosted.coati.experience_maker.base",
    description: "bittensor.neurons.text.prompting.miners.self_hosted.coati.experience_maker.base",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.self_hosted.coati.models.loss",
    description: "bittensor.neurons.text.prompting.miners.self_hosted.coati.models.loss",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.self_hosted.coati.models.lora",
    description: "bittensor.neurons.text.prompting.miners.self_hosted.coati.models.lora",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.self_hosted.coati.models.llama.llama_rm",
    description: "bittensor.neurons.text.prompting.miners.self_hosted.coati.models.llama.llama_rm",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.self_hosted.coati.models.llama.llama_critic",
    description: "bittensor.neurons.text.prompting.miners.self_hosted.coati.models.llama.llama_critic",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.self_hosted.coati.models.llama.llama_lm",
    description: "bittensor.neurons.text.prompting.miners.self_hosted.coati.models.llama.llama_lm",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.self_hosted.coati.models.llama.llama_actor",
    description: "bittensor.neurons.text.prompting.miners.self_hosted.coati.models.llama.llama_actor",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.self_hosted.coati.models.bloom.bloom_lm",
    description: "bittensor.neurons.text.prompting.miners.self_hosted.coati.models.bloom.bloom_lm",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.self_hosted.coati.models.bloom.bloom_rm",
    description: "bittensor.neurons.text.prompting.miners.self_hosted.coati.models.bloom.bloom_rm",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.self_hosted.coati.models.bloom.bloom_critic",
    description: "bittensor.neurons.text.prompting.miners.self_hosted.coati.models.bloom.bloom_critic",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.self_hosted.coati.models.bloom.bloom_actor",
    description: "bittensor.neurons.text.prompting.miners.self_hosted.coati.models.bloom.bloom_actor",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.self_hosted.coati.models.auto.actor",
    description: "bittensor.neurons.text.prompting.miners.self_hosted.coati.models.auto.actor",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.self_hosted.coati.models.auto.critic",
    description: "bittensor.neurons.text.prompting.miners.self_hosted.coati.models.auto.critic",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.self_hosted.coati.models.auto.reward_model",
    description: "bittensor.neurons.text.prompting.miners.self_hosted.coati.models.auto.reward_model",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.self_hosted.coati.models.auto.lm",
    description: "bittensor.neurons.text.prompting.miners.self_hosted.coati.models.auto.lm",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.self_hosted.coati.models.gpt.gpt_lm",
    description: "bittensor.neurons.text.prompting.miners.self_hosted.coati.models.gpt.gpt_lm",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.self_hosted.coati.models.gpt.gpt_rm",
    description: "bittensor.neurons.text.prompting.miners.self_hosted.coati.models.gpt.gpt_rm",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.self_hosted.coati.models.gpt.gpt_actor",
    description: "bittensor.neurons.text.prompting.miners.self_hosted.coati.models.gpt.gpt_actor",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.self_hosted.coati.models.gpt.gpt_critic",
    description: "bittensor.neurons.text.prompting.miners.self_hosted.coati.models.gpt.gpt_critic",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.self_hosted.coati.models.base.actor",
    description: "bittensor.neurons.text.prompting.miners.self_hosted.coati.models.base.actor",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.self_hosted.coati.models.base.critic",
    description: "bittensor.neurons.text.prompting.miners.self_hosted.coati.models.base.critic",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.self_hosted.coati.models.base.reward_model",
    description: "bittensor.neurons.text.prompting.miners.self_hosted.coati.models.base.reward_model",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.self_hosted.coati.models.base.lm",
    description: "bittensor.neurons.text.prompting.miners.self_hosted.coati.models.base.lm",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.self_hosted.coati.models.roberta.roberta_critic",
    description: "bittensor.neurons.text.prompting.miners.self_hosted.coati.models.roberta.roberta_critic",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.self_hosted.coati.models.roberta.roberta_rm",
    description: "bittensor.neurons.text.prompting.miners.self_hosted.coati.models.roberta.roberta_rm",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.self_hosted.coati.models.roberta.roberta_actor",
    description: "bittensor.neurons.text.prompting.miners.self_hosted.coati.models.roberta.roberta_actor",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.self_hosted.coati.models.opt.opt_lm",
    description: "bittensor.neurons.text.prompting.miners.self_hosted.coati.models.opt.opt_lm",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.self_hosted.coati.models.opt.opt_critic",
    description: "bittensor.neurons.text.prompting.miners.self_hosted.coati.models.opt.opt_critic",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.self_hosted.coati.models.opt.opt_actor",
    description: "bittensor.neurons.text.prompting.miners.self_hosted.coati.models.opt.opt_actor",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.self_hosted.coati.models.opt.opt_rm",
    description: "bittensor.neurons.text.prompting.miners.self_hosted.coati.models.opt.opt_rm",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.self_hosted.coati.models.deberta.deberta_rm",
    description: "bittensor.neurons.text.prompting.miners.self_hosted.coati.models.deberta.deberta_rm",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.self_hosted.coati.models.deberta.deberta_critic",
    description: "bittensor.neurons.text.prompting.miners.self_hosted.coati.models.deberta.deberta_critic",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.self_hosted.coati.trainer.rm",
    description: "bittensor.neurons.text.prompting.miners.self_hosted.coati.trainer.rm",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.self_hosted.coati.trainer.ppo",
    description: "bittensor.neurons.text.prompting.miners.self_hosted.coati.trainer.ppo",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.self_hosted.coati.trainer.base",
    description: "bittensor.neurons.text.prompting.miners.self_hosted.coati.trainer.base",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.self_hosted.coati.trainer.sft",
    description: "bittensor.neurons.text.prompting.miners.self_hosted.coati.trainer.sft",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.self_hosted.coati.trainer.callbacks.save_checkpoint",
    description: "bittensor.neurons.text.prompting.miners.self_hosted.coati.trainer.callbacks.save_checkpoint",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.self_hosted.coati.trainer.callbacks.base",
    description: "bittensor.neurons.text.prompting.miners.self_hosted.coati.trainer.callbacks.base",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.self_hosted.coati.trainer.callbacks.performance_evaluator",
    description: "bittensor.neurons.text.prompting.miners.self_hosted.coati.trainer.callbacks.performance_evaluator",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.self_hosted.coati.trainer.strategies.colossalai",
    description: "bittensor.neurons.text.prompting.miners.self_hosted.coati.trainer.strategies.colossalai",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.self_hosted.coati.trainer.strategies.naive",
    description: "bittensor.neurons.text.prompting.miners.self_hosted.coati.trainer.strategies.naive",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.self_hosted.coati.trainer.strategies.ddp",
    description: "bittensor.neurons.text.prompting.miners.self_hosted.coati.trainer.strategies.ddp",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.self_hosted.coati.trainer.strategies.base",
    description: "bittensor.neurons.text.prompting.miners.self_hosted.coati.trainer.strategies.base",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.self_hosted.ppo.actor",
    description: "bittensor.neurons.text.prompting.miners.self_hosted.ppo.actor",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.self_hosted.ppo.loss",
    description: "bittensor.neurons.text.prompting.miners.self_hosted.ppo.loss",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.self_hosted.ppo.lora",
    description: "bittensor.neurons.text.prompting.miners.self_hosted.ppo.lora",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.self_hosted.ppo.base.actor",
    description: "bittensor.neurons.text.prompting.miners.self_hosted.ppo.base.actor",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.self_hosted.ppo.base.critic",
    description: "bittensor.neurons.text.prompting.miners.self_hosted.ppo.base.critic",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.self_hosted.ppo.base.reward_model",
    description: "bittensor.neurons.text.prompting.miners.self_hosted.ppo.base.reward_model",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.self_hosted.ppo.base.lm",
    description: "bittensor.neurons.text.prompting.miners.self_hosted.ppo.base.lm",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.self_hosted.ppo.strategies.naive",
    description: "bittensor.neurons.text.prompting.miners.self_hosted.ppo.strategies.naive",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.self_hosted.ppo.strategies.base",
    description: "bittensor.neurons.text.prompting.miners.self_hosted.ppo.strategies.base",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.AlephAlpha.neuron",
    description: "bittensor.neurons.text.prompting.miners.AlephAlpha.neuron",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.AI21.neuron",
    description: "bittensor.neurons.text.prompting.miners.AI21.neuron",
    address: "158.247.70.45:8888"
  },
  {
    name: "neuron",
    description: "neuron",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.huggingface.dromedary.neuron",
    description: "bittensor.neurons.text.prompting.miners.huggingface.dromedary.neuron",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.huggingface.neoxt.neuron",
    description: "bittensor.neurons.text.prompting.miners.huggingface.neoxt.neuron",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.huggingface.dolly.neuron",
    description: "bittensor.neurons.text.prompting.miners.huggingface.dolly.neuron",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.huggingface.wizard_vicuna.neuron",
    description: "bittensor.neurons.text.prompting.miners.huggingface.wizard_vicuna.neuron",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.huggingface.stabilityai.neuron",
    description: "bittensor.neurons.text.prompting.miners.huggingface.stabilityai.neuron",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.huggingface.pythia.neuron",
    description: "bittensor.neurons.text.prompting.miners.huggingface.pythia.neuron",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.huggingface.open_llama.neuron",
    description: "bittensor.neurons.text.prompting.miners.huggingface.open_llama.neuron",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.huggingface.guanaco.neuron",
    description: "bittensor.neurons.text.prompting.miners.huggingface.guanaco.neuron",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.huggingface.robertmyers.neuron",
    description: "bittensor.neurons.text.prompting.miners.huggingface.robertmyers.neuron",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.huggingface.chat_glm.neuron",
    description: "bittensor.neurons.text.prompting.miners.huggingface.chat_glm.neuron",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.huggingface.gpt4_x_vicuna.neuron",
    description: "bittensor.neurons.text.prompting.miners.huggingface.gpt4_x_vicuna.neuron",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.huggingface.fastchat_t5.neuron",
    description: "bittensor.neurons.text.prompting.miners.huggingface.fastchat_t5.neuron",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.huggingface.oasst_pythia.neuron",
    description: "bittensor.neurons.text.prompting.miners.huggingface.oasst_pythia.neuron",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.huggingface.raven.neuron",
    description: "bittensor.neurons.text.prompting.miners.huggingface.raven.neuron",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.huggingface.vicuna.neuron",
    description: "bittensor.neurons.text.prompting.miners.huggingface.vicuna.neuron",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.huggingface.koala.neuron",
    description: "bittensor.neurons.text.prompting.miners.huggingface.koala.neuron",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.huggingface.mpt_chat.neuron",
    description: "bittensor.neurons.text.prompting.miners.huggingface.mpt_chat.neuron",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.textgen.neuron",
    description: "bittensor.neurons.text.prompting.miners.textgen.neuron",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.openai.neuron",
    description: "bittensor.neurons.text.prompting.miners.openai.neuron",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.miners.cohere.neuron",
    description: "bittensor.neurons.text.prompting.miners.cohere.neuron",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.validators.core.reward",
    description: "bittensor.neurons.text.prompting.validators.core.reward",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.validators.core.gating",
    description: "bittensor.neurons.text.prompting.validators.core.gating",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.validators.core.neuron",
    description: "bittensor.neurons.text.prompting.validators.core.neuron",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.neurons.text.prompting.validators.constitution.neuron",
    description: "bittensor.neurons.text.prompting.validators.constitution.neuron",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.miner.neuron",
    description: "bittensor.miner.neuron",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.miner.server",
    description: "bittensor.miner.server",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.axon.__init__",
    description: "bittensor.axon.__init__",
    address: "158.247.70.45:8888"
  },
  {
    name: "bittensor.axon.axon_impl",
    description: "bittensor.axon.axon_impl",
    address: "158.247.70.45:8888"
  },
  {
    name: "combook",
    description: "combook",
    address: "158.247.70.45:8888"
  },
  {
    name: "repo",
    description: "repo",
    address: "158.247.70.45:8888"
  },
  {
    name: "model",
    description: "model",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.layer",
    description: "model.layer",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.text2image",
    description: "model.text2image",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.chat_models.litellm",
    description: "model.chat_models.litellm",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.chat_models.azureml_endpoint",
    description: "model.chat_models.azureml_endpoint",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.chat_models.human",
    description: "model.chat_models.human",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.chat_models.promptlayer_openai",
    description: "model.chat_models.promptlayer_openai",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.chat_models.openai",
    description: "model.chat_models.openai",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.chat_models.ollama",
    description: "model.chat_models.ollama",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.chat_models.mlflow_ai_gateway",
    description: "model.chat_models.mlflow_ai_gateway",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.chat_models.ernie",
    description: "model.chat_models.ernie",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.chat_models.anyscale",
    description: "model.chat_models.anyscale",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.chat_models.google_palm",
    description: "model.chat_models.google_palm",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.chat_models.anthropic",
    description: "model.chat_models.anthropic",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.chat_models.jinachat",
    description: "model.chat_models.jinachat",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.chat_models.azure_openai",
    description: "model.chat_models.azure_openai",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.chat_models.base",
    description: "model.chat_models.base",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.chat_models.fake",
    description: "model.chat_models.fake",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.chat_models.vertexai",
    description: "model.chat_models.vertexai",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.zapier",
    description: "model.zapier",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.bitapai",
    description: "model.bitapai",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.imageupscaler",
    description: "model.imageupscaler",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.yolo",
    description: "model.yolo",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.gpts",
    description: "model.gpts",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.roleplay.miner",
    description: "model.roleplay.miner",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.nodes",
    description: "model.ComfyUI.nodes",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.cuda_malloc",
    description: "model.ComfyUI.cuda_malloc",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.server",
    description: "model.ComfyUI.server",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.latent_preview",
    description: "model.ComfyUI.latent_preview",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.tests.inference",
    description: "model.ComfyUI.tests.inference",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.comfy_extras.nodes_compositing",
    description: "model.ComfyUI.comfy_extras.nodes_compositing",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.comfy_extras.nodes_model_advanced",
    description: "model.ComfyUI.comfy_extras.nodes_model_advanced",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.comfy_extras.chainner_models.model_loading",
    description: "model.ComfyUI.comfy_extras.chainner_models.model_loading",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.comfy_extras.chainner_models.architecture.DAT",
    description: "model.ComfyUI.comfy_extras.chainner_models.architecture.DAT",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.comfy_extras.chainner_models.architecture.SwiftSRGAN",
    description: "model.ComfyUI.comfy_extras.chainner_models.architecture.SwiftSRGAN",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.comfy_extras.chainner_models.architecture.HAT",
    description: "model.ComfyUI.comfy_extras.chainner_models.architecture.HAT",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.comfy_extras.chainner_models.architecture.block",
    description: "model.ComfyUI.comfy_extras.chainner_models.architecture.block",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.comfy_extras.chainner_models.architecture.SRVGG",
    description: "model.ComfyUI.comfy_extras.chainner_models.architecture.SRVGG",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.comfy_extras.chainner_models.architecture.RRDB",
    description: "model.ComfyUI.comfy_extras.chainner_models.architecture.RRDB",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.comfy_extras.chainner_models.architecture.Swin2SR",
    description: "model.ComfyUI.comfy_extras.chainner_models.architecture.Swin2SR",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.comfy_extras.chainner_models.architecture.LaMa",
    description: "model.ComfyUI.comfy_extras.chainner_models.architecture.LaMa",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.comfy_extras.chainner_models.architecture.SPSR",
    description: "model.ComfyUI.comfy_extras.chainner_models.architecture.SPSR",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.comfy_extras.chainner_models.architecture.SCUNet",
    description: "model.ComfyUI.comfy_extras.chainner_models.architecture.SCUNet",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.comfy_extras.chainner_models.architecture.SwinIR",
    description: "model.ComfyUI.comfy_extras.chainner_models.architecture.SwinIR",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.comfy_extras.chainner_models.architecture.timm.drop",
    description: "model.ComfyUI.comfy_extras.chainner_models.architecture.timm.drop",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.comfy_extras.chainner_models.architecture.OmniSR.OSA",
    description: "model.ComfyUI.comfy_extras.chainner_models.architecture.OmniSR.OSA",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.comfy_extras.chainner_models.architecture.OmniSR",
    description: "model.ComfyUI.comfy_extras.chainner_models.architecture.OmniSR",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.comfy_extras.chainner_models.architecture.OmniSR.esa",
    description: "model.ComfyUI.comfy_extras.chainner_models.architecture.OmniSR.esa",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.comfy_extras.chainner_models.architecture.OmniSR.ChannelAttention",
    description: "model.ComfyUI.comfy_extras.chainner_models.architecture.OmniSR.ChannelAttention",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.comfy_extras.chainner_models.architecture.OmniSR.OSAG",
    description: "model.ComfyUI.comfy_extras.chainner_models.architecture.OmniSR.OSAG",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.comfy_extras.chainner_models.architecture.OmniSR.layernorm",
    description: "model.ComfyUI.comfy_extras.chainner_models.architecture.OmniSR.layernorm",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.comfy_extras.chainner_models.architecture.face.upfirdn2d",
    description: "model.ComfyUI.comfy_extras.chainner_models.architecture.face.upfirdn2d",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.comfy_extras.chainner_models.architecture.face.fused_act",
    description: "model.ComfyUI.comfy_extras.chainner_models.architecture.face.fused_act",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.comfy_extras.chainner_models.architecture.face.gfpganv1_arch",
    description: "model.ComfyUI.comfy_extras.chainner_models.architecture.face.gfpganv1_arch",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.comfy_extras.chainner_models.architecture.face.arcface_arch",
    description: "model.ComfyUI.comfy_extras.chainner_models.architecture.face.arcface_arch",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.comfy_extras.chainner_models.architecture.face.stylegan2_arch",
    description: "model.ComfyUI.comfy_extras.chainner_models.architecture.face.stylegan2_arch",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.comfy_extras.chainner_models.architecture.face.gfpgan_bilinear_arch",
    description: "model.ComfyUI.comfy_extras.chainner_models.architecture.face.gfpgan_bilinear_arch",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.comfy_extras.chainner_models.architecture.face.stylegan2_clean_arch",
    description: "model.ComfyUI.comfy_extras.chainner_models.architecture.face.stylegan2_clean_arch",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.comfy_extras.chainner_models.architecture.face.restoreformer_arch",
    description: "model.ComfyUI.comfy_extras.chainner_models.architecture.face.restoreformer_arch",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.comfy_extras.chainner_models.architecture.face.stylegan2_bilinear_arch",
    description: "model.ComfyUI.comfy_extras.chainner_models.architecture.face.stylegan2_bilinear_arch",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.comfy_extras.chainner_models.architecture.face.gfpganv1_clean_arch",
    description: "model.ComfyUI.comfy_extras.chainner_models.architecture.face.gfpganv1_clean_arch",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.comfy_extras.chainner_models.architecture.face.codeformer",
    description: "model.ComfyUI.comfy_extras.chainner_models.architecture.face.codeformer",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.comfy.cli_args",
    description: "model.ComfyUI.comfy.cli_args",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.comfy.samplers",
    description: "model.ComfyUI.comfy.samplers",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.comfy.sd1_clip",
    description: "model.ComfyUI.comfy.sd1_clip",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.comfy.model_management",
    description: "model.ComfyUI.comfy.model_management",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.comfy.sdxl_clip",
    description: "model.ComfyUI.comfy.sdxl_clip",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.comfy.model_base",
    description: "model.ComfyUI.comfy.model_base",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.comfy.latent_formats",
    description: "model.ComfyUI.comfy.latent_formats",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.comfy.ops",
    description: "model.ComfyUI.comfy.ops",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.comfy.supported_models",
    description: "model.ComfyUI.comfy.supported_models",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.comfy.clip_vision",
    description: "model.ComfyUI.comfy.clip_vision",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.comfy.controlnet",
    description: "model.ComfyUI.comfy.controlnet",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.comfy.checkpoint_pickle",
    description: "model.ComfyUI.comfy.checkpoint_pickle",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.comfy.conds",
    description: "model.ComfyUI.comfy.conds",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.comfy.model_sampling",
    description: "model.ComfyUI.comfy.model_sampling",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.comfy.sd2_clip",
    description: "model.ComfyUI.comfy.sd2_clip",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.comfy.gligen",
    description: "model.ComfyUI.comfy.gligen",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.comfy.clip_model",
    description: "model.ComfyUI.comfy.clip_model",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.comfy.sd",
    description: "model.ComfyUI.comfy.sd",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.comfy.t2i_adapter.adapter",
    description: "model.ComfyUI.comfy.t2i_adapter.adapter",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.comfy.taesd",
    description: "model.ComfyUI.comfy.taesd",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.comfy.cldm",
    description: "model.ComfyUI.comfy.cldm",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.comfy.ldm.util",
    description: "model.ComfyUI.comfy.ldm.util",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.comfy.ldm.ema",
    description: "model.ComfyUI.comfy.ldm.ema",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.comfy.ldm.temporal_ae",
    description: "model.ComfyUI.comfy.ldm.temporal_ae",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.comfy.ldm.attention",
    description: "model.ComfyUI.comfy.ldm.attention",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.comfy.ldm.sub_quadratic_attention",
    description: "model.ComfyUI.comfy.ldm.sub_quadratic_attention",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.comfy.ldm.encoders.noise_aug_modules",
    description: "model.ComfyUI.comfy.ldm.encoders.noise_aug_modules",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.comfy.ldm.diffusionupscaling",
    description: "model.ComfyUI.comfy.ldm.diffusionupscaling",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.comfy.ldm.diffusionopenaimodel",
    description: "model.ComfyUI.comfy.ldm.diffusionopenaimodel",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.comfy.ldm.diffusionmodel",
    description: "model.ComfyUI.comfy.ldm.diffusionmodel",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.comfy.ldm.diffusionutil",
    description: "model.ComfyUI.comfy.ldm.diffusionutil",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.comfy.ldm.distributions",
    description: "model.ComfyUI.comfy.ldm.distributions",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.comfy.ldm.models.autoencoder",
    description: "model.ComfyUI.comfy.ldm.models.autoencoder",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.comfy.k_diffusion.utils",
    description: "model.ComfyUI.comfy.k_diffusion.utils",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ComfyUI.comfy.k_diffusion.sampling",
    description: "model.ComfyUI.comfy.k_diffusion.sampling",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.llama",
    description: "model.llama",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.llama.tokenization_llama_fast",
    description: "model.llama.tokenization_llama_fast",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.stock",
    description: "model.stock",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.text2video",
    description: "model.text2video",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.musicgen",
    description: "model.musicgen",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.image2video",
    description: "model.image2video",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.finetune",
    description: "model.finetune",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.apify",
    description: "model.apify",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.chat.litellm",
    description: "model.chat.litellm",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.chat.azureml_endpoint",
    description: "model.chat.azureml_endpoint",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.chat.human",
    description: "model.chat.human",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.chat.promptlayer_openai",
    description: "model.chat.promptlayer_openai",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.chat.openai",
    description: "model.chat.openai",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.chat.ollama",
    description: "model.chat.ollama",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.chat.mlflow_ai_gateway",
    description: "model.chat.mlflow_ai_gateway",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.chat.ernie",
    description: "model.chat.ernie",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.chat.anyscale",
    description: "model.chat.anyscale",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.chat.google_palm",
    description: "model.chat.google_palm",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.chat.anthropic",
    description: "model.chat.anthropic",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.chat",
    description: "model.chat",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.chat.azure_openai",
    description: "model.chat.azure_openai",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.chat.base",
    description: "model.chat.base",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.chat.fake",
    description: "model.chat.fake",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.chat.vertexai",
    description: "model.chat.vertexai",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.metamask",
    description: "model.metamask",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.metamask.components.metamask",
    description: "model.metamask.components.metamask",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.gptq",
    description: "model.gptq",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.bt",
    description: "model.bt",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.bt.server",
    description: "model.bt.server",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.bt.dendrite",
    description: "model.bt.dendrite",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.litellm",
    description: "model.litellm",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.music_mixer",
    description: "model.music_mixer",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.corcel",
    description: "model.corcel",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.timeseries_prediction",
    description: "model.timeseries_prediction",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.video2text",
    description: "model.video2text",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.adapter",
    description: "model.adapter",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.adapter.block.AdapterTransformerBlock",
    description: "model.adapter.block.AdapterTransformerBlock",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.adapter.block.AdapterBlock",
    description: "model.adapter.block.AdapterBlock",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.nodes",
    description: "model.comfyui.nodes",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.cuda_malloc",
    description: "model.comfyui.cuda_malloc",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.server",
    description: "model.comfyui.server",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.latent_preview",
    description: "model.comfyui.latent_preview",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.tests.inference",
    description: "model.comfyui.tests.inference",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.comfy_extras.nodes_compositing",
    description: "model.comfyui.comfy_extras.nodes_compositing",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.comfy_extras.nodes_model_advanced",
    description: "model.comfyui.comfy_extras.nodes_model_advanced",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.comfy_extras.chainner_models.model_loading",
    description: "model.comfyui.comfy_extras.chainner_models.model_loading",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.comfy_extras.chainner_models.architecture.DAT",
    description: "model.comfyui.comfy_extras.chainner_models.architecture.DAT",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.comfy_extras.chainner_models.architecture.SwiftSRGAN",
    description: "model.comfyui.comfy_extras.chainner_models.architecture.SwiftSRGAN",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.comfy_extras.chainner_models.architecture.HAT",
    description: "model.comfyui.comfy_extras.chainner_models.architecture.HAT",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.comfy_extras.chainner_models.architecture.block",
    description: "model.comfyui.comfy_extras.chainner_models.architecture.block",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.comfy_extras.chainner_models.architecture.SRVGG",
    description: "model.comfyui.comfy_extras.chainner_models.architecture.SRVGG",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.comfy_extras.chainner_models.architecture.RRDB",
    description: "model.comfyui.comfy_extras.chainner_models.architecture.RRDB",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.comfy_extras.chainner_models.architecture.Swin2SR",
    description: "model.comfyui.comfy_extras.chainner_models.architecture.Swin2SR",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.comfy_extras.chainner_models.architecture.LaMa",
    description: "model.comfyui.comfy_extras.chainner_models.architecture.LaMa",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.comfy_extras.chainner_models.architecture.SPSR",
    description: "model.comfyui.comfy_extras.chainner_models.architecture.SPSR",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.comfy_extras.chainner_models.architecture.SCUNet",
    description: "model.comfyui.comfy_extras.chainner_models.architecture.SCUNet",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.comfy_extras.chainner_models.architecture.SwinIR",
    description: "model.comfyui.comfy_extras.chainner_models.architecture.SwinIR",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.comfy_extras.chainner_models.architecture.timm.drop",
    description: "model.comfyui.comfy_extras.chainner_models.architecture.timm.drop",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.comfy_extras.chainner_models.architecture.OmniSR.OSA",
    description: "model.comfyui.comfy_extras.chainner_models.architecture.OmniSR.OSA",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.comfy_extras.chainner_models.architecture.OmniSR",
    description: "model.comfyui.comfy_extras.chainner_models.architecture.OmniSR",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.comfy_extras.chainner_models.architecture.OmniSR.esa",
    description: "model.comfyui.comfy_extras.chainner_models.architecture.OmniSR.esa",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.comfy_extras.chainner_models.architecture.OmniSR.ChannelAttention",
    description: "model.comfyui.comfy_extras.chainner_models.architecture.OmniSR.ChannelAttention",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.comfy_extras.chainner_models.architecture.OmniSR.OSAG",
    description: "model.comfyui.comfy_extras.chainner_models.architecture.OmniSR.OSAG",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.comfy_extras.chainner_models.architecture.OmniSR.layernorm",
    description: "model.comfyui.comfy_extras.chainner_models.architecture.OmniSR.layernorm",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.comfy_extras.chainner_models.architecture.face.upfirdn2d",
    description: "model.comfyui.comfy_extras.chainner_models.architecture.face.upfirdn2d",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.comfy_extras.chainner_models.architecture.face.fused_act",
    description: "model.comfyui.comfy_extras.chainner_models.architecture.face.fused_act",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.comfy_extras.chainner_models.architecture.face.gfpganv1_arch",
    description: "model.comfyui.comfy_extras.chainner_models.architecture.face.gfpganv1_arch",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.comfy_extras.chainner_models.architecture.face.arcface_arch",
    description: "model.comfyui.comfy_extras.chainner_models.architecture.face.arcface_arch",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.comfy_extras.chainner_models.architecture.face.stylegan2_arch",
    description: "model.comfyui.comfy_extras.chainner_models.architecture.face.stylegan2_arch",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.comfy_extras.chainner_models.architecture.face.gfpgan_bilinear_arch",
    description: "model.comfyui.comfy_extras.chainner_models.architecture.face.gfpgan_bilinear_arch",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.comfy_extras.chainner_models.architecture.face.stylegan2_clean_arch",
    description: "model.comfyui.comfy_extras.chainner_models.architecture.face.stylegan2_clean_arch",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.comfy_extras.chainner_models.architecture.face.restoreformer_arch",
    description: "model.comfyui.comfy_extras.chainner_models.architecture.face.restoreformer_arch",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.comfy_extras.chainner_models.architecture.face.stylegan2_bilinear_arch",
    description: "model.comfyui.comfy_extras.chainner_models.architecture.face.stylegan2_bilinear_arch",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.comfy_extras.chainner_models.architecture.face.gfpganv1_clean_arch",
    description: "model.comfyui.comfy_extras.chainner_models.architecture.face.gfpganv1_clean_arch",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.comfy_extras.chainner_models.architecture.face.codeformer",
    description: "model.comfyui.comfy_extras.chainner_models.architecture.face.codeformer",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.comfy.cli_args",
    description: "model.comfyui.comfy.cli_args",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.comfy.samplers",
    description: "model.comfyui.comfy.samplers",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.comfy.sd1_clip",
    description: "model.comfyui.comfy.sd1_clip",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.comfy.model_management",
    description: "model.comfyui.comfy.model_management",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.comfy.sdxl_clip",
    description: "model.comfyui.comfy.sdxl_clip",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.comfy.model_base",
    description: "model.comfyui.comfy.model_base",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.comfy.latent_formats",
    description: "model.comfyui.comfy.latent_formats",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.comfy.ops",
    description: "model.comfyui.comfy.ops",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.comfy.supported_models",
    description: "model.comfyui.comfy.supported_models",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.comfy.clip_vision",
    description: "model.comfyui.comfy.clip_vision",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.comfy.controlnet",
    description: "model.comfyui.comfy.controlnet",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.comfy.checkpoint_pickle",
    description: "model.comfyui.comfy.checkpoint_pickle",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.comfy.conds",
    description: "model.comfyui.comfy.conds",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.comfy.model_sampling",
    description: "model.comfyui.comfy.model_sampling",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.comfy.sd2_clip",
    description: "model.comfyui.comfy.sd2_clip",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.comfy.gligen",
    description: "model.comfyui.comfy.gligen",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.comfy.clip_model",
    description: "model.comfyui.comfy.clip_model",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.comfy.sd",
    description: "model.comfyui.comfy.sd",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.comfy.t2i_adapter.adapter",
    description: "model.comfyui.comfy.t2i_adapter.adapter",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.comfy.taesd",
    description: "model.comfyui.comfy.taesd",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.comfy.cldm",
    description: "model.comfyui.comfy.cldm",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.comfy.ldm.util",
    description: "model.comfyui.comfy.ldm.util",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.comfy.ldm.ema",
    description: "model.comfyui.comfy.ldm.ema",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.comfy.ldm.temporal_ae",
    description: "model.comfyui.comfy.ldm.temporal_ae",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.comfy.ldm.attention",
    description: "model.comfyui.comfy.ldm.attention",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.comfy.ldm.sub_quadratic_attention",
    description: "model.comfyui.comfy.ldm.sub_quadratic_attention",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.comfy.ldm.encoders.noise_aug_modules",
    description: "model.comfyui.comfy.ldm.encoders.noise_aug_modules",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.comfy.ldm.diffusionupscaling",
    description: "model.comfyui.comfy.ldm.diffusionupscaling",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.comfy.ldm.diffusionopenaimodel",
    description: "model.comfyui.comfy.ldm.diffusionopenaimodel",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.comfy.ldm.diffusionmodel",
    description: "model.comfyui.comfy.ldm.diffusionmodel",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.comfy.ldm.diffusionutil",
    description: "model.comfyui.comfy.ldm.diffusionutil",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.comfy.ldm.distributions",
    description: "model.comfyui.comfy.ldm.distributions",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.comfy.ldm.models.autoencoder",
    description: "model.comfyui.comfy.ldm.models.autoencoder",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.comfy.k_diffusion.utils",
    description: "model.comfyui.comfy.k_diffusion.utils",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.comfyui.comfy.k_diffusion.sampling",
    description: "model.comfyui.comfy.k_diffusion.sampling",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.demusics",
    description: "model.demusics",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.block.activations",
    description: "model.block.activations",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.openrouter",
    description: "model.openrouter",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.captcha",
    description: "model.captcha",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.beautifulsoap",
    description: "model.beautifulsoap",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.beautifulsoap.google_search",
    description: "model.beautifulsoap.google_search",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.beautifulsoap.random_website",
    description: "model.beautifulsoap.random_website",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.beautifulsoap.image_fetcher",
    description: "model.beautifulsoap.image_fetcher",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ijepa",
    description: "model.ijepa",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ijepa.src.transforms",
    description: "model.ijepa.src.transforms",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ijepa.src.datasets.imagenet1k",
    description: "model.ijepa.src.datasets.imagenet1k",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ijepa.src.masks.random",
    description: "model.ijepa.src.masks.random",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ijepa.src.masks.multiblock",
    description: "model.ijepa.src.masks.multiblock",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ijepa.src.masks.default",
    description: "model.ijepa.src.masks.default",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ijepa.src.utils.distributed",
    description: "model.ijepa.src.utils.distributed",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ijepa.src.utils.logging",
    description: "model.ijepa.src.utils.logging",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ijepa.src.utils.schedulers",
    description: "model.ijepa.src.utils.schedulers",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ijepa.src.models.vision_transformer",
    description: "model.ijepa.src.models.vision_transformer",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.talker",
    description: "model.talker",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.talker.src.test_audio2coeff",
    description: "model.talker.src.test_audio2coeff",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.talker.src.gradio_demo",
    description: "model.talker.src.gradio_demo",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.talker.src.facerender.animate",
    description: "model.talker.src.facerender.animate",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.talker.src.facerender.sync_batchnorm.unittest",
    description: "model.talker.src.facerender.sync_batchnorm.unittest",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.talker.src.facerender.sync_batchnorm.replicate",
    description: "model.talker.src.facerender.sync_batchnorm.replicate",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.talker.src.facerender.sync_batchnorm.comm",
    description: "model.talker.src.facerender.sync_batchnorm.comm",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.talker.src.facerender.sync_batchnorm.batchnorm",
    description: "model.talker.src.facerender.sync_batchnorm.batchnorm",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.talker.src.facerender.keypoint_detector",
    description: "model.talker.src.facerender.keypoint_detector",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.talker.src.facerender.discriminator",
    description: "model.talker.src.facerender.discriminator",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.talker.src.facerender.util",
    description: "model.talker.src.facerender.util",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.talker.src.facerender.make_animation",
    description: "model.talker.src.facerender.make_animation",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.talker.src.facerender.dense_motion",
    description: "model.talker.src.facerender.dense_motion",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.talker.src.facerender.generator",
    description: "model.talker.src.facerender.generator",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.talker.src.facerender.mapping",
    description: "model.talker.src.facerender.mapping",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.talker.src.face3d.extract_kp_videos_safe",
    description: "model.talker.src.face3d.extract_kp_videos_safe",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.talker.src.face3d.extract_kp_videos",
    description: "model.talker.src.face3d.extract_kp_videos",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.talker.src.face3d.data.template_dataset",
    description: "model.talker.src.face3d.data.template_dataset",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.talker.src.face3d.data.__init__",
    description: "model.talker.src.face3d.data.__init__",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.talker.src.face3d.data.flist_dataset",
    description: "model.talker.src.face3d.data.flist_dataset",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.talker.src.face3d.data.image_folder",
    description: "model.talker.src.face3d.data.image_folder",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.talker.src.face3d.data.base_dataset",
    description: "model.talker.src.face3d.data.base_dataset",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.talker.src.face3d.models.facerecon_model",
    description: "model.talker.src.face3d.models.facerecon_model",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.talker.src.face3d.models.losses",
    description: "model.talker.src.face3d.models.losses",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.talker.src.face3d.models.networks",
    description: "model.talker.src.face3d.models.networks",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.talker.src.face3d.models.template_model",
    description: "model.talker.src.face3d.models.template_model",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.talker.src.face3d.models.base_model",
    description: "model.talker.src.face3d.models.base_model",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.talker.src.face3d.models.arcface_torch.losses",
    description: "model.talker.src.face3d.models.arcface_torch.losses",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.talker.src.face3d.models.arcface_torch.eval_ijbc",
    description: "model.talker.src.face3d.models.arcface_torch.eval_ijbc",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.talker.src.face3d.models.arcface_torch.partial_fc",
    description: "model.talker.src.face3d.models.arcface_torch.partial_fc",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.talker.src.face3d.models.arcface_torch.onnx_ijbc",
    description: "model.talker.src.face3d.models.arcface_torch.onnx_ijbc",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.talker.src.face3d.models.arcface_torch.dataset",
    description: "model.talker.src.face3d.models.arcface_torch.dataset",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.talker.src.face3d.models.arcface_torch.utils.utils_callbacks",
    description: "model.talker.src.face3d.models.arcface_torch.utils.utils_callbacks",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.talker.src.face3d.models.arcface_torch.utils.utils_amp",
    description: "model.talker.src.face3d.models.arcface_torch.utils.utils_amp",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.talker.src.face3d.models.arcface_torch.utils.utils_logging",
    description: "model.talker.src.face3d.models.arcface_torch.utils.utils_logging",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.talker.src.face3d.models.arcface_torch.backbones.mobilefacenet",
    description: "model.talker.src.face3d.models.arcface_torch.backbones.mobilefacenet",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.talker.src.face3d.models.arcface_torch.backbones.iresnet2060",
    description: "model.talker.src.face3d.models.arcface_torch.backbones.iresnet2060",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.talker.src.face3d.models.arcface_torch.backbones.iresnet",
    description: "model.talker.src.face3d.models.arcface_torch.backbones.iresnet",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.talker.src.face3d.util.nvdiffrast",
    description: "model.talker.src.face3d.util.nvdiffrast",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.talker.src.face3d.util.my_awing_arch",
    description: "model.talker.src.face3d.util.my_awing_arch",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.talker.src.face3d.util.visualizer",
    description: "model.talker.src.face3d.util.visualizer",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.talker.src.face3d.util",
    description: "model.talker.src.face3d.util",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.talker.src.face3d.options",
    description: "model.talker.src.face3d.options",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.talker.src.utils.preprocess",
    description: "model.talker.src.utils.preprocess",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.talker.src.utils.model2safetensor",
    description: "model.talker.src.utils.model2safetensor",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.talker.src.utils.face_enhancer",
    description: "model.talker.src.utils.face_enhancer",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.talker.src.utils.text2speech",
    description: "model.talker.src.utils.text2speech",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.talker.src.audio2pose_models.cvae",
    description: "model.talker.src.audio2pose_models.cvae",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.talker.src.audio2pose_models.res_unet",
    description: "model.talker.src.audio2pose_models.res_unet",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.talker.src.audio2pose_models.discriminator",
    description: "model.talker.src.audio2pose_models.discriminator",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.talker.src.audio2pose_models.networks",
    description: "model.talker.src.audio2pose_models.networks",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.talker.src.audio2pose_models.audio2pose",
    description: "model.talker.src.audio2pose_models.audio2pose",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.talker.src.audio2pose_models.audio_encoder",
    description: "model.talker.src.audio2pose_models.audio_encoder",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.talker.src.audio2exp_models.networks",
    description: "model.talker.src.audio2exp_models.networks",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.talker.src.audio2exp_models.audio2exp",
    description: "model.talker.src.audio2exp_models.audio2exp",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.ocr",
    description: "model.ocr",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.image2image",
    description: "model.image2image",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.stabletune",
    description: "model.stabletune",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.translation",
    description: "model.translation",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.deepface",
    description: "model.deepface",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.sentence",
    description: "model.sentence",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.image2text",
    description: "model.image2text",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.com_twscrape",
    description: "model.com_twscrape",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.com_twscrape.imap",
    description: "model.com_twscrape.imap",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.com_twscrape.account",
    description: "model.com_twscrape.account",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.com_twscrape.models",
    description: "model.com_twscrape.models",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.com_twscrape.api",
    description: "model.com_twscrape.api",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.com_twscrape.queue_client",
    description: "model.com_twscrape.queue_client",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.com_twscrape.accounts_pool",
    description: "model.com_twscrape.accounts_pool",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.litgpt",
    description: "model.litgpt",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.litgpt.tests.test_utils",
    description: "model.litgpt.tests.test_utils",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.litgpt.tests.test_lora",
    description: "model.litgpt.tests.test_lora",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.litgpt.tests.test_packed_dataset",
    description: "model.litgpt.tests.test_packed_dataset",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.litgpt.tests.test_model",
    description: "model.litgpt.tests.test_model",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.litgpt.quantize.bnb",
    description: "model.litgpt.quantize.bnb",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.litgpt.quantize.gptq",
    description: "model.litgpt.quantize.gptq",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.litgpt.lit_gpt.adapter",
    description: "model.litgpt.lit_gpt.adapter",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.litgpt.lit_gpt.utils",
    description: "model.litgpt.lit_gpt.utils",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.litgpt.lit_gpt.model",
    description: "model.litgpt.lit_gpt.model",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.litgpt.lit_gpt.speed_monitor",
    description: "model.litgpt.lit_gpt.speed_monitor",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.litgpt.lit_gpt.rmsnorm",
    description: "model.litgpt.lit_gpt.rmsnorm",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.litgpt.lit_gpt.packed_dataset",
    description: "model.litgpt.lit_gpt.packed_dataset",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.litgpt.lit_gpt.lora",
    description: "model.litgpt.lit_gpt.lora",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.litgpt.pretrain.openwebtext_trainer",
    description: "model.litgpt.pretrain.openwebtext_trainer",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.lora",
    description: "model.lora",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.openai",
    description: "model.openai",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.openai.experimental",
    description: "model.openai.experimental",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.openai.free",
    description: "model.openai.free",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.imageedit",
    description: "model.imageedit",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.imageedit.scheduling_dpmsolver_multistep_inject",
    description: "model.imageedit.scheduling_dpmsolver_multistep_inject",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.imageedit.pipeline_semantic_stable_diffusion_img2img_solver",
    description: "model.imageedit.pipeline_semantic_stable_diffusion_img2img_solver",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.remote",
    description: "model.remote",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.gan",
    description: "model.gan",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.gan.esrgan.datasets",
    description: "model.gan.esrgan.datasets",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.gan.esrgan.models",
    description: "model.gan.esrgan.models",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.gan.stargan.datasets",
    description: "model.gan.stargan.datasets",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.gan.stargan.models",
    description: "model.gan.stargan.models",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.gan.pix2pix.datasets",
    description: "model.gan.pix2pix.datasets",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.gan.pix2pix",
    description: "model.gan.pix2pix",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.gan.pix2pix.models",
    description: "model.gan.pix2pix.models",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.gan.bicyclegan.datasets",
    description: "model.gan.bicyclegan.datasets",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.gan.bicyclegan.models",
    description: "model.gan.bicyclegan.models",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.gan.cluster_gan.clustergan",
    description: "model.gan.cluster_gan.clustergan",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.gan.wgan_gp",
    description: "model.gan.wgan_gp",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.gan.ccgan.datasets",
    description: "model.gan.ccgan.datasets",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.gan.ccgan.models",
    description: "model.gan.ccgan.models",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.gan.munit.datasets",
    description: "model.gan.munit.datasets",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.gan.munit.models",
    description: "model.gan.munit.models",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.gan.munit",
    description: "model.gan.munit",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.gan.aae",
    description: "model.gan.aae",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.gan.unit.datasets",
    description: "model.gan.unit.datasets",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.gan.unit.models",
    description: "model.gan.unit.models",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.gan.unit",
    description: "model.gan.unit",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.gan.discogan.datasets",
    description: "model.gan.discogan.datasets",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.gan.discogan.models",
    description: "model.gan.discogan.models",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.gan.srgan.datasets",
    description: "model.gan.srgan.datasets",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.gan.srgan.models",
    description: "model.gan.srgan.models",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.gan.cogan.mnistm",
    description: "model.gan.cogan.mnistm",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.gan.dualgan.datasets",
    description: "model.gan.dualgan.datasets",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.gan.dualgan.models",
    description: "model.gan.dualgan.models",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.gan.pixelda.mnistm",
    description: "model.gan.pixelda.mnistm",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.gan.pixelda",
    description: "model.gan.pixelda",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.gan.context_encoder.datasets",
    description: "model.gan.context_encoder.datasets",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.gan.context_encoder",
    description: "model.gan.context_encoder",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.gan.context_encoder.models",
    description: "model.gan.context_encoder.models",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.gan.wgan_div",
    description: "model.gan.wgan_div",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.gan.cyclegan.datasets",
    description: "model.gan.cyclegan.datasets",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.gan.cyclegan.models",
    description: "model.gan.cyclegan.models",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.text2speech",
    description: "model.text2speech",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.text2speech.video.pipeline",
    description: "model.text2speech.video.pipeline",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.text2speech.video.models.unet_blocks",
    description: "model.text2speech.video.models.unet_blocks",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.text2speech.video.models.resnet",
    description: "model.text2speech.video.models.resnet",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.text2speech.video.models.transformers",
    description: "model.text2speech.video.models.transformers",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.text2speech.video.models.unet",
    description: "model.text2speech.video.models.unet",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.gemini",
    description: "model.gemini",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.langchain_agent",
    description: "model.langchain_agent",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.langchain_agent.test_agent",
    description: "model.langchain_agent.test_agent",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.dalle",
    description: "model.dalle",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.diffusion.pipeline",
    description: "model.diffusion.pipeline",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.diffusion",
    description: "model.diffusion",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.diffusion.stable",
    description: "model.diffusion.stable",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.diffusion.dreambooth",
    description: "model.diffusion.dreambooth",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.diffusion.dreambooth.dataset",
    description: "model.diffusion.dreambooth.dataset",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.supdawg",
    description: "model.supdawg",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.hf",
    description: "model.hf",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.hf.textgen",
    description: "model.hf.textgen",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.hf.textgen.clients.python.tests.test_client",
    description: "model.hf.textgen.clients.python.tests.test_client",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.hf.textgen.clients.python.text_generation.types",
    description: "model.hf.textgen.clients.python.text_generation.types",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.hf.textgen.clients.python.text_generation.errors",
    description: "model.hf.textgen.clients.python.text_generation.errors",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.hf.textgen.clients.python.text_generation.client",
    description: "model.hf.textgen.clients.python.text_generation.client",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.hf.textgen.clients.python.text_generation.inference_api",
    description: "model.hf.textgen.clients.python.text_generation.inference_api",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.hf.textgen.integration-tests.conftest",
    description: "model.hf.textgen.integration-tests.conftest",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.hf.textgen.server.tests.models.test_model",
    description: "model.hf.textgen.server.tests.models.test_model",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.hf.textgen.server.text_generation_server.interceptor",
    description: "model.hf.textgen.server.text_generation_server.interceptor",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.hf.textgen.server.text_generation_server.tracing",
    description: "model.hf.textgen.server.text_generation_server.tracing",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.hf.textgen.server.text_generation_server.server",
    description: "model.hf.textgen.server.text_generation_server.server",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.hf.textgen.server.text_generation_server.cli",
    description: "model.hf.textgen.server.text_generation_server.cli",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.hf.textgen.server.text_generation_server.utils.logits_process",
    description: "model.hf.textgen.server.text_generation_server.utils.logits_process",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.hf.textgen.server.text_generation_server.utils.watermark",
    description: "model.hf.textgen.server.text_generation_server.utils.watermark",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.hf.textgen.server.text_generation_server.utils.layers",
    description: "model.hf.textgen.server.text_generation_server.utils.layers",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.hf.textgen.server.text_generation_server.utils.gptq.custom_autotune",
    description: "model.hf.textgen.server.text_generation_server.utils.gptq.custom_autotune",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.hf.textgen.server.text_generation_server.utils.gptq.quant_linear",
    description: "model.hf.textgen.server.text_generation_server.utils.gptq.quant_linear",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.hf.textgen.server.text_generation_server.utils.gptq.quantize",
    description: "model.hf.textgen.server.text_generation_server.utils.gptq.quantize",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.hf.textgen.server.text_generation_server.models.types",
    description: "model.hf.textgen.server.text_generation_server.models.types",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.hf.textgen.server.text_generation_server.models.galactica",
    description: "model.hf.textgen.server.text_generation_server.models.galactica",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.hf.textgen.server.text_generation_server.models.model",
    description: "model.hf.textgen.server.text_generation_server.models.model",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.hf.textgen.server.text_generation_server.models.bloom",
    description: "model.hf.textgen.server.text_generation_server.models.bloom",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.hf.textgen.server.text_generation_server.models.mpt",
    description: "model.hf.textgen.server.text_generation_server.models.mpt",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.hf.textgen.server.text_generation_server.models.rw",
    description: "model.hf.textgen.server.text_generation_server.models.rw",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.hf.textgen.server.text_generation_server.models.causal_lm",
    description: "model.hf.textgen.server.text_generation_server.models.causal_lm",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.hf.textgen.server.text_generation_server.models.t5",
    description: "model.hf.textgen.server.text_generation_server.models.t5",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.hf.textgen.server.text_generation_server.models.flash_llama",
    description: "model.hf.textgen.server.text_generation_server.models.flash_llama",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.hf.textgen.server.text_generation_server.models.gpt_neox",
    description: "model.hf.textgen.server.text_generation_server.models.gpt_neox",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.hf.textgen.server.text_generation_server.models.santacoder",
    description: "model.hf.textgen.server.text_generation_server.models.santacoder",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.hf.textgen.server.text_generation_server.models.flash_neox",
    description: "model.hf.textgen.server.text_generation_server.models.flash_neox",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.hf.textgen.server.text_generation_server.models.opt",
    description: "model.hf.textgen.server.text_generation_server.models.opt",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.hf.textgen.server.text_generation_server.models.flash_causal_lm",
    description: "model.hf.textgen.server.text_generation_server.models.flash_causal_lm",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.hf.textgen.server.text_generation_server.models.flash_rw",
    description: "model.hf.textgen.server.text_generation_server.models.flash_rw",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.hf.textgen.server.text_generation_server.models.seq2seq_lm",
    description: "model.hf.textgen.server.text_generation_server.models.seq2seq_lm",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.hf.textgen.server.text_generation_server.models.flash_santacoder",
    description: "model.hf.textgen.server.text_generation_server.models.flash_santacoder",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.hf.textgen.server.text_generation_server.models.custom_modeling.mpt_modeling",
    description: "model.hf.textgen.server.text_generation_server.models.custom_modeling.mpt_modeling",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.hf.textgen.server.text_generation_server.models.custom_modeling.t5_modeling",
    description: "model.hf.textgen.server.text_generation_server.models.custom_modeling.t5_modeling",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.hf.textgen.server.text_generation_server.models.custom_modeling.flash_santacoder_modeling",
    description: "model.hf.textgen.server.text_generation_server.models.custom_modeling.flash_santacoder_modeling",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.hf.textgen.server.text_generation_server.models.custom_modeling.flash_neox_modeling",
    description: "model.hf.textgen.server.text_generation_server.models.custom_modeling.flash_neox_modeling",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.hf.textgen.server.text_generation_server.models.custom_modeling.flash_rw_modeling",
    description: "model.hf.textgen.server.text_generation_server.models.custom_modeling.flash_rw_modeling",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.hf.textgen.server.text_generation_server.models.custom_modeling.bloom_modeling",
    description: "model.hf.textgen.server.text_generation_server.models.custom_modeling.bloom_modeling",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.hf.textgen.server.text_generation_server.models.custom_modeling.neox_modeling",
    description: "model.hf.textgen.server.text_generation_server.models.custom_modeling.neox_modeling",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.hf.textgen.server.text_generation_server.models.custom_modeling.opt_modeling",
    description: "model.hf.textgen.server.text_generation_server.models.custom_modeling.opt_modeling",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.hf.textgen.server.text_generation_server.models.custom_modeling.flash_llama_modeling",
    description: "model.hf.textgen.server.text_generation_server.models.custom_modeling.flash_llama_modeling",
    address: "158.247.70.45:8888"
  },
  {
    name: "model.vectorstore",
    description: "model.vectorstore",
    address: "158.247.70.45:8888"
  },
  {
    name: "router.task",
    description: "router.task",
    address: "158.247.70.45:8888"
  },
  {
    name: "router",
    description: "router",
    address: "158.247.70.45:8888"
  },
  {
    name: "router.ex",
    description: "router.ex",
    address: "158.247.70.45:8888"
  },
  {
    name: "router.dashboard",
    description: "router.dashboard",
    address: "158.247.70.45:8888"
  },
  {
    name: "pool",
    description: "pool",
    address: "158.247.70.45:8888"
  },
  {
    name: "server",
    description: "server",
    address: "158.247.70.45:8888"
  },
  {
    name: "server.dashboard",
    description: "server.dashboard",
    address: "158.247.70.45:8888"
  },
  {
    name: "server.grpc",
    description: "server.grpc",
    address: "158.247.70.45:8888"
  },
  {
    name: "server.grpc.interceptor",
    description: "server.grpc.interceptor",
    address: "158.247.70.45:8888"
  },
  {
    name: "server.grpc.serializer",
    description: "server.grpc.serializer",
    address: "158.247.70.45:8888"
  },
  {
    name: "server.grpc.client",
    description: "server.grpc.client",
    address: "158.247.70.45:8888"
  },
  {
    name: "server.grpc.client.pool",
    description: "server.grpc.client.pool",
    address: "158.247.70.45:8888"
  },
  {
    name: "server.grpc.proto.server_pb2_grpc",
    description: "server.grpc.proto.server_pb2_grpc",
    address: "158.247.70.45:8888"
  },
  {
    name: "server.ucall",
    description: "server.ucall",
    address: "158.247.70.45:8888"
  },
  {
    name: "server.ucall.client",
    description: "server.ucall.client",
    address: "158.247.70.45:8888"
  },
  {
    name: "server.http",
    description: "server.http",
    address: "158.247.70.45:8888"
  },
  {
    name: "server.ws.filetransfer",
    description: "server.ws.filetransfer",
    address: "158.247.70.45:8888"
  },
  {
    name: "server.ws",
    description: "server.ws",
    address: "158.247.70.45:8888"
  },
  {
    name: "server.ws.client",
    description: "server.ws.client",
    address: "158.247.70.45:8888"
  },
  {
    name: "server.access",
    description: "server.access",
    address: "158.247.70.45:8888"
  },
  {
    name: "server.access.subspace",
    description: "server.access.subspace",
    address: "158.247.70.45:8888"
  },
  {
    name: "server.access.base",
    description: "server.access.base",
    address: "158.247.70.45:8888"
  },
  {
    name: "dashboard",
    description: "dashboard",
    address: "158.247.70.45:8888"
  },
  {
    name: "coder",
    description: "coder",
    address: "158.247.70.45:8888"
  },
  {
    name: "pm2",
    description: "pm2",
    address: "158.247.70.45:8888"
  },
  {
    name: "thread",
    description: "thread",
    address: "158.247.70.45:8888"
  },
  {
    name: "thread.pool",
    description: "thread.pool",
    address: "158.247.70.45:8888"
  },
  {
    name: "streamlit",
    description: "streamlit",
    address: "158.247.70.45:8888"
  },
  {
    name: "streamlit.auth",
    description: "streamlit.auth",
    address: "158.247.70.45:8888"
  },
  {
    name: "streamlit.watchdog",
    description: "streamlit.watchdog",
    address: "158.247.70.45:8888"
  },
  {
    name: "tool.registry",
    description: "tool.registry",
    address: "158.247.70.45:8888"
  },
  {
    name: "tool.read_file",
    description: "tool.read_file",
    address: "158.247.70.45:8888"
  },
  {
    name: "tool.write_file",
    description: "tool.write_file",
    address: "158.247.70.45:8888"
  },
  {
    name: "tool.get_best_apy",
    description: "tool.get_best_apy",
    address: "158.247.70.45:8888"
  },
  {
    name: "tool.swap",
    description: "tool.swap",
    address: "158.247.70.45:8888"
  },
  {
    name: "tool.compare_token_price",
    description: "tool.compare_token_price",
    address: "158.247.70.45:8888"
  },
  {
    name: "tool",
    description: "tool",
    address: "158.247.70.45:8888"
  },
  {
    name: "tool.search",
    description: "tool.search",
    address: "158.247.70.45:8888"
  },
  {
    name: "tool.defi.read_file",
    description: "tool.defi.read_file",
    address: "158.247.70.45:8888"
  },
  {
    name: "tool.defi.write_file",
    description: "tool.defi.write_file",
    address: "158.247.70.45:8888"
  },
  {
    name: "tool.defi.get_best_apy",
    description: "tool.defi.get_best_apy",
    address: "158.247.70.45:8888"
  },
  {
    name: "tool.defi.swap",
    description: "tool.defi.swap",
    address: "158.247.70.45:8888"
  },
  {
    name: "tool.defi.compare_token_price",
    description: "tool.defi.compare_token_price",
    address: "158.247.70.45:8888"
  },
  {
    name: "tool.defi.tool",
    description: "tool.defi.tool",
    address: "158.247.70.45:8888"
  },
  {
    name: "tool.defi.defillama.aave",
    description: "tool.defi.defillama.aave",
    address: "158.247.70.45:8888"
  },
  {
    name: "tool.defi.defillama.lido",
    description: "tool.defi.defillama.lido",
    address: "158.247.70.45:8888"
  },
  {
    name: "tool.defi.defillama",
    description: "tool.defi.defillama",
    address: "158.247.70.45:8888"
  },
  {
    name: "tool.defi.defillama.rocketpool",
    description: "tool.defi.defillama.rocketpool",
    address: "158.247.70.45:8888"
  },
  {
    name: "tool.defi.inch.balances",
    description: "tool.defi.inch.balances",
    address: "158.247.70.45:8888"
  },
  {
    name: "tool.defi.inch.gasprice",
    description: "tool.defi.inch.gasprice",
    address: "158.247.70.45:8888"
  },
  {
    name: "tool.defi.inch.prices",
    description: "tool.defi.inch.prices",
    address: "158.247.70.45:8888"
  },
  {
    name: "tool.defi.inch",
    description: "tool.defi.inch",
    address: "158.247.70.45:8888"
  },
  {
    name: "tool.web",
    description: "tool.web",
    address: "158.247.70.45:8888"
  },
  {
    name: "user",
    description: "user",
    address: "158.247.70.45:8888"
  },
  {
    name: "user.app",
    description: "user.app",
    address: "158.247.70.45:8888"
  },
  {
    name: "history",
    description: "history",
    address: "158.247.70.45:8888"
  },
  {
    name: "text",
    description: "text",
    address: "158.247.70.45:8888"
  },
  {
    name: "remote",
    description: "remote",
    address: "158.247.70.45:8888"
  },
  {
    name: "remote.dashboard",
    description: "remote.dashboard",
    address: "158.247.70.45:8888"
  },
  {
    name: "auth",
    description: "auth",
    address: "158.247.70.45:8888"
  },
  {
    name: "storage.new_storage",
    description: "storage.new_storage",
    address: "158.247.70.45:8888"
  },
  {
    name: "storage",
    description: "storage",
    address: "158.247.70.45:8888"
  },
  {
    name: "storage.vali",
    description: "storage.vali",
    address: "158.247.70.45:8888"
  },
  {
    name: "storage.vector",
    description: "storage.vector",
    address: "158.247.70.45:8888"
  },
  {
    name: "evm",
    description: "evm",
    address: "158.247.70.45:8888"
  },
  {
    name: "evm.contract",
    description: "evm.contract",
    address: "158.247.70.45:8888"
  },
  {
    name: "evm.key",
    description: "evm.key",
    address: "158.247.70.45:8888"
  },
  {
    name: "evm.network",
    description: "evm.network",
    address: "158.247.70.45:8888"
  },
  {
    name: "trainer",
    description: "trainer",
    address: "158.247.70.45:8888"
  },
  {
    name: "trainer.bittensor",
    description: "trainer.bittensor",
    address: "158.247.70.45:8888"
  },
  {
    name: "trainer.dream",
    description: "trainer.dream",
    address: "158.247.70.45:8888"
  },
  {
    name: "plotly",
    description: "plotly",
    address: "158.247.70.45:8888"
  },
  {
    name: "diffusion.projected-gan.legacy",
    description: "diffusion.projected-gan.legacy",
    address: "158.247.70.45:8888"
  },
  {
    name: "diffusion.projected-gan.training.loss",
    description: "diffusion.projected-gan.training.loss",
    address: "158.247.70.45:8888"
  },
  {
    name: "diffusion.projected-gan.training.dataset",
    description: "diffusion.projected-gan.training.dataset",
    address: "158.247.70.45:8888"
  },
  {
    name: "diffusion.projected-gan.dnnlib.util",
    description: "diffusion.projected-gan.dnnlib.util",
    address: "158.247.70.45:8888"
  },
  {
    name: "diffusion.projected-gan.torch_utils.misc",
    description: "diffusion.projected-gan.torch_utils.misc",
    address: "158.247.70.45:8888"
  },
  {
    name: "diffusion.projected-gan.torch_utils.persistence",
    description: "diffusion.projected-gan.torch_utils.persistence",
    address: "158.247.70.45:8888"
  },
  {
    name: "diffusion.projected-gan.torch_utils.ops.bias_act",
    description: "diffusion.projected-gan.torch_utils.ops.bias_act",
    address: "158.247.70.45:8888"
  },
  {
    name: "diffusion.projected-gan.torch_utils.ops.upfirdn2d",
    description: "diffusion.projected-gan.torch_utils.ops.upfirdn2d",
    address: "158.247.70.45:8888"
  },
  {
    name: "diffusion.projected-gan.torch_utils.ops.filtered_lrelu",
    description: "diffusion.projected-gan.torch_utils.ops.filtered_lrelu",
    address: "158.247.70.45:8888"
  },
  {
    name: "diffusion.projected-gan.torch_utils.ops.grid_sample_gradfix",
    description: "diffusion.projected-gan.torch_utils.ops.grid_sample_gradfix",
    address: "158.247.70.45:8888"
  },
  {
    name: "diffusion.projected-gan.torch_utils.ops.fma",
    description: "diffusion.projected-gan.torch_utils.ops.fma",
    address: "158.247.70.45:8888"
  },
  {
    name: "diffusion.projected-gan.torch_utils.ops.conv2d_gradfix",
    description: "diffusion.projected-gan.torch_utils.ops.conv2d_gradfix",
    address: "158.247.70.45:8888"
  },
  {
    name: "diffusion.projected-gan.pg_discriminator",
    description: "diffusion.projected-gan.pg_discriminator",
    address: "158.247.70.45:8888"
  },
  {
    name: "diffusion.projected-gan.pg_networks_stylegan2",
    description: "diffusion.projected-gan.pg_networks_stylegan2",
    address: "158.247.70.45:8888"
  },
  {
    name: "diffusion.projected-gan.pg_blocks",
    description: "diffusion.projected-gan.pg_blocks",
    address: "158.247.70.45:8888"
  },
  {
    name: "diffusion.projected-gan.pg_projector",
    description: "diffusion.projected-gan.pg_projector",
    address: "158.247.70.45:8888"
  },
  {
    name: "diffusion.projected-gan.pg_diffusion",
    description: "diffusion.projected-gan.pg_diffusion",
    address: "158.247.70.45:8888"
  },
  {
    name: "diffusion.projected-gan.pg_networks_fastgan",
    description: "diffusion.projected-gan.pg_networks_fastgan",
    address: "158.247.70.45:8888"
  },
  {
    name: "diffusion.projected-gan.metrics.perceptual_path_length",
    description: "diffusion.projected-gan.metrics.perceptual_path_length",
    address: "158.247.70.45:8888"
  },
  {
    name: "os",
    description: "os",
    address: "158.247.70.45:8888"
  },
  {
    name: "subprocess",
    description: "subprocess",
    address: "158.247.70.45:8888"
  },
  {
    name: "executor.task",
    description: "executor.task",
    address: "158.247.70.45:8888"
  },
  {
    name: "executor",
    description: "executor",
    address: "158.247.70.45:8888"
  },
  {
    name: "executor.process",
    description: "executor.process",
    address: "158.247.70.45:8888"
  },
  {
    name: "executor.thread",
    description: "executor.thread",
    address: "158.247.70.45:8888"
  },
  {
    name: "selenium",
    description: "selenium",
    address: "158.247.70.45:8888"
  },
  {
    name: "appagent",
    description: "appagent",
    address: "158.247.70.45:8888"
  },
  {
    name: "hf",
    description: "hf",
    address: "158.247.70.45:8888"
  },
  {
    name: "utils.network",
    description: "utils.network",
    address: "158.247.70.45:8888"
  },
  {
    name: "module",
    description: "module",
    address: "158.247.70.45:8888"
  },
  {
    name: "module.peers",
    description: "module.peers",
    address: "158.247.70.45:8888"
  },
  {
    name: "module.tree",
    description: "module.tree",
    address: "158.247.70.45:8888"
  },
  {
    name: "module.tests",
    description: "module.tests",
    address: "158.247.70.45:8888"
  },
  {
    name: "module.wrap",
    description: "module.wrap",
    address: "158.247.70.45:8888"
  },
  {
    name: "module.manager",
    description: "module.manager",
    address: "158.247.70.45:8888"
  },
  {
    name: "module.config",
    description: "module.config",
    address: "158.247.70.45:8888"
  },
  {
    name: "module.watchdog",
    description: "module.watchdog",
    address: "158.247.70.45:8888"
  }
];

export default class ModulesService {
  static getModulesList = async (searchQuery = "") => {
    if (!searchQuery) {
      return modulesList;
    }

    return modulesList.filter((module) => module.name.includes(searchQuery) || (module.description && module.description.includes(searchQuery)));
  };


  static getModuleDetailsByName = async (name: string) => {
    const moduleDetails = modulesList.find((module) => module.name === name);

    if (!moduleDetails) {
      return null;
    }

    return moduleDetails;
  };
}
