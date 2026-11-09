---
title: "LLM Inference Engine Learning Notes"
description: "Follow one request from prompt to first token, then work out how much memory its KV cache needs."
---

Why can a chatbot pause before its first word, then stream the rest of the sentence quickly? To see what changes after the first token, trace a single request through a decoder-only language model.

## Where does the first token come from?

Suppose a prompt is already tokenized as `[A, B, C]`. During **prefill**, the model processes the three known tokens. At each attention layer, it saves keys and values for their positions. The model's output at position `C` gives scores for the next token. Suppose the sampler picks `D`: the server can stream it immediately.

Now the user has seen `D`, but the cache still covers only `A, B, C`. During the next **decode** step, the engine feeds in `D`, adds its keys and values, and uses the scores at that position to pick `E`. This one-token offset matters when reading a trace: emitting a token and caching that token happen at different steps in ordinary autoregressive decoding.

## What does the cache cost?

Caching saves the work of computing old keys and values again. The new query still uses those earlier positions when computing full attention. The [Hugging Face cache guide](https://huggingface.co/docs/transformers/en/cache_explanation) shows how the new and cached tensors fit together.

Consider one sequence in a model with 32 full-attention layers, 8 KV heads per layer, 128 values per head, and a two-byte cache dtype. Each cached token needs:

```text
2 (key and value) × 32 layers × 8 KV heads × 128 × 2 bytes
= 131,072 bytes = 128 KiB
```

At 4,096 cached positions, that is **512 MiB for one sequence**. This is a size estimate for the KV tensors alone; model weights, temporary buffers, and allocation overhead are separate. A serving engine has to budget for many such sequences, each growing at a different rate. [PagedAttention](https://arxiv.org/abs/2309.06180) addresses that allocation problem by letting a sequence grow in blocks.

## A small experiment to try

Send the same short question twice to a warmed-up engine: once by itself, once after a long passage. Keep the model and generation settings fixed, disable prefix reuse, run one request at a time, and ask for the same number of output tokens. Record two measurements:

- **Time to first token:** from sending the request to receiving its first generated token. This includes prompt processing as well as any queueing and transport time.
- **Mean time between subsequent tokens:** divide the time from the first token to the last by the number of output tokens minus one. Use token timestamps; a network chunk can contain several tokens.

The prediction is that the longer prompt will take longer to produce its first token. Subsequent tokens may slow down too, because they attend over a longer context. Keeping these two timings separate makes it possible to tell which wait actually changed. [DistServe](https://arxiv.org/abs/2401.09670) uses this distinction to study how a serving engine meets latency targets.
