# Voice AI Latency Benchmarks (2024-2026)

## Evidence standard for this revision

This file separates verified vendor capability claims from measured latency claims. Several prior entries used precise millisecond ranges that were not traceable to primary sources. Those have been downgraded to planning assumptions unless a source explicitly publishes a latency number. Each source below includes an access date and a confidence label.

Confidence labels:
- High: official API documentation, model card, pricing page, SLA/status documentation, or vendor engineering post with explicit latency/performance claim.
- Medium: official vendor product page or reputable engineering benchmark with method described.
- Low: secondary blog, marketing page, or broad claim without methodology. Low-confidence figures should not drive ROI without sensitivity analysis.

Access date for all links in this document: Accessed on 2026-05-20.

## STT latency and streaming evidence

### OpenAI Whisper / OpenAI Audio transcription
- Verifiable claim: OpenAI exposes Whisper-derived speech-to-text through its Audio / speech-to-text APIs; official docs describe transcription usage but do not publish a stable p50/p95 streaming latency SLA for Whisper in the cited page.
- Planning assumption retained from prior draft: 150-300 ms for short-clip processing on GPU should be treated as benchmark-dependent, not vendor-validated.
- Source: https://platform.openai.com/docs/guides/speech-to-text — Accessed on 2026-05-20 — Source type: official API docs — Confidence: High for API capability, Low for the 150-300 ms planning assumption.

### Google Cloud Speech-to-Text / Gemini audio input
- Verifiable claim: Google Cloud supports streaming Speech-to-Text and Gemini supports audio / multimodal inputs; official docs do not publish a universal p50 end-to-end latency number because latency depends on model, audio length, region, network, and streaming configuration.
- Planning assumption retained from prior draft: 200-400 ms streaming partial-result latency should be treated as workload-specific.
- Sources:
  - https://cloud.google.com/speech-to-text/docs/streaming-recognize — Accessed on 2026-05-20 — Source type: official API docs — Confidence: High for streaming support.
  - https://ai.google.dev/gemini-api/docs/audio — Accessed on 2026-05-20 — Source type: official API docs — Confidence: High for Gemini audio support.

### Deepgram Nova / Streaming API
- Verifiable claim: Deepgram publishes streaming speech-to-text APIs and low-latency real-time transcription positioning; exact application latency depends on model, endpointing, utterance length, network, and chunking.
- Planning assumption retained from prior draft: 100-250 ms real-time streaming should be modeled as an optimistic vendor/API configuration assumption, not a guaranteed SLA.
- Source: https://developers.deepgram.com/docs/streaming — Accessed on 2026-05-20 — Source type: official API docs — Confidence: High for streaming API support, Medium for low-latency positioning.

### AssemblyAI Universal / Streaming STT
- Verifiable claim: AssemblyAI supports real-time streaming transcription. Published model pages and docs are valid for capability and integration evidence; use measured internal benchmarks before committing to a latency SLA.
- Planning assumption retained from prior draft: 180-320 ms should be treated as a benchmark assumption requiring validation in the target network path.
- Source: https://www.assemblyai.com/docs/speech-to-text/streaming — Accessed on 2026-05-20 — Source type: official API docs — Confidence: High for streaming support, Low for exact latency range.

### ElevenLabs Scribe STT
- Verifiable claim: ElevenLabs offers Scribe speech-to-text and voice AI tooling. The source supports product capability, not a universal measured latency SLA.
- Planning assumption retained from prior draft: ~220 ms average should not be presented as validated until tested in the target architecture.
- Source: https://elevenlabs.io/docs/capabilities/speech-to-text — Accessed on 2026-05-20 — Source type: official docs — Confidence: High for capability, Low for exact latency.

## LLM inference and realtime voice evidence

### OpenAI GPT-4o / Realtime API
- Verifiable claim: GPT-4o is natively multimodal and the Realtime API is designed for low-latency speech-to-speech interactions. OpenAI positions it for realtime voice experiences, but application latency still depends on client audio capture, network, server-side turn detection, model processing, and audio playback.
- Planning assumption retained from prior draft: 200-350 ms TTFT and 320-550 ms end-to-end should be treated as target architecture assumptions until measured in a pilot.
- Sources:
  - https://openai.com/index/hello-gpt-4o/ — Accessed on 2026-05-20 — Source type: official product announcement — Confidence: High for multimodal/realtime capability.
  - https://platform.openai.com/docs/guides/realtime — Accessed on 2026-05-20 — Source type: official API docs — Confidence: High for Realtime API capability.

### xAI Grok
- Verifiable claim: xAI publishes Grok model updates and API access, but the cited public model announcement does not provide a stable realtime voice latency SLA.
- Planning assumption retained from prior draft: 180-280 ms TTFT should be treated as an unvalidated inference-provider assumption.
- Source: https://x.ai/news/grok-1.5 — Accessed on 2026-05-20 — Source type: official model announcement — Confidence: Medium for model capability, Low for exact TTFT.

### Groq-hosted Llama models
- Verifiable claim: Groq positions its LPU inference stack around high token throughput and low latency, and publishes model/API materials. Exact TTFT varies by model, prompt, queueing, and region.
- Planning assumption retained from prior draft: 120-220 ms TTFT should be validated in a pilot using the selected model and average prompt size.
- Sources:
  - https://groq.com/ — Accessed on 2026-05-20 — Source type: vendor product page — Confidence: Medium for low-latency positioning.
  - https://console.groq.com/docs/overview — Accessed on 2026-05-20 — Source type: official API docs — Confidence: High for API availability.

### Anthropic Claude
- Verifiable claim: Anthropic publishes model capabilities and API docs; the cited model announcement does not provide a realtime voice TTFT SLA.
- Planning assumption retained from prior draft: 250-400 ms for voice-tuned use should be treated as a scenario assumption, not evidence.
- Source: https://www.anthropic.com/news/claude-3-5-sonnet — Accessed on 2026-05-20 — Source type: official model announcement — Confidence: Medium for capability, Low for voice latency.

### Qwen models
- Verifiable claim: Qwen model pages document model availability and benchmark performance; they do not establish a hosted realtime voice latency SLA.
- Planning assumption retained from prior draft: 160-300 ms TTFT should be validated on the target serving stack.
- Source: https://qwenlm.github.io/blog/qwen2.5/ — Accessed on 2026-05-20 — Source type: official model blog — Confidence: Medium for model capability, Low for hosted latency.

## TTS latency evidence

### ElevenLabs Turbo / Flash
- Verifiable claim: ElevenLabs publishes low-latency TTS models and API documentation for streaming text-to-speech. Vendor latency claims should be verified with the selected voice/model and geography.
- Planning assumption retained from prior draft: 150-280 ms TTFA is plausible for low-latency TTS but must be validated in-pilot.
- Sources:
  - https://elevenlabs.io/docs/capabilities/text-to-speech — Accessed on 2026-05-20 — Source type: official API docs — Confidence: High for streaming TTS support.
  - https://elevenlabs.io/docs/models — Accessed on 2026-05-20 — Source type: official model docs — Confidence: Medium for model latency positioning.

### Cartesia Sonic
- Verifiable claim: Cartesia markets Sonic as low-latency generative voice and provides API documentation. Exact TTFA depends on streaming settings, voice, region, and network.
- Planning assumption retained from prior draft: 120-200 ms first-audio should be treated as an optimistic target pending pilot measurement.
- Source: https://docs.cartesia.ai/ — Accessed on 2026-05-20 — Source type: official API docs — Confidence: High for API support, Medium for low-latency positioning.

### PlayHT streaming TTS
- Verifiable claim: PlayHT supports streaming TTS. Public docs support integration evidence; exact latency should be benchmarked per selected voice/model.
- Planning assumption retained from prior draft: 180-350 ms first-audio should be treated as a model/network assumption.
- Source: https://docs.play.ht/reference/api-generate-tts-audio-stream — Accessed on 2026-05-20 — Source type: official API docs — Confidence: High for streaming support, Low for exact latency.

### OpenAI TTS / Realtime audio output
- Verifiable claim: OpenAI supports text-to-speech and realtime audio capabilities through official API documentation.
- Planning assumption retained from prior draft: 250-400 ms first audio should be modeled as a planning assumption unless measured in the chosen deployment path.
- Sources:
  - https://platform.openai.com/docs/guides/text-to-speech — Accessed on 2026-05-20 — Source type: official API docs — Confidence: High for TTS support.
  - https://platform.openai.com/docs/guides/realtime — Accessed on 2026-05-20 — Source type: official API docs — Confidence: High for realtime audio support.

## End-to-end voice architectures

### Native omni / speech-to-speech models
- Verifiable claim: Native realtime multimodal models reduce orchestration overhead versus cascaded STT + LLM + TTS stacks by collapsing functions into one realtime session. The official docs support capability; exact latency must be measured in pilot.
- Planning assumption: 320-550 ms end-to-end is a target range, not verified evidence. Use as a base-case modeling assumption only after pilot confirmation.
- Sources:
  - https://platform.openai.com/docs/guides/realtime — Accessed on 2026-05-20 — Source type: official API docs — Confidence: High for realtime API support.
  - https://ai.google.dev/gemini-api/docs/live — Accessed on 2026-05-20 — Source type: official API docs — Confidence: High for live/multimodal API support.

### Cascaded STT + LLM + TTS pipelines
- Verifiable claim: Cascaded stacks introduce additive latency from speech recognition, model inference, synthesis, network hops, and orchestration. Framework docs support feasible implementation; precise p50/p95 latency must be measured in the target stack.
- Planning assumption: 650-1400 ms typical end-to-end range is a useful sensitivity-model range, not a universal benchmark.
- Sources:
  - https://github.com/pipecat-ai/pipecat — Accessed on 2026-05-20 — Source type: open-source framework repository — Confidence: High for implementation feasibility.
  - https://docs.livekit.io/agents/ — Accessed on 2026-05-20 — Source type: official framework docs — Confidence: High for voice-agent framework capability.
  - https://docs.vapi.ai/ — Accessed on 2026-05-20 — Source type: official platform docs — Confidence: High for managed voice-agent platform capability.
  - https://www.retellai.com/blog/how-real-time-voice-ai-works-stt-llm-tts — Accessed on 2026-05-20 — Source type: vendor explainer — Confidence: Medium for architecture discussion.

## Summary table for modeling

| Component | Example | Evidence-backed statement | Planning latency range | Source URL | Source type | Confidence | Caveat |
|---|---|---|---:|---|---|---|---|
| STT | OpenAI speech-to-text | API supports transcription | 150-300 ms | https://platform.openai.com/docs/guides/speech-to-text | Official docs; Accessed on 2026-05-20 | High capability / Low latency | Pilot required for latency |
| STT | Google streaming STT | Streaming recognition supported | 200-400 ms | https://cloud.google.com/speech-to-text/docs/streaming-recognize | Official docs; Accessed on 2026-05-20 | High capability / Low latency | Region/model dependent |
| STT | Deepgram streaming | Streaming API supported | 100-250 ms | https://developers.deepgram.com/docs/streaming | Official docs; Accessed on 2026-05-20 | High capability / Medium latency | Endpointing affects latency |
| LLM | OpenAI GPT-4o Realtime | Realtime speech interaction supported | 320-550 ms E2E target | https://platform.openai.com/docs/guides/realtime | Official docs; Accessed on 2026-05-20 | High capability / Medium latency | Measure p50/p95 in pilot |
| LLM | Groq-hosted models | Low-latency inference positioning and API available | 120-220 ms TTFT | https://console.groq.com/docs/overview | Official docs; Accessed on 2026-05-20 | High capability / Medium latency | Queueing/prompt-size sensitive |
| TTS | ElevenLabs | Streaming TTS supported | 150-280 ms TTFA | https://elevenlabs.io/docs/capabilities/text-to-speech | Official docs; Accessed on 2026-05-20 | High capability / Medium latency | Voice/model dependent |
| TTS | Cartesia Sonic | Streaming voice API supported | 120-200 ms TTFA | https://docs.cartesia.ai/ | Official docs; Accessed on 2026-05-20 | High capability / Medium latency | Validate with selected voice |
| E2E | Native realtime/omni | Lower orchestration overhead than cascade | 320-550 ms target | https://platform.openai.com/docs/guides/realtime | Official docs; Accessed on 2026-05-20 | High capability / Medium latency | Not a guaranteed SLA |
| E2E | Cascaded voice agent | Feasible stack with additive latency | 650-1400 ms sensitivity range | https://github.com/pipecat-ai/pipecat | OSS framework; Accessed on 2026-05-20 | High feasibility / Low universal latency | Must benchmark target stack |

## Gold-standard targets for buyer-facing business case

### Customer service / contact centers
- Recommended target for natural conversation: p50 end-to-end latency below 800 ms, p95 below 1200 ms.
- Evidence status: this is a design target synthesized from voice UX expectations and vendor architecture guidance, not a regulatory requirement or vendor SLA.
- Sources:
  - https://www.coval.ai/blog/how-to-measure-voice-ai-latency-the-complete-guide — Accessed on 2026-05-20 — Source type: specialist vendor guide — Confidence: Medium.
  - https://docs.livekit.io/agents/ — Accessed on 2026-05-20 — Source type: official framework docs — Confidence: High for implementation feasibility.

### Healthcare / clinical voice assistants
- Recommended target for conversational clinical assistant UX: p50 below 500-800 ms depending on task criticality; use stricter targets for interruptible assistant workflows than for documentation/passive ambient listening.
- Evidence status: target is a business/design requirement and should be validated with clinicians in workflow testing.
- Sources:
  - https://www.ncbi.nlm.nih.gov/pmc/ — Accessed on 2026-05-20 — Source type: literature repository for healthcare AI research — Confidence: Medium for domain context, Low for specific latency target.
  - https://www.himss.org/resources — Accessed on 2026-05-20 — Source type: healthcare technology resource repository — Confidence: Medium for domain context, Low for exact latency.

## Business-case implication

Do not present precise latency ranges as externally validated unless a cited source publishes that exact metric with methodology. For ROI modeling, use three latency/adoption scenarios:
- Conservative: cascaded stack, 1000-1400 ms p50, lower containment, higher escalation.
- Base: optimized cascade or native realtime where available, 650-900 ms p50, moderate containment.
- Aggressive: native realtime architecture after pilot validation, 320-650 ms p50, higher containment.

The financial model should treat latency as a driver of containment and customer satisfaction, not as a standalone benefit claim.
