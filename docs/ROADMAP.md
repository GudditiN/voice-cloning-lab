# Roadmap: from cloned voice to a real-time avatar

Phase 1 (this repo): record, fine-tune and synthesize your voice (XTTS-v2).

## Phase 2: conversation loop
Mic -> STT (faster-whisper) -> LLM (Claude API or a local model) -> TTS (`TTSEngine.stream`) -> speaker.
Target under 1s to first audio: stream LLM tokens, synthesize per sentence, start playback on the first chunk.
Add an XTTS streaming path to `XTTSEngine.stream` (currently yields the whole utterance).

## Phase 3: avatar
Audio-driven talking head, fed by the TTS audio:
- Realistic: MuseTalk, LivePortrait or SadTalker (open source, need a GPU, ~real-time on good hardware).
- Low latency / cheap: rigged 2D/3D avatar with viseme lip-sync in the browser.
Start with a single portrait photo of the person; get their consent for likeness as well as voice.

## Phase 4: real-time delivery to customers
WebRTC stack (LiveKit or Pipecat) wiring STT, LLM, TTS and avatar video, with interruption (barge-in) handling.
Needs an always-on cloud GPU; Colab and Codespaces are for development only.

## Before customer use
- XTTS-v2 weights are CPML (non-commercial). For commercial use, license from Coqui's successors or swap `TTSEngine` for a permissively licensed model (candidates: F5-TTS, Piper, or a commercial API).
- Disclose to customers that the voice/avatar is AI-generated, and keep consent records.
