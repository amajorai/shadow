# Shadow

> Your computer was paying attention the whole time — screen/audio/input capture, OCR, and
> semantic search, all on-device.

[![License](https://shieldcn.dev/badge/License-Apache--2.0-73DC8C.svg?logo=apache&logoColor=white)](./LICENSE)
[![Discord](https://shieldcn.dev/discord/1439211418724597800.svg?logo=discord&logoColor=white&color=4B78E6)](https://ryuhq.com/discord)

Shadow is a Rust capture-and-intelligence sidecar (~23k LOC): screen/audio/input capture, OCR, a
proactive engine, and semantic memory + search. It records pure-Rust JPEG keyframes (no ffmpeg
required). A component of [Ryu](https://github.com/amajorai/ryu), it runs standalone on `:3030`.

## Layout

| Path | What |
|---|---|
| `apps/shadow` | the capture sidecar (HTTP server, `:3030`) |
| `crates/shadow-core` | capture + semantic-search engine |
| `crates/ghost-{core,eyes,hands}` | shared automation crates (vendored from [Ghost](https://github.com/amajorai/ghost)) |

## Build

```bash
cd apps/shadow && cargo build --release
```

## Dual-use & consent

Continuous screen + audio capture is inherently sensitive. Shadow is a device-bound sensor: it runs
locally, never routes capture off-device, and gates every source behind explicit consent and a pause
control. It's open-source so this is fully auditable. See [SECURITY](./apps/shadow/SECURITY.md).

## Credits & license

Shadow is derived from [Shadow](https://github.com/ghostwright/shadow) by Ghostwright (MIT) — the
original copyright + license notice are retained in [NOTICE](./NOTICE). Licensed under **Apache-2.0**
(see [LICENSE](./LICENSE)) with MIT-licensed portions per NOTICE. © 2026 A Major Pte. Ltd.
