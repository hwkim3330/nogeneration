---
title: No Generation
emoji: ⚖️
colorFrom: gray
colorTo: green
sdk: static
app_file: index.html
pinned: false
license: mit
---

# No Generation

**Two systems that decide without producing a single token.** Both run in your browser.
Nothing is sent anywhere.

- **A language model read at its logits.** The answer to a multiple-choice question is
  already present in the first forward pass. Read it there instead of spelling it out:
  the output cannot be malformed, and you get calibrated confidence for free.
- **A real fruit-fly connectome.** FlyWire v783, 138,639 neurons, simulated as a spiking
  network in the page. Light hits one eye, current spreads through actual wiring,
  descending neurons fire harder on one side. One volley. No tokens.

## Why put them together

They are the same shape. Neither generates. Both answer in one pass, and in both the
answer is a comparison rather than a composition. That is what makes the output format
unbreakable — there is no string being assembled that could come out wrong.

The fly adds something the model cannot: **you can cut it.** Severing the 40 neurons that
carry the left/right difference — 0.03% of the brain — takes the decision to exactly zero,
and restoring them brings it back. An implementation that scripted its output would sail
through that test.

## Honest limits

- Reading logits does not make a model smarter. A 0.5B model asked something it does not
  know is confidently wrong, faster.
- For a *single* choice the speed gap is small — sometimes generating is faster, and the
  page says so when it happens. The gap widens with structured output, where every field
  is another decision.
- Small models carry position bias. We measured Qwen2.5-1.5B answering four spatial
  questions with four different correct answers; it picked the same option every time.
- The fly is deciding which side is brighter, which is what optic lobes are for. We have
  separately measured that this connectome cannot steer through a maze — it loses to
  smoothed random noise — and published that result rather than hiding it.

## Measured elsewhere

On an M4 Max with a 1.5B model (MLX, 4-bit), four-field JSON extraction:
**1,617ms → 355ms**, **288 forward passes → 1**. A 28-field schema: 1,900ms → 270ms.
Four schemas, 4.4× to 6.0×.

## Sources

Connectome: FlyWire v783 (CC-BY 4.0) · Model: `onnx-community/Qwen2.5-0.5B-Instruct`
LIF parameters: Shiu et al. connectome-based model · Code: MIT
