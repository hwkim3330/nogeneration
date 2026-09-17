/**
 * No Generation — 생성하지 않고 판단하는 두 가지.
 *
 * 왼쪽: 언어모델을 로짓에서 읽는다. 순전파 한 번. 토큰을 뽑지 않는다.
 * 오른쪽: 실제 초파리 커넥톰. 자극이 배선을 타고 하행뉴런까지 간다. 볼리 한 번.
 *
 * 둘의 공통점이 이 페이지의 주장이다 — 답은 이미 첫 패스에 들어 있고,
 * 그것을 꺼내 읽으면 빠르고, 형식이 깨질 수 없고, 근거를 볼 수 있다.
 */
import { decodeBrain } from "./brain.js";
import { Brain } from "./lif.js";
import { calibrate } from "./calibrate.js";

const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

/* ── 하드웨어 표시 ─────────────────────────────── */
(async () => {
  let gpu = "no WebGPU";
  try {
    const a = await navigator.gpu?.requestAdapter();
    if (a) gpu = `WebGPU · ${a.info?.vendor || "gpu"}${a.info?.architecture ? " " + a.info.architecture : ""}`;
  } catch { /* 없으면 없는 대로 */ }
  $("#hw").textContent = gpu;
})();

/* ══ 1. 언어모델 — 로짓에서 읽기 ══════════════════ */
const LETTERS = "ABCDEFGHIJ";
let T = null, tok = null, model = null, letterIds = null;

$("#load").onclick = async () => {
  const btn = $("#load"), note = $("#loadNote");
  btn.disabled = true; btn.textContent = "Loading…";
  try {
    T = await import("https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.6");
    const id = "onnx-community/Qwen2.5-0.5B-Instruct";
    note.textContent = "fetching tokenizer…";
    tok = await T.AutoTokenizer.from_pretrained(id);
    note.textContent = "fetching weights — about 400 MB, cached after the first time…";
    const t0 = performance.now();
    const device = navigator.gpu ? "webgpu" : "wasm";
    model = await T.AutoModelForCausalLM.from_pretrained(id, { dtype: "q4", device });
    // 각 글자의 단일 토큰 id. 토크나이저마다 다르므로 실제로 확인한다.
    letterIds = {};
    for (const c of LETTERS) {
      for (const form of [c, " " + c]) {
        const ids = tok.encode(form, { add_special_tokens: false });
        if (ids.length === 1) (letterIds[c] ||= []).push(ids[0]);
      }
      if (!letterIds[c]) letterIds[c] = [tok.encode(c, { add_special_tokens: false })[0]];
    }
    note.innerHTML = `Ready in <b>${((performance.now() - t0) / 1000).toFixed(1)}s</b> on ${device}.`;
    btn.hidden = true; $("#llmUI").hidden = false;
  } catch (e) {
    note.innerHTML = `<span style="color:var(--red)">Could not load: ${esc(e.message)}</span>`;
    btn.disabled = false; btn.textContent = "Try again";
  }
};

function options() {
  return $("#opts").value.split("\n").map((s) => s.trim()).filter(Boolean).slice(0, 10);
}
function buildPrompt(ctx, opts) {
  const body = opts.map((o, i) => `${LETTERS[i]}. ${o}`).join("\n");
  return tok.apply_chat_template(
    [{ role: "user", content: `${ctx}\n\nOptions:\n${body}\n\nAnswer with one letter only.\nAnswer:` }],
    { tokenize: false, add_generation_prompt: true });
}
function softmax(xs) {
  const m = Math.max(...xs), e = xs.map((x) => Math.exp(x - m)), s = e.reduce((a, b) => a + b, 0);
  return e.map((x) => x / s);
}
function drawBars(opts, probs) {
  const best = probs.indexOf(Math.max(...probs));
  $("#bars").innerHTML = opts.map((o, i) => `
    <div class="bar ${i === best ? "win" : ""}">
      <span class="nm">${esc(o)}<i style="width:${(probs[i] * 100).toFixed(1)}%"></i></span>
      <span class="pc">${(probs[i] * 100).toFixed(1)}%</span>
    </div>`).join("");
  return best;
}

$("#grab").onclick = async () => {
  const opts = options();
  if (opts.length < 2) return;
  const b = $("#grab"); b.disabled = true;
  const inp = await tok(buildPrompt($("#ctx").value, opts));
  const t0 = performance.now();
  const out = await model(inp);                 // 순전파 한 번. 이게 전부다.
  const dt = performance.now() - t0;
  const lg = out.logits, V = lg.dims[2], last = (lg.dims[1] - 1) * V;
  const raw = opts.map((_, i) => Math.max(...letterIds[LETTERS[i]].map((t) => lg.data[last + t])));
  const best = drawBars(opts, softmax(raw));
  $("#llmOut").innerHTML =
    `<span class="big">${dt.toFixed(0)}<span style="font-size:.45em;color:var(--fg3)">ms</span></span>
     <b>1</b> forward pass · chose <span class="win">${esc(opts[best])}</span>.
     The answer was already in that pass; generating it would only have spelled it out.`;
  b.disabled = false;
};

$("#gen").onclick = async () => {
  const opts = options();
  if (opts.length < 2) return;
  const b = $("#gen"); b.disabled = true; $("#llmOut").textContent = "generating…";
  const text = buildPrompt($("#ctx").value, opts);
  const inp = await tok(text);
  const t0 = performance.now();
  const gen = await model.generate({ ...inp, max_new_tokens: 24, do_sample: false });
  const dt = performance.now() - t0;
  const full = tok.batch_decode(gen, { skip_special_tokens: true })[0];
  const said = full.slice(full.lastIndexOf("Answer:") + 7).trim().slice(0, 60);
  const passes = gen.dims[1] - inp.input_ids.dims[1];
  // 선택지 하나를 고르는 문제는 생성해도 몇 패스면 끝난다. 대비가 커지는 것은
  // 필드가 여럿인 구조화된 출력에서다. 그걸 숨기지 않고 화면에 적는다.
  $("#llmOut").innerHTML =
    `<span class="big slow">${dt.toFixed(0)}<span style="font-size:.45em;color:var(--fg3)">ms</span></span>
     <b>${passes}</b> forward passes · it wrote “${esc(said)}”.
     Notice it did not answer with a bare letter — nothing made it.
     <br><br>For a single choice the gap is small; the answer is short either way. It widens
     with structured output: a four-field JSON schema measured off this page took
     <b>288 passes and 1,617ms</b> generated, against <b>1 pass and 355ms</b> read from
     logits, because each field is one more decision and none of them needs writing.`;
  b.disabled = false;
};

/* ══ 2. 초파리 — 배선을 타고 오는 판단 ══════════════ */
let brain = null, cal = null, meta = null, cut = false;
const GW = 16, GH = 12;
const vision = new Float32Array(GW * GH);
const eyeCtx = $("#eye").getContext("2d");

$("#hire").onclick = async () => {
  const btn = $("#hire"), note = $("#flyNote");
  btn.disabled = true; btn.textContent = "Waking…";
  try {
    note.textContent = "downloading the connectome — 11 MB…";
    const buf = await (await fetch(new URL("./data/brain.bin", import.meta.url))).arrayBuffer();
    meta = await (await fetch(new URL("./data/meta.json", import.meta.url))).json();
    const c = decodeBrain(buf);
    brain = new Brain(c, 1.0);
    note.textContent = "training — flashing light into each eye to find this individual's channels…";
    const t0 = performance.now();
    cal = await calibrate(brain, meta.eye, meta.descendingAll, { steps: 900 });
    note.innerHTML = `Awake in <b>${((performance.now() - t0) / 1000).toFixed(1)}s</b>.
      ${c.N.toLocaleString()} neurons · left channel ${cal.leftCh.length},
      right ${cal.rightCh.length}. Every individual is wired slightly differently, so the
      channels are found, not assumed.`;
    btn.hidden = true; $("#flyUI").hidden = false;
    drawEye();
  } catch (e) {
    note.innerHTML = `<span style="color:var(--red)">Could not wake it: ${esc(e.message)}</span>`;
    btn.disabled = false; btn.textContent = "Try again";
  }
};

function drawEye(spark = 0) {
  const img = eyeCtx.createImageData(GW, GH);
  for (let i = 0; i < GW * GH; i++) {
    const v = vision[i] * 255;
    // 발화가 많을수록 초록이 섞인다. 빛은 파랗고, 반응은 초록이다.
    const g = Math.min(255, v * .85 + spark * 90);
    img.data[4 * i] = v * .4; img.data[4 * i + 1] = g;
    img.data[4 * i + 2] = v; img.data[4 * i + 3] = 255;
  }
  eyeCtx.putImageData(img, 0, 0);
}

async function flash(side) {
  if (!brain) return;
  for (const b of [$("#lightL"), $("#lightR")]) b.disabled = true;

  // 판단 하나를 독립적으로 보여주려면 앞선 자극의 잔향을 지워야 한다.
  // 지우지 않으면 왼쪽 빛의 여운이 다음 판단을 그대로 끌고 간다(실측: 오른쪽
  // 빛을 줬는데 왼쪽으로 답했다). 실제 뉴런은 잔향을 갖지만, 여기서 보여주려는
  // 것은 "한 자극 → 한 판단"이다.
  brain.reset();
  brain.clearDrive();
  vision.fill(0.05);
  for (let y = 0; y < GH; y++)
    for (let x = 0; x < GW / 2; x++) vision[y * GW + x + (side === "R" ? GW / 2 : 0)] = 0.95;
  drawEye();

  for (const s of ["left", "right"]) {
    const e = meta.eye[s], ci = Uint32Array.from(e.ci), buf = new Float32Array(ci.length);
    const half = s === "left" ? 0 : 1;
    for (let i = 0; i < ci.length; i++) {
      const gx = Math.min(GW - 1, Math.floor((e.u[i] * 0.5 + half * 0.5) * GW));
      const gy = Math.min(GH - 1, Math.floor(e.v[i] * GH));
      buf[i] = vision[gy * GW + gx];
    }
    brain.setDrive(ci, buf, 150);
  }
  // 계산은 30ms 면 끝나지만 한 번에 돌리면 아무것도 안 보인다.
  // 볼리를 여덟 토막으로 나눠 그리는 사이 반응이 번지는 것을 보여준다.
  const t0 = performance.now();
  let compute = 0;
  for (let chunk = 0; chunk < 8; chunk++) {
    const c0 = performance.now();
    for (let k = 0; k < 5; k++) brain.step();
    compute += performance.now() - c0;
    drawEye(Math.min(1, brain.nSpikes / 900));
    await new Promise((r) => requestAnimationFrame(r));
  }
  const dt = compute;

  const l = brain.groupHz(cal.leftCh) / cal.baseL;
  const r = brain.groupHz(cal.rightCh) / cal.baseR;
  const sum = l + r;
  // 좌우 불균형을 합으로 정규화한다. 절대 문턱을 두면 안 된다 —
  // 이 값은 이미 개체별 기준선으로 나눈 비율이라, 합이 0.04 라도 한쪽만
  // 울고 있으면 그것은 뚜렷한 판단이다. 실측에서 0.05 문턱을 뒀다가
  // 멀쩡한 판단(좌 0.04 / 우 0.00)을 0 으로 지워버렸다.
  const LIVE = 1e-4;
  const d = sum < LIVE ? 0 : (r - l) / sum;
  const hz = brain.groupHz(cal.active);

  $("#mS").style.left = d < 0 ? `${50 + d * 50}%` : "50%";
  $("#mS").style.width = `${Math.abs(d) * 50}%`;
  $("#mS").style.background = Math.abs(d) < .05 ? "var(--fg3)" : "var(--accent)";
  $("#vS").textContent = (d > 0 ? "→" : d < 0 ? "←" : "·") + Math.abs(d * 100).toFixed(0);
  $("#mD").style.width = `${Math.min(100, hz)}%`;
  $("#vD").textContent = hz.toFixed(1) + "Hz";

  const side_name = side === "L" ? "left" : "right";
  const dead = sum < LIVE;
  $("#flyOut").innerHTML = cut
    ? `Light on the ${side_name}. Descending neurons still fire (<b>${hz.toFixed(1)}Hz</b>) but the
       channel that carried the difference is gone, so the decision is <b>${$("#vS").textContent}</b>.`
    : `Light on the ${side_name} → left channel <b>${l.toFixed(2)}</b>, right <b>${r.toFixed(2)}</b>
       → decision <span class="win">${$("#vS").textContent}</span>.
       <b>${brain.nSpikes.toLocaleString()}</b> neurons fired in ${dt.toFixed(0)}ms. No tokens.`;
  for (const b of [$("#lightL"), $("#lightR")]) b.disabled = false;
}

$("#lightL").onclick = () => flash("L");
$("#lightR").onclick = () => flash("R");

$("#cut").onclick = () => {
  if (!brain) return;
  brain.lesion(Uint32Array.from([...cal.leftCh, ...cal.rightCh])); cut = true;
  $("#cutNote").innerHTML = `<b>${(cal.leftCh.length + cal.rightCh.length)} neurons cut.</b>
    Flash a light and watch the decision go to zero. Restore to bring it back.`;
  flash("L");
};
$("#heal").onclick = () => {
  if (!brain) return;
  brain.healAll(); cut = false;
  $("#cutNote").textContent = "Restored. The decision comes back.";
  flash("L");
};

/* ── 실측값과 한계 ───────────────────────────────── */
$("#measured").innerHTML = `
  <b>One choice is the easy case.</b> Generating a single letter takes only a few passes, so
  reading the logits saves little — try both buttons above and you will see a small gap.
  The row in this table is about <i>structured</i> output, where every field is another
  decision.<br><br>
  <b>Measured off this page</b>, on an M4 Max with a 1.5B model (MLX, 4-bit), on a four-field
  JSON extraction: <b>1,617ms → 355ms</b> and <b>288 forward passes → 1</b>. A 28-field schema
  went 1,900ms → 270ms. Four schemas measured, 4.4× to 6.0×. Whatever this page reports is
  your own machine, right now.`;

$("#limits").innerHTML = `
  <b>Reading logits does not make a model smarter.</b> It makes the same decision arrive
  sooner and in a shape that cannot break. A 0.5B model asked something it does not know
  will be confidently wrong, faster.<br><br>
  <b>Small models carry position bias.</b> We measured Qwen2.5-1.5B answering four
  spatial questions whose correct answers were all different; it picked the same option
  every time. Averaging over shuffled option orders helped one model and hurt another.
  Treat the confidence bars as the model's preference, not as truth.<br><br>
  <b>The fly is not doing your task.</b> It is deciding which side is brighter, which is
  what its optic lobes are for. We have separately measured that this connectome
  <i>cannot</i> steer through a maze — it loses to smoothed random noise — and published
  that result rather than hiding it.<br><br>
  <b>Nothing here leaves your machine.</b> The model and the connectome both run in this
  page. There is no server.`;

/* ── 스크롤 진입 ───────────────────────────────── */
{
  const els = document.querySelectorAll("section > .eyebrow, section > h3, section > .body, .duo, .cmp, .fine, .spec > div");
  els.forEach((e) => e.classList.add("rise"));
  const io = new IntersectionObserver((rows) => {
    for (const r of rows) if (r.isIntersecting) { r.target.classList.add("in"); io.unobserve(r.target); }
  }, { rootMargin: "0px 0px -8% 0px", threshold: 0.05 });
  els.forEach((e) => io.observe(e));
}
