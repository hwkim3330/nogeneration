/**
 * 초파리 직원 — 수습 교육 (캘리브레이션)
 *
 * 개체마다 어떤 하행뉴런이 "왼쪽 시야"에 반응하고 어떤 뉴런이 "오른쪽 시야"에
 * 반응하는지는 미리 알 수 없다. DNa01/DNa02 같은 이름 붙은 조향 뉴런은
 * 좌우 각 1개뿐이라 표본이 너무 작아 신호가 안 잡힌다.
 *
 * 그래서 고용 직후 좌/우 눈에 번갈아 빛을 비춰 차등 반응하는 하행뉴런을
 * 골라낸다. 이것이 그 개체의 조향 채널이 된다. 집단 부호화다.
 */

/**
 * @param {import('./lif.js').Brain} brain
 * @param {{left:{ci:number[]}, right:{ci:number[]}}} eye  좌/우 눈 (meta.eye 형태)
 * @param {number[]} dn  하행뉴런 인덱스
 * @param {{steps?:number, hz?:number, top?:number, minDiff?:number, onProgress?:Function}} opt
 */
export async function calibrate(brain, eye, dn, opt = {}) {
  const steps = opt.steps ?? 1200;
  const hz = opt.hz ?? 150;
  const top = opt.top ?? 20;
  const minDiff = opt.minDiff ?? 3;      // Hz
  const prog = opt.onProgress;

  // meta.eye 는 {left:{ci,u,v}, right:{...}} 형태다. 배열로 착각하면 조용히 빈 배열이 되어
  // 자극이 하나도 안 들어가고 조향 채널이 0개가 된다.
  const pick = (o) => Uint32Array.from(Array.isArray(o) ? o : (o?.ci ?? []));
  const L = pick(eye.left);
  const R = pick(eye.right);
  if (!L.length || !R.length) throw new Error("눈 뉴런 목록이 비어 있습니다");
  const onesL = new Float32Array(L.length).fill(1);
  const onesR = new Float32Array(R.length).fill(1);

  const run = async (which) => {
    brain.reset();
    if (which === "L") brain.setDrive(L, onesL, hz);
    else brain.setDrive(R, onesR, hz);
    for (let t = 0; t < steps; t++) {
      brain.step();
      if (prog && (t & 255) === 255) {
        prog(which, t / steps);
        await Promise.resolve();          // 워커가 멈춰 보이지 않게 양보
      }
    }
    return dn.map((i) => brain.rate[i] * 1000 / brain.dt);   // Hz
  };

  const onlyL = await run("L");
  const onlyR = await run("R");

  // 개체 고유 편향: 좌우를 똑같이 비춰도 한쪽으로 치우친다.
  // 이 값을 빼주지 않으면 편향이 좌우 신호를 덮어버려 게임에서 한쪽으로만 돈다.
  brain.reset();
  brain.setDrive(L, onesL, hz);
  brain.setDrive(R, onesR, hz);
  for (let t = 0; t < steps; t++) brain.step();
  const balanced = dn.map((i) => brain.rate[i] * 1000 / brain.dt);

  const scored = dn.map((ci, k) => ({ ci, diff: onlyL[k] - onlyR[k] }));
  scored.sort((a, b) => b.diff - a.diff);

  const leftCh = scored.filter((s) => s.diff > minDiff).slice(0, top).map((s) => s.ci);
  const rightCh = scored.filter((s) => s.diff < -minDiff).slice(-top).map((s) => s.ci);

  // 활동적인 하행뉴런 (버튼 채널 후보)
  const active = dn.filter((ci, k) => Math.max(onlyL[k], onlyR[k]) > 1);

  // 균형 조명에서의 좌우 채널 세기 → 편향
  const idxOf = new Map(dn.map((ci, k) => [ci, k]));
  const chMean = (arr) => {
    let s = 0;
    for (const ci of arr) s += balanced[idxOf.get(ci)] || 0;
    return arr.length ? s / arr.length : 0;
  };
  // 각 채널이 "균형 조명일 때" 내는 값. 이걸 기준으로 나눠 쓰면
  // 개체마다 다른 채널 이득 차이가 자동으로 상쇄된다.
  const baseL = Math.max(1e-3, chMean(leftCh));
  const baseR = Math.max(1e-3, chMean(rightCh));
  const baseActive = Math.max(1e-3, chMean(active));

  brain.reset();
  return {
    leftCh, rightCh, active, baseL, baseR, baseActive,
    quality: leftCh.length >= 5 && rightCh.length >= 5 ? "good" : "weak",
    span: scored.length ? Math.round(scored[0].diff - scored[scored.length - 1].diff) : 0,
  };
}

/**
 * 조향을 -1(좌) ~ +1(우)로 돌려준다.
 *
 * 각 채널을 자기 자신의 균형 조명 반응(baseL/baseR)으로 나눈 뒤 비교한다.
 * 그래야 "왼쪽 채널이 원래 더 세게 운다" 같은 개체차가 상쇄되고,
 * 전체 밝기가 오르내려도 좌우 비율만 남는다.
 */
export function steering(brain, cal, gain = 2.0, adapt = 1 / 40) {
  const l = brain.groupHz(cal.leftCh) / cal.baseL;
  const r = brain.groupHz(cal.rightCh) / cal.baseR;
  const s = l + r;
  if (s < 0.05) return 0;                      // 활동이 너무 적으면 방향을 말하지 않는다
  const raw = (r - l) / s;

  /*
   * 적응형 기준선.
   *
   * 개체마다 좌우 채널의 기저 활동이 다르다. 수습 교육에서 잰 값으로 나눠도
   * 실제 게임의 자극 세기가 교육 때와 달라 편향이 남는다. 실측에서 조향이
   * 항상 +0.5 근처에 붙어 초파리가 제자리를 빙빙 돌았다.
   *
   * 그래서 자기 자신의 평균을 빼고 "평소와 얼마나 다른가"만 본다.
   * 감각 적응과 같은 원리다. 일정한 치우침은 스스로 사라진다.
   *
   * 적응 시간과 이득은 탐색 범위를 기준으로 실측해 정했다.
   * 1500프레임 4시드 평균 방문 칸 수:
   *   gain 4.0 / adapt 1/150  →  16.5칸,  회전 3.7바퀴  (제자리에서 돌았다)
   *   gain 2.0 / adapt 1/40   →  23.8칸,  회전 1.3바퀴
   * QA 퍼저에게 중요한 것은 목표 도달이 아니라 얼마나 넓게 두들기느냐다.
   */
  if (cal._ema === undefined) cal._ema = raw;
  cal._ema += (raw - cal._ema) * adapt;
  const dev = raw - cal._ema;

  /*
   * 짧은 평활. 한 샘플의 잡음 표준편차가 0.08이라 그대로 쓰면 방향이 떨린다.
   * 10샘플쯤 평균 내면 0.025 수준으로 내려가 신호가 드러난다.
   * 초파리의 조향도 개별 스파이크가 아니라 집단의 시간 평균으로 결정된다.
   */
  if (cal._sm === undefined) cal._sm = 0;
  cal._sm += (dev - cal._sm) * 0.12;

  return Math.max(-1, Math.min(1, cal._sm * gain));
}

/** 전진 강도 0~1 — 활성 하행뉴런이 평소 대비 얼마나 우는가 */
export function thrust(brain, cal) {
  const a = brain.groupHz(cal.active) / cal.baseActive;
  return Math.max(0.25, Math.min(1, a * 0.8));
}
