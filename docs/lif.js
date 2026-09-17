/**
 * 초파리 직원 — LIF(누출 적분 발화) 엔진
 *
 * Shiu et al. 커넥톰 기반 모델의 표준 파라미터를 따른다.
 *   안정막전위 -52mV, 발화역치 -45mV, 불응기 2.2ms
 *   막 시상수 20ms, 시냅스 시상수 5ms, 시냅스 1개당 0.275mV
 *
 * 한 개체 = Brain 인스턴스 하나. 커넥톰(읽기 전용)은 여러 개체가 공유한다.
 */
const V_REST = -52, V_TH = -45, V_RESET = -52;
const TAU_M = 20, TAU_SYN = 5, REFRAC = 2.2;

/**
 * 시냅스 1개당 전위 상승(mV).
 * Shiu et al. 원 모델은 0.275지만, 우리는 시냅스 5개 미만 연결을 쳐내
 * 전체 시냅스의 62.7%만 남겼다. 남은 시냅스가 그만큼을 대신해야 하므로
 * 0.275 / 0.627 ≈ 0.44 로 보정한다. 임의 조정이 아니라 가지치기 보정이다.
 */
const W_SYN = 0.44;

export class Brain {
  /** @param {{N:number,indptr:Uint32Array,indices:Uint32Array,weights:Int16Array}} c */
  /**
   * @param tonic 모든 뉴런에 주는 기저 전류(mV).
   *
   * 실제 뉴런은 가만히 있어도 조금씩 운다. 억제 시냅스는 그 기저 활동을 눌러서
   * 정보를 전달한다. 기저가 0이면 억제는 아무 일도 하지 못한다.
   *
   * 초파리 ON 경로가 정확히 그런 구조다 — L1 이 글루탐산성 억제로 Mi1 을 누르고,
   * Mi1 이 다시 반전시켜 ON 신호를 만든다. 기저 활동 없이는 ON 경로 전체가
   * 죽고, ON 입력을 받는 T4 도 영영 울지 않는다. 실측으로 확인했다.
   */
  constructor(c, dt = 1.0, wsyn = W_SYN, tonic = 0) {
    /**
     * 난수원. 감각 뉴런의 강제 발화는 포아송 과정이라 난수를 쓴다.
     * 즉 이 뇌는 결정적이지 않다. 측정값을 "재현된다"고 말하려면
     * 난수까지 고정할 수 있어야 하므로 갈아끼울 수 있게 둔다.
     * 화면에서는 Math.random 그대로 쓰고, 벤치에서만 시드 고정 PRNG 를 꽂는다.
     */
    this.rng = Math.random;
    this.wsyn = wsyn;
    this.tonic = tonic;
    this.c = c;
    this.dt = dt;
    const N = c.N;
    this.V = new Float32Array(N).fill(V_REST);
    this.I = new Float32Array(N);        // 시냅스 전류
    this.ext = new Float32Array(N);      // 외부 주입 전류
    this.drive = new Float32Array(N);    // 강제 발화 확률/스텝 (감각 입력)
    this.refr = new Float32Array(N);     // 남은 불응기
    this.rate = new Float32Array(N);     // 발화율 추정 (지수 이동 평균)
    this.spikes = new Uint32Array(N);    // 이번 스텝 발화 목록
    this.nSpikes = 0;
    this.steps = 0;
    this.totalSpikes = 0;

    this.decayM = Math.exp(-dt / TAU_M);
    this.decayS = Math.exp(-dt / TAU_SYN);
    this.decayR = Math.exp(-dt / 50);     // 발화율 EMA
  }

  /** 감각 뉴런에 전류를 준다 (0~1 정규화 값을 mV로) */
  inject(indices, values, gain = 12) {
    const ext = this.ext;
    for (let i = 0; i < indices.length; i++) ext[indices[i]] = values[i] * gain;
  }
  clearInject() { this.ext.fill(0); }

  /**
   * 감각 뉴런을 정해진 발화율로 강제 발화시킨다.
   * 커넥톰 모델에서 표준으로 쓰는 자극 방식이다 (Shiu et al.).
   * @param values 0~1 정규화 세기, maxHz에 곱해진다
   */
  setDrive(indices, values, maxHz = 150) {
    const d = this.drive, p = (maxHz * this.dt) / 1000;
    for (let i = 0; i < indices.length; i++) d[indices[i]] = values[i] * p;
  }
  clearDrive() { this.drive.fill(0); }

  step() {
    const { indptr, indices, weights } = this.c;
    const { V, I, ext, refr, rate } = this;
    const N = this.c.N, dt = this.dt;
    const dm = this.decayM, ds = this.decayS, dr = this.decayR;
    const sp = this.spikes, ws = this.wsyn, cut = this.cut, tn = this.tonic;
    const rnd = this.rng;
    let n = 0;

    const drv = this.drive;

    // 1) 막전위 갱신 + 발화 판정
    for (let i = 0; i < N; i++) {
      if (refr[i] > 0) { refr[i] = refr[i] > dt ? refr[i] - dt : 0; V[i] = V_RESET; continue; }
      // 감각 뉴런 강제 발화
      if (cut && cut[i]) { V[i] = V_REST; continue; }   // 잘린 뉴런은 울지 않는다
      if (drv[i] > 0 && rnd() < drv[i]) {
        V[i] = V_RESET; refr[i] = REFRAC; sp[n++] = i; continue;
      }
      const drive = I[i] + ext[i] + tn;
      // 닫힌 형태 지수 갱신 (오일러보다 안정적)
      V[i] = V_REST + (V[i] - V_REST) * dm + drive * (1 - dm);
      if (V[i] >= V_TH) {
        V[i] = V_RESET;
        refr[i] = REFRAC;
        sp[n++] = i;
      }
    }

    // 2) 시냅스 전류 감쇠
    for (let i = 0; i < N; i++) I[i] *= ds;

    // 3) 발화 전파 (희소 — 활성 뉴런만 순회)
    for (let s = 0; s < n; s++) {
      const src = sp[s];
      const a = indptr[src], b = indptr[src + 1];
      for (let k = a; k < b; k++) I[indices[k]] += weights[k] * ws;
    }

    // 4) 발화율 EMA
    for (let i = 0; i < N; i++) rate[i] *= dr;
    for (let s = 0; s < n; s++) rate[sp[s]] += (1 - dr);

    this.nSpikes = n;
    this.totalSpikes += n;
    this.steps++;
    return n;
  }

  /**
   * 뉴런을 잘라낸다(손상 실험). 해당 뉴런의 출력 연결을 끊고 발화를 막는다.
   *
   * "정말 이 커넥톰이 조종하는 게 맞냐"에 대한 답이다.
   * 조향 채널을 자르면 조향이 죽는다. 잘린 게 회복되지 않는 것도 확인할 수 있다.
   */
  lesion(indices) {
    if (!this.cut) this.cut = new Uint8Array(this.c.N);
    for (const i of indices) {
      if (this.cut[i]) continue;
      this.cut[i] = 1;
      this.nCut = (this.nCut || 0) + 1;
    }
  }
  healAll() {
    if (this.cut) this.cut.fill(0);
    this.nCut = 0;
  }

  /** 뉴런 묶음의 평균 발화율 (스텝당 발화 확률) */
  groupRate(idxArr) {
    let s = 0;
    for (let i = 0; i < idxArr.length; i++) s += this.rate[idxArr[i]];
    return idxArr.length ? s / idxArr.length : 0;
  }

  /** 뉴런 묶음의 평균 발화율을 Hz로 */
  groupHz(idxArr) { return this.groupRate(idxArr) * 1000 / this.dt; }

  /** 이번 스텝에 이 묶음에서 몇 개가 발화했나 */
  countSpikes(set) {
    let n = 0;
    for (let s = 0; s < this.nSpikes; s++) if (set.has(this.spikes[s])) n++;
    return n;
  }

  reset() {
    this.V.fill(V_REST); this.I.fill(0); this.ext.fill(0); this.drive.fill(0);
    this.refr.fill(0); this.rate.fill(0);
    this.steps = 0; this.totalSpikes = 0;
  }
}
