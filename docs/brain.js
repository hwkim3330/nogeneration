/**
 * 초파리 직원 — 커넥톰 디코더
 *
 * brain.bin 을 CSR 배열로 푼다. 브라우저와 Node 양쪽에서 쓴다.
 *   indptr  Uint32Array(N+1)   뉴런 i의 출력 연결은 [indptr[i], indptr[i+1])
 *   indices Uint32Array(M)     대상 뉴런 (파일에는 행 안 델타+varint로 저장)
 *   weights Int16Array(M)      부호 있는 가중치 (음수 = 억제)
 */
export function decodeBrain(buf) {
  const u8 = new Uint8Array(buf);
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  if (String.fromCharCode(u8[0], u8[1], u8[2], u8[3]) !== "FLY1")
    throw new Error("커넥톰 파일 형식이 아닙니다");

  const N = dv.getUint32(4, true);
  const M = dv.getUint32(8, true);
  const varLen = dv.getUint32(12, true);

  let off = 16;
  const indptr = new Uint32Array(N + 1);
  for (let i = 0; i <= N; i++) indptr[i] = dv.getUint32(off + i * 4, true);
  off += (N + 1) * 4;

  // 행 안 델타 + LEB128 → 절대 인덱스
  const indices = new Uint32Array(M);
  let p = off, last = 0, row = 0, k = 0;
  for (let i = 0; i < N; i++) {
    const end = indptr[i + 1];
    last = 0;
    for (; k < end; k++) {
      let shift = 0, d = 0, b;
      do { b = u8[p++]; d |= (b & 0x7f) << shift; shift += 7; } while (b & 0x80);
      last += d;
      indices[k] = last;
    }
  }
  off += varLen;

  const weights = new Int16Array(M);
  for (let i = 0; i < M; i++) weights[i] = dv.getInt16(off + i * 2, true);

  return { N, M, indptr, indices, weights };
}
