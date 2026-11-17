"""各段を起動する前の予算の関門(監査 v2 の指摘 1)。手元で `python gate.py <段>` を実行し、exit 0 のときだけ起動する。

判定: 使用済み + この段の最大額 + まだ一度も起動していない段の最大額 ≤ 上限(30 ドル)
- 使用済み = 今の metered_cost + 最後の「精算」以降に起動した段(やり直しを含む)の最大額の合計。
  請求の反映は遅れるので、精算までは起動した段を最大額で数える。反映済みの分と二重に数えることはあっても(安全側)、
  取りこぼすことはない。
- 精算は時間でも割合でも行わない。段が終わったら `python gate.py done <段> <app_id>` で Modal の app id を記録し、
  `python gate.py settle` が請求レポート(`modal billing report --json`)でその app に載った額 c を読む。
  その段の予約は max(0, 最大額 − c) に減る(c は metered に含まれているので、合計は最大額のまま。未反映分は予約に残る)。
- 自動の再実行: GPU の段は run ディレクトリの started.json、CPU の段と J2 の子ジョブは /vol/guards/*.started で
  二度目を作業前に止める(再実行で消費するのはコンテナの起動分だけ)。
- CPU・メモリは要求量 = 上限に固定してある(cpu=(n, n)、memory=(m, m))。課金は要求量と使用量の大きいほうなので、
  上限を固定して初めて最大額が決まる。丸めはしない。
- 最大額 = timeout いっぱいまで走った場合の GPU + CPU + メモリの料金(2026-11-17 の `modal billing rates`)。
- 通過すると ledger.jsonl に「起動した」と書く。同じ段を二度通すには --retry を付ける。
user 側でも Modal の支出上限(spend limit)を 0 ドルに設定済み(無料クレジットを超えると止まる)。
"""
import json
import math
import subprocess
import sys
import time
from pathlib import Path

LIMIT = 30.0
H100, CPU, MEM = 3.95, 0.0473, 0.008          # $/h(GPU 1 枚、CPU 1 コア、メモリ 1 GiB)


def cost(minutes, gpus=0, cpu=0, mem_gib=0):
    return minutes / 60 * (gpus * H100 + cpu * CPU + mem_gib * MEM)


# 段 → 最大額(関数の timeout と資源の指定から)
STAGES = {
    "S0": cost(60, cpu=2, mem_gib=4) + cost(15, cpu=2, mem_gib=8),           # prepare_data + val_target_bytes en
    "S1": cost(15, 1, 8, 32),                                                # smoke
    "S2": cost(16, 8, 32, 96),                                               # train_en
    "S4": cost(10, 1, 8, 32),                                                # sample_en
    "J0": cost(40, cpu=2, mem_gib=4),                                        # download
    "J1": cost(50, cpu=8, mem_gib=64),                                       # train_tokenizer(1 回目は 32GiB で OOM → 64GiB)
    "J2": cost(40, cpu=2, mem_gib=4) + 41 * cost(25, cpu=2, mem_gib=4) + cost(15, cpu=2, mem_gib=8),  # tokenize(学習の子 40 本 + 検証 1 本、ja_data.MAX_JOBS)+ val bytes ja
    "J2s": cost(15, 1, 8, 32),                                               # smoke_ja
    "J3": cost(16, 8, 32, 96),                                               # train_ja
    "J4": cost(10, 1, 8, 32),                                                # sample_ja
}
ORDER = list(STAGES)
LEDGER = Path(__file__).parent / "ledger.jsonl"


def metered() -> float:
    out = subprocess.run(["modal", "billing", "summary", "--json"], capture_output=True, text=True, check=True).stdout
    m = float(json.loads(out)["metered_cost"])
    if not math.isfinite(m) or m < 0:
        raise RuntimeError(f"metered_cost が不正: {m!r}")
    return m


def _entries():
    return [json.loads(l) for l in LEDGER.read_text().splitlines()] if LEDGER.exists() else []


def report_rows():
    out = subprocess.run(["modal", "billing", "report", "--for", "this month", "--json"], capture_output=True,
                         text=True, check=True).stdout
    return json.loads(out)


def app_cost(rows, app_id: str) -> float:
    """請求レポートの行のうち、どこかの値に app_id を含む行の費用を足す(行の形が想定と違えば例外)。"""
    total, hit = 0.0, False
    for r in rows:
        if not isinstance(r, dict):
            raise RuntimeError(f"請求レポートの形が想定と違う: {r!r}"[:200])
        if any(isinstance(v, str) and app_id in v for v in r.values()):
            cost = r.get("cost")
            if cost is None:
                raise RuntimeError(f"請求レポートに cost が無い: {sorted(r)}")
            c = float(cost)
            if not math.isfinite(c) or c < 0:
                raise RuntimeError(f"請求額が不正: {cost!r}")
            total += c
            hit = True
    return total if hit else 0.0


def reflected(es) -> dict:
    """entry 番号 → 請求に載った額(最新の settle の値)"""
    r = {}
    for e in es:
        if e.get("reflect"):
            c = float(e["cost"])
            if not math.isfinite(c) or c < 0:
                raise RuntimeError(f"台帳の請求額が不正: {e}")
            r[e["entry"]] = c
    return r


def settle(rows=None):
    es = _entries()
    dones = {e["entry"]: e for e in es if e.get("done")}
    rows = report_rows() if rows is None else rows
    costs = {i: app_cost(rows, d["app_id"]) for i, d in dones.items()}   # 全部読めてから書く(途中で止まったら何も書かない)
    with LEDGER.open("a") as f:
        for i, c in costs.items():
            f.write(json.dumps({"reflect": True, "entry": i, "cost": c, "at": time.time()}) + "\n")
            print(f"{es[i]['stage']} app={dones[i]['app_id']} 請求に載った額 {c:.4f} → 予約 {max(0.0, es[i]['max_cost'] - c):.4f}")


def done(stage: str, app_id: str):
    es = _entries()
    idx = [i for i, e in enumerate(es) if e.get("stage") == stage]
    assert idx, f"{stage} は起動していない"
    with LEDGER.open("a") as f:
        f.write(json.dumps({"done": True, "entry": idx[-1], "app_id": app_id, "at": time.time()}) + "\n")


def main():
    if sys.argv[1] == "settle":
        return settle()
    if sys.argv[1] == "done":
        return done(sys.argv[2], sys.argv[3])
    stage = sys.argv[1]
    retry = "--retry" in sys.argv
    assert stage in STAGES, f"段は {ORDER}"
    es = _entries()
    started = [e for e in es if "stage" in e]
    done_stages = {e["stage"] for e in started}
    if stage in done_stages and not retry:
        sys.exit(f"REFUSED: {stage} は起動済み(やり直すなら --retry)")
    refl = reflected(es)
    unsettled = sum(max(0.0, e["max_cost"] - refl.get(i, 0.0)) for i, e in enumerate(es) if "stage" in e)
    m = metered()
    spent = m + unsettled
    later = [s for s in ORDER if s not in done_stages and s != stage]
    reserve = sum(STAGES[s] for s in later)
    total = spent + STAGES[stage] + reserve
    print(f"metered={m:.4f} 未精算の予約={unsettled:.4f} → 使用済み {spent:.4f} + {stage} 最大 {STAGES[stage]:.4f}"
          f" + 未起動の段 {reserve:.4f} ({','.join(later)}) = {total:.4f} / 上限 {LIMIT:.2f}")
    if total > LIMIT:
        sys.exit("REFUSED: 予算の上限を超えうる")
    with LEDGER.open("a") as f:
        f.write(json.dumps({"stage": stage, "max_cost": STAGES[stage], "metered_at_start": m, "at": time.time(),
                            "retry": retry}) + "\n")
    print("OK")


if __name__ == "__main__":
    main()
