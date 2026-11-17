"""GPU なしで走らせる検査(監査 v3 の正常・異常ケース)。python tests/test_cpu_checks.py"""
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import modal_app as ma  # noqa: E402
spec = importlib.util.spec_from_file_location("ja", HERE / "ja_data.py"); ja = importlib.util.module_from_spec(spec); spec.loader.exec_module(ja)

NORMAL = """step:0/20 val_loss:10.8258 train_time:0ms step_avg:nanms
step:1/20 train_loss:10.8258 train_time:300ms step_avg:nanms
step:20/20 train_loss:6.1000 train_time:4000ms step_avg:400.00ms
step:20/20 val_loss:6.0512 train_time:4100ms step_avg:410.00ms
valraw 20 6.05123456 65536
params: 162201642
"""


def mkrun(log, ckpts=(20,)):
    d = Path(tempfile.mkdtemp())
    (d / "stdout.txt").write_text(log)
    (d / "logs" / "abc").mkdir(parents=True)
    for s in ckpts:
        (d / "logs" / "abc" / ("state_step%06d.pt" % s)).write_bytes(b"x")
    return d


def expect_fail(log, ckpts=(20,), why=""):
    try:
        ma._check_train(mkrun(log, ckpts), 20, [20], 65536)
    except RuntimeError as e:
        print("  OK 拒否:", why, "→", str(e)[:60]); return
    raise AssertionError("通ってしまった: " + why)


r = ma._check_train(mkrun(NORMAL), 20, [20], 65536)
assert r["params"] == 162201642 and r["final_val_loss_raw"] == 6.05123456
print("正常ログ(step_avg:nan を含む)→ 通過")
expect_fail(NORMAL.replace("valraw 20 6.05123456", "valraw 20 nan"), why="valraw が nan")
expect_fail(NORMAL.replace("6.05123456 65536", "6.05123456 1"), why="採点数が 1")
expect_fail(NORMAL.replace("6.05123456", "6.10000000"), why="表示と raw の不一致")
expect_fail(NORMAL.replace("train_loss:6.1000", "train_loss:nan"), why="学習損失 nan")
expect_fail(NORMAL.replace("step:20/20 val_loss:6.0512 train_time:4100ms step_avg:410.00ms\n", ""), why="最終の検証なし")
expect_fail(NORMAL, ckpts=(), why="重みなし")
expect_fail(NORMAL, ckpts=(10, 20), why="余分な重み")

# 期限: 出力のない子プロセスも止まる
d = Path(tempfile.mkdtemp())
res = ma._run([sys.executable, "-c", "import time; time.sleep(30)"], str(d), d, dict(os.environ), 2)
assert res["killed_by_deadline"] and res["proc_sec"] < 10, res
print("無出力の子を期限で停止 →", res)
# SIGTERM を無視する子は SIGKILL
res = ma._run([sys.executable, "-c", "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"],
              str(d), d, dict(os.environ), 2)
assert res["killed_by_deadline"] and res["proc_sec"] < 25, res
print("SIGTERM を無視する子を SIGKILL →", res)
res = ma._run([sys.executable, "-c", "print('hi')"], str(d), d, dict(os.environ), 30)
assert res["rc"] == 0 and not res["killed_by_deadline"] and (d / "stdout.txt").read_text() == "hi\n"
print("正常な子 →", res)

# ローダーの循環
assert ja._loader_batches([10_944_513], 8, 65536) == 20 and ja._loader_batches([10_944_512], 8, 65536) == 19
print("ローダーの境目 20/19 batch → 一致")
print("ALL OK")

# --- 監査 v4 の反例 ---
import json, subprocess, time
import numpy as np

# 孫プロセス: 親だけ先に終わっても、TERM を無視する孫を止める
d = Path(tempfile.mkdtemp())
code = ("import subprocess,sys,time;"
        "subprocess.Popen([sys.executable,'-c','import signal,time,os;signal.signal(signal.SIGTERM,signal.SIG_IGN);"
        "open(\\'%s/gc.pid\\',\\'w\\').write(str(os.getpid()));time.sleep(120)']);time.sleep(60)") % d
res = ma._run([sys.executable, "-c", code], str(d), d, dict(os.environ), 2)
time.sleep(0.5)
gpid = int((d / "gc.pid").read_text())
alive = subprocess.run(["kill", "-0", str(gpid)], capture_output=True).returncode == 0
assert res["killed_by_deadline"] and not alive, (res, alive)
print("TERM を無視する孫も停止 →", res)


class DummyVol:
    def reload(self): pass
    def commit(self): pass


ma.vol = DummyVol()


def write_bin(path, ntok):
    h = np.zeros(256, dtype=np.int32); h[0], h[1], h[2] = 20240520, 1, ntok
    with open(path, "wb") as f:
        f.write(h.tobytes()); f.write(np.full(ntok, 50256, dtype=np.uint16).tobytes())


# _binding: 検収後に検証 bin のトークン数が変わったら拒否
root = Path(tempfile.mkdtemp())
ja_dir = root / "ja"; (ja_dir / "val").mkdir(parents=True); (ja_dir / "train").mkdir()
(ja_dir / "tokenizer.json").write_text("{}")
import hashlib
sha = hashlib.sha256((ja_dir / "tokenizer.json").read_bytes()).hexdigest()
write_bin(ja_dir / "val/ja_val_000000.bin", 10_944_513)
write_bin(ja_dir / "train/ja_train_000001.bin", 600_000)
dm = {"val": {"path": str(ja_dir / "val/ja_val_000000.bin"), "tokens": 10_944_513},
      "train": [{"path": str(ja_dir / "train/ja_train_000001.bin"), "tokens": 600_000}],
      "tokenizer_sha256": sha, "validated": {"val_tokens": 10_944_513}}
(ja_dir / "manifest.json").write_text(json.dumps(dm))
ma.JA_DATA, ma.JA_TOK = str(ja_dir), str(ja_dir / "tokenizer.json")
env = {"DATA_VAL": str(ja_dir / "val/ja_val_*.bin"), "DATA_TRAIN": str(ja_dir / "train/ja_train_*.bin")}
b = ma._binding("ja", env)
assert b["val_ntok"] == 10_944_513 and b["tokenizer"] == "sha256:" + sha
print("_binding 正常 → 通過")
write_bin(ja_dir / "val/ja_val_000000.bin", 10_944_514)
try:
    ma._binding("ja", env); raise AssertionError("通ってしまった")
except RuntimeError as e:
    print("  OK 拒否: 検証 bin の差し替え →", str(e)[:50])
write_bin(ja_dir / "val/ja_val_000000.bin", 10_944_513)
write_bin(ja_dir / "train/ja_train_000001.bin", 600_001)
try:
    ma._binding("ja", env); raise AssertionError("通ってしまった")
except RuntimeError as e:
    print("  OK 拒否: 学習 bin の差し替え →", str(e)[:50])

# _sample: 元の manifest に data が無い → run ディレクトリを作る前に止まる(started.json も残らない)
ma.VOL = str(root)
(root / "runs/tr").mkdir(parents=True)
(root / "runs/tr/manifest.json").write_text(json.dumps({"status": "ok", "checkpoints": [], "script_sha256": "x"}))
try:
    ma._sample("tr", "out", "gpt2", 1, ["a"], 10, [1875]); raise AssertionError("通ってしまった")
except KeyError:
    assert not (root / "runs/out").exists()
    print("  OK 拒否: data の無い学習 manifest → run ディレクトリを作らずに停止")

# gate: 未反映の初回とやり直しを両方予約する
spec = importlib.util.spec_from_file_location("gate", HERE / "gate.py"); gate = importlib.util.module_from_spec(spec); spec.loader.exec_module(gate)
gate.LEDGER = Path(tempfile.mkdtemp()) / "ledger.jsonl"
_m = gate.LIMIT - sum(gate.STAGES.values()) - gate.STAGES["S0"] / 2   # 初回は通り、未反映の初回を数えればやり直しは超える額
gate.metered = lambda: _m
sys.argv = ["gate.py", "S0"]; gate.main()
sys.argv = ["gate.py", "S0", "--retry"]
try:
    gate.main(); raise AssertionError("通ってしまった")
except SystemExit as e:
    print("  OK 拒否: 既使用 + 未反映の S0 + S0 やり直し + 全段 →", e)
# 精算: 予約 = max(0, 最大額 − 請求に載った額)。時間や割合で外さない
gate.LEDGER = Path(tempfile.mkdtemp()) / "ledger.jsonl"
gate.metered = lambda: 0.65
for st in ["S0", "S1"]:
    sys.argv = ["gate.py", st]; gate.main()
gate.done("S1", "ap-TEST1")
def unsettled():
    es = gate._entries(); r = gate.reflected(es)
    return sum(max(0.0, e["max_cost"] - r.get(i, 0.0)) for i, e in enumerate(es) if "stage" in e)
full = gate.STAGES["S0"] + gate.STAGES["S1"]
gate.settle(rows=[])
assert abs(unsettled() - full) < 1e-9
print("  OK 未反映 → 予約は最大額のまま")
gate.settle(rows=[{"object_id": "ap-TEST1", "cost": "1.10"}])   # 最大額 1.1461 の 96% が反映
assert abs(unsettled() - (full - 1.10)) < 1e-9 and unsettled() > gate.STAGES["S0"]
print("  OK 96% 反映 → 残り", round(gate.STAGES["S1"] - 1.10, 4), "を予約に残す(全額は外さない)")
gate.settle(rows=[{"object_id": "ap-TEST1", "cost": "0.40"}, {"object_id": "ap-OTHER", "cost": "5"}])
assert abs(unsettled() - (full - 0.40)) < 1e-9
print("  OK 生成込みで請求が 0.40 → 予約は 最大額 − 0.40(時間の自己申告に頼らない)")
for bad in ["NaN", "inf", "-1"]:
    try:
        gate.settle(rows=[{"object_id": "ap-TEST1", "cost": bad}]); raise AssertionError("通ってしまった")
    except RuntimeError:
        pass
assert abs(unsettled() - (full - 0.40)) < 1e-9
print("  OK 請求額 NaN / inf / 負 → 例外で止まり、予約は変わらない")
print("ALL OK (v4)")
