"""Modal で NanoGPT スピードランの record #12 を動かし、英語版の再現と日本語版の学習をする。

段(各段は前の段が成功してから。走らせる前後に `modal billing summary` で使用額を見る):
  modal run modal_app.py::prepare_data        # S0: CPU。英語 FineWeb のトークン(0.9B)を Volume へ、header と大きさを検査
  modal run modal_app.py::smoke               # S1: H100 x1。20 step(検証は 1 batch)→ 保存 → その重みで短く生成
  modal run modal_app.py::train_en            # S2: H100 x8。record #12 の本番(1875 step)
  modal run modal_app.py::sample_en           # S4: H100 x1。S2 の重みで英文を生成
  (日本語版のデータは ja_data.py)
  modal run modal_app.py::train_ja            # J3: H100 x8。同じ record #12 を日本語のデータで、375 step ごとに保存
  modal run modal_app.py::sample_ja           # J4: H100 x1。日本語版の各チェックポイントで日本語を生成
  modal run modal_app.py::val_target_bytes --lang en|ja   # CPU。採点されるトークン列のバイト数(bits per byte の分母)
  modal run modal_app.py::smoke_ja            # J2.5: H100 x1。日本語 bin とトークナイザーで 20 step → 生成

各段の起動前に `python gate.py <段>` を通す(実使用額 + その段の最大額 + 残りの必須段の最大額 ≤ 30 ドル)。

守ること:
- 予算は Starter の月30ドルの無料クレジット内。各段の最大費用(timeout まで走った場合)は gate.py の STAGES。
- 同じ段を二度走らせない: 段ごとの run ディレクトリに started.json を置き、あれば即座に止める(Modal が preemption などで
  自動で再実行しても、学習を始める前に止まる)。やり直すときは run 名を変えて明示的に起動する。
- 失敗は失敗として返す: 終了コード・最終 step・NaN・重みの有無を検査し、満たさなければ例外にする。
- 学習曲線と GPU の使用率は Trackio(無料・オープンソース)で記録する。記録はコンテナ内の /tmp/trackio に書き、
  終了時に run ディレクトリへ写す(SQLite を Volume の上で直接開かないため)。手元では `trackio show` で見る。
"""
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import modal

HERE = Path(__file__).parent
app = modal.App("nanogpt-speedrun")
vol = modal.Volume.from_name("nanogpt-speedrun", create_if_missing=True)
VOL = "/vol"
EN_DIR = f"{VOL}/data/fineweb10B"
EN_REVISION = "889765ea1f903759787add96995d81171b632d0c"   # kjj0/fineweb10B-gpt2(2026-11-17 時点)

r12_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy==2.5.5", "tiktoken==0.14.0", "huggingface-hub==2.4.0", "tokenizers==0.23.3", "trackio==0.45.1", "nvidia-ml-py==13.615.71")
    .pip_install("torch==2.10", index_url="https://download.pytorch.org/whl/cu128")
    .env({"TIKTOKEN_CACHE_DIR": f"{VOL}/tiktoken"})
    .add_local_file(HERE / "record12_train_gpt.py", "/r12/train_gpt.py")
    .add_local_file(HERE / "sample_r12.py", "/r12/sample_r12.py")
    .add_local_file(HERE / "nccl_check.py", "/r12/nccl_check.py")
)
cpu_image = (modal.Image.debian_slim(python_version="3.12")
             .pip_install("huggingface-hub==2.4.0", "numpy==2.5.5", "tiktoken==0.14.0", "tokenizers==0.23.3"))
SECRETS = []   # W&B は試用期間切れで使わない。学習曲線は Trackio で Volume に残す

H100X1 = dict(gpu="H100!", cpu=(8, 8), memory=(32768, 32768))        # "!" で H200 への自動の格上げを止め、H100 に固定
H100X8 = dict(gpu="H100!:8", cpu=(32, 32), memory=(98304, 98304))   # 8 プロセスのコンパイルに各 4 コア


# --- 共通 -------------------------------------------------------------------

def _once(key: str):
    """同じ処理の二度目を止める(Modal がプリエンプションなどで自動で再実行しても、作業を始める前に止まる)。
    やり直すときは attempt を変えて明示的に起動する(gate.py --retry を通してから)。"""
    vol.reload()
    g = Path(VOL) / "guards" / f"{key}.started"
    if g.exists():
        raise RuntimeError(f"{key} は既に開始済み(自動の再実行を止めた)")
    g.parent.mkdir(parents=True, exist_ok=True)
    g.write_text(json.dumps({"at": time.time(), "task": os.environ.get("MODAL_TASK_ID")}))
    vol.commit()

def _start_run(name: str) -> Path:
    vol.reload()
    run = Path(VOL) / "runs" / name
    if (run / "started.json").exists():
        raise RuntimeError(f"run {name} は既に開始済み(二重実行の防止)。やり直すなら別の run 名で")
    run.mkdir(parents=True, exist_ok=True)
    (run / "started.json").write_text(json.dumps({"at": time.time(), "task": os.environ.get("MODAL_TASK_ID")}))
    vol.commit()
    return run


def _gpu_info() -> str:
    return subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap,driver_version", "--format=csv"],
                          capture_output=True, text=True).stdout


def _killpg(pgid: int, sig):
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass


def _run(cmd: list[str], cwd: str, run: Path, env: dict, deadline_sec: int, out_name: str = "stdout.txt") -> dict:
    """子プロセスを動かし、標準出力を run ディレクトリへ直接書く。出力が無くても期限で止める
    (読み取りは別スレッド、本体は 1 秒ごとに期限を見る)。止めるときは SIGTERM → 15 秒待って SIGKILL。"""
    import threading
    out = run / out_name
    t0 = time.perf_counter()
    killed = False
    with open(out, "w", buffering=1) as f:
        p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, start_new_session=True)

        def pump():
            for line in p.stdout:
                print(line, end="", flush=True)
                f.write(line)

        th = threading.Thread(target=pump, daemon=True)
        th.start()
        while p.poll() is None:
            if time.perf_counter() - t0 > deadline_sec:
                killed = True
                _killpg(p.pid, signal.SIGTERM)
                try:
                    p.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    pass
                break
            time.sleep(1)
        # 親が終わっても孫が残りうるので、プロセスグループ全体を必ず止める(残っていなければ何もしない)
        _killpg(p.pid, signal.SIGKILL)
        p.wait(timeout=15)
        th.join(timeout=15)
        rc = p.returncode
    return {"rc": rc, "killed_by_deadline": killed, "proc_sec": round(time.perf_counter() - t0, 1)}


def _check_train(run: Path, expect_steps: int, need_ckpt_steps: list[int], val_tokens: int) -> dict:
    """ログから最終の検証損失を取り、完走・損失の有限性・valraw・重みの有無を確かめる。満たさなければ例外。"""
    import math
    import re
    lines = (run / "stdout.txt").read_text().splitlines()
    pat = re.compile(r"step:(\d+)/(\d+) (train_loss|val_loss):(\S+) train_time:(\d+)ms")
    recs = [m for m in map(pat.match, lines) if m]
    if not recs:
        raise RuntimeError("学習ログの行がない")
    for m in recs:   # 損失の値だけを見る(step 0 の step_avg:nan は元の仕様)
        if not math.isfinite(float(m[4])):
            raise RuntimeError(f"損失が有限でない: {m[0]}")
    vals = [m for m in recs if m[3] == "val_loss"]
    if not vals or int(vals[-1][1]) != expect_steps or int(vals[-1][2]) != expect_steps:
        raise RuntimeError(f"最終 step の検証がない: {vals[-1][0] if vals else None}")
    loss = float(vals[-1][4])
    train_time_ms = int(vals[-1][5])
    raws = [l.split() for l in lines if l.startswith("valraw ")]
    if not raws or len(raws[-1]) != 4 or int(raws[-1][1]) != expect_steps:
        raise RuntimeError("丸める前の検証損失(valraw)が最終 step にない")
    raw, scored = float(raws[-1][2]), int(raws[-1][3])
    if not math.isfinite(raw) or scored != val_tokens or abs(raw - loss) > 6e-5:
        raise RuntimeError(f"valraw が不正: raw={raw} scored={scored} 表示={loss}")
    ckpts = sorted(str(p.relative_to(run)) for p in run.glob("logs/*/state_step*.pt"))
    for s in need_ckpt_steps:
        if sum(c.endswith("state_step%06d.pt" % s) for c in ckpts) != 1:
            raise RuntimeError(f"step {s} の重みが 1 個ではない: {ckpts}")
    if len(ckpts) != len(need_ckpt_steps):
        raise RuntimeError(f"想定外の重みがある: {ckpts}")
    params = [l for l in lines if l.startswith("params: ")]
    return {"final_val_loss": loss, "final_val_loss_raw": raw, "val_scored_tokens": scored,
            "params": int(params[0].split()[1]) if params else None,
            "train_time_ms": train_time_ms, "checkpoints": ckpts,
            "val_curve": [f"step:{m[1]} val_loss:{m[4]}" for m in vals]}


def _binding(lang: str, env: dict) -> dict:
    """学習・bits per byte・生成を同じデータとトークナイザーに結び付ける。glob が採用ファイルと一致することも確かめる。"""
    import glob
    import hashlib
    import numpy as np
    vals = sorted(glob.glob(env["DATA_VAL"]))
    trains = sorted(glob.glob(env["DATA_TRAIN"]))
    if lang == "en":
        em = dict(json.loads(Path(f"{EN_DIR}/manifest.json").read_text())["files"])
        want_val = [f"{EN_DIR}/fineweb_val_000000.bin"]
        want_train = [f"{EN_DIR}/fineweb_train_%06d.bin" % i for i in range(1, 10)]
        want_val_ntok = em["fineweb_val_000000.bin"]
        want_train_ntok = {f"{EN_DIR}/{k}": v for k, v in em.items() if k.startswith("fineweb_train_")}
        tok = "tiktoken-gpt2"
    else:
        dm = json.loads(Path(f"{JA_DATA}/manifest.json").read_text())
        if "validated" not in dm:
            raise RuntimeError("日本語データが検収されていない(ja_data.py::validate)")
        want_val = [dm["val"]["path"]]
        want_train = sorted(r["path"] for r in dm["train"])
        want_val_ntok = dm["val"]["tokens"]
        if want_val_ntok != dm["validated"]["val_tokens"]:
            raise RuntimeError("検証 bin のトークン数が検収値と違う")
        want_train_ntok = {r["path"]: r["tokens"] for r in dm["train"]}
        sha = hashlib.sha256(open(JA_TOK, "rb").read()).hexdigest()
        if sha != dm["tokenizer_sha256"]:
            raise RuntimeError("トークナイザーがデータを作ったときと違う")
        tok = "sha256:" + sha
    if vals != want_val or trains != want_train:
        raise RuntimeError(f"glob が採用ファイルと一致しない: val={vals} train={len(trains)} 本")
    hdr = lambda f: int(np.fromfile(f, dtype=np.int32, count=256)[2])
    ntok = hdr(vals[0])
    if ntok != want_val_ntok or ntok < 10_944_513 or os.path.getsize(vals[0]) != 1024 + 2 * ntok:
        raise RuntimeError(f"検証 bin が記録と違う {ntok} != {want_val_ntok}")
    for f in trains:
        if hdr(f) != want_train_ntok[f] or os.path.getsize(f) != 1024 + 2 * want_train_ntok[f]:
            raise RuntimeError(f"学習 bin が記録と違う {f}")
    return {"lang": lang, "val_bin": vals[0], "val_ntok": ntok, "train_files": len(trains), "tokenizer": tok}


def _write_manifest(run: Path, manifest: dict):
    (run / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    vol.commit()


def _train(name: str, lang: str, nproc: int, extra_env: dict, deadline_sec: int, expect_steps: int,
           ckpt_steps: list[int], smoke: bool = False, val_tokens: int = 10_485_760) -> tuple[Path, dict]:
    """学習を走らせて検査する。manifest は呼び出し側が最後に書く(smoke は生成まで含めて成否を決めるため)。"""
    t0 = time.perf_counter()
    run = _start_run(name)
    manifest = {"run": name, "status": "running", "started_at": time.time()}
    try:
        manifest["data"] = _binding(lang, extra_env)
        (run / "logs").mkdir(exist_ok=True)
        Path("/r12/logs").unlink(missing_ok=True)   # コンテナが再利用された場合に前の run の接続を外す
        Path("/r12/logs").symlink_to(run / "logs", target_is_directory=True)   # 学習スクリプトは cwd の logs/ に書く
        env = dict(os.environ, RUN_NAME=name, TRACKIO_DIR="/tmp/trackio", TOKENIZER_ID=manifest["data"]["tokenizer"],
                   **extra_env)
        cmd = ["torchrun", "--standalone", f"--nproc_per_node={nproc}", "train_gpt.py"] + (["--smoke"] if smoke else [])
        manifest |= {"cmd": cmd, "env": dict(extra_env), "gpu": _gpu_info(),
                     "torch": subprocess.run(["python", "-c", "import torch;print(torch.__version__, torch.version.cuda)"],
                                             capture_output=True, text=True).stdout.strip(),
                     "script_sha256": subprocess.run(["sha256sum", "/r12/train_gpt.py"], capture_output=True,
                                                     text=True).stdout.split()[0]}
        if nproc > 1:   # 8 GPU の通信だけを先に確かめる(数十秒)。失敗したら学習を始めない
            nc = _run(["torchrun", "--standalone", f"--nproc_per_node={nproc}", "nccl_check.py"], "/r12", run,
                      dict(os.environ), 120, out_name="nccl_stdout.txt")
            manifest["nccl_check"] = nc
            if nc["rc"] != 0 or nc["killed_by_deadline"]:
                raise RuntimeError(f"NCCL の確認に失敗 {nc}")
        manifest["train_proc"] = _run(cmd, "/r12", run, env, deadline_sec)
        tp = manifest["train_proc"]
        if tp["rc"] != 0 or tp["killed_by_deadline"]:
            raise RuntimeError(f"学習が異常終了 {tp}")
        manifest |= _check_train(run, expect_steps, ckpt_steps, val_tokens)
        manifest["status"] = "train_ok"
    except Exception as e:
        manifest["status"] = f"failed: {e}"
        manifest["function_sec_until_fail"] = round(time.perf_counter() - t0, 1)
        _write_manifest(run, manifest)
        raise
    finally:
        subprocess.run(["cp", "-r", "/tmp/trackio", f"{run}/trackio"], check=False)  # 失敗時は記録が無いこともある
    manifest["train_function_sec"] = round(time.perf_counter() - t0, 1)
    return run, manifest


# --- S0 データ ---------------------------------------------------------------
@app.function(image=cpu_image, volumes={VOL: vol}, cpu=(2, 2), memory=(4096, 4096), timeout=3600)
def prepare_data(num_chunks: int = 9, attempt: int = 1):
    """英語: README と同じく学習 9 チャンク(各 1 億トークン)+ 検証 1。revision を固定し、header と大きさを検査。
    tiktoken の GPT-2 辞書も Volume に置いておく(GPU の時間に取りに行かない)。"""
    import numpy as np
    import tiktoken
    from huggingface_hub import hf_hub_download
    _once(f"S0-prepare-{attempt}")
    os.environ["TIKTOKEN_CACHE_DIR"] = f"{VOL}/tiktoken"
    tiktoken.get_encoding("gpt2")
    d = Path(EN_DIR)
    d.mkdir(parents=True, exist_ok=True)
    names = ["fineweb_val_%06d.bin" % 0] + ["fineweb_train_%06d.bin" % i for i in range(1, num_chunks + 1)]
    report = []
    for n in names:
        p = d / n
        if not p.exists():
            hf_hub_download("kjj0/fineweb10B-gpt2", n, repo_type="dataset", revision=EN_REVISION, local_dir=str(d))
        h = np.fromfile(p, dtype=np.int32, count=256)
        assert h[0] == 20240520 and h[1] == 1, n
        assert p.stat().st_size == 256 * 4 + int(h[2]) * 2, n
        toks = np.memmap(p, dtype=np.uint16, mode="r", offset=1024)
        assert int(toks.max()) <= 50256, n
        if "val" in n:
            assert int(h[2]) >= 10_944_513, n   # 10,485,760 トークンを循環なしに採点できる長さ
        report.append((n, int(h[2])))
    (d / "manifest.json").write_text(json.dumps({"repo": "kjj0/fineweb10B-gpt2", "revision": EN_REVISION,
                                                 "files": report}))
    vol.commit()
    return report


JA_DATA = f"{VOL}/ja/fw2jpn-v1"
VAL_TOKENS = 10_485_760


@app.function(image=cpu_image, volumes={VOL: vol}, cpu=(2, 2), memory=(8192, 8192), timeout=15 * 60)
def val_target_bytes(lang: str, attempt: int = 1):
    """bits per byte の分母。学習スクリプトの検証が採点するターゲット(検証 bin の tokens[1 : 10,485,761])について、
    各トークンの元のバイト列の長さを足す。終端トークンは損失に含め、バイト数は 0 とする(英語・日本語で同じ扱い)。"""
    import numpy as np
    _once(f"bytes-{lang}-{attempt}")
    lens = np.zeros(50257, dtype=np.int64)
    if lang == "en":
        import tiktoken
        os.environ["TIKTOKEN_CACHE_DIR"] = f"{VOL}/tiktoken"
        enc = tiktoken.get_encoding("gpt2")
        for i in range(50256):
            lens[i] = len(enc.decode_single_token_bytes(i))
        path = f"{EN_DIR}/fineweb_val_000000.bin"
    else:
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(f"{JA_DATA}/tokenizer.json")
        for tok_str, i in tk.get_vocab().items():
            if i != 50256:
                lens[i] = len(tok_str)   # byte-level BPE: 1 文字 = 元の 1 バイト
        path = json.loads(Path(f"{JA_DATA}/manifest.json").read_text())["val"]["path"]
    env = EN_ENV if lang == "en" else JA_ENV
    bind = _binding(lang, env)   # 学習と同じ検証 bin・トークナイザーであることを記録する
    assert bind["val_bin"] == path
    lens[50256] = 0
    toks = np.memmap(path, dtype=np.uint16, mode="r", offset=1024)
    assert len(toks) >= 10_944_513, len(toks)
    tgt = np.asarray(toks[1:VAL_TOKENS + 1])
    r = {"lang": lang, "val_bin": path, "val_ntok": bind["val_ntok"], "tokenizer": bind["tokenizer"],
         "targets": int(len(tgt)), "bytes": int(lens[tgt].sum()),
         "eos_targets": int((tgt == 50256).sum())}
    r["bytes_per_token"] = round(r["bytes"] / r["targets"], 5)
    Path(f"{VOL}/runs").mkdir(exist_ok=True)
    Path(f"{VOL}/runs/val_target_bytes_{lang}.json").write_text(json.dumps(r))
    vol.commit()
    return r


EN_ENV = {"DATA_TRAIN": f"{EN_DIR}/fineweb_train_*.bin", "DATA_VAL": f"{EN_DIR}/fineweb_val_*.bin", "SEED": "0"}
JA_ENV = {"DATA_TRAIN": f"{JA_DATA}/train/ja_train_*.bin", "DATA_VAL": f"{JA_DATA}/val/ja_val_*.bin", "SEED": "0",
          "SAVE_EVERY": "375"}
JA_TOK = f"{JA_DATA}/tokenizer.json"


# --- S1 smoke ----------------------------------------------------------------
def _smoke(run_name: str, lang: str, env: dict, tok: str, prompt: str):
    run, m = _train(run_name, lang, 1, env, deadline_sec=11 * 60, expect_steps=20, ckpt_steps=[20], smoke=True,
                    val_tokens=65536)
    try:
        senv = dict(os.environ, SAMPLES_OUT=str(run / "samples.json"), EXPECT_SCRIPT_SHA256=m["script_sha256"],
                    EXPECT_TOKENIZER=m["data"]["tokenizer"], EXPECT_RUN=run_name, EXPECT_VAL=m["data"]["val_bin"])
        m["sample_proc"] = r = _run(["python", "sample_r12.py", str(run / m["checkpoints"][-1]), tok, "1", prompt],
                                    "/r12", run, senv, 3 * 60, out_name="sample_stdout.txt")
        if r["rc"] != 0 or r["killed_by_deadline"]:
            raise RuntimeError(f"生成の確認に失敗 {r}")
        m["sample"] = json.loads((run / "samples.json").read_text())["samples"]
        if len(m["sample"]) != 1:
            raise RuntimeError("生成が 1 本ではない")
        m["status"] = "ok"
    except Exception as e:
        m["status"] = f"failed: {e}"
        raise
    finally:
        _write_manifest(run, m)
    return m


@app.function(image=r12_image, volumes={VOL: vol}, timeout=15 * 60, secrets=SECRETS, **H100X1)
def smoke(run_name: str = "smoke-en-h100x1"):
    return _smoke(run_name, "en", EN_ENV, "gpt2", "Hello")


@app.function(image=r12_image, volumes={VOL: vol}, timeout=15 * 60, secrets=SECRETS, **H100X1)
def smoke_ja(run_name: str = "smoke-ja-h100x1"):
    return _smoke(run_name, "ja", {k: v for k, v in JA_ENV.items() if k != "SAVE_EVERY"}, JA_TOK, "日本の首都は")


# --- S2 / J3 本番 -------------------------------------------------------------
def _main_train(run_name: str, lang: str, env: dict, ckpt_steps: list[int]):
    run, m = _train(run_name, lang, 8, env, deadline_sec=14 * 60, expect_steps=1875, ckpt_steps=ckpt_steps)
    m["status"] = "ok"
    _write_manifest(run, m)
    return m


@app.function(image=r12_image, volumes={VOL: vol}, timeout=16 * 60, secrets=SECRETS, **H100X8)
def train_en(run_name: str = "record12-en-h100x8"):
    return _main_train(run_name, "en", EN_ENV, [1875])


@app.function(image=r12_image, volumes={VOL: vol}, timeout=16 * 60, secrets=SECRETS, **H100X8)
def train_ja(run_name: str = "record12-ja-h100x8"):
    return _main_train(run_name, "ja", JA_ENV, [0, 375, 750, 1125, 1500, 1875])


# --- S4 / J4 生成 -------------------------------------------------------------
def _sample(train_run: str, out_name: str, tok: str, n: int, prompts: list[str], deadline_sec: int,
            need_steps: list[int]):
    t0 = time.perf_counter()
    vol.reload()
    # 学習 run の検査は run ディレクトリを作る前に済ませる(ここで失敗しても何も残らない = やり直してよい)
    src = Path(VOL) / "runs" / train_run
    m = json.loads((src / "manifest.json").read_text())
    if m.get("status") != "ok":
        raise RuntimeError(f"{train_run} が成功していない: {m.get('status')}")
    tok_id, script_sha, val_bin = m["data"]["tokenizer"], m["script_sha256"], m["data"]["val_bin"]
    ckpts = [str(src / c) for c in m["checkpoints"]]
    if len(ckpts) != len(need_steps) or not all(c.endswith("state_step%06d.pt" % s) for c, s in zip(ckpts, need_steps)):
        raise RuntimeError(f"重みの並びが想定と違う: {ckpts}")
    run = _start_run(out_name)
    out_m = {"run": out_name, "train_run": train_run, "status": "running", "prompts": prompts, "n": n,
             "tokenizer": tok_id, "script_sha256": script_sha}
    try:
        env = dict(os.environ, RUN_NAME=out_name, TRACKIO_DIR="/tmp/trackio", SAMPLES_OUT=str(run / "samples.json"),
                   EXPECT_SCRIPT_SHA256=script_sha, EXPECT_TOKENIZER=tok_id, EXPECT_RUN=train_run, EXPECT_VAL=val_bin)
        out_m["proc"] = r = _run(["python", "sample_r12.py", ",".join(ckpts), tok, str(n)] + prompts, "/r12", run,
                                 env, deadline_sec)
        if r["rc"] != 0 or r["killed_by_deadline"]:
            raise RuntimeError(f"生成が異常終了 {r}")
        out = json.loads((run / "samples.json").read_text())
        if len(out["samples"]) != len(need_steps) * len(prompts) * n:
            raise RuntimeError(f"生成の本数が違う: {len(out['samples'])}")
        out_m["n_samples"] = len(out["samples"])
        out_m["status"] = "ok"
    except Exception as e:
        out_m["status"] = f"failed: {e}"
        raise
    finally:
        subprocess.run(["cp", "-r", "/tmp/trackio", f"{run}/trackio"], check=False)
        out_m["function_sec"] = round(time.perf_counter() - t0, 1)
        _write_manifest(run, out_m)
    return out_m


@app.function(image=r12_image, volumes={VOL: vol}, timeout=10 * 60, secrets=SECRETS, **H100X1)
def sample_en(train_run: str = "record12-en-h100x8"):
    return _sample(train_run, "samples-en", "gpt2", 3, ["The meaning of life is", "Once upon a time,"], 8 * 60,
                   need_steps=[1875])


@app.function(image=r12_image, volumes={VOL: vol}, timeout=10 * 60, secrets=SECRETS, **H100X1)
def sample_ja(train_run: str = "record12-ja-h100x8"):
    return _sample(train_run, "samples-ja", JA_TOK, 2, ["日本の首都は", "昔々、ある村に", "今日の天気は"], 8 * 60,
                   need_steps=[0, 375, 750, 1125, 1500, 1875])


@app.local_entrypoint()
def main():
    print(__doc__)
