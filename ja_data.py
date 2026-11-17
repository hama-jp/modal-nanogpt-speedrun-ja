"""日本語版のデータ準備(Modal の CPU コンテナだけで行う)。

  modal run ja_data.py::download         # J0: FineWeb-2 jpn_Jpan の train 1 本(4.84GB)と test 1 本を Volume へ(revision 固定)
  modal run ja_data.py::train_tokenizer  # J1: 日本語の byte-level BPE を自作(語彙 50,256 + 終端 1 = 50,257。GPT-2 と同じ数)
  modal run ja_data.py::tokenize         # J2: 検証 bin と学習 bin を作る。必要量に届いたら次を投入しない
  modal run ja_data.py::validate         # J2 の検収だけをやり直す(CPU)

.bin の形式は modded-nanogpt / llm.c と同じ(int32 のヘッダー 256 個: 20240520・版 1・トークン数、続いて uint16)。
各文書の先頭に終端トークン(id 50256)を置く。FineWeb の GPT-2 版と同じ並べ方で、学習スクリプトの文書境界のマスク
(idx == 50256)がそのまま効く。データは DATA_ID ごとのディレクトリに置き、採用したファイルは manifest.json に書く。

監査 v2(GPT-6 Astra)の指摘 3・4・5・11・17 への対応:
- 行グループを小分けに符号化して uint16 を逐次書き足す(Python の整数リストに溜めない)。
- 最初に 1 ジョブで量と時間とメモリを測り、その見積もりで必要な分だけ投入する。足りなければ追加で投入する。
- 並列数は max_containers で抑える。子の書き込みのあと親は vol.reload() してから読む。
- 各 bin を検査する(ヘッダー・実サイズ・ID の範囲)。学習スクリプトのローダーと同じ読み方を CPU でたどり、
  8 GPU で 1876 batch を循環なしで読めること、検証で 10,485,760 トークンを循環なしで採点できることを確かめる。
"""
import json
import os
import resource
import time
from pathlib import Path

import modal

app = modal.App("nanogpt-ja-data")
vol = modal.Volume.from_name("nanogpt-speedrun", create_if_missing=True)
VOL = "/vol"
RAW = f"{VOL}/ja/raw"
DATA_ID = "fw2jpn-v1"
DATA = f"{VOL}/ja/{DATA_ID}"
TOK = f"{DATA}/tokenizer.json"
REPO = "HuggingFaceFW/fineweb-2"
REVISION = "af9c13333eb981300149d5ca60a8e9d659b276b9"   # 2026-11-17 時点
TRAIN_FILE = "data/jpn_Jpan/train/000_00000.parquet"
TEST_FILE = "data/jpn_Jpan/test/000_00000.parquet"
TRAIN_SIZE, TEST_SIZE = 4_840_000_000, 48_000_000          # おおよその大きさ(桁の確認用)
EOS = 50256
TOKENIZER_GROUPS = 120                    # 先頭のこの範囲は学習 bin に使わない
TOKENIZER_SAMPLE_GROUPS = 40              # うちトークナイザーの学習に使う行グループ(120 は 32GiB で OOM、2026-11-17)
GROUPS_PER_JOB = 40
MAX_CONTAINERS = 4                        # Volume v1 への同時書き込みを抑える
MAX_JOBS = 40                             # 学習 bin の子ジョブの総数の上限(予算 gate.py と同じ数)
# 学習ローダーの条件(record #12 の 8 GPU): 1 batch = 8 GPU × 65,536 トークン。1875 step + 最初の 1 回 = 1876 batch
WORLD, T = 8, 65536
TRAIN_BATCHES = 1876
VAL_TOKENS = 10_485_760
VAL_MIN = 10_944_513                      # 元のローダーで 10,485,760 を循環なしに採点できる最小の長さ(監査 v2 の再現)
TARGET_TRAIN_TOKENS = 1_050_000_000       # 983M + ファイル末尾で読まれない分の余裕。不足は validate が検出する

image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("huggingface-hub==2.4.0", "pyarrow==26.0.1", "tokenizers==0.23.3", "numpy==2.5.5"))


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


def _rss_mb() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024


@app.function(image=image, volumes={VOL: vol}, cpu=(2, 2), memory=(4096, 4096), timeout=40 * 60)
def download(attempt: int = 1):
    _once(f"J0-download-{attempt}")
    from huggingface_hub import hf_hub_download
    Path(RAW).mkdir(parents=True, exist_ok=True)
    out = {}
    for f, approx in ((TRAIN_FILE, TRAIN_SIZE), (TEST_FILE, TEST_SIZE)):
        p = Path(RAW) / f
        if not p.exists():
            hf_hub_download(REPO, f, repo_type="dataset", revision=REVISION, local_dir=RAW)
        size = p.stat().st_size
        assert 0.8 * approx < size < 1.2 * approx, (f, size)
        out[f] = size
    vol.commit()
    return {"revision": REVISION, "files": out}


@app.function(image=image, volumes={VOL: vol}, cpu=(8, 8), memory=(65536, 65536), timeout=50 * 60)
def train_tokenizer(attempt: int = 1):
    """train の先頭 120 行グループ(約 1.8 億文字)で byte-level BPE を学習。test と学習 bin の範囲は使わない。"""
    import hashlib
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    _once(f"J1-tokenizer-{attempt}")
    pf = pq.ParquetFile(f"{RAW}/{TRAIN_FILE}")
    stats = {"docs": 0, "chars": 0}

    def it():
        for g in range(TOKENIZER_SAMPLE_GROUPS):
            texts = pf.read_row_group(g, columns=["text"]).column("text").to_pylist()
            stats["docs"] += len(texts)
            stats["chars"] += sum(map(len, texts))
            yield texts

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=50256, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    t0 = time.time()
    tok.train_from_iterator(it(), trainer=tr)
    tok.add_special_tokens(["<|endoftext|>"])
    assert tok.get_vocab_size() == 50257, tok.get_vocab_size()
    assert tok.token_to_id("<|endoftext|>") == EOS
    # 本文中の文字列 "<|endoftext|>" は特殊 id にしない(encode_special_tokens=True)。往復で元に戻ることも確かめる
    tok.encode_special_tokens = True
    for s in ["日本の首都は東京です。", "  空白と\n改行\tタブ", "絵文字😀と記号①㈱", "文中の<|endoftext|>は文字列のまま", "English 123"]:
        ids = tok.encode(s, add_special_tokens=False).ids
        assert EOS not in ids, s
        assert tok.decode(ids) == s, (s, tok.decode(ids))
    Path(DATA).mkdir(parents=True, exist_ok=True)
    tok.save(TOK)
    vol.commit()
    return {"train_sec": round(time.time() - t0, 1), "row_groups": TOKENIZER_SAMPLE_GROUPS, **stats,
            "max_rss_mb": _rss_mb(), "sha256": hashlib.sha256(open(TOK, "rb").read()).hexdigest()}


def _load_tok():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOK)
    tok.encode_special_tokens = True
    assert tok.get_vocab_size() == 50257 and tok.token_to_id("<|endoftext|>") == EOS
    return tok


@app.function(image=image, volumes={VOL: vol}, cpu=(2, 2), memory=(4096, 4096), timeout=25 * 60, max_containers=MAX_CONTAINERS)
def tokenize_groups(job: tuple[str, int, int, str]) -> dict:
    """job = (parquet の相対パス, 開始の行グループ, 終了の行グループ, 出力パス)。行グループごとに符号化して書き足す。"""
    import numpy as np
    import pyarrow.parquet as pq
    rel, g0, g1, out = job
    _once(f"J2-child-{DATA_ID}-" + Path(out).name)
    tok = _load_tok()
    pf = pq.ParquetFile(f"{RAW}/{rel}")
    t0 = time.time()
    ntok = chars = nbytes = docs = 0
    tmp = out + ".tmp"
    with open(tmp, "wb") as f:
        f.write(np.zeros(256, dtype=np.int32).tobytes())
        for g in range(g0, min(g1, pf.num_row_groups)):
            texts = pf.read_row_group(g, columns=["text"]).column("text").to_pylist()
            for i in range(0, len(texts), 250):
                part = texts[i:i + 250]
                encs = tok.encode_batch(part, add_special_tokens=False)
                arr = np.fromiter((x for e in encs for x in ([EOS] + e.ids)), dtype=np.uint16,
                                  count=sum(len(e.ids) + 1 for e in encs))
                f.write(arr.tobytes())
                ntok += len(arr)
                chars += sum(map(len, part))
                nbytes += sum(len(t.encode("utf-8")) for t in part)
            docs += len(texts)
        header = np.zeros(256, dtype=np.int32)
        header[0], header[1], header[2] = 20240520, 1, ntok
        f.seek(0)
        f.write(header.tobytes())
    os.replace(tmp, out)
    vol.commit()
    return {"path": out, "groups": [g0, min(g1, pf.num_row_groups)], "tokens": ntok, "chars": chars,
            "bytes": nbytes, "docs": docs, "sec": round(time.time() - t0, 1), "max_rss_mb": _rss_mb(),
            "container": os.environ.get("MODAL_TASK_ID")}


def _check_bin(path: str) -> int:
    import numpy as np
    h = np.fromfile(path, dtype=np.int32, count=256)
    assert h[0] == 20240520 and h[1] == 1, path
    ntok = int(h[2])
    assert os.path.getsize(path) == 1024 + 2 * ntok, path
    toks = np.memmap(path, dtype=np.uint16, mode="r", offset=1024)
    assert int(toks.max()) <= EOS, (path, int(toks.max()))
    assert toks[0] == EOS, path
    return ntok


def _loader_batches(lengths: list[int], world: int, t: int) -> int:
    """学習スクリプトの DistributedDataLoader と同じ進み方で、全ファイルを一巡するまでに出せる batch 数を数える。
    rank r の position は r*t から始まり、next_batch のたびに world*t 進み、position + world*t >= len で次のファイルへ。
    判定は rank ごとに自分の position で行うので、後ろの rank ほど早くファイルを替える。いちばん少ない rank の数を返す。"""
    counts = []
    for r in range(world):
        n = 0
        for L in lengths:
            assert L >= world * t + 1
            pos = r * t
            while True:
                n += 1
                pos += world * t
                if pos + world * t >= L:
                    break
        counts.append(n)
    return min(counts)


@app.function(image=image, volumes={VOL: vol}, cpu=(2, 2), memory=(4096, 4096), timeout=15 * 60)
def validate() -> dict:
    vol.reload()
    m = json.loads(Path(f"{DATA}/manifest.json").read_text())
    import hashlib
    val_n = _check_bin(m["val"]["path"])
    assert val_n == m["val"]["tokens"], "検証 bin のトークン数が記録と違う"
    assert sorted(str(p) for p in Path(f"{DATA}/val").glob("ja_val_*.bin")) == [m["val"]["path"]], "val/ に余分なファイル"
    assert hashlib.sha256(open(TOK, "rb").read()).hexdigest() == m["tokenizer_sha256"], "トークナイザーが違う"
    assert val_n >= VAL_MIN, f"検証 bin が短い {val_n} < {VAL_MIN}"
    lens = [_check_bin(r["path"]) for r in m["train"]]
    assert lens == [r["tokens"] for r in m["train"]]
    nb = _loader_batches(lens, WORLD, T)
    assert nb >= TRAIN_BATCHES, f"学習 bin で循環なしに読めるのは {nb} batch(必要 {TRAIN_BATCHES})"
    # 余分なファイルが学習の glob に混ざっていないこと
    on_disk = sorted(str(p) for p in Path(f"{DATA}/train").glob("ja_train_*.bin"))
    assert on_disk == sorted(r["path"] for r in m["train"]), "train/ に manifest 外のファイルがある"
    r = {"val_tokens": val_n, "train_files": len(lens), "train_tokens": sum(lens), "loader_batches_no_wrap": nb}
    m["validated"] = r
    Path(f"{DATA}/manifest.json").write_text(json.dumps(m, ensure_ascii=False, indent=1))
    vol.commit()
    return r


@app.function(image=image, volumes={VOL: vol}, cpu=(2, 2), memory=(4096, 4096), timeout=40 * 60)
def tokenize(attempt: int = 1) -> dict:
    import hashlib
    import pyarrow.parquet as pq
    _once(f"J2-tokenize-{attempt}")
    for sub in ("train", "val", "surplus"):
        Path(f"{DATA}/{sub}").mkdir(parents=True, exist_ok=True)
    assert not list(Path(f"{DATA}/train").glob("*.bin")), "train/ が空ではない(作り直すなら DATA_ID を変える)"
    vol.commit()
    n_groups = pq.ParquetFile(f"{RAW}/{TRAIN_FILE}").num_row_groups
    t0 = time.time()
    # 検証: test 1 本を丸ごと(並行して走らせる)
    val_call = tokenize_groups.spawn((TEST_FILE, 0, 10**9, f"{DATA}/val/ja_val_000000.bin"))
    # 学習: 先頭 120 行グループ(トークナイザー用)を避け、その後ろから。最初の 1 ジョブで量を測る
    g, idx, results = TOKENIZER_GROUPS, 1, []

    def job():
        nonlocal g, idx
        j = (TRAIN_FILE, g, g + GROUPS_PER_JOB, f"{DATA}/train/ja_train_%06d.bin" % idx)
        g += GROUPS_PER_JOB
        idx += 1
        return j

    probe = tokenize_groups.remote(job())
    results.append(probe)
    # 必要量に届くまで、同時実行数ぶんずつ投入する(一度に投げるのは最大 MAX_CONTAINERS 本、総数は MAX_JOBS 本まで)
    while sum(r["tokens"] for r in results) < TARGET_TRAIN_TOKENS:
        need = TARGET_TRAIN_TOKENS - sum(r["tokens"] for r in results)
        per_job = min(r["tokens"] for r in results)
        k = min(MAX_CONTAINERS, max(1, -(-need // per_job)), MAX_JOBS - len(results))
        if k <= 0 or g >= n_groups:
            raise RuntimeError(f"必要量に届かない: {sum(r['tokens'] for r in results)} トークンで子ジョブ {len(results)} 本")
        jobs = [job() for _ in range(k) if g < n_groups]
        results += list(tokenize_groups.map(jobs, order_outputs=True))
    val = val_call.get()
    vol.reload()
    # 必要量を超えた分(最後のまとまりの余り)は surplus/ へ移し、学習の glob に入れない
    keep, acc = [], 0
    for r in results:
        if acc >= TARGET_TRAIN_TOKENS:
            dst = f"{DATA}/surplus/{Path(r['path']).name}"
            os.replace(r["path"], dst)
            r["path"] = dst
            continue
        keep.append(r)
        acc += r["tokens"]
    surplus = [r for r in results if r not in keep]
    m = {"data_id": DATA_ID, "repo": REPO, "revision": REVISION, "eos": EOS,
         "tokenizer_sha256": hashlib.sha256(open(TOK, "rb").read()).hexdigest(),
         "tokenizer_groups": [0, TOKENIZER_GROUPS], "wall_sec": round(time.time() - t0, 1),
         "max_containers": MAX_CONTAINERS, "containers_seen": sorted({r["container"] for r in results + [val]} - {None}),
         "val": val, "train": keep, "surplus": surplus, "train_tokens": acc,
         "train_chars_per_token": round(sum(r["chars"] for r in keep) / acc, 4),
         "child_wall_sec_total": round(sum(r["sec"] for r in results + [val]), 1), "child_jobs": len(results) + 1}
    Path(f"{DATA}/manifest.json").write_text(json.dumps(m, ensure_ascii=False, indent=1))
    vol.commit()
    return {"wall_sec": m["wall_sec"], "train_files": len(keep), "train_tokens": acc, "surplus_files": len(surplus),
            "val_tokens": val["tokens"], "child_wall_sec_total": m["child_wall_sec_total"], "validated": validate.local()}
