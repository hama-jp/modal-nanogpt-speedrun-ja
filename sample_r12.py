"""record #12 の構造で学習した重みから文章を生成する(動画で「学習したモデルが何を書くか」を見せる用)。

学習スクリプトのモデル定義部分だけを読み込み、損失ではなくロジットを返すように forward の末尾を差し替える。
使い方: python sample_r12.py <重みのパスをカンマでつないだもの | .pt> <gpt2 | tokenizer.json> <n_samples> <prompt> [...]
条件(事前登録): 温度 0.8 / top-k 50 / 最大 80 トークン / seed = 1234 + 本数の番号 / 書き出しの前に終端トークン 50256。
step 0 の出力層はゼロ初期化なので確率が一様になり、top-k は同率の中からどれを選ぶかが実装まかせになる
(torch.topk は同率の順序を保証しない)。step 0 の生成は「学習前は意味のない並び」を示す参考で、採点の比較には使わない。
結果は SAMPLES_OUT(既定 samples.json)に条件・token id・終了理由つきで保存する。
"""
import hashlib
import json
import os
import re
import sys

import torch
import torch.nn.functional as F

SRC = "/r12/train_gpt.py"
src = open(SRC).read()
model_part = src.split("# Our own simple Distributed Data Loader")[0]
assert src.count("# Our own simple Distributed Data Loader") == 1
old_code = "with open(sys.argv[0]) as f:\n    code = f.read() # read the code of this file ASAP, for logging"
assert model_part.count(old_code) == 1
model_part = model_part.replace(old_code, "code = ''")
old = "loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))\n        return loss"
assert model_part.count(old) == 1, "forward の末尾が想定と違う"
model_part = model_part.replace(old, "return logits")
ns = {"__name__": "r12_model"}
exec(compile(model_part, "r12_model", "exec"), ns)
script_sha = hashlib.sha256(src.encode()).hexdigest()
if os.environ.get("EXPECT_SCRIPT_SHA256") and os.environ["EXPECT_SCRIPT_SHA256"] != script_sha:
    raise SystemExit("学習に使ったスクリプトとモデル定義が違う")

ckpt_arg, tok_arg, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
prompts = sys.argv[4:]
assert prompts, "書き出しがない"
if tok_arg == "gpt2":
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    encode, decode = enc.encode_ordinary, enc.decode
    tok_id = "tiktoken-gpt2"
else:
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(tok_arg)
    tk.encode_special_tokens = True   # 書き出し中の文字列を特殊トークンにしない
    assert tk.get_vocab_size() == 50257 and tk.token_to_id("<|endoftext|>") == 50256
    encode = lambda s: tk.encode(s, add_special_tokens=False).ids
    decode = lambda ids: tk.decode(ids)
    tok_id = "sha256:" + hashlib.sha256(open(tok_arg, "rb").read()).hexdigest()

if os.environ.get("EXPECT_TOKENIZER") and os.environ["EXPECT_TOKENIZER"] != tok_id:
    raise SystemExit(f"トークナイザーが学習時と違う: {tok_id} != {os.environ['EXPECT_TOKENIZER']}")
paths = [p for p in ckpt_arg.split(",") if p]
if not paths or not all(os.path.isfile(p) for p in paths):
    raise SystemExit(f"重みが見つからない: {paths}")
paths = sorted(paths, key=lambda p: int(re.search(r"state_step(\d+)", p).group(1)))
COND = {"temperature": 0.8, "top_k": 50, "max_new_tokens": 80, "seed_base": 1234, "pad_len": 256,
        "tokenizer": tok_id, "script_sha256": script_sha}
results = []
for path in paths:
    step = int(re.search(r"state_step(\d+)", path).group(1))
    state = torch.load(path, map_location="cpu", weights_only=False)  # 最適化の状態は GPU に載せない
    assert state["step"] == step, (path, state["step"])
    if hashlib.sha256(state["code"].encode()).hexdigest() != script_sha:
        raise SystemExit(f"{path} は別のスクリプトで学習した重み")
    for key, env_key in (("run", "EXPECT_RUN"), ("tokenizer", "EXPECT_TOKENIZER"), ("data_val", "EXPECT_VAL")):
        if os.environ.get(env_key):
            got = state.get(key)
            want = os.environ[env_key]
            ok = (got == want) if key != "data_val" else (got is not None and __import__("glob").glob(got) == [want])
            if not ok:
                raise SystemExit(f"{path} の {key} が違う: {got!r} != {want!r}")
    model = ns["GPT"](ns["GPTConfig"](vocab_size=50304, n_layer=12, n_head=6, n_embd=768)).cuda().bfloat16()
    for m in model.modules():
        if isinstance(m, ns["CastedLinear"]):
            m.float()
    model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in state["model"].items()})
    model.eval()
    del state
    for prompt in prompts:
        for s in range(n):
            torch.manual_seed(1234 + s)
            ids = [50256] + encode(prompt)   # 学習データと同じく文書の先頭に終端トークン
            n_prompt = len(ids)
            stop = "max_new_tokens"
            with torch.no_grad():
                for _ in range(80):
                    L = len(ids)
                    assert L <= 256
                    x = torch.tensor(ids + [0] * (256 - L), dtype=torch.long, device="cuda")  # 長さ固定で再コンパイルを避ける
                    logits = model(x, None)[0, L - 1].float()
                    logits[50257:] = -float("inf")
                    top = torch.topk(F.softmax(logits / 0.8, dim=-1), 50)
                    nxt = top.indices[torch.multinomial(top.values / top.values.sum(), 1)].item()
                    if nxt == 50256:
                        stop = "eos"
                        break
                    ids.append(nxt)
            text = decode(ids[1:])
            results.append({"step": step, "prompt": prompt, "sample": s + 1, "text": text, "stop": stop,
                            "new_token_ids": ids[n_prompt:], "ckpt": path})
            print(f"--- step {step} / sample {s + 1} ({stop})\n{text}\n", flush=True)
    del model
    torch.cuda.empty_cache()
out = os.environ.get("SAMPLES_OUT", "samples.json")
with open(out, "w") as f:
    json.dump({"conditions": COND, "samples": results}, f, ensure_ascii=False, indent=1)

# Trackio: 生成文を表として残す(TRACKIO_DIR があるときだけ)
if os.environ.get("TRACKIO_DIR"):
    import trackio
    trackio.init(project=os.environ.get("TRACKIO_PROJECT", "modal-nanogpt-speedrun"),
                 name=os.environ.get("RUN_NAME", "samples-" + ("en" if tok_arg == "gpt2" else "ja")))
    trackio.log({"samples": trackio.Table(columns=["step", "prompt", "sample", "text"],
                                          data=[[r["step"], r["prompt"], r["sample"], r["text"]] for r in results])})
    trackio.finish()
