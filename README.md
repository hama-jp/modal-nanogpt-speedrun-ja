# Modal で H100 を 8 枚借りて、言語モデルを一から学習する(英語の再現と、トークナイザーから自作した日本語版)

YouTube「海外テックの『これマジ?』」ep536 で使ったコードと結果です。
動画: https://youtu.be/Lxr-2E3WVas

[Modal](https://modal.com) で NVIDIA H100 80GB を 8 枚借り、[NanoGPT スピードラン](https://github.com/KellerJordan/modded-nanogpt)の record #12(2024-11-19 FlexAttention、約 1.6 億パラメータ)の学習スクリプトを、設定を変えずに動かしました。

1. **英語版**: record #12 を再現して、手順が正しいかを確かめる(記録を出すことが目的ではありません。n=1)。
2. **日本語版**: FineWeb-2 の日本語から byte-level BPE のトークナイザー(語彙 50,257)を自作し、同じレシピで学習する。

すべて Modal の Starter プランの月 30 ドルの無料クレジット内で実行しました(2026-11-17〜18)。

## 結果

| | 英語版(record #12 の再現) | 日本語版(自作トークナイザー) |
|---|---|---|
| データ | FineWeb10B の GPT-2 トークン | FineWeb-2 `jpn_Jpan`(revision `af9c1333`)、学習 約 10.8 億トークン |
| 最終の検証損失 | 3.2805(目標 3.28、record #12 の公表平均 3.279) | 10.83 → 3.5924 |
| 学習の時間(ログの train_time) | 296.9 秒 | 300.8 秒 |
| bits per byte | 1.0669 | 0.8567 |
| 費用(Modal の請求、app 別) | 約 5.8 ドル(全工程) | 約 7.02 ドル(全工程。うち 8 枚での学習 6.34) |

- 英語版と日本語版は辞書もコーパスも違うので、損失や bits per byte の大小から言語間の能力差は言えません。
- 日本語版の途中の重み(375 step ごと)で書かせた文章を、step を伏せて Claude Sonnet が採点しました(文法・つながり・繰り返しの少なさ、各 0〜2 点、1 step あたり 6 本)。375 が最低、1500 が最高、1875 は 1500 より低く、この標本数では step ごとの差ははっきりしません。損失の改善ははっきりしています。
- 学習データに Claude や GPT が書いた文章は使っていません(FineWeb / FineWeb-2 のみ)。

生の結果は `results/` にあります(`bpb.json`、各段の `manifest.json`、生成文 `samples-*/`、採点 `samples-ja/blind_*`、トークナイザー `ja-data/tokenizer.json`)。

## ファイル

| ファイル | 中身 |
|---|---|
| `modal_app.py` | 英語データの準備、試運転、学習(H100 × 8)、生成、bits per byte の分母の計算 |
| `ja_data.py` | 日本語データの取得、トークナイザーの学習、`.map()` で CPU コンテナを並べたトークン化、検収 |
| `record12_train_gpt.py` | record #12 の学習スクリプト(公式ログに埋め込まれたコード)に、番号つきの差分だけを足したもの。差分は `record12.patch` |
| `sample_r12.py` | 保存した重みから文章を生成(温度 0.8、top-k 50、最大 80 トークン) |
| `nccl_check.py` | 8 枚の GPU 間の通信の確認 |
| `gate.py` | 各段を起動する前の予算の関門(実使用額 + 未精算の予約 + その段と残りの段の最大額 ≤ 30 ドル) |
| `tests/` | GPU なしで走る検査(`python tests/test_cpu_checks.py`) |

学習の中身(モデル・最適化・スケジュール)は変えていません。差分は、試運転モード、データの場所の環境変数、途中のチェックポイント、Trackio での記録(W&B の代わり)、シード、速さだけ測るモード、検証損失の丸める前の値の出力、保存の安全化などです(`record12.patch`)。

## 動かし方

```bash
pip install "modal>=1.6" && modal setup
# 英語版
python gate.py S0 && modal run modal_app.py::prepare_data
modal run modal_app.py::val_target_bytes --lang en
python gate.py S1 && modal run modal_app.py::smoke
python gate.py S2 && modal run modal_app.py::train_en
python gate.py S4 && modal run modal_app.py::sample_en
# 日本語版
modal run ja_data.py::download
modal run ja_data.py::train_tokenizer
modal run ja_data.py::tokenize
modal run modal_app.py::val_target_bytes --lang ja
modal run modal_app.py::smoke_ja
modal run modal_app.py::train_ja
modal run modal_app.py::sample_ja
```

- `gpu="H100!:8"` の `!` は、H200 への自動の格上げを止める指定です。
- CPU とメモリは要求量 = 上限に固定しています(`cpu=(n, n)`、`memory=(m, m)`)。課金は要求量と使用量の大きいほうなので、これで最大額が決まります。
- 同じ段を二度走らせないよう、Volume 上の印(`started.json` など)で Modal の自動再実行を作業前に止めます。
- 学習曲線と GPU の使用率は [Trackio](https://github.com/gradio-app/trackio) で記録し、手元では `trackio show` で見ます。
- 料金は 2026-11-17 時点(H100 1 枚 3.95 ドル/時)。最新の料金は Modal のページで確認してください。

## ライセンスと出典

- このリポジトリのコード: MIT(`LICENSE`)。
- `record12_train_gpt.py` は [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)(MIT、Copyright (c) 2024 Keller Jordan、commit 4ea6b937 の `records/track_1_short/2024-11-19_FlexAttention`)に基づきます。元のライセンスは `LICENSE-modded-nanogpt`。
- データ: [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)(ODC-By)、[FineWeb10B の GPT-2 トークン](https://huggingface.co/datasets/kjj0/fineweb10B-gpt2)。`results/ja-data/tokenizer.json` は FineWeb-2 の日本語から学習したものです。
- 生成文(`results/samples-*`)はモデルの出力で、事実の誤りを含みます。
