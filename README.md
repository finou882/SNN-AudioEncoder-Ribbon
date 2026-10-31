# SNN-AudioEncoder-Ribbon 🎵⚡

生物の聴覚系における**内有毛細胞（IHC）**と**リボンシナプス（Ribbon Synapse）**、そして**聴神経線維（ANF / SGN）の並列接続構造**に着想を得た、SNN（Spiking Neural Network）専用のオーディオエンコーダです。

---

## 🌟 主な特徴と数理モデル

### 1. 動的帯域伸縮フィルタバンク（受容野チューニング）
- **トノトピー（周波数局在性）とかぶり配置**:
  低音から高音まで、隣り合う周波数チャネルが適度なオーバーラップ（かぶり）を持つように対数スケールで配置。
- **Hodgkin-Huxley風の動的ピント調節**:
  - 音刺激が入ると、活動度（反応頻度）が上昇し、**低い閾値で敏感に帯域幅をギュッと縮小（ピント合わせ・高解像度化）**。
  - 音が止むと、**スピーディーに最大帯域幅へ復帰（パッと広がるアンテナ拡大）**。

### 2. リボンシナプス機構（伝達物質小胞プール）
- 前細胞（有毛細胞）内に**伝達物質プール**をモデル化。
- **アタック感と順応（短期シナプス抑制: STD）**:
  音が鳴った瞬間に一気に小胞が放出されて急峻な立ち上がりを表現。音が続くと在庫が減少して自然な順応が発生し、刺激終了後は補充されるまで一時的な反動（疲労回復）が生じる。

### 3. 1対多の並列シナプス後細胞（10〜20個の聴神経線維）
1つの前細胞に対して 10〜20 個のシナプス後細胞が並列に結合：
- **閾値のグラデーション ➡️ 「音の大きさ（ダイナミックレンジ）」の確保**:
  低閾値（超高感度・微小音用）から高閾値（大音量用）まで段階的に配置することで、単一ニューロンの飽和を防ぎ、ささやき声から爆音まで表現。
- **不応期と交代発火（ボレー原理） ➡️ 「時間的連続性」の確保**:
  1つの細胞が発火後の不応期（休憩）に入っている間、別の細胞が波のピークを捉えることで、集団として途切れず滑らかな連続性をキープ。
- **自発発火ノイズ（確率共鳴）**:
  微小な背景ノイズにより、閾値以下の微弱な音もスパイク列として浮き上がらせる。

---

## 📁 ディレクトリ構成

```
SNN-AudioEncoder-Ribbon/
├── cocktail_party_test.py        # カクテルパーティー効果検証（BGM騒音からの目的音声スパイク抽出）
├── encoder_tradeoffs_and_cost.py # 既存手法得意タスク（Poisson, Tonic）＆計算コスト（RTF）総合検証
├── optimize_params.py            # Optuna による自動ハイパーパラメータ探索（ベイズ最適化）
├── benchmark_comparison.py       # 4大SNNエンコーダ直接対決（MFCC+Poisson, Tonic Delta, LIF, Ribbon）
├── singularity_test.py           # 音響特異点検出ベンチマーク（通常SNN vs リボンSNN 比較検証）
├── launch.py                     # 実験・実行ランチャー（CLI引数で全パラメータを調整可能）
├── config.py                     # ハイパーパラメータの一元管理（デフォルト設定）
├── receptor.py                   # 周波数受容部 ＆ 動的ピント伸縮
├── ribbon_synapse.py             # リボンシナプス小胞プール ＆ 10〜20個の並列後細胞
├── encoder.py                    # 全体を統合した完成オーディオエンコーダ
├── demo.py                       # 検証・可視化スクリプト
├── main.py                       # メインエントリーポイント
├── pyproject.toml                # プロジェクト設定
└── README.md
```

---

## 🎯 Optuna 自動ハイパーパラメータ最適化

ベイズ最適化（TPEアルゴリズム）を用いて、特異点検出率100%・低遅延・省電力を両立する最強パラメータを自動探索します：

```bash
uv run python src/optimize_params.py --trials 30
```
最適化の推移とパラメータ重要度（どのパラメータが性能に最も影響を与えているか）のグラフ `images/optuna_optimization_result.png` が生成されます。

---

## 🏆 4大SNNエンコーダ直接対決ベンチマーク

MFCC+Poisson、Tonic Delta、Standard LIF、Ribbon SNN（Ours）の4手法を横並びで比較します：

```bash
uv run python src/benchmark_comparison.py
```
実行すると、全方式のラスタプロットと下流SNNの検出比較グラフ `images/benchmark_comparison.png` が生成されます。

---

## 🔬 特異点検出ベンチマーク（案A & 案B 検証）

背景ノイズ下での「突発アタック音（衝撃・クリック音）」や「周波数の急変（ピッチホップ）」を、通常の標準SNNエンコーダと比較検証できます：

```bash
uv run python src/singularity_test.py
```
実行すると、両エンコーダのスパイク列と後段SNNの検出比較グラフ `images/singularity_test_result.png` が生成されます。

---

## 📈 音量単体＆音量×周波数2次元特異点ベンチマーク

音量単体の跳躍（微小音量・大音量跳躍・ウェーバー則）に対する閾値ニューロン動員数（15段階）と、背景音下での「周波数×音量」2次元局所バーストを検証します：

```bash
uv run python src/volume_and_2d_singularity_test.py
```
実行すると、閾値別ラスタープロットと2次元時空間ヒートマップ `images/volume_and_2d_singularity_result.png` が生成されます。

---

## 🍸 カクテルパーティー効果＆エンコーダ特性・計算コスト検証

- **カクテルパーティー効果検証（BGM中の突発音声抽出）：**
  ```bash
  uv run python src/cocktail_party_test.py
  ```
  生成画像: `images/cocktail_party_result.png`

- **既存エンコーダ得意タスク（静的スペクトル・差分エッジ）＆レイテンシ・計算コスト：**
  ```bash
  uv run python src/encoder_tradeoffs_and_cost.py
  ```
  生成画像: `images/encoder_cost_and_tradeoffs.png`

---

## 🚀 使い方

### 依存関係のインストール
```bash
uv sync
```

### 実験ランチャー（`main.py` または `src/launch.py`）による実行
ハイパーパラメータ（小胞放出率、順応補充速度、閾値グラデーション、ボレー不応期など）を引数で自由に指定して実験できます：

```bash
# デフォルト設定でデモ実行＆発火マトリックス表示 (main.py 経由)
uv run python main.py --show-table

# 放出率と補充速度（順応スピード）を調整
uv run python src/launch.py --release-rate 2.5 --replenish-rate 0.10

# 閾値グラデーションとボレー不応期を調整
uv run python src/launch.py --min-threshold 0.12 --max-threshold 0.80 --refractory-steps 3

# 独自の音声ファイル (.wav) を流して解析
uv run python src/launch.py --audio sample.wav --output images/result.png
```

#### 主なCLI引数一覧
| 引数 | 説明 | デフォルト値 |
| :--- | :--- | :--- |
| `--release-rate` | 音刺激に対する小胞放出率（アタック強度と高閾値動員） | `2.0` |
| `--replenish-rate` | 小胞補充速度（順応後の持続発火レベルと回復の速さ） | `0.08` |
| `--min-threshold` | 最も敏感な後細胞の閾値（かすかな微小音用） | `0.15` |
| `--max-threshold` | 最も鈍感な後細胞の閾値（大音量アタック用） | `0.85` |
| `--refractory-steps` | 不応期ステップ数（ボレー原理：交代発火の休息間隔） | `2` (約4ms) |
| `--num-post-neurons` | 1チャネルあたりの後細胞数 | `15` |
| `--num-channels` | 周波数チャネル数 | `16` |
| `--audio` | 入力WAVファイルのパス（未指定時は検証用音声を生成） | `None` |
| `--output` | プロット画像の保存先パス | `images/ribbon_encoder_demo.png` |
| `--show-table` | チャネル別・ニューロン別の詳細発火統計表を表示 | `False` |

### コードでの利用例
```python
import numpy as np
from src import RibbonAudioEncoder, EncoderConfig

# エンコーダの初期化
cfg = EncoderConfig()
encoder = RibbonAudioEncoder(cfg)

# 1次元の音声波形 (16kHz)
audio = np.random.randn(16000)

# スパイク列に変換
spikes, info = encoder.encode(audio)
# spikes の形状: (時間ステップ数, チャネル数 × 後細胞数)
```
