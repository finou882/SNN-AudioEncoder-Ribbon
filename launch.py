"""Launch and Experiment Runner for Ribbon SNN Audio Encoder.

コマンドライン引数でハイパーパラメータ（小胞放出率、順応補充速度、閾値グラデーション、
ボレー原理の不応期、動的ピント伸縮など）を柔軟に上書きして検証・実験を実行できます。

使い方例:
    # デフォルト設定でデモ実行
    python launch.py

    # 放出率と補充速度（順応の速さ）を変更して実行
    python launch.py --release-rate 2.5 --replenish-rate 0.10

    # 閾値グラデーションと不応期（ボレー原理）を変更
    python launch.py --min-threshold 0.10 --max-threshold 0.90 --refractory-steps 3

    # 自前の音声ファイルを指定して実行
    python launch.py --audio sample.wav --output result.png
"""

import argparse
import sys
from typing import Optional
import numpy as np

from config import EncoderConfig, ReceptorConfig, RibbonSynapseConfig
from demo import run_demo


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ribbon SNN Audio Encoder - Experiment Launcher",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ----------------------------------------------------
    # 1. リボンシナプス（小胞放出・順応・枯渇）
    # ----------------------------------------------------
    synapse_group = parser.add_argument_group("Ribbon Synapse & Vesicle Pool Parameters")
    synapse_group.add_argument(
        "--release-rate",
        type=float,
        default=2.0,
        help="音刺激に対する小胞放出率（アタック感と高閾値細胞の動員強度）",
    )
    synapse_group.add_argument(
        "--replenish-rate",
        type=float,
        default=0.08,
        help="小胞プールの補充速度（順応後の持続発火レベルと回復の速さ）",
    )
    synapse_group.add_argument(
        "--pool-max",
        type=float,
        default=1.0,
        help="伝達物質小胞プールの最大容量",
    )

    # ----------------------------------------------------
    # 2. 並列後細胞（閾値グラデーション・ボレー原理・ノイズ）
    # ----------------------------------------------------
    post_group = parser.add_argument_group("Post-Synaptic Parallel Neurons Parameters")
    post_group.add_argument(
        "--num-post-neurons",
        type=int,
        default=15,
        help="1前細胞あたりの後細胞（聴神経線維）数 (通常10〜20)",
    )
    post_group.add_argument(
        "--min-threshold",
        type=float,
        default=0.15,
        help="最も敏感な後細胞の閾値（かすかな微小音用）",
    )
    post_group.add_argument(
        "--max-threshold",
        type=float,
        default=0.85,
        help="最も鈍感な後細胞の閾値（大音量アタック用）",
    )
    post_group.add_argument(
        "--threshold-dist",
        type=str,
        default="geometric",
        choices=["geometric", "linear"],
        help="閾値の空間配置分布 (geometric=対数的/べき乗配置, linear=等間隔)",
    )
    post_group.add_argument(
        "--refractory-steps",
        type=int,
        default=2,
        help="不応期ステップ数（ボレー原理：発火後の休息ステップ、交代発火用）",
    )
    post_group.add_argument(
        "--leak-rate",
        type=float,
        default=0.85,
        help="膜電位の保持率（LIFリーク積分、時定数に相当）",
    )
    post_group.add_argument(
        "--noise",
        "--spontaneous-noise",
        dest="spontaneous_noise",
        type=float,
        default=0.03,
        help="自発発火用の背景ガウスノイズ強度（確率共鳴による微小音検出）",
    )

    # ----------------------------------------------------
    # 3. 受容部（周波数チャネル・動的ピント伸縮）
    # ----------------------------------------------------
    receptor_group = parser.add_argument_group("Receptor & Dynamic Focus Parameters")
    receptor_group.add_argument(
        "--num-channels",
        type=int,
        default=16,
        help="周波数チャネル数（有毛細胞数）",
    )
    receptor_group.add_argument(
        "--min-freq",
        type=float,
        default=200.0,
        help="最小周波数 (Hz)",
    )
    receptor_group.add_argument(
        "--max-freq",
        type=float,
        default=7000.0,
        help="最大周波数 (Hz)",
    )
    receptor_group.add_argument(
        "--hop-length-ms",
        type=float,
        default=2.0,
        help="時間解像度・ステップ時間 (ms)",
    )
    receptor_group.add_argument(
        "--narrow-threshold",
        type=float,
        default=0.15,
        help="ピントを絞り始める活動度閾値",
    )
    receptor_group.add_argument(
        "--shrink-rate",
        type=float,
        default=0.05,
        help="ピントをギュッと絞る速度",
    )
    receptor_group.add_argument(
        "--expand-rate",
        type=float,
        default=0.12,
        help="音が止んだときにピントをパッと広げる速度",
    )
    receptor_group.add_argument(
        "--activity-decay",
        type=float,
        default=0.88,
        help="受容活動度の記憶保持率 (H&Hの時定数に相当)",
    )

    # ----------------------------------------------------
    # 4. 入出力・実行オプション
    # ----------------------------------------------------
    io_group = parser.add_argument_group("I/O & Execution Options")
    io_group.add_argument(
        "--audio",
        type=str,
        default=None,
        help="入力音声WAVファイルへのパス（未指定の場合は検証用音声を自動生成）",
    )
    io_group.add_argument(
        "--output",
        type=str,
        default="ribbon_encoder_demo.png",
        help="検証プロット画像の保存先パス",
    )
    io_group.add_argument(
        "--no-plot",
        action="store_true",
        help="画像出力をスキップし、エンコードと統計計算のみ実行",
    )
    io_group.add_argument(
        "--show-table",
        action="store_true",
        help="チャネル別・ニューロン別の詳細発火統計表を表示",
    )

    return parser.parse_args()


def build_config(args) -> EncoderConfig:
    """引数から EncoderConfig を構築する"""
    receptor_cfg = ReceptorConfig(
        sample_rate=16000,
        num_channels=args.num_channels,
        min_freq=args.min_freq,
        max_freq=args.max_freq,
        narrow_threshold=args.narrow_threshold,
        shrink_rate=args.shrink_rate,
        expand_rate=args.expand_rate,
        activity_decay=args.activity_decay,
        hop_length_ms=args.hop_length_ms,
    )

    synapse_cfg = RibbonSynapseConfig(
        num_post_neurons=args.num_post_neurons,
        min_threshold=args.min_threshold,
        max_threshold=args.max_threshold,
        threshold_distribution=args.threshold_dist,
        leak_rate=args.leak_rate,
        refractory_steps=args.refractory_steps,
        pool_max=args.pool_max,
        replenish_rate=args.replenish_rate,
        release_rate=args.release_rate,
        spontaneous_noise=args.spontaneous_noise,
    )

    return EncoderConfig(receptor=receptor_cfg, synapse=synapse_cfg)


def print_summary_table(spikes_3d: np.ndarray, center_freqs: np.ndarray, thresholds: np.ndarray):
    """チャネル別・ニューロン別の発火統計テーブルを見やすく表示"""
    steps, num_ch, num_n = spikes_3d.shape
    print("\n" + "=" * 78)
    print("【ニューロン発火分布マトリックス (横: 閾値順ニューロン 0〜N-1, 縦: 周波数チャネル)】")
    print("=" * 78)

    header = f"{'Ch (Hz)':>14} | " + " ".join([f"{i:>3}" for i in range(num_n)]) + " |   Total"
    print(header)
    print("-" * len(header))

    for ch in range(num_ch):
        fc = center_freqs[ch]
        row_spikes = [np.sum(spikes_3d[:, ch, n]) for n in range(num_n)]
        row_str = " ".join([f"{int(s):>3}" for s in row_spikes])
        print(f"Ch{ch:02d} ({fc:4.0f}Hz) | {row_str} | {sum(row_spikes):>7}")

    print("-" * len(header))
    col_totals = [np.sum(spikes_3d[:, :, n]) for n in range(num_n)]
    col_str = " ".join([f"{int(s):>3}" for s in col_totals])
    print(f"{'Total':>14} | {col_str} | {sum(col_totals):>7}")
    print("=" * 78)


def main():
    # Windows CP932 環境対策
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    args = parse_args()
    cfg = build_config(args)

    print("\n[+] Ribbon SNN Audio Encoder - Launching with Configuration")
    print(f"  * 放出率 (release_rate)     : {cfg.synapse.release_rate}")
    print(f"  * 補充速度 (replenish_rate) : {cfg.synapse.replenish_rate} (順応・回復スピード)")
    print(f"  * 閾値レンジ (min ~ max)    : {cfg.synapse.min_threshold:.2f} ~ {cfg.synapse.max_threshold:.2f} ({cfg.synapse.threshold_distribution})")
    print(f"  * 不応期 (refractory_steps) : {cfg.synapse.refractory_steps} steps (約 {cfg.synapse.refractory_steps * cfg.receptor.hop_length_ms:.1f} ms)")
    print(f"  * 後細胞数 / チャネル数     : {cfg.synapse.num_post_neurons} 並列 / {cfg.receptor.num_channels} ch")
    print(f"  * 自発発火ノイズ (noise)    : {cfg.synapse.spontaneous_noise}")
    print(f"  * 時間解像度 (hop_length)   : {cfg.receptor.hop_length_ms} ms")
    print("-" * 60)

    spikes_flat, info = run_demo(
        cfg=cfg,
        output_filename=args.output,
        audio_path=args.audio,
        save_plot=not args.no_plot,
    )

    if args.show_table or "--show-table" in sys.argv:
        print_summary_table(
            spikes_3d=info["spikes_3d"],
            center_freqs=info["center_freqs"],
            thresholds=info.get("thresholds", np.zeros(cfg.synapse.num_post_neurons)),
        )

    print("\n[+] 実行が正常に完了しました！")


if __name__ == "__main__":
    main()
