"""Encoder Trade-offs and Computational Cost Benchmark.

【既存手法が得意なタスク ＆ 計算コストの客観的比較】
研究としての公平性（Fairness）を担保するため、既存手法の「得意分野」でのベンチマークおよび
実測計算コスト（処理時間、Real-Time Factor、スパイク消費電力）を徹底比較します。

【比較する3つの検証タスク】
1. 【Poissonが得意】定常スペクトル形状の静的再構成 (Static Spectral Fidelity)
   - 目的: 500ms鳴り続ける持続和音のスペクトルパワーを、スパイクの時間積分からどれだけ正確に復元できるか
   - 評価指標: コサイン類似度 (Cosine Similarity: 1.0に近いほど高忠実度)
2. 【Tonic Deltaが得意】急激なステップ変化の超省エネ記録 (Ultra-low-power Edge Recording)
   - 目的: 激しい無音/音圧ステップ変化において、最小限のスパイク数でエッジを記録できるか
   - 評価指標: 総スパイク数 (Total Spikes: 少ないほど超低消費電力)
3. 【計算コスト】実行時間・スループット・リアルタイム性 (Computational Cost & RTF)
   - 目的: 5.0秒の音声に対するエンコード実時間 (ms) および Real-Time Factor (RTF)
   - 評価指標: 実行時間 (ms), RTF (1.0未満でリアルタイム処理可能)
"""

import sys
import time
from typing import Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from config import EncoderConfig
from encoder import RibbonAudioEncoder
from benchmark_comparison import MFCCPoissonEncoder, TonicDeltaEncoder, StandardLIFEncoder


# Windows CP932 対策
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def run_tradeoffs_and_cost_benchmark():
    print("=" * 84)
    print("【SNN オーディオエンコーダ: トレードオフ特性 ＆ 計算コスト総合ベンチマーク】")
    print("=" * 84)
    
    sr = 16000
    encoders = {
        "MFCC + Poisson": MFCCPoissonEncoder(sample_rate=sr, num_channels=16),
        "Tonic Delta": TonicDeltaEncoder(sample_rate=sr, num_channels=16, threshold=0.025),
        "Standard LIF": StandardLIFEncoder(sample_rate=sr, num_channels=16, threshold=0.20),
        "Ribbon SNN (Ours)": RibbonAudioEncoder(EncoderConfig()),
    }
    
    # ==========================================================================
    # Task 1: 【Poissonが得意】定常スペクトルの静的再構成 (Static Spectral Fidelity)
    # ==========================================================================
    print("\n--- [Task 1] 定常スペクトルの静的再構成テスト (Poisson の得意分野) ---")
    print("  * 定常的な持続和音 (500Hz + 1200Hz + 2800Hz) を 1.0 秒入力")
    print("  * スパイク頻度の時間平均から、入力スペクトル形状のコサイン類似度を計測")
    
    t_stat = np.linspace(0, 1.0, sr, endpoint=False)
    stat_audio = (
        0.30 * np.sin(2 * np.pi * 500.0 * t_stat) +
        0.25 * np.sin(2 * np.pi * 1200.0 * t_stat) +
        0.20 * np.sin(2 * np.pi * 2800.0 * t_stat)
    )
    
    # 理想的な目標スペクトル（フィルタ通過後のエネルギープロファイル）
    center_freqs = np.geomspace(200.0, 7000.0, 16)
    freqs, _, Zxx = signal.stft(stat_audio, fs=sr, nperseg=256, noverlap=256-32, boundary=None)
    spec_mean = np.mean(np.abs(Zxx), axis=1)
    target_profile = np.zeros(16)
    for ch in range(16):
        fc = center_freqs[ch]
        sigma = max(1.0, (fc * 0.20) / 2.355)
        k = np.exp(-0.5 * ((freqs - fc) / sigma) ** 2)
        target_profile[ch] = np.sum(spec_mean * k) / np.sum(k)
    target_norm = target_profile / (np.linalg.norm(target_profile) + 1e-8)
    
    task1_sims = {}
    for name, enc in encoders.items():
        if name == "Ribbon SNN (Ours)":
            sp_3d, _ = enc.encode(stat_audio, flatten_channels=False)
            # 各チャネルのスパイク頻度プロファイル
            rec_profile = np.sum(sp_3d, axis=(0, 2)).astype(float)
        elif name == "Tonic Delta":
            sp, _ = enc.encode(stat_audio)
            rec_profile = np.sum(sp[:, :16], axis=0).astype(float)
        else:
            sp, _ = enc.encode(stat_audio)
            rec_profile = np.sum(sp, axis=0).astype(float)
            
        rec_norm = rec_profile / (np.linalg.norm(rec_profile) + 1e-8)
        cos_sim = float(np.dot(target_norm, rec_norm))
        task1_sims[name] = max(0.0, cos_sim)
        
    for name, sim in task1_sims.items():
        tag = " (Poissonの真骨頂: 忠実なスペクトル復元)" if "Poisson" in name else ""
        print(f"  - {name:<20}: コサイン類似度 = {sim:.4f}{tag}")

    # ==========================================================================
    # Task 2: 【Tonic Deltaが得意】矩形ステップ変化の超省エネスパイク化
    # ==========================================================================
    print("\n--- [Task 2] 矩形ステップ音の超省エネスパイク化テスト (Tonic Delta の得意分野) ---")
    print("  * 無音とトーンが急激にON/OFFする矩形波状の音響刺激 (1.0秒)")
    print("  * エッジだけを拾い、無駄なスパイクを出さない省電力性を計測")
    
    t_step = np.linspace(0, 1.0, sr, endpoint=False)
    step_audio = np.zeros_like(t_step)
    # 0.2s〜0.4s と 0.6s〜0.8s に 1000Hz 矩形トーン
    step_audio[(t_step >= 0.2) & (t_step <= 0.4)] = 0.50 * np.sin(2 * np.pi * 1000.0 * t_step[(t_step >= 0.2) & (t_step <= 0.4)])
    step_audio[(t_step >= 0.6) & (t_step <= 0.8)] = 0.50 * np.sin(2 * np.pi * 1000.0 * t_step[(t_step >= 0.6) & (t_step <= 0.8)])
    
    task2_spikes = {}
    for name, enc in encoders.items():
        if name == "Ribbon SNN (Ours)":
            sp, _ = enc.encode(step_audio, flatten_channels=True)
        else:
            sp, _ = enc.encode(step_audio)
        task2_spikes[name] = int(np.sum(sp))
        
    for name, sp_count in task2_spikes.items():
        tag = " (Tonic Deltaの真骨頂: 超極小スパイクで省エネ)" if "Tonic" in name else ""
        print(f"  - {name:<20}: 総スパイク数 = {sp_count:>6,} 発{tag}")

    # ==========================================================================
    # Task 3: 【計算コスト】実行速度・Real-Time Factor (RTF)
    # ==========================================================================
    print("\n--- [Task 3] 計算コスト・処理速度ベンチマーク (5.0秒音声) ---")
    bench_duration = 5.0
    t_bench = np.linspace(0, bench_duration, int(sr * bench_duration), endpoint=False)
    bench_audio = np.sin(2 * np.pi * 800.0 * t_bench) * 0.30
    
    # ウォームアップ
    for enc in encoders.values():
        if hasattr(enc, "reset"): enc.reset()
        enc.encode(bench_audio[:sr])
        
    task3_times = {}
    task3_rtf = {}
    num_runs = 5
    
    for name, enc in encoders.items():
        t0 = time.perf_counter()
        for _ in range(num_runs):
            if hasattr(enc, "reset"): enc.reset()
            if name == "Ribbon SNN (Ours)":
                enc.encode(bench_audio, flatten_channels=True)
            else:
                enc.encode(bench_audio)
        elapsed_sec = (time.perf_counter() - t0) / num_runs
        elapsed_ms = elapsed_sec * 1000.0
        rtf = elapsed_sec / bench_duration  # 1.0未満でリアルタイム処理可能
        
        task3_times[name] = elapsed_ms
        task3_rtf[name] = rtf
        print(f"  - {name:<20}: 実行時間 = {elapsed_ms:>6.2f} ms (5.0秒音声を処理) | RTF = {rtf:.4f} (×{1.0/rtf:.1f}倍速)")

    # ==========================================================================
    # 総合サマリー表示
    # ==========================================================================
    print("\n" + "=" * 90)
    print("【4大エンコーダの適性・トレードオフ・計算コスト総括テーブル】")
    print("=" * 90)
    print(f"{'方式 (Encoder)':<20} | {'スペクトル再現度':<16} | {'ステップ時スパイク数':<18} | {'処理時間 (5s)':<14} | {'RTF (倍速)':<12}")
    print("-" * 90)
    for name in encoders.keys():
        sim_s = f"{task1_sims[name]:.3f} (類似度)"
        sp_s = f"{task2_spikes[name]:,} 発"
        t_s = f"{task3_times[name]:.1f} ms"
        rtf_s = f"×{1.0/task3_rtf[name]:.0f}倍速"
        print(f"{name:<20} | {sim_s:<16} | {sp_s:<18} | {t_s:<14} | {rtf_s:<12}")
    print("=" * 90)
    
    # 4. 可視化グラフの作成
    out_img = "encoder_cost_and_tradeoffs.png"
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    plt.subplots_adjust(wspace=0.32)
    
    names = list(encoders.keys())
    colors = ["#6c757d", "#17a2b8", "#e76f51", "#2a9d8f"]
    
    # (a) スペクトル忠実度 (Poissonが得意)
    axes[0].bar(names, [task1_sims[n] for n in names], color=colors, alpha=0.85)
    axes[0].set_title("(a) Static Spectrum Fidelity (Cosine Sim)\n[MFCC+Poisson's Strength]", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("Cosine Similarity (Higher is Better)")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, alpha=0.3, axis="y")
    axes[0].tick_params(axis="x", rotation=25)
    
    # (b) エッジ記録時の総スパイク数 (Tonic Deltaが得意)
    axes[1].bar(names, [task2_spikes[n] for n in names], color=colors, alpha=0.85)
    axes[1].set_title("(b) Spikes in Step Audio (Lower is Energy-Efficient)\n[Tonic Delta's Strength]", fontsize=10, fontweight="bold")
    axes[1].set_ylabel("Total Spikes (Fewer is Lower Power)")
    axes[1].grid(True, alpha=0.3, axis="y")
    axes[1].tick_params(axis="x", rotation=25)
    
    # (c) 計算時間 (5秒音声のエンコード所要時間)
    axes[2].bar(names, [task3_times[n] for n in names], color=colors, alpha=0.85)
    axes[2].set_title("(c) Compute Latency for 5.0s Audio (ms)\n[Lower is Faster, All Real-Time]", fontsize=10, fontweight="bold")
    axes[2].set_ylabel("Execution Time (ms)")
    axes[2].grid(True, alpha=0.3, axis="y")
    axes[2].tick_params(axis="x", rotation=25)
    
    plt.tight_layout()
    plt.savefig(out_img, dpi=180)
    print(f"\n[4] トレードオフ・計算コスト比較グラフを保存しました: {out_img}")
    plt.close()


if __name__ == "__main__":
    run_tradeoffs_and_cost_benchmark()
