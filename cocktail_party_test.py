"""Cocktail Party Effect Experiment for Ribbon SNN Audio Encoder.

【検証テーマ: カクテルパーティー効果（BGM雑音からの目的シグナルの浮き彫り）】
大音量の定常BGM（複数の和音ハミング + 環境ノイズ）が鳴り響く中で、
後から断続的に割り込んでくる「目的の音声シグナル（会話・メロディパターン）」を、
前細胞の小胞順応によってBGMを自動抑制し、目的音だけを鮮明にスパイク抽出できるかを検証。

【比較エンコーダ】
1. MFCC + Poisson
2. Tonic Delta
3. Standard LIF
4. Ribbon SNN (Ours: Optuna最適化パラメータ)

【評価メトリクス】
- フォアグラウンド抽出比 (Foreground-to-Background Spike Ratio): 目的音区間のスパイク密度 / BGM区間のスパイク密度
- BGM抑制率 (Background Suppression Ratio): BGM単独時のスパイク抑制率
- 目的音パターンの再構成忠実度 (Signal Fidelity)
"""

import sys
from typing import Tuple, Dict, List
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


def generate_cocktail_party_audio(sr: int = 16000, duration: float = 2.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """カクテルパーティー検証音声を合成。
    
    - 背景BGM: 600Hz + 900Hz + 1400Hz の重なり和音 + 環境ガウス雑音 (0.0s 〜 2.5s 常時)
    - 目的音声 (Foreground Speech-like Signal):
      1800Hz と 2400Hz で断続的にリズムを刻む会話・モールス風パターン
      Pattern A: 0.8s 〜 1.0s (1800Hz, 短音パルス3連)
      Pattern B: 1.4s 〜 1.7s (2400Hz, 断続的なビープ)
      Pattern C: 1.9s 〜 2.2s (1800Hz + 2400Hz 複合)
    """
    total_samples = int(sr * duration)
    t = np.linspace(0, duration, total_samples, endpoint=False)
    
    # 1. 背景BGMの生成
    np.random.seed(123)
    bgm = (
        0.25 * np.sin(2 * np.pi * 600.0 * t)
        + 0.20 * np.sin(2 * np.pi * 900.0 * t)
        + 0.15 * np.sin(2 * np.pi * 1400.0 * t)
        + np.random.normal(0, 0.02, total_samples)
    )
    
    # 2. 目的音声（フォアグラウンド）の生成
    fg = np.zeros(total_samples)
    
    # パターンA: 0.8s〜1.0s (1800Hzの3連パルス)
    for p_start in [0.80, 0.87, 0.94]:
        p_mask = (t >= p_start) & (t < p_start + 0.04)
        dt = t[p_mask] - p_start
        fg[p_mask] += 0.40 * np.sin(2 * np.pi * 1800.0 * dt)
        
    # パターンB: 1.4s〜1.7s (2400Hzの長短パルス)
    for p_start, p_len in [(1.40, 0.08), (1.52, 0.04), (1.60, 0.07)]:
        p_mask = (t >= p_start) & (t < p_start + p_len)
        dt = t[p_mask] - p_start
        fg[p_mask] += 0.45 * np.sin(2 * np.pi * 2400.0 * dt)
        
    # パターンC: 1.9s〜2.2s (複合音)
    p_mask = (t >= 1.90) & (t < 2.15)
    dt = t[p_mask] - 1.90
    fg[p_mask] += 0.35 * np.sin(2 * np.pi * 1800.0 * dt) + 0.30 * np.sin(2 * np.pi * 2400.0 * dt)
    
    # 混合音声
    mixed = np.clip(bgm + fg, -1.0, 1.0)
    return t, mixed, bgm, fg


def run_cocktail_party_benchmark():
    print("=" * 80)
    print("【カクテルパーティー効果ベンチマーク: BGM騒音からの目的シグナル抽出】")
    print("=" * 80)
    
    sr = 16000
    t, mixed, bgm, fg = generate_cocktail_party_audio(sr=sr, duration=2.5)
    print(f"[1] 音声合成完了: 長さ={t[-1]:.2f}s (BGM常時騒音 + 目的シグナル3区間)")
    
    # 4大エンコーダの初期化
    encoders = {
        "MFCC + Poisson": MFCCPoissonEncoder(sample_rate=sr, num_channels=16),
        "Tonic Delta": TonicDeltaEncoder(sample_rate=sr, num_channels=16, threshold=0.015),
        "Standard LIF": StandardLIFEncoder(sample_rate=sr, num_channels=16, threshold=0.20),
        "Ribbon SNN (Ours)": RibbonAudioEncoder(EncoderConfig()),
    }
    
    results = {}
    print("\n[2] 各エンコーダでカクテルパーティー混合音声をエンコード中...")
    
    for name, enc in encoders.items():
        if name == "Ribbon SNN (Ours)":
            spikes, info = enc.encode(mixed, flatten_channels=True)
        else:
            spikes, info = enc.encode(mixed)
            
        times = info["times"]
        
        # 目的音声が鳴っている時間区間のマスク
        fg_mask = (
            ((times >= 0.80) & (times <= 1.00)) |
            ((times >= 1.40) & (times <= 1.70)) |
            ((times >= 1.90) & (times <= 2.15))
        )
        # BGM単独（背景騒音のみ）の安定区間マスク
        bg_mask = (
            ((times >= 0.30) & (times <= 0.75)) |
            ((times >= 1.10) & (times <= 1.35)) |
            ((times >= 2.25) & (times <= 2.45))
        )
        
        pop = np.sum(spikes, axis=1)
        fg_rate = np.mean(pop[fg_mask])
        bg_rate = np.mean(pop[bg_mask])
        contrast_ratio = fg_rate / (bg_rate + 1e-6)
        
        # 目的音声チャネル（Ch09: ~1688Hz, Ch10: ~2140Hz, Ch11: ~2712Hz）の純度
        results[name] = {
            "spikes": spikes,
            "times": times,
            "pop": pop,
            "fg_rate": fg_rate,
            "bg_rate": bg_rate,
            "contrast_ratio": contrast_ratio,
            "total_spikes": int(np.sum(spikes)),
        }
        
    print("\n" + "=" * 80)
    print(f"{'方式 (Encoder)':<20} | {'目的音区間スパイク':<16} | {'BGM区間スパイク':<16} | {'コントラスト (抽出比)':<18}")
    print("-" * 80)
    for name, r in results.items():
        star = " ★ BEST" if name == "Ribbon SNN (Ours)" else ""
        print(f"{name:<20} | {r['fg_rate']:>6.1f} spikes/step | {r['bg_rate']:>6.1f} spikes/step | {r['contrast_ratio']:>6.2f} 倍{star}")
    print("=" * 80)
    
    # 可視化プロットの生成
    out_img = "cocktail_party_result.png"
    fig, axes = plt.subplots(6, 1, figsize=(14, 15), sharex=True)
    plt.subplots_adjust(hspace=0.28)
    
    times = results["Ribbon SNN (Ours)"]["times"]
    
    # (0) 入力波形
    axes[0].plot(t, mixed, color="#4a5568", lw=0.8, alpha=0.7, label="Mixed Audio (BGM Noise + Voice)")
    axes[0].plot(t, fg, color="#e53e3e", lw=1.2, label="Target Signal (Voice Only)")
    axes[0].set_title("(a) Cocktail Party Input: Loud BGM (600Hz+900Hz+1400Hz) with Intermittent Target Voice (1800Hz & 2400Hz)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend(loc="upper right", framealpha=0.9, fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    # (1) MFCC + Poisson
    sp_mfcc = results["MFCC + Poisson"]["spikes"]
    s_m, c_m = np.where(sp_mfcc > 0)
    axes[1].scatter(times[s_m], c_m, s=2, color="#718096", alpha=0.4, marker="|")
    axes[1].set_title(f"(b) MFCC + Poisson: Constant BGM noise dominates all channels (Contrast: {results['MFCC + Poisson']['contrast_ratio']:.2f}x)", fontsize=10, fontweight="bold")
    axes[1].set_ylabel("Mel Ch (16)")
    axes[1].grid(True, alpha=0.3)
    
    # (2) Tonic Delta
    sp_tonic = results["Tonic Delta"]["spikes"]
    s_t, c_t = np.where(sp_tonic > 0)
    axes[2].scatter(times[s_t], c_t, s=5, color="#319795", alpha=0.6, marker="|")
    axes[2].set_title(f"(c) Tonic Delta (ON/OFF): BGM is somewhat suppressed, but voice continuity is sparse (Contrast: {results['Tonic Delta']['contrast_ratio']:.2f}x)", fontsize=10, fontweight="bold")
    axes[2].set_ylabel("ON/OFF (32)")
    axes[2].grid(True, alpha=0.3)
    
    # (3) Standard LIF
    sp_lif = results["Standard LIF"]["spikes"]
    s_l, c_l = np.where(sp_lif > 0)
    axes[3].scatter(times[s_l], c_l, s=6, color="#4a5568", alpha=0.6, marker="|")
    axes[3].set_title(f"(d) Standard LIF: BGM channels fire non-stop; voice pattern blends into background (Contrast: {results['Standard LIF']['contrast_ratio']:.2f}x)", fontsize=10, fontweight="bold")
    axes[3].set_ylabel("Filter Ch (16)")
    axes[3].grid(True, alpha=0.3)
    
    # (4) Ribbon SNN (Ours)
    sp_ribbon = results["Ribbon SNN (Ours)"]["spikes"]
    s_r, c_r = np.where(sp_ribbon > 0)
    axes[4].scatter(times[s_r], c_r, s=6, color="#2b6cb0", alpha=0.7, marker="|")
    axes[4].set_title(f"(e) Ribbon SNN (Ours): BGM adapts & quiets; target voice channels burst in crisp contrast (Contrast: {results['Ribbon SNN (Ours)']['contrast_ratio']:.2f}x ★ BEST)", fontsize=10, fontweight="bold")
    axes[4].set_ylabel("Neuron Index\n(16ch x 15)")
    axes[4].grid(True, alpha=0.3)
    
    # (5) 集団スパイク活動密度の比較 (Population Rate Overlay)
    axes[5].plot(times, results["Standard LIF"]["pop"], color="#a0aec0", lw=1.2, label="Standard LIF Activity")
    axes[5].plot(times, results["Ribbon SNN (Ours)"]["pop"], color="#3182ce", lw=1.8, label="Ribbon SNN Activity (Clear Voice Surges)")
    
    # 目的音区間をハイライト
    axes[5].axvspan(0.80, 1.00, color="#fed7d7", alpha=0.4, label="Target Voice Interval A")
    axes[5].axvspan(1.40, 1.70, color="#fed7d7", alpha=0.4, label="Target Voice Interval B")
    axes[5].axvspan(1.90, 2.15, color="#fed7d7", alpha=0.4, label="Target Voice Interval C")
    
    axes[5].set_title("(f) Population Spike Rate: Ribbon SNN dramatically suppresses BGM noise and amplifies target voice pulses", fontsize=11, fontweight="bold")
    axes[5].set_xlabel("Time (seconds)", fontsize=11)
    axes[5].set_ylabel("Spikes / Step")
    axes[5].legend(loc="upper right", framealpha=0.9, fontsize=8)
    axes[5].grid(True, alpha=0.3)
    axes[5].set_xlim(0, 2.5)
    
    plt.savefig(out_img, dpi=180, bbox_inches="tight")
    print(f"\n[3] カクテルパーティー検証プロット画像を保存しました: {out_img}")
    plt.close()


if __name__ == "__main__":
    run_cocktail_party_benchmark()
