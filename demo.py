"""Demo and Visualization for Ribbon SNN Audio Encoder.

テスト音声を流して、以下の挙動をグラフで一括検証・可視化します：
1. 入力音波形（パルス音 ＆ 周波数変化）
2. 周波数チャネルの受容エネルギー
3. 帯域幅（ピント）の動的伸縮（敏感に狭まり、すぐ広がるかの確認）
4. リボン伝達物質プールの枯渇と回復（アタック感と順応）
5. 10〜20個の後細胞のスパイク列（ラスタプロット：音の大きさと連続性の確認）
"""

from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from config import EncoderConfig
from encoder import RibbonAudioEncoder


def generate_test_audio(sr: int = 16000, duration_sec: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """検証用のテスト音声を生成する。
    
    - 0.0s〜0.2s: 無音（自発発火ノイズの確認）
    - 0.2s〜0.4s: 弱めの 1000 Hz トーン（微小音：低閾値細胞のみ発火＆ピント絞り）
    - 0.4s〜0.5s: 無音（帯域幅が素早く広がるかの確認）
    - 0.5s〜0.8s: 強めの 1000 Hz トーン（大音量：高閾値細胞まで総動員＆プールの枯渇反動）
    - 0.8s〜1.0s: 500 Hz から 2500 Hz への周波数スイープ（他チャネルへの移行確認）
    """
    total_samples = int(sr * duration_sec)
    t = np.linspace(0, duration_sec, total_samples, endpoint=False)
    audio = np.zeros(total_samples)
    
    # 0.2s〜0.4s: 弱めの 1000 Hz (振幅 0.2)
    mask1 = (t >= 0.2) & (t < 0.4)
    audio[mask1] = 0.2 * np.sin(2 * np.pi * 1000.0 * t[mask1])
    
    # 0.5s〜0.8s: 強めの 1000 Hz (振幅 0.8)
    mask2 = (t >= 0.5) & (t < 0.8)
    audio[mask2] = 0.8 * np.sin(2 * np.pi * 1000.0 * t[mask2])
    
    # 0.8s〜1.0s: 500Hz -> 2500Hz スイープ (振幅 0.6)
    mask3 = (t >= 0.8) & (t < 1.0)
    t_sweep = t[mask3] - 0.8
    f_instant = 500.0 + (2500.0 - 500.0) * (t_sweep / 0.2)
    phase = 2 * np.pi * (500.0 * t_sweep + 0.5 * (2500.0 - 500.0) * (t_sweep**2) / 0.2)
    audio[mask3] = 0.6 * np.sin(phase)
    
    return t, audio


def main():
    print("=" * 60)
    print("Ribbon SNN Audio Encoder - Verification & Demo")
    print("=" * 60)
    
    # 1. 設定とエンコーダ初期化
    cfg = EncoderConfig()
    encoder = RibbonAudioEncoder(cfg)
    
    # 2. テスト音声生成
    sr = cfg.receptor.sample_rate
    time_audio, audio = generate_test_audio(sr=sr, duration_sec=1.0)
    
    print(f"[1] テスト音声生成完了: サンプル数={len(audio)}, サンプリング周波数={sr}Hz")
    
    # 3. エンコード実行
    spikes_flat, info = encoder.encode(audio, flatten_channels=True)
    spikes_3d = info["spikes_3d"]  # (Steps, Channels, PostNeurons)
    times = info["times"]
    center_freqs = info["center_freqs"]
    width_hist = info["width_history"]
    pool_hist = info["pool_history"]
    
    print(f"[2] エンコード完了:")
    print(f"    - 時間ステップ数: {spikes_flat.shape[0]} (各ステップ約 {cfg.receptor.hop_length_ms} ms)")
    print(f"    - 全出力ニューロン数: {spikes_flat.shape[1]} ({cfg.receptor.num_channels} チャネル × {cfg.synapse.num_post_neurons} 後細胞)")
    print(f"    - 総スパイク数: {np.sum(spikes_flat)}")
    
    # 4. 結果のプロット・可視化
    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)
    plt.subplots_adjust(hspace=0.25)
    
    # (a) 入力音声波形
    axes[0].plot(time_audio, audio, color="royalblue", lw=1.0)
    axes[0].set_title("(a) Input Audio Waveform", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.3)
    
    # 1000Hzに一番近いチャネルのインデックスを特定
    target_ch = int(np.argmin(np.abs(center_freqs - 1000.0)))
    target_fc = center_freqs[target_ch]
    
    # (b) 受容帯域幅（ピント）の動的伸縮（1000Hzチャネル）
    axes[1].plot(times, width_hist[:, target_ch], color="crimson", lw=1.8, label=f"Ch {target_ch} (Fc={target_fc:.0f}Hz) Bandwidth")
    axes[1].axhline(encoder.receptor_bank.min_widths[target_ch], color="gray", ls="--", alpha=0.7, label="Min Width (Narrow)")
    axes[1].axhline(encoder.receptor_bank.max_widths[target_ch], color="darkgray", ls=":", alpha=0.7, label="Max Width (Broad)")
    axes[1].set_title(f"(b) Dynamic Bandwidth Tuning (Focus Effect for ~1000Hz Channel)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Bandwidth (Hz)")
    axes[1].legend(loc="upper right", framealpha=0.9)
    axes[1].grid(True, alpha=0.3)
    
    # (c) リボン伝達物質プール残量（アタック感・順応・枯渇反動）
    axes[2].plot(times, pool_hist[:, target_ch], color="darkorange", lw=1.8, label="Vesicle Pool Level")
    axes[2].set_title("(c) Ribbon Vesicle Pool Dynamics (Depletion & Recovery)", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("Pool Level [0-1]")
    axes[2].legend(loc="upper right", framealpha=0.9)
    axes[2].grid(True, alpha=0.3)
    
    # (d) 1000Hzチャネルの後細胞15個のスパイク（閾値グラデーションと音の大きさ）
    spikes_target = spikes_3d[:, target_ch, :]  # (Steps, 15)
    step_indices, neuron_indices = np.where(spikes_target > 0)
    spike_times = times[step_indices]
    
    axes[3].scatter(spike_times, neuron_indices, s=12, c=neuron_indices, cmap="plasma", marker="|")
    axes[3].set_title(f"(d) Post-Synaptic Neurons (Ch {target_ch}) - Threshold Gradation (0=Sensitive, 14=High Threshold)", fontsize=12, fontweight="bold")
    axes[3].set_ylabel("Neuron Index")
    axes[3].set_ylim(-0.5, cfg.synapse.num_post_neurons - 0.5)
    axes[3].grid(True, alpha=0.3)
    
    # (e) 全チャネル × 全後細胞の全体ラスタプロット
    all_steps, all_neurons = np.where(spikes_flat > 0)
    axes[4].scatter(times[all_steps], all_neurons, s=4, color="black", alpha=0.6, marker="|")
    axes[4].set_title("(e) Population Raster Plot (All Channels & All Post-Synaptic Neurons)", fontsize=12, fontweight="bold")
    axes[4].set_ylabel("Global Neuron Index")
    axes[4].set_xlabel("Time (seconds)", fontsize=11)
    axes[4].set_xlim(0, 1.0)
    axes[4].grid(True, alpha=0.3)
    
    # 画像の保存
    out_filename = "ribbon_encoder_demo.png"
    plt.savefig(out_filename, dpi=180, bbox_inches="tight")
    print(f"[3] 検証プロット画像を保存しました: {out_filename}")
    plt.close()


if __name__ == "__main__":
    main()
