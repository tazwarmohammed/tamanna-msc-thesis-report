import wandb
import pandas as pd
import matplotlib.pyplot as plt
import os

try:
    api = wandb.Api()
    entity = "rahman-aryan07-e"
    project = "continuous-latent-autoencoder"
    
    # We want these three runs in order of execution:
    # 1. 20260717_124435 (Initial)
    # 2. large-2kh-packed-recovery-lr-half (Recovery 1)
    # 3. large-2kh-packed-300k-tail-lr-1e4 (Recovery 2)
    run_names = ["20260717_124435", "large-2kh-packed-recovery-lr-half", "large-2kh-packed-300k-tail-lr-1e4"]
    
    runs = api.runs(f"{entity}/{project}")
    target_runs = {run.name: run for run in runs if run.name in run_names}
    
    print(f"Found runs: {list(target_runs.keys())}")
    
    # Fetch history
    histories = []
    step_offset = 0
    for name in run_names:
        if name in target_runs:
            run = target_runs[name]
            # Fetch history, limit to a reasonable number to avoid huge data
            hist = run.history(samples=1000)
            if not hist.empty:
                # Add step offset to make it continuous
                hist['continuous_step'] = hist['_step'] + step_offset
                step_offset = hist['continuous_step'].max()
                histories.append(hist)
    
    if histories:
        full_history = pd.concat(histories, ignore_index=True)
        
        # Plot Loss
        plt.figure(figsize=(10, 6))
        if 'loss' in full_history.columns:
            plt.plot(full_history['continuous_step'], full_history['loss'], label='Total Loss', alpha=0.8)
        if 'loss_mel' in full_history.columns:
            plt.plot(full_history['continuous_step'], full_history['loss_mel'], label='Mel Loss', alpha=0.5)
        
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title('Training Loss over 170k Steps (Continuous Latent Autoencoder)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        out_path = '/Users/tazwar/Desktop/tamanna thesis book/tamanna-msc-thesis-report/figure/training_loss.png'
        plt.savefig(out_path)
        print(f"Saved plot to {out_path}")
        
        print(f"Final step: {full_history['continuous_step'].max()}")
        if 'loss' in full_history.columns:
            print(f"Final loss: {full_history['loss'].iloc[-1]}")
    else:
        print("No history fetched.")
except Exception as e:
    print(f"Exception: {e}")
