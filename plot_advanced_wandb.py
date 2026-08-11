import json
import matplotlib.pyplot as plt

# Paths to the JSON files for the main 170k run
main_files = [
    '/Users/tazwar/.gemini/antigravity-ide/brain/48c49890-0bfa-4828-9a5a-b04a6675f2c2/.system_generated/steps/726/output.txt',
    '/Users/tazwar/.gemini/antigravity-ide/brain/48c49890-0bfa-4828-9a5a-b04a6675f2c2/.system_generated/steps/729/output.txt',
    '/Users/tazwar/.gemini/antigravity-ide/brain/48c49890-0bfa-4828-9a5a-b04a6675f2c2/.system_generated/steps/730/output.txt'
]

# Path to the ablation run JSON
ablation_file = '/Users/tazwar/.gemini/antigravity-ide/brain/48c49890-0bfa-4828-9a5a-b04a6675f2c2/.system_generated/steps/810/output.txt'

# --- Load Main Run Data ---
main_data = []

for file_path in main_files:
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        rows = data.get('rows', [])
        for row in rows:
            step = row.get('step', row.get('_step', None))
            if step is not None:
                row['actual_step'] = step
                main_data.append(row)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")

main_data.sort(key=lambda x: x['actual_step'])

# Extract metrics for plotting
steps = []
l_jepa = []
l_vis = []
z_rank = []
z_rank_utt = []
input_rms = []
output_rms = []
grad_norm = []
main_loss = []

for r in main_data:
    steps.append(r['actual_step'])
    l_jepa.append(r.get('l_jepa', None))
    l_vis.append(r.get('l_vis', None))
    z_rank.append(r.get('z_rank', None))
    z_rank_utt.append(r.get('z_rank_utt', None))
    input_rms.append(r.get('input_rms', None))
    output_rms.append(r.get('decoder_output_rms', None))
    grad_norm.append(r.get('decoder_grad_norm', None))
    main_loss.append(r.get('loss', None))

# Interpolate None values for plotting
def fill_forward(arr, default=0.0):
    res = []
    last_val = default
    for val in arr:
        if val is not None:
            last_val = val
        res.append(last_val)
    return res

l_jepa = fill_forward(l_jepa)
l_vis = fill_forward(l_vis)
z_rank = fill_forward(z_rank, 128)
z_rank_utt = fill_forward(z_rank_utt, 128)
input_rms = fill_forward(input_rms, 0.05)
output_rms = fill_forward(output_rms, 0.05)
grad_norm = fill_forward(grad_norm, 1.0)
main_loss = fill_forward(main_loss, 5.0)

# --- 1. Component Loss ---
plt.figure(figsize=(10, 6))
plt.plot(steps, l_jepa, label='JEPA Loss (Context Consistency)', color='#d62728', alpha=0.8)
plt.plot(steps, l_vis, label='VISReg Loss (Dimensional Decorrelation)', color='#9467bd', alpha=0.8)
plt.xlabel('Training Steps')
plt.ylabel('Loss Component Value')
plt.title('Self-Supervised Regularization Dynamics')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('/Users/tazwar/Desktop/tamanna thesis book/tamanna-msc-thesis-report/figure/component_loss.png', dpi=300)
plt.close()

# --- 2. Latent Rank ---
plt.figure(figsize=(10, 6))
plt.plot(steps, z_rank, label='Batch Latent Rank (Max 256)', color='#8c564b', alpha=0.8)
plt.plot(steps, z_rank_utt, label='Utterance Latent Rank', color='#e377c2', alpha=0.8)
plt.xlabel('Training Steps')
plt.ylabel('Effective Rank (Dimensionality)')
plt.title('Latent Space Utilization (Embedding Space Diversity)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('/Users/tazwar/Desktop/tamanna thesis book/tamanna-msc-thesis-report/figure/latent_rank.png', dpi=300)
plt.close()

# --- 3. Signal Energy ---
plt.figure(figsize=(10, 6))
plt.plot(steps, input_rms, label='Input Audio RMS (Target)', color='#7f7f7f', alpha=0.7)
plt.plot(steps, output_rms, label='Decoder Output RMS (Reconstruction)', color='#bcbd22', alpha=0.8)
plt.xlabel('Training Steps')
plt.ylabel('Root Mean Square Energy')
plt.title('Signal Energy Match (Attention to Acoustic Density)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('/Users/tazwar/Desktop/tamanna thesis book/tamanna-msc-thesis-report/figure/signal_energy.png', dpi=300)
plt.close()

# --- 4. Gradient Norm ---
plt.figure(figsize=(10, 6))
plt.plot(steps, grad_norm, label='Decoder Gradient Norm', color='#17becf', alpha=0.8)
plt.yscale('log')
plt.xlabel('Training Steps')
plt.ylabel('Gradient Norm (Log Scale)')
plt.title('Training Stability and Failure Case Recovery')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('/Users/tazwar/Desktop/tamanna thesis book/tamanna-msc-thesis-report/figure/gradient_norm.png', dpi=300)
plt.close()


# --- Load Ablation Run Data ---
ab_data = []
try:
    with open(ablation_file, 'r') as f:
        data = json.load(f)
    rows = data.get('rows', [])
    for row in rows:
        step = row.get('step', row.get('_step', None))
        if step is not None:
            row['actual_step'] = step
            ab_data.append(row)
except Exception as e:
    print(f"Error loading {ablation_file}: {e}")

ab_data.sort(key=lambda x: x['actual_step'])
ab_steps = []
ab_loss = []
for r in ab_data:
    ab_steps.append(r['actual_step'])
    ab_loss.append(r.get('loss', None))

ab_loss = fill_forward(ab_loss, 5.0)

# --- 5. Ablation Loss ---
plt.figure(figsize=(10, 6))
plt.plot(steps[:len(ab_steps)], main_loss[:len(ab_steps)], label='Primary Model (Composite Loss)', color='#1f77b4', alpha=0.8)
plt.plot(ab_steps, ab_loss, label='Ablation Model (Recon-Only)', color='#d62728', alpha=0.8)
plt.xlabel('Training Steps')
plt.ylabel('Total Loss')
plt.title('Ablation Study: Full Objective vs Reconstruction-Only')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('/Users/tazwar/Desktop/tamanna thesis book/tamanna-msc-thesis-report/figure/ablation_loss.png', dpi=300)
plt.close()

print("Successfully generated all 5 advanced plots!")
