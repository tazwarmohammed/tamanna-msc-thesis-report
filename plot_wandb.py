import json
import matplotlib.pyplot as plt

files = [
    '/Users/tazwar/.gemini/antigravity-ide/brain/48c49890-0bfa-4828-9a5a-b04a6675f2c2/.system_generated/steps/726/output.txt',
    '/Users/tazwar/.gemini/antigravity-ide/brain/48c49890-0bfa-4828-9a5a-b04a6675f2c2/.system_generated/steps/729/output.txt',
    '/Users/tazwar/.gemini/antigravity-ide/brain/48c49890-0bfa-4828-9a5a-b04a6675f2c2/.system_generated/steps/730/output.txt'
]

all_steps = []
all_losses = []
all_mel = []
all_lr = []

current_step_offset = 0

for file_path in files:
    with open(file_path, 'r') as f:
        data = json.load(f)
        
    rows = data.get('rows', [])
    rows = sorted(rows, key=lambda x: x.get('_step', 0))
    
    last_step_in_file = 0
    for row in rows:
        step = row.get('_step')
        if step is None: continue
        
        # If the run resumed with the exact step number instead of starting from 0,
        # we don't want to add current_step_offset. Let's check if step < last_step_in_file.
        # Actually, wandb resuming sometimes continues the step number natively.
        # Let's just use row['step'] if available, else _step.
        actual_step = row.get('step', step)
        
        # For our 3 runs, let's see if they resumed cleanly:
        if 'loss' in row and 'l_mel' in row and row['loss'] is not None and row['l_mel'] is not None:
            all_steps.append(actual_step)
            all_losses.append(row['loss'])
            all_mel.append(row['l_mel'])
            all_lr.append(row.get('lr', all_lr[-1] if len(all_lr) > 0 else 0.0001))

# Sort everything by actual step just in case they overlap
combined = list(zip(all_steps, all_losses, all_mel, all_lr))
combined.sort(key=lambda x: x[0])

clean_steps = [x[0] for x in combined]
clean_losses = [x[1] for x in combined]
clean_mel = [x[2] for x in combined]
clean_lr = [x[3] for x in combined]

# Plotting Loss
plt.figure(figsize=(10, 6))
plt.plot(clean_steps, clean_losses, label='Total Loss', color='#1f77b4', alpha=0.8)
plt.plot(clean_steps, clean_mel, label='Mel Reconstruction Loss', color='#ff7f0e', alpha=0.8)
plt.xlabel('Training Steps')
plt.ylabel('Loss Value')
plt.title('CLAE Pre-training Loss Convergence (170k Steps)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('/Users/tazwar/Desktop/tamanna thesis book/tamanna-msc-thesis-report/figure/training_loss.png', dpi=300)
plt.close()

# Plotting LR
plt.figure(figsize=(10, 6))
plt.plot(clean_steps, clean_lr, label='Learning Rate', color='#2ca02c')
plt.xlabel('Training Steps')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule across Resumed Runs')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('/Users/tazwar/Desktop/tamanna thesis book/tamanna-msc-thesis-report/figure/learning_rate.png', dpi=300)
plt.close()

print(f"Total points plotted: {len(clean_steps)}")
if len(clean_steps) > 0:
    print(f"Initial loss: {clean_losses[0]} at step {clean_steps[0]}")
    print(f"Final loss: {clean_losses[-1]} at step {clean_steps[-1]}")
