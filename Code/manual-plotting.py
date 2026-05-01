import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import ast

# ==========================================
# 1. THE RAW DATA
# ==========================================
raw_logs = """
{'loss': 1.22, 'grad_norm': 0.404296875, 'learning_rate': 0.0002, 'entropy': 1.2376341208815576, 'num_tokens': 80671.0, 'mean_token_accuracy': 0.7038635179400444, 'epoch': 0.03}
{'loss': 1.2112, 'grad_norm': 0.416015625, 'learning_rate': 0.0002, 'entropy': 1.3445072785019874, 'num_tokens': 135821.0, 'mean_token_accuracy': 0.7009090319275856, 'epoch': 0.06}
{'loss': 1.2412, 'grad_norm': 0.76953125, 'learning_rate': 0.0002, 'entropy': 1.4808235079050065, 'num_tokens': 167387.0, 'mean_token_accuracy': 0.6995216742157936, 'epoch': 0.1}
{'loss': 1.2684, 'grad_norm': 1.3515625, 'learning_rate': 0.0002, 'entropy': 1.902634710073471, 'num_tokens': 181904.0, 'mean_token_accuracy': 0.7066691190004348, 'epoch': 0.13}
{'loss': 1.3819, 'grad_norm': 4.375, 'learning_rate': 0.0002, 'entropy': 2.220099702477455, 'num_tokens': 190182.0, 'mean_token_accuracy': 0.6976225823163986, 'epoch': 0.16}
{'loss': 1.1507, 'grad_norm': 0.302734375, 'learning_rate': 0.0002, 'entropy': 1.2836984872817994, 'num_tokens': 270197.0, 'mean_token_accuracy': 0.7110754564404488, 'epoch': 0.19}
{'loss': 1.1693, 'grad_norm': 0.478515625, 'learning_rate': 0.0002, 'entropy': 1.3447961956262589, 'num_tokens': 327541.0, 'mean_token_accuracy': 0.7093142285943032, 'epoch': 0.22}
{'loss': 1.2251, 'grad_norm': 0.7421875, 'learning_rate': 0.0002, 'entropy': 1.5337384909391403, 'num_tokens': 362042.0, 'mean_token_accuracy': 0.6977292478084565, 'epoch': 0.26}
{'loss': 1.1939, 'grad_norm': 1.0234375, 'learning_rate': 0.0002, 'entropy': 1.8417538225650787, 'num_tokens': 379583.0, 'mean_token_accuracy': 0.709807388484478, 'epoch': 0.29}
{'loss': 1.1651, 'grad_norm': 3.015625, 'learning_rate': 0.0002, 'entropy': 2.3250756353139876, 'num_tokens': 388712.0, 'mean_token_accuracy': 0.7311770141124725, 'epoch': 0.32}
{'loss': 1.1157, 'grad_norm': 0.326171875, 'learning_rate': 0.0002, 'entropy': 1.2405297696590423, 'num_tokens': 468992.0, 'mean_token_accuracy': 0.7151216387748718, 'epoch': 0.35}
{'loss': 1.2072, 'grad_norm': 0.453125, 'learning_rate': 0.0002, 'entropy': 1.389373740553856, 'num_tokens': 524487.0, 'mean_token_accuracy': 0.7025148451328278, 'epoch': 0.38}
{'loss': 1.2301, 'grad_norm': 0.83203125, 'learning_rate': 0.0002, 'entropy': 1.5801038801670075, 'num_tokens': 554361.0, 'mean_token_accuracy': 0.7016839608550072, 'epoch': 0.42}
{'loss': 1.2308, 'grad_norm': 1.46875, 'learning_rate': 0.0002, 'entropy': 1.9565806448459626, 'num_tokens': 569401.0, 'mean_token_accuracy': 0.7090479701757431, 'epoch': 0.45}
{'loss': 1.1668, 'grad_norm': 2.625, 'learning_rate': 0.0002, 'entropy': 2.415874993801117, 'num_tokens': 577757.0, 'mean_token_accuracy': 0.734724098443985, 'epoch': 0.48}
{'loss': 1.0762, 'grad_norm': 0.3359375, 'learning_rate': 0.0002, 'entropy': 1.2004819095134736, 'num_tokens': 657860.0, 'mean_token_accuracy': 0.7236034348607063, 'epoch': 0.51}
{'loss': 1.1634, 'grad_norm': 0.5078125, 'learning_rate': 0.0002, 'entropy': 1.3487437188625335, 'num_tokens': 710649.0, 'mean_token_accuracy': 0.7114080026745796, 'epoch': 0.54}
{'loss': 1.1686, 'grad_norm': 0.86328125, 'learning_rate': 0.0002, 'entropy': 1.5083874583244323, 'num_tokens': 740437.0, 'mean_token_accuracy': 0.7104368597269058, 'epoch': 0.58}
{'loss': 1.1644, 'grad_norm': 1.375, 'learning_rate': 0.0002, 'entropy': 1.8165470868349076, 'num_tokens': 755243.0, 'mean_token_accuracy': 0.7205680176615715, 'epoch': 0.61}
{'loss': 1.1581, 'grad_norm': 1.84375, 'learning_rate': 0.0002, 'entropy': 2.372552013397217, 'num_tokens': 763733.0, 'mean_token_accuracy': 0.7364309072494507, 'epoch': 0.64}
{'loss': 1.1325, 'grad_norm': 0.333984375, 'learning_rate': 0.0002, 'entropy': 1.240130227804184, 'num_tokens': 844364.0, 'mean_token_accuracy': 0.7130555897951126, 'epoch': 0.67}
{'loss': 1.1052, 'grad_norm': 0.412109375, 'learning_rate': 0.0002, 'entropy': 1.2983531266450883, 'num_tokens': 898135.0, 'mean_token_accuracy': 0.7232835903763771, 'epoch': 0.7}
{'loss': 1.1613, 'grad_norm': 0.8046875, 'learning_rate': 0.0002, 'entropy': 1.5156183570623398, 'num_tokens': 928252.0, 'mean_token_accuracy': 0.7136634767055512, 'epoch': 0.74}
{'loss': 1.1636, 'grad_norm': 1.28125, 'learning_rate': 0.0002, 'entropy': 1.9855878412723542, 'num_tokens': 943098.0, 'mean_token_accuracy': 0.7228619188070298, 'epoch': 0.77}
{'loss': 1.1635, 'grad_norm': 2.78125, 'learning_rate': 0.0002, 'entropy': 2.4665081918239595, 'num_tokens': 951028.0, 'mean_token_accuracy': 0.73317796215415, 'epoch': 0.8}
{'loss': 1.1106, 'grad_norm': 0.392578125, 'learning_rate': 0.0002, 'entropy': 1.2462620228528976, 'num_tokens': 1031343.0, 'mean_token_accuracy': 0.7171276211738586, 'epoch': 0.83}
{'loss': 1.1384, 'grad_norm': 0.46484375, 'learning_rate': 0.0002, 'entropy': 1.3329086989164352, 'num_tokens': 1088787.0, 'mean_token_accuracy': 0.7108286827802658, 'epoch': 0.86}
{'loss': 1.1291, 'grad_norm': 0.65234375, 'learning_rate': 0.0002, 'entropy': 1.4327853173017502, 'num_tokens': 1121472.0, 'mean_token_accuracy': 0.7142648190259934, 'epoch': 0.9}
{'loss': 1.1836, 'grad_norm': 1.3125, 'learning_rate': 0.0002, 'entropy': 1.8512447357177735, 'num_tokens': 1136759.0, 'mean_token_accuracy': 0.7170016571879387, 'epoch': 0.93}
{'loss': 1.234, 'grad_norm': 6.4375, 'learning_rate': 0.0002, 'entropy': 2.2536168903112412, 'num_tokens': 1145371.0, 'mean_token_accuracy': 0.7192930847406387, 'epoch': 0.96}
{'loss': 1.1533, 'grad_norm': 1.546875, 'learning_rate': 0.0002, 'entropy': 1.4891109317541122, 'num_tokens': 1193713.0, 'mean_token_accuracy': 0.7136688619852066, 'epoch': 0.99}
"""

# ==========================================
# 2. PARSE DATA
# ==========================================
data = []
for line in raw_logs.strip().split('\n'):
    # Safe parsing of python-dict style strings
    if line.startswith("{'loss'"):
        try:
            entry = ast.literal_eval(line)
            data.append(entry)
        except:
            pass

df = pd.DataFrame(data)

# Create a Step column (since logging_steps=10)
df['Step'] = range(10, (len(df) + 1) * 10, 10)

# ==========================================
# 3. PLOTTING
# ==========================================
# Set publication style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# --- Plot 1: Training Loss ---
sns.lineplot(data=df, x='Step', y='loss', ax=axes[0, 0], color='#e74c3c', linewidth=2, marker='o')
axes[0, 0].set_title('Training Loss (Cross-Entropy)', fontweight='bold')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_xlabel('Steps')

# --- Plot 2: Token Accuracy ---
sns.lineplot(data=df, x='Step', y='mean_token_accuracy', ax=axes[0, 1], color='#2ecc71', linewidth=2)
axes[0, 1].set_title('Mean Token Accuracy', fontweight='bold')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].set_xlabel('Steps')
axes[0, 1].set_ylim(0.65, 0.80) # Zoom in to see the trend

# --- Plot 3: Gradient Norm ---
# Using Log scale for Y because of spikes
sns.lineplot(data=df, x='Step', y='grad_norm', ax=axes[1, 0], color='#f39c12', linewidth=2)
axes[1, 0].set_title('Gradient Norm (Stability)', fontweight='bold')
axes[1, 0].set_ylabel('L2 Norm')
axes[1, 0].set_xlabel('Steps')
axes[1, 0].set_yscale('log') # Log scale helps visualize the spikes better

# --- Plot 4: Entropy ---
sns.lineplot(data=df, x='Step', y='entropy', ax=axes[1, 1], color='#9b59b6', linewidth=2)
axes[1, 1].set_title('Model Entropy (Uncertainty)', fontweight='bold')
axes[1, 1].set_ylabel('Entropy')
axes[1, 1].set_xlabel('Steps')

plt.tight_layout()
plt.savefig('figures/training_metrics.png', dpi=300)

print("Plot saved as figures/training_metrics.png")