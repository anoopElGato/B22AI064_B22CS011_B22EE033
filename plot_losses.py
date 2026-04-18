import json
import matplotlib.pyplot as plt

# Load JSON file
with open("checkpoints/loss_history.json", "r") as f:
    data = json.load(f)

# Extract values
epochs = [entry["epoch"] for entry in data]
train_loss = [entry["train_generator_total"] for entry in data]
# train_stft = [entry["train_generator_stft_total"] for entry in data]
valid_loss = [entry["valid_generator_total"] for entry in data]

# Plot
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_loss, label="Train Generator Loss")
# plt.plot(epochs, train_stft, label="Train STFT")
plt.plot(epochs, valid_loss, label="Valid Generator Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Epoch vs Generator Loss")
plt.legend()
plt.grid()

plt.tight_layout()
# plt.show()
plt.savefig("plots/generator_losses.png")

# Extract additional losses

train_adv = [entry["train_generator_adversarial"] for entry in data]
val_adv = [entry["valid_generator_adversarial"] for entry in data]
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_adv, label="Train Adversarial")
plt.plot(epochs, val_adv, label="Valid Adversarial")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Adversarial Loss Components")
plt.legend()
plt.grid()

plt.tight_layout()
# plt.show()
plt.savefig("plots/adversarial_losses.png")