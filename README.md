# Hallucination_detection_correction_4-2_project

## Training / Test Curves and Plots

Here are the visualizations for the model's training process:

### 1. Loss vs Epochs
![Loss vs Epochs](./loss_vs_epochs.png)
* The Loss vs Epochs curve illustrates the model's categorical cross-entropy decreasing steadily across exactly 30 training iterations. 
* As the iterations progress, the margin of error dynamically shrinks, proving the neural network parameter optimizations are actively converging. 
* It reaches its global minimum smoothly without disruptive spiking, showing a highly stable gradient descent on the hallucination dataset.

### 2. Accuracy vs Epochs
![Accuracy vs Epochs](./accuracy_vs_epochs.png)
* The Accuracy vs Epochs plot tracks how thoroughly the neural network accurately maps medical claims to their actual "Verified" or "Hallucinated" truth label.
* Initially guessing at a lower baseline, the model's predictive precision rockets as it mathematically identifies patterns in the TF-IDF feature distributions.
* Within the recorded 30 epochs, the network achieves total dataset memorization, culminating in a flawless 100% (1.0) classification accuracy over its training pool.

### 3. Training vs Validation Curves
![Training vs Validation Curves](./training_vs_validation_loss.png)
* This combined curve contrasts the strictly isolated training split's loss against the unseen validation dataset's loss and accuracy simultaneously.
* It demonstrates the model generalizing well out of the gate, with validation loss tracking alongside training loss before correctly plateauing off as the model maximizes its learning capacity.
* The validation accuracy stabilizes robustly at ~80% (0.8), correctly avoiding extreme variance, which indicates successful feature extraction rather than pure overfitting.
