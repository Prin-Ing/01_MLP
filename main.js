import { NeuralNetwork } from './src/network.js';
import { MSE } from './src/loss.js';
import { ReLU } from './src/activations.js';

// Configuration
const MAX_VALUE = 255;
const INPUT = [0, 255];
const NORMALIZED_INPUT = INPUT.map(x => x / MAX_VALUE);
const TARGET_DATA = [0, 1];
const LEARNING_RATE = 0.01;
const EPOCHS = 1000;

// Create network: 2 inputs -> 5 hidden -> 2 outputs
const network = new NeuralNetwork([2, 5, 2], ReLU);

// Training loop
function train() {
  for (let epoch = 0; epoch < EPOCHS; epoch++) {
    // Forward pass
    const activations = network.forward(NORMALIZED_INPUT);
    const output = activations[activations.length - 1];

    // Backward pass
    const grads = network.backward(activations, TARGET_DATA, MSE.gradient);
    network.updateWeights(grads.weightsGrad, grads.biasesGrad, LEARNING_RATE);

    // Logging
    if ((epoch + 1) % 100 === 0) {
      const loss = MSE.loss(output, TARGET_DATA);
      console.log(`Epoch ${epoch + 1}: Loss: ${loss.toFixed(6)}`);
    }
  }
}

// Prediction
function predict(input) {
  const normalized = input.map(x => x / MAX_VALUE);
  return network.predict(normalized);
}

// Run training
train();
console.log('\n--- Predictions ---');
console.log('[0, 255]:', predict([0, 255]));
console.log('[0, 0]:', predict([0, 0]));
console.log('[255, 255]:', predict([255, 255]));