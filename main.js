import { NeuralNetwork } from './src/network.js';
import { MSE } from './src/loss.js';
import { ReLU } from './src/activations.js';
import readline from 'readline';

// Configuration
const MAX_VALUE = 255;
const INPUT = [0, 255];
const NORMALIZED_INPUT = INPUT.map(x => x / MAX_VALUE);
const TARGET_DATA = [0, 1];
const LEARNING_RATE = 0.01;
let EPOCHS = 100;

// Create network: 2 inputs -> 5 hidden -> 2 outputs
let network = new NeuralNetwork([2, 5, 2], ReLU);

// Training loop
function train() {
  console.log('\n🚀 Training the network...\n');
  for (let epoch = 0; epoch < EPOCHS; epoch++) {
    // Forward pass
    const activations = network.forward(NORMALIZED_INPUT);
    const output = activations[activations.length - 1];

    // Backward pass
    const grads = network.backward(activations, TARGET_DATA, MSE.gradient);
    network.updateWeights(grads.weightsGrad, grads.biasesGrad, LEARNING_RATE);

    // Logging
    if ((epoch + 1) % (EPOCHS / 5) === 0 || (epoch + 1) === 1) {
      const loss = MSE.loss(output, TARGET_DATA);
      console.log(`Epoch ${epoch + 1}: Loss: ${loss.toFixed(6)}`);
    }
  }
  console.log('\n✅ Training completed!\n');
}

// Reset network
function resetNetwork() {
  network = new NeuralNetwork([2, 5, 2], ReLU);
  console.log('\n🔄 Network has been reset with new random weights!\n');
}

// Prediction
function predict(input) {
  const normalized = input.map(x => x / MAX_VALUE);
  return network.predict(normalized);
}

// Print predictions
function showPredictions() {
  console.log('\n📊 --- Predictions ---');
  console.log('[0, 255]:', predict([0, 255]));
  console.log('[0, 0]:', predict([0, 0]));
  console.log('[255, 255]:', predict([255, 255]));
  console.log('');
}

// Menu
function showMenu(page = 1) {
  if (page === 1) {
    console.log('\n=== MLP Neural Network Menu ===');
    console.log('1. Training');
    console.log('2. Predicting');
    console.log('3. Reset Network');
    console.log('0. Exit');
    console.log('==============================\n');
  }
}

// Main loop
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

function startMenu() {
  showMenu();
  rl.question('Press number to start: ', (answer) => {
    switch (answer.trim()) {
      case '1':
        rl.question('How many times would you like to train? (default 100): ', (trainCount) => {
          EPOCHS = parseInt(trainCount) || 100;
          train();
          startMenu();
        });
        break;
      case '2':
        showPredictions();
        startMenu();
        break;
      case '3':
        resetNetwork();
        startMenu();
        break;
      case '0':
        console.log('👋 Goodbye!\n');
        rl.close();
        process.exit(0);
        break;
      default:
        console.log('❌ Invalid input. Please press 0, 1, 2, or 3.\n');
        startMenu();
        break;
    }
  });
}

startMenu();