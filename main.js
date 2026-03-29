import * as tf from '@tensorflow/tfjs';
import readline from 'readline';

// Configuration
const MAX_VALUE = 255;
const INPUT = [0, 255];
const NORMALIZED_INPUT = INPUT.map(x => x / MAX_VALUE);
const TARGET_DATA = [0, 1];
let EPOCHS = 100;

// TensorFlow model
const model = tf.sequential();
model.add(tf.layers.dense({ units: 5, inputShape: [2], activation: 'relu' }));
model.add(tf.layers.dense({ units: 2, activation: 'softmax' }));
model.compile({ optimizer: tf.train.adam(0.01), loss: 'meanSquaredError' });

// Input tensors
const xs = tf.tensor2d([NORMALIZED_INPUT]);
const ys = tf.tensor2d([TARGET_DATA]);

async function train() {
  console.log('\n🚀 Training the TensorFlow model...\n');
  console.log('training now...');
  console.log('');

  await model.fit(xs, ys, {
    epochs: EPOCHS,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        if (epoch === 0 || epoch === EPOCHS - 1 || (epoch + 1) % Math.max(1, Math.round(EPOCHS / 100)) === 0) {
          console.log(`epoch ${epoch + 1} (${epoch + 1}/${EPOCHS}) : loss : ${logs.loss.toFixed(6)}`);
        }
      }
    }
  });

  console.log('\n\nTraining completed!\n');
}

function predict(input) {
  const normalized = input.map(x => x / MAX_VALUE);
  const inputTensor = tf.tensor2d([normalized]);
  const outputTensor = model.predict(inputTensor);
  const output = outputTensor.arraySync()[0];
  inputTensor.dispose();
  outputTensor.dispose();
  return output;
}

function showPredictions() {
  console.log('\n--- Predictions ---');
  console.log('[0, 255]:', predict([0, 255]));
  console.log('[0, 0]:', predict([0, 0]));
  console.log('[255, 255]:', predict([255, 255]));
  console.log('');
}

function showMenu() {
  console.log('\n=== TensorFlow MLP Menu ===');
  console.log('1. Training');
  console.log('2. Predicting');
  console.log('3. Reset model (reinitialize weights)');
  console.log('0. Exit');
  console.log('===========================\n');
}

const rl = readline.createInterface({ input: process.stdin, output: process.stdout });

function resetModel() {
  const newWeights = model.getWeights().map(w => tf.randomNormal(w.shape));
  model.setWeights(newWeights);
  console.log('\nModel weights reset!\n');
}

function startMenu() {
  showMenu();
  rl.question('Press number to start: ', (answer) => {
    switch (answer.trim()) {
      case '1':
        rl.question('How many epochs to train? (default 100): ', async (epochs) => {
          EPOCHS = parseInt(epochs, 10) || 100;
          await train();
          startMenu();
        });
        break;
      case '2':
        showPredictions();
        startMenu();
        break;
      case '3':
        resetModel();
        startMenu();
        break;
      case '0':
        console.log('Exiting...');
        rl.close();
        process.exit(0);
        break;
      default:
        console.log('Invalid choice. Use 0,1,2,3.');
        startMenu();
        break;
    }
  });
}

startMenu();
