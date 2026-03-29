import { ReLU } from './activations.js';

export class NeuralNetwork {
  constructor(layerSizes, activation = ReLU) {
    this.layerSizes = layerSizes;
    this.activation = activation;
    this.weights = [];
    this.biases = [];
    this.initialize();
  }

  initialize() {
    // Initialize weights and biases
    for (let l = 0; l < this.layerSizes.length - 1; l++) {
      const inputSize = this.layerSizes[l];
      const outputSize = this.layerSizes[l + 1];

      // Xavier initialization
      const limit = Math.sqrt(6 / (inputSize + outputSize));

      this.weights[l] = [];
      for (let i = 0; i < outputSize; i++) {
        this.weights[l][i] = [];
        for (let j = 0; j < inputSize; j++) {
          this.weights[l][i][j] = Math.random() * 2 * limit - limit;
        }
      }

      this.biases[l] = Array(outputSize).fill(0.01);
    }
  }

  forward(input) {
    let activations = [input];
    for (let l = 0; l < this.weights.length; l++) {
      let layer_input = activations[l];
      let layer_output = [];
      for (let i = 0; i < this.weights[l].length; i++) {
        let sum = this.biases[l][i];
        for (let j = 0; j < this.weights[l][i].length; j++) {
          sum += layer_input[j] * this.weights[l][i][j];
        }
        layer_output[i] = this.activation.activate(sum);
      }
      activations.push(layer_output);
    }
    return activations;
  }

  predict(input) {
    const activations = this.forward(input);
    return activations[activations.length - 1];
  }

  backward(activations, data, lossGradient) {
    let weightsGrad = [];
    let biasesGrad = [];

    // Output layer gradient
    let delta = lossGradient(activations[activations.length - 1], data);
    const lastLayerIdx = this.weights.length - 1;
    for (let i = 0; i < delta.length; i++) {
      delta[i] *= this.activation.derivative(activations[lastLayerIdx + 1][i]);
    }

    biasesGrad.unshift([...delta]);

    // Calculate output layer weight gradients
    let layerGrad = [];
    for (let i = 0; i < this.weights[lastLayerIdx].length; i++) {
      layerGrad[i] = [];
      for (let j = 0; j < this.weights[lastLayerIdx][i].length; j++) {
        layerGrad[i][j] = delta[i] * activations[lastLayerIdx][j];
      }
    }
    weightsGrad.unshift(layerGrad);

    // Backpropagation
    for (let l = this.weights.length - 2; l >= 0; l--) {
      let nextDelta = delta;
      delta = [];
      for (let j = 0; j < this.weights[l].length; j++) {
        let sum = 0;
        for (let i = 0; i < this.weights[l + 1].length; i++) {
          sum += nextDelta[i] * this.weights[l + 1][i][j];
        }
        delta[j] = sum * this.activation.derivative(activations[l + 1][j]);
      }
      biasesGrad.unshift([...delta]);

      layerGrad = [];
      for (let i = 0; i < this.weights[l].length; i++) {
        layerGrad[i] = [];
        for (let j = 0; j < this.weights[l][i].length; j++) {
          layerGrad[i][j] = delta[i] * activations[l][j];
        }
      }
      weightsGrad.unshift(layerGrad);
    }

    return { weightsGrad, biasesGrad };
  }

  updateWeights(weightsGrad, biasesGrad, learningRate) {
    for (let l = 0; l < this.weights.length; l++) {
      for (let i = 0; i < this.weights[l].length; i++) {
        for (let j = 0; j < this.weights[l][i].length; j++) {
          this.weights[l][i][j] -= learningRate * weightsGrad[l][i][j];
        }
      }
      for (let i = 0; i < this.biases[l].length; i++) {
        this.biases[l][i] -= learningRate * biasesGrad[l][i];
      }
    }
  }
}
