// update.js

export function update_weights(weights, gradients, learning_rate) {
  for (let l = 0; l < weights.length; l++) {
    for (let i = 0; i < weights[l].length; i++) {
      for (let j = 0; j < weights[l][i].length; j++) {
        weights[l][i][j] -= learning_rate * gradients[l][i][j];
      }
    }
  }
  return weights;
}

export function update_bias(biases, gradients, learning_rate) {
  for (let l = 0; l < biases.length; l++) {
    for (let i = 0; i < biases[l].length; i++) {
      biases[l][i] -= learning_rate * gradients[l][i];
    }
  }
  return biases;
}
