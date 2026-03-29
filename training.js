// training.js
export default function training() {
  let activations = forward_propagation_full(normalized_input, weights, biases);
  let output = activations[activations.length - 1];
  let grads = get_gradient(activations, weights, data, output);
  weights = update_weights(weights, grads.weights_grad, 0.01);
  biases = update_bias(biases, grads.biases_grad, 0.01);
  const loss = output.reduce((sum, val, i) => sum + Math.pow(data[i] - val, 2), 0);
  return loss;
}