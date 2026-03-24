import random_weights from "./random_weights.js"

// Hyperparameters
let epochs = 100000

// Input and output data
const input = [
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1]
]

const data = [
  [0],
  [1],
  [1],
  [0]
]

const learning_rate = 0.1

// Weights
const init_weights = random_weights(2)
let weights = init_weights

// bias
const init_bias = Math.random()
let bias = init_bias

//
function get_error(prediction, data) {
  let error = prediction - data[0]
  return error
}

function get_loss(error) {
  let loss = Math.pow(error, 2)
  return loss
}

function edit_weights(weights, learning_rate, error, input) {
  let new_weights = []
  for (let i = 0; i < weights.length; i++) {
    new_weights.push(weights[i] - learning_rate * error * input[i])
  }
  return new_weights
}

function edit_bias(bias, learning_rate, error) {
  let new_bias = bias - learning_rate * error
  return new_bias
}

// Forward pass
function forward(input, weights, bias) {
  let result = input[0] * weights[0] + input[1] * weights[1] + bias
  return result
}

// Backward pass

// Training

const output = [0, 0, 0, 0]

for (let epoch = 0; epoch < epochs; epoch++) {
  let epochLoss = 0
  let lastPrediction = 0
  let lastError = 0
  let lastInput = null
  let lastTarget = null

  for (let i = 0; i < input.length; i++) {
    const prediction = forward(input[i], weights, bias)
    const error = get_error(prediction, data[i])
    const loss = get_loss(error)

    epochLoss += loss
    lastPrediction = prediction
    lastError = error
    lastInput = input[i]
    lastTarget = data[i][0]

    weights = edit_weights(weights, learning_rate, error, input[i])
    bias = edit_bias(bias, learning_rate, error)
  }

  const avgLoss = epochLoss / input.length

  console.log(`####### TRAINING ${epoch + 1} TIMES #########`)
  console.log(`최종 샘플: Input: ${lastInput}, Prediction: ${lastPrediction.toFixed(4)}, Error: ${lastError.toFixed(4)}, 평균 Loss: ${avgLoss.toFixed(4)}`)
  console.log("##########################")
  console.log("")
  console.log("##### RESULTS ########")
  console.log(`  PREDICTION: ${lastPrediction.toFixed(4)}, TARGET: ${lastTarget}`)
  console.log("#################")
  console.log("")
}


// 출력 확인
for (let i = 0; i < input.length; i++) {
  const y_pred = forward(input[i], weights, bias)
  console.log(`Input: ${input[i]} -> Prediction: ${y_pred}`)
}

console.log("Learned Weights: ", weights)
console.log("Learned Bias: ", bias)