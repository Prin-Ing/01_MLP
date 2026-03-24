export default function random_weights(dimension) {
  let weights = []
  for (let i = 0; i < dimension; i++) {
    weights.push(Math.random())
  }
  return weights
}
