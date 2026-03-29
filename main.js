const x = [0, 1]

const w = [
  [0.5, 0.5],
  [0.5, 0.5]
]

const bias = 0

function calculate(x, w) {
  let ret = []

  ret[0] = x[0] * w[0][0] + x[1] * w[0][1] + bias
  ret[1] = x[0] * w[1][0] + x[1] * w[1][1] + bias

  ret[0] = Math.max(0, ret[0])
  ret[1] = Math.max(0, ret[1])

  return ret
}

console.log(calculate(x, w))