// Loss functions
export const MSE = {
  loss: (output, target) => {
    let sum = 0;
    for (let i = 0; i < output.length; i++) {
      sum += Math.pow(target[i] - output[i], 2);
    }
    return sum;
  },
  gradient: (output, target) => {
    let grad = [];
    for (let i = 0; i < output.length; i++) {
      grad[i] = -2 * (target[i] - output[i]);
    }
    return grad;
  }
};

export const MAE = {
  loss: (output, target) => {
    let sum = 0;
    for (let i = 0; i < output.length; i++) {
      sum += Math.abs(target[i] - output[i]);
    }
    return sum;
  },
  gradient: (output, target) => {
    let grad = [];
    for (let i = 0; i < output.length; i++) {
      grad[i] = output[i] > target[i] ? 1 : -1;
    }
    return grad;
  }
};
