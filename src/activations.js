// Activation functions
export const ReLU = {
  activate: (x) => Math.max(0, x),
  derivative: (x) => (x > 0 ? 1 : 0)
};

export const Sigmoid = {
  activate: (x) => 1 / (1 + Math.exp(-x)),
  derivative: (x) => {
    const sig = 1 / (1 + Math.exp(-x));
    return sig * (1 - sig);
  }
};

export const Linear = {
  activate: (x) => x,
  derivative: (x) => 1
};
