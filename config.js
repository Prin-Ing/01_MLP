// Configuration and constants
export const MAX_VALUE = 255;
export const INPUT = [0, 255];
export const NORMALIZED_INPUT = INPUT.map(x => x / MAX_VALUE);
export const TARGET_DATA = [0, 1];
export const LEARNING_RATE = 0.01;
export const EPOCHS = 1000;

// Initial weights for layer 1 (input: 2, hidden: 5)
export const INITIAL_WEIGHTS = [
  [
    [0.5, 0.5],
    [0.5, 0.5],
    [0.5, 0.5],
    [0.5, 0.5],
    [0.5, 0.5]
  ],
  // Layer 2 (input: 5, output: 2)
  [
    [0.5, 0.5, 0.5, 0.5, 0.5],
    [0.5, 0.5, 0.5, 0.5, 0.5]
  ]
];

// Initial biases
export const INITIAL_BIASES = [
  [0.5, 0.5, 0.5, 0.5, 0.5],
  [0.5, 0.5]
];
