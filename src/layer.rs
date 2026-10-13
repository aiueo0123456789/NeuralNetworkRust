#[derive(Clone, Copy)]
pub enum ActivateFunction {
  None,
  Sigmoid,
  ReLU,
  LeakyReLU,
  Softmax,
}

pub struct Layer {
  pub input_size: usize,
  pub output_size: usize,
  pub weights: Box<[f32]>,
  pub bias: Box<[f32]>,
  pub outputs: Box<[f32]>,
  pub pre_activation: Box<[f32]>,
  pub gradients: Box<[f32]>,
  pub old_weights: Box<[f32]>,
  pub activate_function: ActivateFunction,
}

impl Layer {
  pub fn new(input_size: usize, output_size: usize, activate_function: ActivateFunction) -> Self {
    Self {
      input_size,
      output_size,
      weights: vec![0.0; input_size * output_size].into_boxed_slice(),
      bias: vec![0.0; output_size].into_boxed_slice(),
      outputs: vec![0.0; output_size].into_boxed_slice(),
      pre_activation: vec![0.0; output_size].into_boxed_slice(),
      gradients: vec![0.0; output_size].into_boxed_slice(),
      old_weights: vec![0.0; input_size * output_size].into_boxed_slice(),
      activate_function,
    }
  }
}