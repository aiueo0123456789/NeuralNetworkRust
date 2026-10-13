use crate::layer::Layer;
use crate::layer::ActivateFunction;
use rand::distr::{Distribution, Uniform};

pub struct NeuralNetwork {
  layers: Box<[Layer]>,
}


fn sigmoid(x: f32) -> f32 {
  return 1.0 / (1.0 + (-x).exp());
}
fn relu(x: f32) -> f32 {
  return x.max(0.0);
}
fn leaky_relu(x: f32) -> f32 {
  if x > 0.0 {x}
  else {0.01 * x}
}
fn sigmoid_derivative(x: f32) -> f32 {
  let a = 1.0 / (1.0 + (-x).exp());
  return a * (1.0 - a);
}
fn relu_derivative(x: f32) -> f32 {
  if x > 0.0 {1.0}
  else {0.0}
}
fn leaky_relu_derivative(x: f32) -> f32 {
  if x > 0.0 {1.0}
  else {0.01}
}

impl NeuralNetwork {
  pub fn new(layers: Vec<Layer>) -> Self {
    Self { layers: layers.into_boxed_slice() }
  }

  fn layer_size(&self) -> usize {
    return self.layers.len();
  }

  pub fn init_parmater(&mut self) {
    let mut rng = rand::rng();
    let between = Uniform::new(-0.5, 0.5).unwrap();

    for layer in &mut self.layers[1..] {
      for weight in layer.weights.iter_mut() { // .iter_mut()は要素をループないで&mutと宣言する
        *weight = between.sample(&mut rng);
      }
      for bias in layer.bias.iter_mut() {
        *bias = between.sample(&mut rng);
      }
    }
  }

  pub fn output(&self) -> Box<[f32]> {
    return self.layers[self.layer_size() - 1].outputs.clone();
  }

  pub fn forward(&mut self, inputs: &[f32]) { // selfで自身を変更可能に
    assert_eq!(
      inputs.len(),
      self.layers[0].output_size,
      "入力数が一致していません"
    );

    for neuron_index in 0..self.layers[0].output_size {
      self.layers[0].outputs[neuron_index] = inputs[neuron_index];
    }
    for layer_index in 1..self.layer_size() {
      let (previous_layers, current_layers) = self.layers.split_at_mut(layer_index); // lastとcurrentが違うものの証明のためにlayersをlayer_indexで区切って2つの配列に分ける

      let befor_layer = &previous_layers[layer_index - 1];
      let layer = &mut current_layers[0];

      for neuron_index in 0..layer.output_size {
        // ニューロンの値を計算
        let base_weight_index = neuron_index * layer.input_size;
        let mut sum_value: f32 = layer.bias[neuron_index];
        for input_neuron_index in 0..layer.input_size {
          sum_value += befor_layer.outputs[input_neuron_index] * layer.weights[base_weight_index + input_neuron_index];
        }
        layer.pre_activation[neuron_index] = sum_value;
        layer.outputs[neuron_index] = match layer.activate_function {
          ActivateFunction::None => sum_value,
          ActivateFunction::Sigmoid => sigmoid(sum_value),
          ActivateFunction::ReLU => relu(sum_value),
          ActivateFunction::LeakyReLU => leaky_relu(sum_value),
          ActivateFunction::Softmax => {panic!("Softmaxはレイヤー全体で計算する必要があります")}
        };
      }
    }
  }

  pub fn backward(&mut self, cost_gradients: &[f32], learning_rate: f32) {
    assert_eq!(
      cost_gradients.len(),
      self.layers[self.layer_size() - 1].output_size,
      "入力数が一致していません"
    );

    { // 出力層の学習
      let output_index = self.layer_size() - 1;
      let (previous_layers, output_layers) = self.layers.split_at_mut(output_index);
      let befor_layer = &previous_layers[output_index - 1];
      let layer = &mut output_layers[0];

      for neuron_index in 0..layer.output_size {
        // ニューロンの値を計算
        let mut node_gradient: f32 = cost_gradients[neuron_index]; // 自分が出力に与える影響

        let activation_derivative = match layer.activate_function {
          ActivateFunction::None => 1.0,
          ActivateFunction::Sigmoid => {sigmoid_derivative(layer.pre_activation[neuron_index])}
          ActivateFunction::ReLU => {relu_derivative(layer.pre_activation[neuron_index])}
          ActivateFunction::LeakyReLU => {leaky_relu_derivative(layer.pre_activation[neuron_index])}
          ActivateFunction::Softmax => {panic!("Softmaxの微分は別途実装が必要です")}
        };
        node_gradient *= activation_derivative;

        let step_size = node_gradient * learning_rate;

        let base_weight_index = neuron_index * layer.input_size;
        for input_neuron_index in 0..layer.input_size {
          layer.old_weights[base_weight_index + input_neuron_index] = layer.weights[base_weight_index + input_neuron_index];
          layer.weights[base_weight_index + input_neuron_index] += befor_layer.outputs[input_neuron_index] * step_size;
        }
        layer.bias[neuron_index] += step_size;

        layer.gradients[neuron_index] = node_gradient;
      }
    }

    for layer_index in (1..self.layer_size() - 1).rev() { // 出力層と入力層以外を後ろから学習
      let (previous_layers, current_and_next_layers) = self.layers.split_at_mut(layer_index);
      let befor_layer = &previous_layers[layer_index - 1];
      let (layer_slice, next_layers) = current_and_next_layers.split_at_mut(1);
      let layer = &mut layer_slice[0];
      let next_layer = &next_layers[0];

      for neuron_index in 0..layer.output_size {
        // ニューロンの値を計算
        let mut node_gradient: f32 = 0.0; // 自分が出力に与える影響

        for output_neuron_index in 0..next_layer.output_size {
          node_gradient += next_layer.old_weights[output_neuron_index * next_layer.input_size + neuron_index] * // 自身に対する重みのindexは相手のindex * 相手のinputの想定 + 自分nのindex
                            next_layer.gradients[output_neuron_index]; // 相手の出力に対する影響度をかける
        }

        let activation_derivative = match layer.activate_function {
          ActivateFunction::None => 1.0,
          ActivateFunction::Sigmoid => {sigmoid_derivative(layer.pre_activation[neuron_index])}
          ActivateFunction::ReLU => {relu_derivative(layer.pre_activation[neuron_index])}
          ActivateFunction::LeakyReLU => {leaky_relu_derivative(layer.pre_activation[neuron_index])}
          ActivateFunction::Softmax => {panic!("Softmaxの微分は別途実装が必要です")}
        };
        node_gradient *= activation_derivative;

        let step_size = node_gradient * learning_rate;

        let base_weight_index = neuron_index * layer.input_size;
        for input_neuron_index in 0..layer.input_size {
          layer.old_weights[base_weight_index + input_neuron_index] = layer.weights[base_weight_index + input_neuron_index];
          layer.weights[base_weight_index + input_neuron_index] += befor_layer.outputs[input_neuron_index] * step_size;
        }
        layer.bias[neuron_index] += step_size;

        layer.gradients[neuron_index] = node_gradient;
      }
    }
  }
}