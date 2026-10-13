mod layer;
mod neural_network;

use layer::Layer;
use layer::ActivateFunction;
use neural_network::NeuralNetwork;
use rand::distr::{Distribution, Uniform};

fn main() {
  println!("Hello, world!");
  const input_size: usize = 10 * 10;
  const output_size: usize = 5;
  let mut net = NeuralNetwork::new(vec![
    Layer::new(0, input_size, ActivateFunction::None),
    Layer::new(input_size, 100, ActivateFunction::LeakyReLU),
    Layer::new(100, 8, ActivateFunction::LeakyReLU),
    Layer::new(8, output_size, ActivateFunction::None),
  ]);
  net.init_parmater();

  let mut test_input = [0.0; input_size];
  let mut rng = rand::rng();
  let between = Uniform::new(-1.0, 1.0).unwrap();
  for value in test_input.iter_mut() {
    *value = between.sample(&mut rng);
  }

  let mut cost_gradients = [0.0; output_size];
  net.forward(&test_input);
  println!("{:?}", net.output());

  for learning_count in 0..1000 {
    net.forward(&test_input);
    let output = net.output();
    for output_index in 0..output_size {
      cost_gradients[output_index] = 2.0 * (1.0 - output[output_index]);
    }
    net.backward(&cost_gradients, 0.001);
  }
  println!("{:?}", net.output());
}
