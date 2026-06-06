use serde::{Deserialize, Serialize};

use crate::{layers::{activation::Activate, dense::Dense}, math::matrix::Matrix, network::Adam};

// traits all layer types should have

pub trait Layerneeds {
    fn forward(&mut self,input:&mut Matrix);
    fn back(&mut self,dl_da:&mut Matrix,buffer:&mut Matrix);
    fn new(neurons:usize,input_neurons:usize) -> Self;
    fn update_grads(&mut self);
}

// activation structs. (in layer for api i think it looks better);

#[derive(Clone,Debug)]
#[derive(Serialize, Deserialize)]
pub struct Relu(pub Vec<f32>);
#[derive(Serialize, Deserialize)]
#[derive(Clone,Debug)]
pub struct Sin(pub Vec<f32>);

pub enum Init {
    Random,
    Xavier(usize,usize),
    He(usize),
}

#[derive(Clone,Debug)]
#[derive(Serialize, Deserialize)]
pub enum Layer {
    Dense(Dense),
    Relu(Relu),
    Sin(Sin),
}

impl Layer {
    pub fn dense(input_neurons:usize,neurons_in_layer:usize) -> Layer {
        Layer::Dense(Dense::new(input_neurons, neurons_in_layer))
    }

    pub fn relu() -> Layer {
        Layer::Relu(Relu::new())
    }

    pub fn sin() -> Layer {
        Layer::Sin(Sin::new())
    }
}

