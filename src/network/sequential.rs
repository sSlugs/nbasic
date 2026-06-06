use std::fs;

use rand::{RngExt, seq::SliceRandom};

use crate::{layers::{activation::Activate, layer::{Layer, Layerneeds}}, math::matrix::Matrix, network::{Dataset, Loss, Network}};
use serde::{Serialize, Deserialize};

#[derive(Clone)]
#[derive(Serialize, Deserialize)]
pub struct Sequential {
    tag: String,
    pub layers: Vec<Layer>,
    pub loss: Loss,
}

impl Sequential {
    pub fn new(layers: Vec<Layer>,loss:Loss) -> Self {
        Self { tag: "Sequential".to_string(), layers, loss }
    }
}

impl Network for Sequential {
    // hard code instead of using forward since we dont wanna overwrite anything
    fn inference(&self, input:&Matrix) -> Matrix {
        let mut input = input.clone();

        for layer in self.layers.iter() {
            match layer {
                Layer::Dense(d) => {
                    d.weights.matmul(&mut input.clone(), &mut input);
                    input.add(&d.bias);
                }

                Layer::Relu(r) => {
                    r.nactivate(&mut input);
                }

                Layer::Sin(s) => {
                    s.nactivate(&mut input);
                }
            }
        }
        input
    }

    // takes in dl_da which is how cost changes with respect to final output.
    fn backprop(&mut self,dl_da:&mut Matrix,buffer: &mut Matrix) {
        // backward pass keep sending dl_da back and layers will save their gradiants
        for layer in self.layers.iter_mut().rev() {
            match layer {
                Layer::Dense(d) => {
                    d.back(dl_da,buffer);
                }

                Layer::Relu(r) => {
                    r.back(dl_da);
                }

                Layer::Sin(s) => {
                    s.back(dl_da);
                }
            }
        }
    }

    fn train_epochs(&mut self,dat:&Dataset,epochs:i32) {
        // create buffer
        let mut buffer = Matrix::new_empty(1, 1);
        let mut rng = rand::rng();

        // how many times we want to train on dataset
        for _ in 0..epochs {
            // shuffle the data set to avoid memorizing weird patterns
            let mut data = dat.clone();
            let n = data.inputs.len();

            for i in (1..n).rev() {
                let j = rng.random_range(0..=i);

                data.inputs.swap(i, j);
                data.targets.swap(i, j);
            }

            for di in 0..n {
                // forward pass
                for layer in self.layers.iter_mut() {
                    match layer {
                        Layer::Dense(d) => {
                            d.forward(&mut data.inputs[di]);
                        }

                        Layer::Relu(r) => {
                            r.activate(&mut data.inputs[di]);
                        }

                        Layer::Sin(s) => {
                            s.activate(&mut data.inputs[di]);
                        }
                    }
                } 

                // data.inputs[di] is our output vector. since we might aswell reuse
                // calculate dl_da.

                match self.loss {
                    // mean squared error
                    Loss::MSE => {
                        data.inputs[di].sub(&data.targets[di]);
                        data.inputs[di].scale(2.0);
                    }
                }

                self.backprop(&mut data.inputs[di], &mut buffer);
                
                // for now a tight SGD where we apply after every backprop
                for layer in self.layers.iter_mut() {
                    if let Layer::Dense(l) = layer {
                        l.update_grads();
                    }
                }

                // println!("avg loss: {}",self.avg_loss(&dat,Loss::MSE));
            }

        }
    }

    // computes average loss from a dataset
    fn avg_loss(&self,data:&Dataset,loss:Loss) -> f32 {
        match loss {
            Loss::MSE => {
                let mut loss = 0.0;

                // add all losses
                for i in 0..data.inputs.len() {

                    let mut result = self.inference(&data.inputs[i]);

                    // MSE
                    result.sub(&data.targets[i]);
                    result.sqr();

                    // add all output values and add mean to the main loss var
                    let mut temp = 0.0;
                    for val in result.data {
                        temp += val;
                    }

                    loss += temp / result.rows as f32
                }

                // return mean loss
                loss / data.inputs.len() as f32
            }
        }
    }

    fn save_json(&self,name:&str) {
        let data_dir = dirs::data_local_dir().unwrap().join("nbasic");

        // create data directory if its first time saving
        fs::create_dir_all(&data_dir).unwrap();

        // turn network into jason and save
        let data = self.clone();

        let json = serde_json::to_string_pretty(&data).unwrap();

        fs::write(data_dir.join(&format!("{}.json",name)), &json).unwrap();
    }
}
