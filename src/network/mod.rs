use std::fs;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{math::matrix::Matrix, network::sequential::Sequential};

pub mod sequential;

#[derive(Clone)]
pub struct Dataset {
    pub inputs: Vec<Matrix>,
    pub targets: Vec<Matrix>
}

impl Dataset {
    pub fn from_function<F>(f: F,samples: usize,min: f32,max: f32,) -> Self
    where F: Fn(f32) -> f32 {
        
        let mut inputs = Vec::with_capacity(samples);
        let mut targets = Vec::with_capacity(samples);

        let step = (max - min) / samples as f32;

        for i in 0..samples {
            let x = min + i as f32 * step;
            let y = f(x);

            inputs.push(Matrix::from(1,1,vec![x]));   
            targets.push(Matrix::from(1,1,vec![y]));
        }

        Dataset { inputs, targets }
    }
}

#[derive(Clone)]
#[derive(Serialize,Deserialize)]
pub enum Loss {
    MSE
}

pub trait Network {
    /// returns prediction from given input
    fn inference(&self, input:&Matrix) -> Matrix;

    /// runs just backprop NO forward pass, assumes forward pass has been done.
    fn backprop(&mut self,dl_da:&mut Matrix,buffer: &mut Matrix);

    /// takes in a vector of data and returns the current loss based of data.
    fn avg_loss(&self,data:&Dataset,loss:Loss) -> f32;

    /// trains over dataset epoch times.
    fn train_epochs(&mut self,data:&Dataset,epochs:i32);

    /// saves network to file accesable with given ID
    /// (Overwrites the file be careful.)
    fn save_json(&self,name:&str);
}

/// loads network from nbasic local dir.
pub fn load(name:&str) -> Sequential {
    let data_dir = dirs::data_local_dir().unwrap().join("nbasic");

    let content = fs::read_to_string(data_dir.join(&format!("{}.json",name))).expect("Failed to read file");

    let v: Value = serde_json::from_str(&content).unwrap();
    let tag = &v["tag"];
    match tag.as_str().unwrap() {
        "Sequential" => {
            let data: Sequential = serde_json::from_str(&content).expect("Invalid JSON format");
            return data; 
        }

        _ => panic!("found unknown tag while loading network")
    }
}

/// lists saved network from nbasic local dir.
pub fn list_saved() {
    
}

// will be path buf later dw
pub fn load_json(path:&str) {

}

// ADAM
#[derive(Clone,Debug)]
#[derive(Serialize,Deserialize)]
pub struct Adam {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    t: usize,

    m: Matrix,
    v: Matrix,
}

impl Adam {
    pub fn new(rows: usize, cols: usize, lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.99,
            eps: 1e-8,
            t: 0,
            m: Matrix::new_empty(rows, cols),
            v: Matrix::new_empty(rows, cols),
        }
    }
}

impl Adam {
    pub fn update(&mut self, w: &mut Matrix, grad: &Matrix) {
        self.t += 1;

        for i in 0..w.data.len() {
            let g = grad.data[i];

            // 1. update momentum
            self.m.data[i] = self.beta1 * self.m.data[i] + (1.0 - self.beta1) * g;

            // 2. update variance
            self.v.data[i] = self.beta2 * self.v.data[i] + (1.0 - self.beta2) * g * g;

            // 3. bias correction
            let m_hat = self.m.data[i] / (1.0 - self.beta1.powi(self.t as i32));
            let v_hat = self.v.data[i] / (1.0 - self.beta2.powi(self.t as i32));

            // 4. update weight
            w.data[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
        }
    }
}
