use rand::random_range;
use serde::{Deserialize, Serialize};

use crate::{layers::layer::Layerneeds, math::matrix::Matrix, network::Adam};


// normal network layer of neurons with weights and bias
#[derive(Clone,Debug)]
#[derive(Serialize, Deserialize)]
pub struct Dense {
    // matrix of all weights that connect to input.
    pub weights: Matrix,

    // bias of all neurons
    pub bias: Matrix,

    // the inputs given before weights and bias
    input_cache: Matrix,

    // gradiant caches
    pub gw: Matrix,
    pub gb: Matrix,

    adam_w: Adam,
    adam_b: Adam,
}

// Dense implementation
impl Layerneeds for Dense {
    /// input_neurons is the number of neurons in the previous layer (or the expected input if
    /// first layer). neurons is the amount of neurons in the layer
    fn new(input_neurons:usize,neurons:usize) -> Self {
        let mut weights = Matrix::new_empty(neurons, input_neurons);
        let bias = Matrix::new_empty(neurons, 1);

        // initilize weights
        for row in 0..neurons {
            for col in 0..input_neurons {
                let value = random_range(-1.0..1.0);
                weights.set(value, row, col);
            }
        }

        Dense { 
            weights,
            bias,
            gw: Matrix::new_empty(neurons,input_neurons),
            gb: Matrix::new_empty(neurons,1),
            input_cache:Matrix::new_empty(input_neurons, 1),
            adam_w: Adam::new(neurons, input_neurons, 1e-4),
            adam_b: Adam::new(neurons, 1, 1e-4),
        }        
    }

    /// takes input returns it weighted + biased, updates cache
    #[inline(always)]
    fn forward(&mut self,mut input: &mut Matrix) {
        self.input_cache = input.clone();

        self.weights.matmul(&self.input_cache, &mut input);
        input.add(&self.bias);
    }

    /// takes in previous dl_da, and 1 mutable refrance to overwrite for the gradiants tempararily
    /// buffer must be size of weights in layer
    #[inline(always)]
    fn back(&mut self,dl_da:&mut Matrix,buffer:&mut Matrix) {

        // correct buffer
        buffer.resize(self.weights.rows, self.weights.cols, 0.0);

        // compute gradiants for output with respect to weights. this can be done by taking do/dw
        // which = the input * how our output effects loss which we already have. we must transpose
        // so rxc cxr
        dl_da.matmul_factor_transposed(&self.input_cache,buffer);
        self.gw.add(&buffer);

        // since bias  gradiant is literally just dl_da accumalted to the cache
        self.gb.add(&dl_da);

        // get the next dl_da so that we can continue backprop. ask ourselves how do each of the
        // input neurons change our loss? this is simply dl/di = dl_da * da/di. a can be thought of
        // as what our layer outputs after weights and bias. and i is what our input was before
        // weights and bias, or what the output was of the previous layer, which is our dl_da from
        // the layer above (previous iteration in backprop).
        // dl/di simply equals the weight of our layer, so if we can get dot product of how each
        // inputs weight effects the loss then we have solved our problem.
        self.weights.transposed_matmul(&dl_da,buffer);

        dl_da.data = buffer.data.clone();
        dl_da.rows = buffer.rows;
        dl_da.cols = buffer.cols;
    }


    #[inline(always)]
    fn update_grads(&mut self) {
        // scale by as adam
        self.adam_w.update(&mut self.weights, &self.gw);
        self.adam_b.update(&mut self.bias, &self.gb);
        
        
        // clear caches
        self.gw.data.fill(0.0);
        self.gb.data.fill(0.0);
    }
}

