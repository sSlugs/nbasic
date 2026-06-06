use crate::{layers::layer::{Relu, Sin}, math::matrix::Matrix};

pub trait Activate {
    /// element-wise activation via function
    fn activate(&mut self, input:&mut Matrix);

    /// same as activate but we dont write cache;
    fn nactivate(&self,input:&mut Matrix);

    // element-wise multiplication by derivative of function
    fn back(&self, input: &mut Matrix);

    // makes new instance
    fn new() -> Self;
}

impl Activate for Relu {
    fn activate(&mut self, input: &mut Matrix) {
        self.0.clear();
        for value in input.data.iter_mut() {
            if *value < 0.0 {
                *value = 0.0;
            }
            self.0.push(*value);
        }
    }

    fn nactivate(&self, input: &mut Matrix) {
        for value in input.data.iter_mut() {
            if *value < 0.0 {
                *value = 0.0;
            }
        }
    }

    fn back(&self, dl_da: &mut Matrix) {
        for (grad, a) in dl_da.data.iter_mut().zip(self.0.iter()) {
            if *a <= 0.0 {
                *grad = 0.0;
            } 
        }
    }

    fn new() -> Self {
        Relu(Vec::new())
    }
}

impl Activate for Sin {
    fn activate(&mut self, input:&mut Matrix) {
        self.0.clear();
        for value in input.data.iter_mut() {
            *value = value.sin();
            self.0.push(*value);
        }
    }

    fn nactivate(&self,input:&mut Matrix) {
        for value in input.data.iter_mut() {
            *value = value.sin();
        }
    }

    fn back(&self, dl_da: &mut Matrix) {
        for (grad, a) in dl_da.data.iter_mut().zip(self.0.iter()) {
            *grad *= a.cos();
        }
    }
    
    fn new() -> Self {
        Sin(Vec::new())
    }
}
