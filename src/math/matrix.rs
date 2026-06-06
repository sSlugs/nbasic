use rand::random_range;
use serde::{Deserialize, Serialize};


// the matrix structure, can do matrix math. very awesome. uses row major indexing
#[derive(Clone,Debug)]
#[derive(Serialize, Deserialize)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Matrix {
    /// multiplies self * factor. we use buffer to save memory, note buffer must have same rows as
    /// self and columns of factor.
    #[inline(always)]
    pub fn matmul(&self,factor:&Matrix,buffer: &mut Matrix) {
        // this should probably be optimized in future
        
        buffer.resize(self.rows, factor.cols, 0.0);

        for r in 0..self.rows {
            // for each row compute dot product with each column of factor
            for c in 0..factor.cols {
                let mut dot = 0.0;
                // gets dot product by summing each value in self row with respective value in factor col
                for i in 0..self.cols {
                    dot += self.get(r, i)*factor.get(i,c);
                }

                buffer.set(dot, r, c); 
            }
        }
    }

    // same as matmul but self is treated as transposed
    #[inline(always)]
    pub fn transposed_matmul(&self, factor: &Matrix, buffer: &mut Matrix) {

        buffer.resize(self.cols, factor.cols, 0.0);

        for r in 0..self.cols {
            for c in 0..factor.cols {
                let mut dot = 0.0;

                let self_col = r; 
                for i in 0..self.rows {
                    dot += self.data[i * self.cols + self_col] * factor.data[i * factor.cols + c];
                }

                buffer.data[r * factor.cols + c] = dot;
            }
        }
    }
    
    // same as matmul but the factor is treated as transposed
    #[inline(always)]
    pub fn matmul_factor_transposed(&self, factor: &Matrix, buffer: &mut Matrix) {
        for r in 0..self.rows {
            for c in 0..factor.rows { 
                let mut dot = 0.0;

                for i in 0..self.cols {
                    dot += self.get(r, i) * factor.get(c, i);
                }

                buffer.set(dot, r, c);
            }
        }
    }

    //scale matrix element-wise by given f32
    #[inline(always)]
    pub fn scale(&mut self,scale:f32) {
        for value in self.data.iter_mut() {
            *value *= scale;
        }
    }

    // element-wise division
    #[inline(always)]
    pub fn div(&mut self,div:f32) {
        for value in self.data.iter_mut() {
            *value /= div;
        }
    }
   
    // element-wise squaring
    #[inline(always)]
    pub fn sqr(&mut self) {
        for value in self.data.iter_mut() {
            *value *= *value;
        }
    }

    // returns sum of all elements
    #[inline(always)]
    pub fn sum(&self) -> f32 {
        let mut sum = 0.0;
        for value in self.data.iter() {
            sum += *value;
        }
        sum
    }

    /// adds two matrices element-wise, self is mutated
    #[inline(always)]
    pub fn add(&mut self,term: &Matrix) {
        for i in 0..self.data.len() {
            self.data[i] += term.data[i];
        }
    }

    /// subs two matrices element-wise, self is mutated
    #[inline(always)]
    pub fn sub(&mut self,term:&Matrix) {
        for i in 0..self.data.len() {
            self.data[i] -= term.data[i];
        }
    }
}

impl Matrix {
    // returns value at given row and column. will panic if OOB.
    #[inline(always)]
    pub fn get(&self,r:usize,c:usize) -> f32 {
        self.data[r * self.cols + c]
    }

    // sets value at given row and column. will panic if OOB.
    #[inline(always)]
    pub fn set(&mut self,value:f32,r:usize,c:usize) {
        self.data[r * self.cols + c] = value;
    }

    pub fn print(&self) {
        for r in 0..self.rows {
            for c in 0..self.cols {
                print!("{} ",self.get(r, c))
            }
            println!();
        }
    }
}

impl Matrix {
    // from given rows and cols and data create a new matrix
    pub fn from(rows:usize,cols:usize,data:Vec<f32>) -> Self {
        Self { rows, cols, data } 
    }

    // creates new matrix and fills with random values ranging from -1..=1
    pub fn new_random(rows:usize,cols:usize) -> Self {
        let total = rows*cols;
        let mut data = Vec::with_capacity(total);
        for _ in 0..(total) {
            data.push(random_range(-1.0..1.0));  
        }
        Self { rows, cols, data }
    }
    
    // creates new matrix and fills with 0.0
    pub fn new_empty(rows:usize,cols:usize) -> Self {
        let total = rows*cols;
        let mut data = Vec::with_capacity(total);
        for _ in 0..(total) {
            data.push(0.0);  
        }
        Self { rows, cols, data }
    }

    /// changes size of matrix and fills new with given value.
    #[inline(always)]
    pub fn resize(&mut self,rows:usize,cols:usize,fill:f32) {
        self.rows = rows;
        self.cols = cols;

        let needed = self.rows * self.cols;

        if self.data.len() < needed {
            self.data.resize(needed, fill); // grows only when needed
        } else {
            self.data.truncate(needed);
        }
    }
}
