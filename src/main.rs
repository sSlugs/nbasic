use nbasic::{layers::{self, activation::Activate, layer::{Init, Layer, Layerneeds, Relu, Sin}}, math::matrix::Matrix, network::{Dataset, Loss, Network, load, sequential::Sequential}};

fn main() {

    // data
    let data = Dataset::from_function(
        |x| x * x,
        1000,
        -10.0,
        10.0,
    );   
    
    let mut network = load("test 1");

    let epochs = 1000;

    println!("average loss: {}\n",network.avg_loss(&mut data.clone(), Loss::MSE));
    network.train_epochs(&data.clone(), epochs);
    println!("trained for {} epochs",epochs);
    println!("\naverage loss: {}",network.avg_loss(&mut data.clone(), Loss::MSE));

    network.save_json("test 1");
    // average loss: 0.0010765417 this was saved in test 1
}
