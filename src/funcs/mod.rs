#[cfg(feature = "gpu")]
use crate::graph::TensorId;

#[cfg(feature = "gpu")]
mod gpu;

#[cfg(feature = "gpu")]
pub use gpu::{GpuFunction, GpuFunctionGroup};

mod add;
mod cat;
mod coeff;
mod crossentropy;
mod dropout;
mod gelu;
mod layer_norm;
mod mask;
mod matmul;
mod relu;
mod softmax;
mod transpose;

pub use add::*;
pub use cat::*;
pub use coeff::*;
pub use crossentropy::*;
pub use dropout::*;
pub use gelu::*;
pub use layer_norm::*;
pub use mask::*;
pub use matmul::*;
pub use relu::*;
pub use softmax::*;
pub use transpose::*;

use super::tensor::*;

pub trait Function: std::fmt::Debug {
    fn clone_box(&self) -> Box<dyn Function>;
    fn run(&mut self, inps: &[&Tensor<f32>], training: bool) -> Result<Tensor<f32>, TensorError>;
    fn grad(
        &self,
        inps: &[&Tensor<f32>],
        out_grad: &Tensor<f32>,
    ) -> Result<Vec<Tensor<f32>>, TensorError>;

    #[cfg(feature = "gpu")]
    fn gpu_run(&self, _out_id: TensorId, _inp_shapes: &[Vec<usize>]) -> GpuFunction {
        unimplemented!()
    }

    #[cfg(feature = "gpu")]
    fn gpu_grad(&self, _out_id: TensorId, _inp_shapes: &[Vec<usize>]) -> GpuFunctionGroup {
        unimplemented!()
    }
}

pub trait Loss: std::fmt::Debug {
    fn run(&self, inp: &Tensor<f32>) -> Result<(Tensor<f32>, Tensor<f32>), TensorError>;
}
