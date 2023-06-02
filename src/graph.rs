use crate::funcs::{Function, Loss};
use crate::optimizer::Optimizer;
use crate::tensor::*;
use rand::Rng;
use std::collections::{BTreeMap, HashSet};

pub type TensorId = usize;

struct Computation {
    inps: Vec<TensorId>,
    out: TensorId,
    func: Box<dyn Function>,
}

pub struct Graph {
    tensors: BTreeMap<TensorId, Tensor<f32>>,
    grads: BTreeMap<TensorId, Tensor<f32>>,
    computations: Vec<Computation>,
}

impl Graph {
    pub fn new() -> Self {
        Self {
            tensors: Default::default(),
            grads: Default::default(),
            computations: Default::default(),
        }
    }
    pub fn alloc_rand<R: Rng>(&mut self, rng: &mut R, shape: &[usize]) -> TensorId {
        self.alloc(Tensor::<f32>::rand(rng, shape))
    }
    fn alloc(&mut self, t: Tensor<f32>) -> TensorId {
        let id = self.tensors.len();
        self.tensors.insert(id, t);
        id
    }
    pub fn load<T: TensorOps<f32>>(&mut self, tensor_id: TensorId, tensor: &T) {
        self.tensors.insert(tensor_id, tensor.view().into());
    }
    pub fn zero_grad(&mut self) {
        self.grads.clear();
    }
    pub fn add_grad<T: TensorOps<f32>>(&mut self, id: TensorId, add: T) {
        let mut shape = self.get(id).shape().to_vec();
        let grad = self.grads.entry(id).or_insert(Tensor::zeros(&shape));
        if add.dim() >= shape.len() {
            shape.insert(0, 0);
            for t in add.reshape(&shape).inners().iter() {
                *grad = &*grad + t;
            }
        } else {
            *grad = &*grad + &add.view();
        }
    }
    pub fn get(&self, id: TensorId) -> &Tensor<f32> {
        self.tensors.get(&id).expect("Tensor not found!")
    }
    pub fn backward(&mut self, id: TensorId) {
        if let Some(comp) = self.computations.iter().find(|c| c.out == id) {
            for inp in comp.inps.iter() {
                let shape = self.get(*inp).shape().to_vec();
                self.grads
                    .entry(*inp)
                    .or_insert(Tensor::<f32>::zeros(&shape));
            }
            let inps = comp
                .inps
                .iter()
                .map(|id| &self.tensors[id])
                .collect::<Vec<_>>();
            let result_out = &self.tensors[&id];
            let grad_out = &self.grads[&id];
            let grads = comp.func.grad(&inps, result_out, grad_out);
            for (id, grad) in comp.inps.clone().into_iter().zip(grads.into_iter()) {
                self.add_grad(id, grad);
            }
        }
    }
    pub fn backward_all(&mut self, id: TensorId, loss_fn: Box<dyn Loss>) -> f32 {
        let output = self.get(id);
        let (loss, grad) = loss_fn.run(output);
        let mean_coeff = 1. / loss.size() as f32;
        self.add_grad(id, &grad * &Tensor::scalar(mean_coeff));

        let backward_order = self
            .computations
            .iter()
            .map(|c| c.out)
            .rev()
            .collect::<Vec<_>>();
        for id in backward_order {
            self.backward(id);
        }

        loss.mean()
    }
    pub fn forward(&mut self, training: bool) {
        for c in self.computations.iter_mut() {
            let tensors = c
                .inps
                .iter()
                .map(|id| self.tensors.get(id).expect("Tensor not found!"))
                .collect::<Vec<_>>();
            let result = c.func.run(&tensors, training);
            self.tensors.insert(c.out, result);
        }
    }
    pub fn call(&mut self, mut f: Box<dyn Function>, tensor_ids: &[TensorId]) -> TensorId {
        let tensors = tensor_ids
            .iter()
            .map(|id| self.tensors.get(id).expect("Tensor not found!"))
            .collect::<Vec<_>>();
        let out = f.run(&tensors, false);
        let child = self.alloc(out);
        self.computations.push(Computation {
            func: f,
            out: child,
            inps: tensor_ids.to_vec(),
        });
        child
    }
    pub fn optimize<O: Optimizer>(&mut self, opt: &mut O, params: &HashSet<TensorId>) {
        let (params, grads): (Vec<&mut Tensor<f32>>, Vec<&Tensor<f32>>) = self
            .tensors
            .iter_mut()
            .filter(|(id, _)| params.contains(id))
            .map(|(id, params)| {
                let grad = self.grads.get(id).expect("Tensor not found!");
                (params, grad)
            })
            .collect::<Vec<_>>()
            .into_iter()
            .unzip();
        opt.step(params, grads);
    }
}
