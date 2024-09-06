use femto_gpt::gpt::{TrainingState, GPT};
use femto_gpt::graph::GraphError;
use femto_gpt::tokenizer::{SimpleTokenizer, Tokenizer};
use std::fs;
use std::io::prelude::*;
use std::path::Path;

fn main() -> Result<(), GraphError> {
    #[cfg(not(feature = "gpu"))]
    let graph = femto_gpt::graph::CpuGraph::new();
    #[cfg(not(feature = "gpu"))]
    let is_gpu = false;

    #[cfg(feature = "gpu")]
    let graph = femto_gpt::graph::gpu::GpuGraph::new()?;
    #[cfg(feature = "gpu")]
    let is_gpu = true;

    let training_state_path = Path::new("training_state.dat");

    let mut rng = rand::thread_rng();

    // Create a unique char-to-int mapping for all unique characters inside our dataset
    let dataset_char =
        fs::read_to_string("dataset.txt").expect("Should have been able to read the file");
    let tokenizer = SimpleTokenizer::new(&dataset_char);

    let vocab_size = tokenizer.vocab_size();
    let batch_size = 32;
    let num_tokens = 64;
    let embedding_degree = 64;
    let num_layers = 4;
    let num_heads = 4;
    let head_size = embedding_degree / num_heads;
    let dropout = 0.0;

    assert_eq!(num_heads * head_size, embedding_degree);

    println!("Vocab-size: {} unique characters", vocab_size);

    let mut gpt = GPT::new(
        &mut rng,
        graph,
        is_gpu.then(|| batch_size), // Pre-allocate batches only when using GPUs
        vocab_size,
        embedding_degree,
        num_tokens,
        num_layers,
        num_heads,
        head_size,
        dropout,
    )?;

    gpt.sync()?;

    println!("Number of parameters: {}", gpt.num_params());

    // Load training data from train_data directory (If exists)
    if training_state_path.is_file() {
        let mut ts_file = fs::File::open(training_state_path).unwrap();
        let mut bytes = Vec::new();
        ts_file.read_to_end(&mut bytes).unwrap();
        let ts: TrainingState = bincode::deserialize(&bytes).unwrap();
        gpt.set_training_state(ts, true)?;
    }

    let inference_temperature = 0.5; // How creative? 0.0 min 1.0 max

    let prompt = "The description in this text is about the company named";
    let inference = gpt.infer(
        &mut rng,
        &tokenizer.tokenize(prompt),
        100,
        inference_temperature,
        |_ch| {},
    )?;

    println!("{}", tokenizer.untokenize(&inference));

    Ok(())
}