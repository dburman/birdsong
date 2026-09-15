//! Spike: can tract-onnx load, optimise and run an ONNX model? Prints I/O facts and timing.
use std::time::Instant;
use tract_onnx::prelude::*;

fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .expect("usage: load_onnx <model.onnx> [input dims comma-separated]");
    let dims: Option<Vec<usize>> = std::env::args()
        .nth(2)
        .map(|s| s.split(',').map(|d| d.parse().unwrap()).collect());
    let t0 = Instant::now();
    let mut model = tract_onnx::onnx().model_for_path(&path)?;
    println!(
        "loaded in {:?}: {} nodes",
        t0.elapsed(),
        model.nodes().len()
    );
    for i in 0..model.inputs.len() {
        println!("  input {i}: {:?}", model.input_fact(i)?);
    }
    if let Some(d) = &dims {
        model.set_input_fact(0, f32::fact(d.as_slice()).into())?;
    }
    let mut ops = std::collections::BTreeMap::new();
    for n in model.nodes() {
        *ops.entry(n.op.name().to_string()).or_insert(0usize) += 1;
    }
    println!("ops: {ops:?}");
    let t1 = Instant::now();
    let model = model.into_optimized()?;
    println!(
        "optimized in {:?}: {} nodes",
        t1.elapsed(),
        model.nodes().len()
    );
    let runnable = model.into_runnable()?;
    println!("runnable ok");
    if let Some(d) = dims {
        let n: usize = d.iter().product();
        let input = Tensor::from_shape(&d, &vec![0.1f32; n])?;
        let _ = runnable.run(tvec!(input.clone().into()))?;
        let iters = 5;
        let t2 = Instant::now();
        let mut out = None;
        for _ in 0..iters {
            out = Some(runnable.run(tvec!(input.clone().into()))?);
        }
        let per = t2.elapsed() / iters;
        let out = out.unwrap();
        let t: &Tensor = &out[0];
        let v = t.try_as_plain()?.as_slice::<f32>()?;
        println!(
            "ran {iters}x: {per:?} per run; output shape {:?}; first values {:?}",
            t.shape(),
            &v[..4]
        );
    }
    Ok(())
}
