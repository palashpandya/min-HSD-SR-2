mod dataio;
mod gilbert;

use std::path::Path;

use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::Deserialize;

use dataio::{ensure_input, load_density_matrix, save_density_matrix, save_distances_csv};

#[derive(Deserialize)]
struct Config {
    input: InputConfig,
    algorithm: AlgorithmConfig,
    output: OutputConfig,
}

#[derive(Deserialize)]
struct InputConfig {
    matrix_path: String,
    #[serde(rename = "N")]
    n: usize,
    d: usize,
}

#[derive(Deserialize)]
struct AlgorithmConfig {
    optimizer: String,
    max_corr: usize,
    max_trials: usize,
    tol: f64,
    seed: u64,
    #[serde(default = "default_preselection")]
    preselection: bool,
    #[serde(default = "default_gradient_steps")]
    gradient_steps: usize,
    #[serde(default = "default_local_sweeps")]
    local_sweeps: usize,
    reference_distance2: Option<f64>,
    #[serde(default = "default_reference_gap")]
    reference_gap: f64,
}

fn default_preselection() -> bool { true }
fn default_gradient_steps() -> usize { 3 }
fn default_local_sweeps() -> usize { 20 }
fn default_reference_gap() -> f64 { 1e-8 }

#[derive(Deserialize)]
struct OutputConfig {
    dir: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cfg_str = std::fs::read_to_string("config.toml")?;
    let cfg: Config = toml::from_str(&cfg_str)?;

    let input_path = &cfg.input.matrix_path;
    let n = cfg.input.n;
    let d = cfg.input.d;
    let optimizer = &cfg.algorithm.optimizer;
    let max_corr = cfg.algorithm.max_corr;
    let max_trials = cfg.algorithm.max_trials;
    let tol = cfg.algorithm.tol;
    let seed = cfg.algorithm.seed;
    let preselection = cfg.algorithm.preselection;
    let gradient_steps = cfg.algorithm.gradient_steps;
    let local_sweeps = cfg.algorithm.local_sweeps;
    let reference_distance2 = cfg.algorithm.reference_distance2;
    let reference_gap = cfg.algorithm.reference_gap;
    let out_dir = &cfg.output.dir;

    let stem = Path::new(input_path)
        .file_stem()
        .unwrap()
        .to_str()
        .unwrap()
        .to_owned();
    let css_path = format!("{out_dir}/{stem}_css.mtx");
    let csv_path = format!("{out_dir}/{stem}_distances.csv");

    ensure_input(input_path, n, d, seed)?;
    let rho0 = load_density_matrix(input_path)?;
    println!("Input: {input_path}  shape={:?}  N={n}  d={d}", rho0.dim());

    let mut rng = StdRng::seed_from_u64(seed);
    let (css, distances) =
        gilbert::gilbert(&rho0, n, d, max_corr, max_trials, optimizer, tol,
                        preselection, gradient_steps, local_sweeps,
                        reference_distance2, reference_gap, &mut rng);

    println!(
        "optimizer={optimizer}  initial dist={:.6}  final dist={:.6}  corrections={}",
        distances[0],
        distances[distances.len() - 1],
        distances.len() - 1
    );

    save_density_matrix(&css, &css_path)?;
    save_distances_csv(&distances, &csv_path)?;

    println!("CSS matrix  → {css_path}");
    println!("Distances   → {csv_path}");

    Ok(())
}
