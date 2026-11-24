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
}

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
        gilbert::gilbert(&rho0, n, d, max_corr, max_trials, optimizer, tol, &mut rng);

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
