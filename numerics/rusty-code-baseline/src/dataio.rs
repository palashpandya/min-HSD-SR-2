use std::fs;
use std::io::Write;
use std::path::Path;

use num_complex::Complex64;
// use rand::SeedableRng;
// use rand::rngs::StdRng;

use crate::gilbert::{Mat};//, random_density_matrix};

pub fn load_density_matrix(path: &str) -> Result<Mat, Box<dyn std::error::Error>> {
    let content = fs::read_to_string(path)?;
    let mut lines = content.lines().peekable();

    let header = lines.next().ok_or("empty file")?.to_lowercase();
    let is_complex = header.contains("complex");
    let is_array = header.contains("array");
    let is_symmetric = header.contains("symmetric");

    while lines.peek().map(|l| l.starts_with('%')).unwrap_or(false) {
        lines.next();
    }

    let dim_line = lines.next().ok_or("missing dimensions")?;
    println!("Dimensions line: {dim_line}");
    let dims: Vec<usize> = dim_line
        .split_whitespace()
        .map(|s| s.parse::<usize>().unwrap())
        .collect();
    let (rows, cols) = (dims[0], dims[1]);
    println!("Matrix dimensions: {rows} x {cols}");

    let mut mat = ndarray::Array2::<Complex64>::zeros((rows, cols));

    if is_array && is_symmetric {
        // Dense, column-major
        for j in 0..cols {
            for i in j..rows {
                println!("Reading entry ({i}, {j})");
                let line = lines.next().ok_or("missing data, dense")?;
                let vals: Vec<f64> = line
                    .split_whitespace()
                    .map(|s| s.parse::<f64>().unwrap())
                    .collect();
                mat[(i, j)] = if is_complex {
                    Complex64::new(vals[0], vals[1])
                } else {
                    Complex64::new(vals[0], 0.0)
                };
                mat[(j, i)] = if is_complex {
                    Complex64::new(vals[0], vals[1])
                } else {
                    Complex64::new(vals[0], 0.0)
                };
            }
        }
    } else {
        // Coordinate (sparse), 1-indexed
        let nnz = dims[2];
        for _ in 0..nnz {
            let line = lines.next().ok_or("missing data, coordinate")?;
            let vals: Vec<&str> = line.split_whitespace().collect();
            let r: usize = vals[0].parse::<usize>()? - 1;
            let c: usize = vals[1].parse::<usize>()? - 1;
            mat[(r, c)] = if is_complex {
                Complex64::new(vals[2].parse()?, vals[3].parse()?)
            } else {
                Complex64::new(vals[2].parse()?, 0.0)
            };
        }
    }

    Ok(mat)
}

pub fn save_density_matrix(mat: &Mat, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let p = Path::new(path);
    if let Some(parent) = p.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = fs::File::create(p)?;
    let (rows, cols) = mat.dim();
    writeln!(file, "%%MatrixMarket matrix array complex general")?;
    writeln!(file, "{rows} {cols}")?;
    // Column-major order to match scipy mmwrite
    for j in 0..cols {
        for i in 0..rows {
            let v = mat[(i, j)];
            writeln!(file, "{:.15e} {:.15e}", v.re, v.im)?;
        }
    }
    Ok(())
}

pub fn save_distances_csv(distances: &[f64], path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let p = Path::new(path);
    if let Some(parent) = p.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut wtr = csv::Writer::from_path(p)?;
    wtr.write_record(["step", "distance"])?;
    for (i, &d) in distances.iter().enumerate() {
        wtr.write_record([i.to_string(), d.to_string()])?;
    }
    wtr.flush()?;
    Ok(())
}

pub fn ensure_input(
    path: &str,
    n: usize,
    d: usize,
    seed: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    if Path::new(path).exists() {
        return Ok(());
    }
    // let mut rng = StdRng::seed_from_u64(seed);
    // let dim = d.pow(n as u32);
    // let rho = random_density_matrix(dim, &mut rng);
    // save_density_matrix(&rho, path)?;
    // println!("Generated sample input: {path}");
    Ok(())
}
