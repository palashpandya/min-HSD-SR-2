
use ndarray::{Array1, Array2};
use num_complex::Complex64;
use rand::prelude::*;
use rand_distr::StandardNormal;

pub type Mat = Array2<Complex64>;
pub type Vec1 = Array1<Complex64>;

fn c(re: f64) -> Complex64 {
    Complex64::new(re, 0.0)
}

fn conj_t(a: &Mat) -> Mat {
    a.t().mapv(|x| x.conj())
}

fn kron_vec(a: &Vec1, b: &Vec1) -> Vec1 {
    let nb = b.len();
    Array1::from_shape_fn(a.len() * nb, |i| a[i / nb] * b[i % nb])
}

fn outer(a: &Vec1, b: &Vec1) -> Mat {
    let nb = b.len();
    Array2::from_shape_fn((a.len(), nb), |(i, j)| a[i] * b[j].conj())
}

fn trace_real(a: &Mat) -> f64 {
    a.diag().iter().map(|x| x.re).sum()
}

fn trace_of_product(a: &Mat, b: &Mat) -> f64 {
    trace_real(&a.dot(b))
}

pub fn hs_dist_sq(a: &Mat, b: &Mat) -> f64 {
    let diff = a - b;
    trace_of_product(&diff, &diff)
}

pub fn hs_dist(a: &Mat, b: &Mat) -> f64 {
    // Historical API name: this returns the SQUARED Hilbert--Schmidt distance.
    hs_dist_sq(a, b).max(0.0)
}

// pub fn random_density_matrix<R: Rng>(dim: usize, rng: &mut R) -> Mat {
//     let g: Mat = Array2::from_shape_fn((dim, dim), |_| {
//         Complex64::new(StandardNormal.sample_iter(rng).next().unwrap(), StandardNormal.sample_iter(rng).next().unwrap())
//     });
//     let rho = g.dot(&conj_t(&g));
//     let tr: Complex64 = rho.diag().iter().sum();
//     rho.mapv(|x| x / tr)
// }

pub fn random_product_vecs<R: Rng>(n: usize, d: usize, rng: &mut R) -> Vec<Vec1> {
    (0..n)
        .map(|_| {
            let v = Array1::from_shape_fn(d, |_| {
                Complex64::new(
                    StandardNormal.sample(rng),
                    StandardNormal.sample(rng),
                )
            });
            let norm = v.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
            v.mapv(|x| x / norm)
        })
        .collect()
}

pub fn vecs_to_dm(vecs: &[Vec1]) -> Mat {
    let mut ket = Array1::from_elem(1, Complex64::new(1.0, 0.0));
    for v in vecs {
        ket = kron_vec(&ket, v);
    }
    outer(&ket, &ket)
}

pub fn preselection_ok(rho0: &Mat, rho1: &Mat, rho2: &Mat) -> bool {
    let delta = rho0 - rho1;
    let diff = rho2 - rho1;
    trace_of_product(&delta, &diff) > 0.0
}

// Same preselection criterion, evaluated without constructing the product
// projector or multiplying two full matrices on every rejected trial.
fn product_preselection_score(delta: &Mat, vecs: &[Vec1], threshold: f64) -> f64 {
    let mut ket = Array1::from_elem(1, c(1.0));
    for v in vecs { ket = kron_vec(&ket, v); }
    let mut expectation = c(0.0);
    for i in 0..ket.len() {
        for j in 0..ket.len() {
            expectation += ket[i].conj() * delta[(i,j)] * ket[j];
        }
    }
    expectation.re - threshold
}

// Forms L of shape (dl*d*dr, d) where L[i*d*dr + a*dr + l, a] = left[i]*right[l],
// then returns L.H @ delta @ L — the effective d×d operator for one qudit.
fn effective_op(delta: &Mat, left: &Vec1, right: &Vec1, d: usize) -> Mat {
    let dl = left.len();
    let dr = right.len();
    let big_d = dl * d * dr;

    let mut l_mat = Array2::<Complex64>::zeros((big_d, d));
    for i in 0..dl {
        for l in 0..dr {
            let lr = left[i] * right[l];
            for a in 0..d {
                l_mat[(i * d * dr + a * dr + l, a)] = lr;
            }
        }
    }

    let tmp = delta.dot(&l_mat);
    conj_t(&l_mat).dot(&tmp)
}

// Power iteration for the top eigenvector of a Hermitian matrix.
// Shifts by ||A||_F to ensure positive semi-definiteness before iterating.
fn top_eigenvec(a: &Mat) -> Vec1 {
    let d = a.nrows();
    let shift = a.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
    let mut a_shifted = a.clone();
    for i in 0..d {
        a_shifted[(i, i)] += c(shift);
    }

    let mut v = Array1::from_elem(d, c(1.0 / (d as f64).sqrt()));
    for _ in 0..300 {
        let av = a_shifted.dot(&v);
        let norm = av.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
        if norm < 1e-15 {
            break;
        }
        v = av.mapv(|x| x / norm);
    }
    v
}

pub fn optimize_product_state(delta: &Mat, vecs: &[Vec1], n: usize, d: usize, n_iter: usize) -> Vec<Vec1> {
    let mut vecs: Vec<Vec1> = vecs.to_vec();

    for _ in 0..n_iter {
        for k in 0..n {
            let mut left = Array1::from_elem(1, Complex64::new(1.0, 0.0));
            for j in 0..k {
                left = kron_vec(&left, &vecs[j]);
            }
            let mut right = Array1::from_elem(1, Complex64::new(1.0, 0.0));
            for j in (k + 1)..n {
                right = kron_vec(&right, &vecs[j]);
            }

            let eff = effective_op(delta, &left, &right, d);
            let eff_sym = (&eff + &conj_t(&eff)).mapv(|x| x / 2.0);
            vecs[k] = top_eigenvec(&eff_sym);
        }
    }
    vecs
}

pub fn optimize_product_state_grad(
    delta: &Mat,
    vecs: &[Vec1],
    n: usize,
    d: usize,
    n_iter: usize,
    grad_steps: usize,
    lr: f64,
) -> Vec<Vec1> {
    let mut vecs: Vec<Vec1> = vecs.to_vec();

    for _ in 0..n_iter {
        // Precompute right environments once per sweep
        let mut right_envs: Vec<Vec1> = vec![Array1::from_elem(1, Complex64::new(1.0, 0.0)); n + 1];
        for j in (0..n).rev() {
            let r = right_envs[j + 1].clone();
            right_envs[j] = kron_vec(&vecs[j], &r);
        }

        let mut left = Array1::from_elem(1, Complex64::new(1.0, 0.0));
        for k in 0..n {
            let right = right_envs[k + 1].clone();
            let eff = effective_op(delta, &left, &right, d);
            let eff_sym = (&eff + &conj_t(&eff)).mapv(|x| x / 2.0);

            let mut v = vecs[k].clone();
            for _ in 0..grad_steps {
                let eucl_grad = eff_sym.dot(&v);
                // Re(<v|eucl_grad>) — projects out the radial component
                let inner_re: f64 = v
                    .iter()
                    .zip(eucl_grad.iter())
                    .map(|(vi, gi)| (vi.conj() * gi).re)
                    .sum();
                let riem_grad = Array1::from_shape_fn(d, |i| eucl_grad[i] - c(inner_re) * v[i]);
                v = &v + &riem_grad.mapv(|x| x * lr);
                let norm = v.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
                v.mapv_inplace(|x| x / norm);
            }
            vecs[k] = v;
            left = kron_vec(&left, &vecs[k]);
        }
    }
    vecs
}

pub fn find_p_crit(rho0: &Mat, rho1: &Mat, rho2: &Mat) -> f64 {
    let diff1 = rho1 - rho2;
    let denom = trace_of_product(&diff1, &diff1);
    if denom < 1e-15 {
        return 1.0;
    }
    let num = trace_of_product(&(rho0 - rho2), &diff1);
    (num / denom).clamp(0.0, 1.0)
}

pub fn gilbert<R: Rng>(
    rho0: &Mat,
    n: usize,
    d: usize,
    max_corr: usize,
    max_trials: usize,
    optimizer: &str,
    tol: f64,
    preselection: bool,
    gradient_steps: usize,
    local_sweeps: usize,
    reference_distance2: Option<f64>,
    reference_gap: f64,
    rng: &mut R,
) -> (Mat, Vec<f64>) {
    // Initial CSS: diagonal of rho0
    let diag = rho0.diag().to_owned();
    let mut rho1 = Array2::from_diag(&diag);
    let mut d_old = hs_dist(rho0, &rho1);
    let mut distances = vec![d_old];
    let mut num_corr = 0usize;
    let mut num_trials = 0usize;
    let mut delta = rho0 - &rho1;
    let mut threshold = trace_of_product(&delta, &rho1);

    if d_old <= tol {
        return (rho1, distances);
    }

    while num_corr < max_corr && num_trials < max_trials {
        // Every attempt consumes a trial and starts from a fresh product state.
        // Otherwise a failed optimized update can repeat forever without
        // advancing either stopping counter.
        let vecs = random_product_vecs(n, d, rng);
        num_trials += 1;
        if preselection && product_preselection_score(&delta, &vecs, threshold) <= 0.0 {
            continue;
        }

        let opt_vecs = if optimizer == "grad" {
            optimize_product_state_grad(&delta, &vecs, n, d, local_sweeps, gradient_steps, 0.3)
        } else {
            optimize_product_state(&delta, &vecs, n, d, local_sweeps)
        };
        let rho2 = vecs_to_dm(&opt_vecs);

        let p = find_p_crit(rho0, &rho1, &rho2);
        let rho_candidate = rho1.mapv(|x| x * p) + rho2.mapv(|x| x * (1.0 - p));
        let d_new = hs_dist(rho0, &rho_candidate);

        if d_new < d_old {
            rho1 = rho_candidate;
            d_old = d_new;
            num_corr += 1;
            if num_corr % 5 == 0 {
                println!("iter:{num_corr}, dist:{d_old}, trials:{num_trials}");
            }
            distances.push(d_old);
            delta = rho0 - &rho1;
            threshold = trace_of_product(&delta, &rho1);
            if d_old < tol {
                break;
            }
            // Optional benchmark stopping rule. The reference is used only
            // to stop an independent trajectory, never to construct a trial.
            if reference_distance2.is_some_and(|r| d_old-r <= reference_gap) {
                break;
            }
        }
    }

    let reason = if d_old < tol { "distance_threshold" }
        else if reference_distance2.is_some_and(|r| d_old-r <= reference_gap) {
            "reference_gap"
        } else if num_corr >= max_corr { "correction_budget" }
        else { "trial_limit" };
    println!("termination: trials={num_trials}, corrections={num_corr}, reason={reason}");
    (rho1, distances)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn cached_preselection_matches_matrix_criterion() {
        let mut rng = StdRng::seed_from_u64(20261124);
        for d in [2, 3] {
            for _ in 0..200 {
                let rho0 = vecs_to_dm(&random_product_vecs(2,d,&mut rng));
                let a = vecs_to_dm(&random_product_vecs(2,d,&mut rng));
                let b = vecs_to_dm(&random_product_vecs(2,d,&mut rng));
                let rho1 = a.mapv(|x| x*0.37) + b.mapv(|x| x*0.63);
                let vecs = random_product_vecs(2,d,&mut rng);
                let rho2 = vecs_to_dm(&vecs);
                let delta = &rho0 - &rho1;
                let full = trace_of_product(&delta,&(&rho2-&rho1));
                let cached = product_preselection_score(
                    &delta,&vecs,trace_of_product(&delta,&rho1));
                assert!((full-cached).abs() < 2e-14);
                assert_eq!(preselection_ok(&rho0,&rho1,&rho2),cached > 0.0);
            }
        }
    }
}
