use ndarray::{Array2, Axis};
use rand::seq::SliceRandom;
use rand::thread_rng;
use rayon::prelude::*;
use std::error::Error;
use csv::ReaderBuilder;
use serde::Deserialize;

#[derive(Deserialize)]
struct Record {
    value: f64,
}

pub fn load_data(file_path: &str) -> Result<Array2<f64>, Box<dyn Error>> {
    let mut rdr = ReaderBuilder::new().from_path(file_path)?;
    let mut records = Vec::new();
    for result in rdr.deserialize() {
        let record: Record = result?;
        records.push(record.value);
    }
    let size = (records.len() as f64).sqrt() as usize;
    Ok(Array2::from_shape_vec((size, size), records)?)
}

pub fn calculate_mntd(comm: &Array2<f64>, pd: &Array2<f64>, abundance_weighted: bool) -> Array2<f64> {
    let n = comm.nrows();
    let mut res = Array2::zeros((n, 1));
    let mut pd = pd.clone();
    for i in 0..n {
        pd[(i, i)] = f64::NAN;
    }

    if abundance_weighted {
        let mut min_d = Array2::zeros(comm.raw_dim());
        for i in 0..n {
            let row_mask = comm.row(i).mapv(|v| v > 0.0);
            let pdx = pd.select(Axis(0), &row_mask.iter().enumerate().filter(|(_, &b)| b).map(|(j, _)| j).collect::<Vec<_>>())
                        .select(Axis(1), &row_mask.iter().enumerate().filter(|(_, &b)| b).map(|(j, _)| j).collect::<Vec<_>>());
            if pdx.nrows() > 1 {
                for (k, row) in pdx.axis_iter(Axis(0)).enumerate() {
                    min_d[(i, row_mask.iter().enumerate().filter(|(_, &b)| b).map(|(j, _)| j).collect::<Vec<_>>()[k])] = row.iter().filter(|&&v| !v.is_nan()).fold(f64::INFINITY, |a, &b| a.min(b));
                }
            }
        }
        let comm_p = comm / comm.sum_axis(Axis(1)).insert_axis(Axis(1));
        res = min_d * &comm_p;
        res = res.sum_axis(Axis(1)).insert_axis(Axis(1));
    } else {
        for i in 0..n {
            let row_mask = comm.row(i).mapv(|v| v > 0.0);
            let pdx = pd.select(Axis(0), &row_mask.iter().enumerate().filter(|(_, &b)| b).map(|(j, _)| j).collect::<Vec<_>>())
                        .select(Axis(1), &row_mask.iter().enumerate().filter(|(_, &b)| b).map(|(j, _)| j).collect::<Vec<_>>());
            if pdx.nrows() > 1 {
                let mut min_dists = Vec::new();
                for row in pdx.axis_iter(Axis(0)) {
                    min_dists.push(row.iter().filter(|&&v| !v.is_nan()).fold(f64::INFINITY, |a, &b| a.min(b)));
                }
                res[(i, 0)] = min_dists.iter().copied().sum::<f64>() / min_dists.len() as f64;
            } else {
                res[(i, 0)] = 0.0;
            }
        }
    }

    res
}

pub fn randomize_matrix(matrix: &Array2<f64>, rand_times: usize) -> Vec<Array2<f64>> {
    let mut rng = thread_rng();
    let size = matrix.ncols();
    let mut permuted_matrices = Vec::new();

    for _ in 0..rand_times {
        let mut indices: Vec<usize> = (0..size).collect();
        indices.shuffle(&mut rng);
        let permuted_matrix = matrix.select(Axis(1), &indices);
        permuted_matrices.push(permuted_matrix);
    }

    permuted_matrices
}

pub fn parallel_random_mntd(comm: &Array2<f64>, dis: &Array2<f64>, permuted_matrices: Vec<Array2<f64>>, weighted: bool) -> Vec<Array2<f64>> {
    permuted_matrices.into_par_iter()
        .map(|permuted_dis| calculate_mntd(comm, &permuted_dis, weighted))
        .collect()
}

pub fn calculate_nti(mntd_obs: &Array2<f64>, mntd_rand: &[Array2<f64>]) -> Array2<f64> {
    let mut mean_rand = Array2::zeros(mntd_obs.raw_dim());
    let mut std_rand = Array2::zeros(mntd_obs.raw_dim());

    for arr in mntd_rand {
        mean_rand = &mean_rand + arr;
    }
    mean_rand = &mean_rand / mntd_rand.len() as f64;

    for arr in mntd_rand {
        let diff = arr - &mean_rand;
        std_rand = &std_rand + &diff.mapv(|v| v * v);
    }
    std_rand = (&std_rand / (mntd_rand.len() - 1) as f64).mapv(f64::sqrt);

    (mntd_obs - mean_rand) / std_rand
}
