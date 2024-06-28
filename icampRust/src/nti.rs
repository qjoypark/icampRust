use ndarray::{Array1, Array2, Axis};
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
    let records: Vec<Record> = rdr.deserialize().collect::<Result<_, _>>()?;
    let size = (records.len() as f64).sqrt() as usize;
    Ok(Array2::from_shape_vec((size, size), records.into_iter().map(|r| r.value).collect())?)
}

pub fn calculate_mntd(comm: &Array2<f64>, dis: &Array2<f64>, weighted: bool) -> Array1<f64> {
    let n_samples = comm.nrows();
    let n_species = comm.ncols();
    let mut result = Array1::zeros(n_samples);

    for i in 0..n_samples {
        let mut min_distances = vec![f64::MAX; n_species];
        let sample = comm.row(i);
        let present_species: Vec<usize> = sample.iter()
            .enumerate()
            .filter(|(_, &abundance)| abundance > 0.0)
            .map(|(idx, _)| idx)
            .collect();

        for &sp1 in &present_species {
            for &sp2 in &present_species {
                if sp1 != sp2 {
                    let distance = dis[[sp1, sp2]];
                    min_distances[sp1] = min_distances[sp1].min(distance);
                }
            }
        }

        if weighted {
            let total_abundance: f64 = sample.sum();
            let weighted_sum: f64 = present_species.iter()
                .map(|&sp| sample[sp] / total_abundance * min_distances[sp])
                .sum();
            result[i] = weighted_sum;
        } else {
            let mean_distance: f64 = present_species.iter()
                .map(|&sp| min_distances[sp])
                .sum::<f64>() / present_species.len() as f64;
            result[i] = mean_distance;
        }
    }

    result
}

pub fn randomize_matrix(matrix: &Array2<f64>, rand_times: usize) -> Vec<Array2<f64>> {
    let mut rng = thread_rng();
    let size = matrix.ncols();
    (0..rand_times)
        .map(|_| {
            let mut indices: Vec<usize> = (0..size).collect();
            indices.shuffle(&mut rng);
            matrix.select(Axis(1), &indices)
        })
        .collect()
}

pub fn parallel_random_mntd(comm: &Array2<f64>, dis: &Array2<f64>, permuted_matrices: Vec<Array2<f64>>, weighted: bool) -> Vec<Array1<f64>> {
    permuted_matrices.into_par_iter()
        .map(|permuted_comm| calculate_mntd(&permuted_comm, dis, weighted))
        .collect()
}

pub fn calculate_nti(mntd_obs: &Array1<f64>, mntd_rand: &[Array1<f64>]) -> Array1<f64> {
    let mean_rand = mntd_rand.iter().fold(Array1::zeros(mntd_obs.raw_dim()), |acc, arr| acc + arr) / mntd_rand.len() as f64;
    
    let std_rand = (mntd_rand.iter()
        .fold(Array1::zeros(mntd_obs.raw_dim()), |acc, arr| {
            let diff = arr - &mean_rand;
            acc + &diff.mapv(|v| v * v)
        }) / (mntd_rand.len() - 1) as f64)
        .mapv(f64::sqrt);

    (mntd_obs - mean_rand) / std_rand
}