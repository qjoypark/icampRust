use ndarray::{Array2, ArrayBase, OwnedRepr, Dim, ArrayView1, s};
use ndarray_stats::QuantileExt;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};

pub fn load_data(filepath: &str) -> Result<ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>>, Box<dyn Error>> {
    let file = File::open(filepath)?;
    let reader = BufReader::new(file);

    let data: Result<Vec<f64>, _> = reader
        .lines()
        .map(|line| -> Result<f64, Box<dyn Error>> {
            Ok(line?.trim().parse()?)
        })
        .collect();

    Ok(ArrayBase::from_vec(data?))
}

fn density(
    x: &ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>>, 
    y: &ArrayView1<f64>
) -> f64 {
    let min = x.min().unwrap_or(&0.0);
    let max = x.max().unwrap_or(&1.0);
    let n = x.len();

    if n == 0 {
        return 0.0;
    }

    let width = (max - min) / (n as f64).sqrt();
    let d = y.iter()
        .filter(|&&val| {
            let low = (val - min) / width;
            (low.floor() as usize) < n
        })
        .count();

    d as f64 / (n as f64).sqrt()
}

pub fn calculate_niche(
    env: &ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
    com: &ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>,
    _niche_name: &str,
    _nworker: usize,
    _use_memmap: bool,
) -> Result<ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>, Box<dyn Error>> {
    let n = env.shape()[1];
    let s = com.shape()[0];
    
    if com.shape()[1] == 0 {
        return Err("Community matrix is empty".into());
    }

    let comts = Array2::from_shape_fn((s, n), |(i, _)| com[[i, 0]]);

    let mut result = Array2::zeros((s, n));
    for j in 0..n {
        let comp = comts.column(j);
        let env_col = env.column(j).to_owned();
        let dens_all = density(&env_col, &comp);
        let max_den = (0..s)
            .map(|i| density(&env_col, &comp.slice(s![i..i+1])))
            .fold(f64::NEG_INFINITY, f64::max);

        if max_den == 0.0 {
            return Err("Max density is zero".into());
        }

        for index in 0..s {
            let dens1 = density(&env_col, &comp.slice(s![index..index+1]));
            let dif_den = dens_all - dens1;
            result[[index, j]] = dif_den / max_den;
        }
    }

    Ok(result)
}