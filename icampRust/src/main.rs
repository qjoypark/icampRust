mod dniche;
mod nti;

use clap::{Arg, Command};
use std::error::Error;
use std::path::Path;
use sysinfo::{System, SystemExt};
// 删除了未使用的导入

fn main() -> Result<(), Box<dyn Error>> {
    let matches = Command::new("NTI Calculator")
        .version("1.0")
        .author("Your Name <youremail@example.com>")
        .about("Calculates the Nearest Taxon Index (NTI) for community data")
        .arg(
            Arg::new("comm")
                .help("Path to the community data CSV file")
                .required(true)
                .index(1),
        )
        .arg(
            Arg::new("dis")
                .help("Path to the distance matrix CSV file")
                .required(true)
                .index(2),
        )
        .arg(
            Arg::new("weighted")
                .short('w')
                .long("weighted")
                .help("Use weighted calculations"),
        )
        .arg(
            Arg::new("rand_times")
                .short('r')
                .long("rand_times")
                .help("Number of randomizations")
                .default_value("1000")
                .value_parser(clap::value_parser!(usize)),
        )
        .get_matches();

    let comm_path = matches.get_one::<String>("comm").unwrap();
    let dis_path = matches.get_one::<String>("dis").unwrap();
    let weighted = matches.get_flag("weighted");
    let rand_times: usize = *matches.get_one("rand_times").unwrap();

    if !Path::new(comm_path).exists() {
        eprintln!("Error: The community data file '{}' does not exist.", comm_path);
        std::process::exit(1);
    }
    if !Path::new(dis_path).exists() {
        eprintln!("Error: The distance matrix file '{}' does not exist.", dis_path);
        std::process::exit(1);
    }

    let available_memory_gb = get_available_memory_gb();
    println!("Available memory: {} GB", available_memory_gb);

    let use_memmap = available_memory_gb < 4.0;
    let nworker = if available_memory_gb < 4.0 { 2 } else { 4 };

    let comm = nti::load_data(comm_path)?;
    let dis = nti::load_data(dis_path)?;

    let permuted_matrices = nti::randomize_matrix(&dis, rand_times);
    let mntd_obs = nti::calculate_mntd(&comm, &dis, weighted);
    
    // 修改了这部分代码，直接赋值而不使用 match
    let mntd_rand = nti::parallel_random_mntd(&comm, &dis, permuted_matrices, weighted);

    let nti_result = nti::calculate_nti(&mntd_obs, &mntd_rand);

    println!("NTI Results:");
    println!("{:?}", nti_result);

    // dniche 函数调用
    let env = dniche::load_data("path/to/env.csv")?; // 请替换为实际路径
    let env = env.into_dimensionality::<ndarray::Ix2>()?;
    let comm = comm.into_dimensionality::<ndarray::Ix2>()?;
    let niche_result = dniche::calculate_niche(&env, &comm, "niche.value", nworker, use_memmap)?;
    println!("Niche Calculation Results:");
    println!("{:?}", niche_result);

    Ok(())
}

fn get_available_memory_gb() -> f64 {
    let mut sys = System::new_all();
    sys.refresh_memory();
    sys.available_memory() as f64 / 1024.0 / 1024.0 / 1024.0
}