use std::path::PathBuf;
use std::{
    env,
    fs::File,
    io::{self, Read, Write},
};
use structopt::StructOpt;

use svg2gcode::tsp_solver;

#[derive(Debug, StructOpt)]
#[structopt(name = "optimizer", author, about)]
struct Opt {
    /// File path to G-Code input
    #[structopt(long)]
    input: PathBuf,

    /// File path to G-Code output
    #[structopt(long)]
    output: PathBuf,
}

fn main() -> io::Result<()> {
    if env::var("RUST_LOG").is_err() {
        env::set_var("RUST_LOG", "optimizer=info");
    }
    env_logger::init();

    let opt = Opt::from_args();

    println!("Input file path: {:?}", opt.input);
    println!("Output file path: {:?}", opt.output);

    // Open the input file
    let mut input_file = File::open(&opt.input)?;

    // Read the contents of the input file into a string
    let mut input_content = String::new();
    input_file.read_to_string(&mut input_content)?;

    // Print the content of the input file to the console
    // println!("Input file content:\n{}", input_content);

    // Call the TSP solver function to optimize the G-Code
    let optimized_content: String = tsp_solver(input_content);

    // Open the output file
    let mut output_file = File::create(&opt.output)?;

    // Write the optimized G-Code to the output file
    output_file.write_all(optimized_content.as_bytes())?;

    Ok(())
}
