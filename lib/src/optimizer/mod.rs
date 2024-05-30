use block::{GCodeBlock, GCodePoint};
use ga::GeneticAlgorithm;
use std::collections::HashMap;
pub mod block;
pub mod ga;

fn remove_comment(gcode_line: &str) -> &str {
    // Remove comments
    let line = gcode_line.split(';').next().unwrap();
    // Trim whitespace
    line.trim()
}

// Optionally return GCodePoint
fn process_line(gcode_line: &str) -> Option<(String, GCodePoint)> {
    // Split the line into parts and remove comments
    let parts: Vec<&str> = remove_comment(gcode_line).split_whitespace().collect();

    // Get the command
    let command = parts[0];

    // Create a hashmap of the arguments
    let mut arguments: HashMap<char, f64> = HashMap::new();
    for part in &parts[1..] {
        let (key, value) = part.split_at(1);
        if let Ok(value) = value.parse::<f64>() {
            arguments.insert(key.chars().next().unwrap(), value);
        }
    }

    // Extract the arguments with default values of None
    let x = arguments.get(&'X').copied().unwrap_or_default();
    let y = arguments.get(&'Y').copied().unwrap_or_default();
    let z = arguments.get(&'Z').copied();
    let f = arguments.get(&'F').copied();

    // Match the command
    match command {
        "G0" | "G1" => Some((command.to_string(), GCodePoint::new(x, y, z, f))),
        _ => None,
    }
}

fn ingest_gcode(gcode: &str) -> Vec<GCodeBlock> {
    // Split the G-Code into lines
    let lines: Vec<&str> = gcode.lines().collect();

    // Process each line
    let mut blocks: Vec<GCodeBlock> = Vec::new();
    let mut current_block: Vec<GCodePoint> = Vec::new();

    for line in lines {
        // Process the line
        if let Some((command, point)) = process_line(line) {
            // If the command is G0, start a new block
            if command == "G0" {
                if !current_block.is_empty() {
                    blocks.push(GCodeBlock::new(current_block));
                    current_block = Vec::new();
                }
            }

            current_block.push(point);
        }
    }

    blocks
}

pub fn tsp_solver(program: String) -> String {
    // Ingest the G-Code
    let blocks: Vec<GCodeBlock> = ingest_gcode(&program);

    // Loop through the blocks and print
    // for block in &blocks {
    //     println!("{:?}", block);
    // }

    // Create a new genetic algorithm
    let mut ga: GeneticAlgorithm = GeneticAlgorithm::new();

    ga.init(blocks);

    // Run the genetic algorithm
    // Measure time per step
    let mut total_time = 0;
    let mut loops = 0;
    while (ga.get_unchanged_gens() < 500) {
        let start = std::time::Instant::now();
        ga.run_generation();
        let duration = start.elapsed();
        total_time += duration.as_millis();
        loops += 1;
    }

    // Print average time per step
    println!("Average time per step: {}", total_time / loops);

    program
}
