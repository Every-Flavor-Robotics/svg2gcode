// use bincode;
use block::{GCodeBlock, GCodePoint};
use ga::GeneticAlgorithm;
use plotters::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Write};
use std::net::TcpStream;
use std::path::PathBuf;

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

    // Push the last block
    if !current_block.is_empty() {
        blocks.push(GCodeBlock::new(current_block));
    }
    blocks
}

fn serialize_solution(points: &Vec<GCodeBlock>, ga: &GeneticAlgorithm) -> (Vec<u8>, Vec<u8>) {
    //
    let solution = ga.get_best_solution();

    println!("Solution: {:?}", solution);
    // Create a vector of GCodeBlocks
    let new_points: Vec<GCodeBlock> = solution.iter().map(|&i| points[i].clone()).collect();

    // Create a hashmap of the solution
    let mut map: HashMap<String, Vec<GCodeBlock>> = HashMap::new();
    // Allocate space length of new_points
    let mut backtrack_data: Vec<(usize, usize, usize)> = Vec::with_capacity(new_points.len());

    // Loop through blocks
    for i in 0..new_points.len() - 1 {
        let backtrack_info = ga.get_shortest_distance_with_backtrack(&solution, i, i + 1);

        // Push the BackTrackInfo struct to the backtrack_data vector
        backtrack_data.push(backtrack_info);
    }

    // Insert vec of GCodeBlocks into the hashmap with key "solution"
    map.insert("solution".to_string(), new_points);

    // Serialize the solution
    let serialized_solution = serde_pickle::to_vec(&map, Default::default()).unwrap();
    let serialized_backtrack_data =
        serde_pickle::to_vec(&backtrack_data, Default::default()).unwrap();

    (serialized_solution, serialized_backtrack_data)
}

// fn send_solution_over_socket(
//     points: &Vec<GCodeBlock>,
//     ga: &GeneticAlgorithm,
//     address: &str,
// ) -> io::Result<()> {
//     let serialized = serialize_solution(points, &ga.get_best_solution()ZZ, ga);
//     let mut stream = TcpStream::connect(address)?;
//     stream.write_all(&serialized)?;
//     Ok(())
// }

fn plot_gcode_blocks(
    blocks: &[GCodeBlock],
    file_name: &str,
    block_order: Option<&[usize]>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Determine the image size and margins dynamically based on the data range
    let all_points: Vec<&GCodePoint> = if let Some(order) = block_order {
        // Log that we are using the block order
        order.iter().flat_map(|&idx| &blocks[idx].points).collect()
    } else {
        blocks.iter().flat_map(|b| &b.points).collect()
    };

    let (min_x, max_x) = all_points
        .iter()
        .map(|p| p.x)
        .fold((f64::MAX, f64::MIN), |(min, max), x| {
            (min.min(x), max.max(x))
        });
    let (min_y, max_y) = all_points
        .iter()
        .map(|p| p.y)
        .fold((f64::MAX, f64::MIN), |(min, max), y| {
            (min.min(y), max.max(y))
        });
    let x_range = max_x - min_x;
    let y_range = max_y - min_y;
    let margin = 0.1 * x_range.max(y_range); // 10% margin
    let scale_factor = 5.0; // Adjust this value to control the scaling

    // Apply scaling to the dimensions
    let width = (scale_factor * (x_range + 2.0 * margin)) as u32;
    let height = (scale_factor * (y_range + 2.0 * margin)) as u32;

    // Create the image file
    let root = BitMapBackend::new(file_name, (width, height)).into_drawing_area();
    root.fill(&WHITE)?;

    // Create a chart context
    let mut chart = ChartBuilder::on(&root)
        .margin(margin as u32)
        .build_cartesian_2d(min_x..max_x, min_y..max_y)?;

    chart
        .configure_mesh()
        .disable_x_axis()
        .disable_y_axis()
        .draw()?;

    // Draw the axes
    chart.configure_mesh().x_desc("X").y_desc("Y").draw()?;

    // Draw the lines between the points in all points
    for (point, next_point) in all_points.iter().zip(all_points.iter().skip(1)) {
        chart.draw_series(LineSeries::new(
            vec![(point.x, point.y), (next_point.x, next_point.y)],
            &BLACK,
        ))?;
    }

    // Save the image
    root.present()?;

    Ok(())
}

pub fn tsp_solver(program: String, temp_dir: &PathBuf) -> (Vec<u8>, Vec<u8>) {
    // Ingest the G-Code
    let blocks: Vec<GCodeBlock> = ingest_gcode(&program);

    // Create a file called "blocks.svg" in the temp directory
    let image_file = temp_dir.join("gcode.png");

    // Plot the G-Code blocks
    plot_gcode_blocks(&blocks, image_file.to_str().unwrap(), None).unwrap();
    // Wait for 2 seconds
    std::thread::sleep(std::time::Duration::from_secs(2));

    // Loop through the blocks and print
    // for block in &blocks {
    //     println!("{:?}", block);
    // }

    // Create a new genetic algorithm
    let mut ga: GeneticAlgorithm = GeneticAlgorithm::new();

    ga.init(&blocks);

    // Run the genetic algorithm
    // Measure time per step
    let mut total_time = 0;
    let mut loops = 0;
    while (ga.get_unchanged_gens() < 1000) {
        let start = std::time::Instant::now();
        ga.run_generation();
        let duration = start.elapsed();
        total_time += duration.as_micros();
        loops += 1;
    }

    // Print average time per step
    println!("Average time per step: {}", total_time / loops);

    // Get a reference to the best solution
    let best_solution = ga.get_best_solution();

    plot_gcode_blocks(
        &blocks,
        image_file.to_str().unwrap(),
        Some(best_solution.as_slice()),
    )
    .unwrap();

    // Get the best solution
    serialize_solution(&blocks, &ga)
}
