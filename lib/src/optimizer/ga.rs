use crate::optimizer::block::{distance, GCodeBlock, GCodePoint};
use ndarray::{s, Array2, ArrayView2};
use rand::seq::SliceRandom;
use rand::Rng;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::time::{Duration, Instant}; // Import the GCodeBlock and GCodePoint structs

pub struct GeneticAlgorithm {
    population_size: usize,
    crossover_probability: f64,
    mutation_probability: f64,
    unchanged_gens: usize,
    mutation_count: usize,
    best_value: Option<f64>, // Option for potential absence of best value
    best: Vec<usize>,
    current_generation: usize,
    current_best_index: Option<usize>, // Option since it might not be set initially
    population: Vec<Vec<usize>>,       // Individuals represented as Vecs of usize indices
    values: Vec<f64>,
    fitness_values: Vec<f64>,
    roulette: Vec<f64>,
    distances: Array2<f64>, // Using ndarray for 2D array of distances
    nearest_points: Array2<(usize, usize)>, // Same for nearest points
    points: Vec<GCodeBlock>, // Assuming you have a 'Point' struct
    num_total_points: usize,
}

impl GeneticAlgorithm {
    pub fn new() -> Self {
        GeneticAlgorithm {
            population_size: 50,
            crossover_probability: 0.9,
            mutation_probability: 0.07,
            unchanged_gens: 0,
            mutation_count: 0,
            best_value: None,
            best: vec![],
            current_generation: 0,
            current_best_index: None,
            population: vec![],
            values: vec![],
            fitness_values: vec![],
            roulette: vec![],
            distances: Array2::default((0, 0)),
            nearest_points: Array2::default((0, 0)),
            points: vec![],
            num_total_points: 0,
        }
    }

    pub fn init(&mut self, points: &[GCodeBlock]) -> () {
        self.points = points.to_vec();

        self.compute_distances(); // You'll need to implement this method

        // Reset population
        self.population = (0..self.population_size)
            .map(|_| self.random_individual(self.points.len()))
            .collect();

        self.evalute_population();

        println!("Done initializing");
    }

    // Main method to run the genetic algorithm
    pub fn run_generation(&mut self) -> () {
        self.current_generation += 1;
        self.selection_step();

        // let now = Instant::now();

        self.crossover_step();
        // let elapsed = now.elapsed();
        // println!("Evaluation time: {:?}", elapsed);

        self.mutation_step();

        self.evalute_population();

        println!(
            "Generation: {}, Best value: {:?}",
            self.current_generation, self.best_value
        );
    }

    pub fn get_unchanged_gens(&self) -> usize {
        self.unchanged_gens
    }

    pub fn get_best_solution(&self) -> Vec<usize> {
        self.best.clone()
    }
}

// Methods for selection step
impl GeneticAlgorithm {
    /// Perform the selection step of the genetic algorithm.
    ///
    /// This method constructs a list of parents by applying various selection techniques.
    /// The selection process includes elitism, mutation, and roulette wheel selection.
    fn selection_step(&mut self) -> () {
        // Create a new vector to store the parents
        let mut parents = Vec::with_capacity(self.population_size);

        // Add the best individual from the previous generation
        if let Some(idx) = self.current_best_index {
            parents.push(self.population[idx].clone());
        }

        // Add the mutated versions of the best individual across all generations
        parents.push(self.inversion_mutate(self.best.clone()));
        parents.push(self.push_mutate(self.best.clone()));
        parents.push(self.best.clone());

        // Update the roulette wheel
        self.update_roulette();

        // Fill the rest of the population with individuals selected from the roulette wheel
        for _ in 4..self.population_size {
            // With a probability of 0.95, select an individual from the roulette wheel, otherwise generate a random individual
            let parent = if rand::random::<f64>() < 0.95 {
                self.population[self.sample_from_roulette()].clone()
            } else {
                self.random_individual(self.points.len())
            };

            // Add the selected individual to the parents vector
            parents.push(parent);
        }

        // Update the population with the new parents
        self.population = parents;
    }

    /// Update the roulette wheel selection probabilities based on the fitness values.
    ///
    /// This method calculates the fitness values for each individual in the population,
    /// normalizes the fitness values, and calculates the cumulative sum of the normalized
    /// values to create the roulette wheel selection probabilities.
    fn update_roulette(&mut self) -> () {
        self.compute_fitness_values();

        // Sum the fitness values
        let fitness_sum: f64 = self.fitness_values.iter().sum();
        self.roulette = self
            .fitness_values
            .iter()
            .map(|&v| v / fitness_sum)
            .collect();

        // Calculate the cumulative sum of the roulette values
        for i in 1..self.roulette.len() {
            self.roulette[i] += self.roulette[i - 1];
        }
    }

    /// Select an individual from the population using roulette wheel selection.
    ///
    /// Returns:
    ///     usize: The index of the selected individual in the population.
    ///
    /// Comments:
    ///     - The method uses the roulette wheel selection probabilities to select an individual.
    ///     - It generates a random number between 0 and 1 and uses it to select an individual.
    fn sample_from_roulette(&self) -> usize {
        let random_value = rand::thread_rng().gen::<f64>();

        self.roulette
            .iter()
            .enumerate()
            .find(|(_, &probability)| probability > random_value)
            .map(|(index, _)| index)
            .unwrap_or_else(|| self.roulette.len() - 1)
    }

    fn compute_fitness_values(&mut self) -> () {
        self.fitness_values = self.values.iter().map(|&v| 1.0 / v).collect();
    }
}

/// Perform the crossover step of the genetic algorithm.
///
/// This method selects individuals from the population for crossover based on a probability threshold.
/// It shuffles the selected individuals and performs crossover between pairs of individuals.
///
/// Crossover is a genetic operator that combines the genetic material of two individuals to create new offspring.
/// In this implementation, the crossover operation is performed using the `crossover` method.
impl GeneticAlgorithm {
    fn crossover_step(&mut self) {
        // Use a filter and enumerate directly on the range, then collect into a `Vec`
        let mut crossover_indices: Vec<usize> = (0..self.population_size)
            .filter(|_| rand::random::<f64>() < self.crossover_probability)
            .collect();

        // Shuffle the crossover indices
        crossover_indices.shuffle(&mut rand::thread_rng());

        // Perform crossover between pairs of individuals
        // Using chunks for cleaner iteration
        crossover_indices.chunks(2).for_each(|pair| {
            if let [a, b] = pair {
                self.crossover(*a, *b);
            }
        });
    }

    /// Perform the crossover operation between two individuals.
    fn crossover(&mut self, x: usize, y: usize) {
        // Reuse existing allocations for lookup tables
        let mut x_lookup = HashMap::with_capacity(self.population[x].len());
        let mut y_lookup = HashMap::with_capacity(self.population[y].len());

        // Populate the lookup tables
        for (i, &point) in self.population[x].iter().enumerate() {
            x_lookup.insert(point, i);
        }
        for (i, &point) in self.population[y].iter().enumerate() {
            y_lookup.insert(point, i);
        }

        // Compute the children using the forward and backward crossover methods
        let child1 = self.compute_child_forward(x, y, &x_lookup, &y_lookup);
        let child2 = self.compute_child_backwards(x, y, &x_lookup, &y_lookup);

        // Assign the new children to the population
        self.population[x] = child1;
        self.population[y] = child2;
    }

    fn compute_child_forward(
        &self,
        x: usize,
        y: usize,
        x_lookup: &HashMap<usize, usize>,
        y_lookup: &HashMap<usize, usize>,
    ) -> Vec<usize> {
        self.compute_child_helper(Self::next, x, y, x_lookup, y_lookup)
    }

    fn compute_child_backwards(
        &self,
        x: usize,
        y: usize,
        x_lookup: &HashMap<usize, usize>,
        y_lookup: &HashMap<usize, usize>,
    ) -> Vec<usize> {
        self.compute_child_helper(Self::prev, x, y, x_lookup, y_lookup)
    }

    /// Compute the child solution by performing crossover between two parent solutions.
    ///
    /// This method performs crossover between two parent solutions to generate a child solution.
    /// It starts by copying the parent solutions and selecting a random point to start the crossover.
    /// It then iteratively selects the next point to add to the child solution based on the distance between the current point and the potential next points.
    /// The process continues until all points are added to the child solution.
    /// The child solution is returned as a vector of indices representing the order of points in the solution.
    fn compute_child_helper(
        &self,
        iterate_fn: fn(usize, &Vec<usize>) -> usize,
        x: usize,
        y: usize,
        x_lookup: &HashMap<usize, usize>,
        y_lookup: &HashMap<usize, usize>,
    ) -> Vec<usize> {
        // TODO: Consider trying to avoid cloning the parent solutions
        // This would require that the parents are modified in place, which may be more complex
        // Check if this is a bottleneck and if it can be optimized
        // Clone the parent solutions

        // Could replace with a linked list if we are cloning anyway
        // This would allow for easier removal of elements
        let mut px = self.population[x].clone();
        let mut py = self.population[y].clone();

        // Create a vector to store the child solution
        let mut solution = vec![0; px.len()];
        let mut rng = rand::thread_rng();

        // Select a random starting point for the child solution
        let mut c = px[rng.gen_range(0..px.len())];

        solution[0] = c;
        let mut i = 1;
        while i < px.len() {
            let px_index = x_lookup[&c];
            let py_index = y_lookup[&c];

            let dx = iterate_fn(px_index, &px);
            let dy = iterate_fn(py_index, &py);

            px[px_index] = usize::MAX;
            py[py_index] = usize::MAX;

            c = if self.distances[[c, dx]] < self.distances[[c, dy]] {
                dx
            } else {
                dy
            };
            solution[i] = c;
            i += 1;
        }

        solution
    }

    /// Helper function to get the next index in the array, skipping over invalid indices.
    ///
    /// This function returns the next valid index in the array, skipping over invalid indices.
    /// If the end of the array is reached, the function wraps around to the beginning.
    fn next(index: usize, array: &Vec<usize>) -> usize {
        let mut next_index = (index + 1) % array.len();
        while array[next_index] == usize::MAX {
            next_index = (next_index + 1) % array.len();
        }

        array[next_index]
    }

    /// Helper function to get the previous index in the array, skipping over invalid indices.
    /// This function returns the previous valid index in the array, skipping over invalid indices.
    /// If the beginning of the array is reached, the function wraps around to the end.
    fn prev(index: usize, array: &Vec<usize>) -> usize {
        let mut prev_index = if index == 0 {
            array.len() - 1
        } else {
            index - 1
        };
        while array[prev_index] == usize::MAX {
            prev_index = if prev_index == 0 {
                array.len() - 1
            } else {
                prev_index - 1
            };
        }

        array[prev_index]
    }
}

// Methods for mutation step
impl GeneticAlgorithm {
    // """
    //     Applies mutation to the individuals in the population.

    //     This function iterates through each individual in the population and applies mutation
    //     with a certain probability. The mutation can be either a push mutation or a do mutation,
    //     determined randomly. The mutated individual replaces the original individual in the population.

    //     Parameters:
    //         None

    //     Returns:
    //         None

    //     Comments:
    //         - The function can mutate the same individual multiple times.
    //     """

    /// Apply mutation to the individuals in the population.
    /// This method iterates through each individual in the population and applies mutation
    /// with a certain probability. The mutation can be either a push mutation or a do mutation,
    /// determined randomly. The mutated individual replaces the original individual in the population.
    /// The function can mutate the same individual multiple times.
    fn mutation_step(&mut self) {
        let mut new_population = Vec::with_capacity(self.population.len());
        for individual in &self.population {
            if rand::random::<f64>() < self.mutation_probability {
                let mutated_individual = if rand::random::<f64>() > 0.5 {
                    self.push_mutate(individual.clone())
                } else {
                    self.inversion_mutate(individual.clone())
                };
                new_population.push(mutated_individual);
            } else {
                new_population.push(individual.clone());
            }
        }
        self.population = new_population;
    }

    /// Performs an inversion mutation on the given individual.
    ///
    /// Reverses a random subset of elements in the individual.
    fn inversion_mutate(&self, mut individual: Vec<usize>) -> Vec<usize> {
        // self.mutation_count += 1;
        let (m, n) = loop {
            let m = rand::random::<usize>() % (individual.len() - 1);
            let n = rand::random::<usize>() % individual.len();
            if m < n {
                break (m, n);
            }
        };

        // Reverse the subset of elements
        individual[m..n].reverse();

        individual
    }

    /// Performs a push mutation on the given individual.
    ///
    /// This mutation operation takes a random subset of elements from the individual and pushes them to the beginning of the individual's sequence.
    fn push_mutate(&self, individual: Vec<usize>) -> Vec<usize> {
        // self.mutation_count += 1;
        let (m, n) = loop {
            let m = rand::random::<usize>() % (individual.len() / 2);
            let n = rand::random::<usize>() % individual.len();
            if m < n {
                break (m, n);
            }
        };
        let mut s1 = individual[0..m].to_vec();
        let mut s2 = individual[m..n].to_vec();
        let mut s3 = individual[n..].to_vec();
        s2.append(&mut s1);
        s2.append(&mut s3);

        s2
    }
}

impl GeneticAlgorithm {
    fn random_individual(&self, num_points: usize) -> Vec<usize> {
        // Generate a random individual
        let mut rng = rand::thread_rng();
        let mut individual: Vec<usize> = (0..num_points).collect();

        // Manually intialize the individual
        individual.shuffle(&mut rng);

        individual
    }

    /// Evaluate the individuals in the population and update the best solution.
    ///
    /// This method evaluates each individual in the population and updates the stored best solution if a better solution is found.
    /// It also keeps track of the number of unchanged generations.
    fn evalute_population(&mut self) {
        // Evaluate the individuals in the population
        self.values = self
            .population
            .par_iter()
            .map(|ind| self.evaluate(ind))
            .collect();

        let (current_best_index, current_best_value) = self.get_current_best();

        // Update the best solution if a better solution is found, or if the best solution is not set
        if current_best_value < self.best_value.unwrap_or(f64::INFINITY) {
            self.current_best_index = Some(current_best_index);
            // Save a copy of the best individual
            self.best = self.population[current_best_index].clone();
            self.best_value = Some(current_best_value);
            println!("Saving new best");
            self.unchanged_gens = 0;
        } else {
            self.unchanged_gens += 1;
        }
    }

    fn get_current_best(&self) -> (usize, f64) {
        self.values
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(index, value)| (index, *value)) // Dereference and copy
            .unwrap()
    }

    // fn evaluate(&self, individual: &[usize]) -> f64 {
    //     let mut sum = 0.0;

    //     // Calculate the distance between the points in the order of the individual
    //     // First, calculate the distance between the first point and (0, 0)
    //     sum += distance(
    //         &self.points[individual[0]].points[0],
    //         &GCodePoint::new(0.0, 0.0, None, None),
    //     );

    //     // for i in 0..individual.len() - 1 {
    //     //     // Get distances for all previous points to the current point

    //     //     let distances: Vec<f64> = individual[..=i]
    //     //         .iter()
    //     //         .map(|&point| self.distances[[point, individual[i + 1]]])
    //     //         .collect();

    //     //     // Get the index of the closest point
    //     //     let closest_index = distances
    //     //         .iter()
    //     //         .enumerate()
    //     //         .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
    //     //         .map(|(index, _)| index)
    //     //         .unwrap();

    //     //     // Get the index of the nearest point to the next group
    //     //     let (nearest_point_to_next, _) =
    //     //         self.nearest_points[[individual[closest_index], individual[i + 1]]];

    //     //     // Sum backtrack costs, the number of individual steps
    //     //     // Between the closest point and the next point
    //     //     let backtrack_cost: usize = individual[closest_index + 1..=i]
    //     //         .iter()
    //     //         .map(|&j| self.points[j].len())
    //     //         .sum::<usize>()
    //     //         + self.points[individual[closest_index]].len()
    //     //         - nearest_point_to_next;

    //     //     // Calculate the sum
    //     //     sum += 100.0 * distances[closest_index] / individual.len() as f64
    //     //         + 40.0 * backtrack_cost as f64 / self.num_total_points as f64;
    //     // }

    //     let sum: f64 = (0..individual.len() - 1)
    //         .into_par_iter()
    //         .map(|i| {
    //             // Get distances for all previous points to the current point
    //             let distances: Vec<f64> = individual[..=i]
    //                 .iter()
    //                 .map(|&point| self.distances[[point, individual[i + 1]]])
    //                 .collect();

    //             // Get the index of the closest point
    //             let closest_index = distances
    //                 .iter()
    //                 .enumerate()
    //                 .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
    //                 .map(|(index, _)| index)
    //                 .unwrap();

    //             // Get the index of the nearest point to the next group
    //             let (nearest_point_to_next, _) =
    //                 self.nearest_points[[individual[closest_index], individual[i + 1]]];

    //             // Sum backtrack costs, the number of individual steps
    //             // Between the closest point and the next point
    //             let backtrack_cost: usize = individual[closest_index + 1..=i]
    //                 .iter()
    //                 .map(|&j| self.points[j].len())
    //                 .sum::<usize>()
    //                 + self.points[individual[closest_index]].len()
    //                 - nearest_point_to_next;

    //             // Calculate the sum for this iteration
    //             100.0 * distances[closest_index] / individual.len() as f64
    //                 + 40.0 * backtrack_cost as f64 / self.num_total_points as f64
    //         })
    //         .sum();

    //     sum
    // }

    fn evaluate(&self, individual: &[usize]) -> f64 {
        let mut sum = 0.0;

        // Calculate the distance between the first point and (0, 0)
        sum += distance(
            &self.points[individual[0]].points[0],
            &GCodePoint::new(0.0, 0.0, None, None),
        );

        // Calculate distances in parallel
        let distances_sum: f64 = (0..individual.len() - 1)
            .into_par_iter()
            .map(|i| {
                // Find the minimum distance to the next point
                let (closest_index, min_distance) = individual[..=i]
                    .iter()
                    .enumerate()
                    .map(|(index, &point)| (index, self.distances[[point, individual[i + 1]]]))
                    .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .unwrap();

                // Get the index of the nearest point to the next group
                let (nearest_point_to_next, _) =
                    self.nearest_points[[individual[closest_index], individual[i + 1]]];

                // Calculate the backtrack cost
                let backtrack_cost: usize = individual[closest_index + 1..=i]
                    .iter()
                    .map(|&j| self.points[j].len())
                    .sum::<usize>()
                    + self.points[individual[closest_index]].len()
                    - nearest_point_to_next;

                // if (i == 10) {
                //     println!("min_distance {}", min_distance);
                //     println!("closest_index {}", closest_index);
                //     let test = self.points[individual[closest_index]].len() - nearest_point_to_next;

                //     println!("test {}", test);
                //     println!("backtrack_cost {}", backtrack_cost);
                //     println!("individual.len() {}", individual.len());
                //     println!("self.num_total_points {}", self.num_total_points);

                //     let value = 100.0 * min_distance / individual.len() as f64
                //         + 40.0 * backtrack_cost as f64 / self.num_total_points as f64;

                //     println!("value {}", value);

                //     panic!("hi");
                // }

                // Calculate the sum for this iteration
                100.0 * min_distance / individual.len() as f64
                    + 15.0 * backtrack_cost as f64 / self.num_total_points as f64
            })
            .sum();

        sum + distances_sum
    }

    /// Gets the shortest distance between two groups of points by comparing all pairs of points.
    /// Returns a tuple containing the shortest distance, the index of the point in the first group, and the index of the point in the second group.
    fn get_shortest_distance(
        &self,
        group_1_points: &GCodeBlock,
        group_2_points: &GCodeBlock,
    ) -> (f64, usize, usize) {
        group_1_points
            .points
            .iter()
            .enumerate()
            .flat_map(|(i, point_1)| {
                group_2_points
                    .points
                    .iter()
                    .enumerate()
                    .map(move |(j, point_2)| (distance(point_1, point_2), i, j))
            })
            .min_by(|a, b| {
                a.0.partial_cmp(&b.0)
                    .unwrap_or(Ordering::Greater)
                    .then_with(|| b.2.cmp(&a.2)) // Ensure greatest group_2 index preference
            })
            .unwrap_or_else(|| panic!("Shortest distance not found"))
    }
    pub fn get_shortest_distance_with_backtrack(
        &self,
        individual: &Vec<usize>,
        start_idx: usize,
        end_idx: usize,
    ) -> (usize, usize, usize) {
        let distances_view = self.distances.view();

        // Collect distances from the start group to the end point
        let distances_slice: Vec<f64> = individual[..=start_idx]
            .iter()
            .map(|&idx| distances_view[[idx, individual[end_idx]]])
            .collect();

        // Find the minimum distance and its index
        let (nearest_idx, _) = distances_slice
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Greater))
            .expect("Error: no distances found");

        // Get the intermediate points for the nearest group and the next group
        let group_1 = &self
            .points
            .get(individual[nearest_idx])
            .expect("Invalid nearest_idx");
        let group_2 = &self
            .points
            .get(individual[end_idx])
            .expect("Invalid end_idx");

        // Find the shortest distance between the groups
        let (_, group_1_index, group_2_index) = self.get_shortest_distance(group_1, group_2);

        // Return index of the nearest group, index of the nearest point in the group, and index of the next group
        (nearest_idx, group_1_index, group_2_index)
    }

    fn compute_distances(&mut self) {
        let length = self.points.len();

        // Initialize arrays (using .into_shape() for clarity)
        self.distances = Array2::zeros((length, length))
            .into_shape((length, length))
            .unwrap();
        self.nearest_points = Array2::from_elem((length, length), (0, 0))
            .into_shape((length, length))
            .unwrap();

        self.num_total_points = self.points.iter().map(|p| p.len()).sum();

        // Iterate over pairs of points using nested iterators
        // TODO: Consider using par_iter , might be faster
        (0..length).for_each(|i| {
            for j in 0..length {
                if i != j {
                    let (dist, point_i_index, point_j_index) =
                        self.get_shortest_distance(&self.points[i], &self.points[j]);

                    self.distances[(i, j)] = dist;
                    self.nearest_points[(i, j)] = (point_i_index, point_j_index);
                }
            }
        });
    }
}
