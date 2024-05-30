use std::ops::Sub;

#[derive(Debug, Clone)]

pub struct GCodePoint {
    // X coordinate of the point
    x: f64,
    // Y coordinate of the point
    y: f64,
    // Z coordinate of the point
    z: Option<f64>,
    // F coordinate of the point
    f: Option<f64>,
}

impl GCodePoint {
    // Create a new GCodePoint
    pub fn new(x: f64, y: f64, z: Option<f64>, f: Option<f64>) -> Self {
        Self { x, y, z, f }
    }
}

impl Sub for GCodePoint {
    type Output = GCodePoint;

    fn sub(self, other: GCodePoint) -> GCodePoint {
        GCodePoint {
            x: self.x - other.x,
            y: self.y - other.y,
            z: match (self.z, other.z) {
                (Some(z1), Some(z2)) => Some(z1 - z2),
                _ => None, // If either z is None, result z is None
            },
            f: match (self.f, other.f) {
                (Some(f1), Some(f2)) => Some(f1 - f2),
                _ => None, // If either f is None, result f is None
            },
        }
    }
}

// Create a distance function that operates on GCodePoints
pub fn distance(a: &GCodePoint, b: &GCodePoint) -> f64 {
    ((a.x - b.x).powi(2) + (a.y - b.y).powi(2)).sqrt()
}

#[derive(Debug)]
pub struct GCodeBlock {
    // Vector of points in the block
    pub points: Vec<GCodePoint>,
}

impl GCodeBlock {
    // Get the length of the GCodeBlock
    pub fn len(&self) -> usize {
        self.points.len()
    }
}

impl GCodeBlock {
    // Create a new GCodeBlock
    pub fn new(points: Vec<GCodePoint>) -> Self {
        Self { points }
    }
}
