#![feature(sort_floats)]
#![no_std]

// use core::cmp::Ordering;
use nalgebra::{ComplexField, SMatrix, SMatrixView, Vector3};

/// Lightweight least squares approach to
/// determining the offset and scaling
/// factors for gyroscope calibration.
/// Also includes the capability to automatically
/// collect good data points, using a `const`-sized
/// buffer matrix, and a k-nearest neighbors
pub struct MagCalibrator<const N: usize, const M: usize> {
    matrix: SMatrix<f32, N, 6>,
    matrix_filled: usize,
    mean_distance: f32,
    pre_scaler: f32,
    k: usize,
}

impl<const N: usize, const M: usize> Default for MagCalibrator<N, M> {
    fn default() -> Self {
        Self {
            matrix: SMatrix::from_element(1.0),
            matrix_filled: Default::default(),
            mean_distance: Default::default(),
            pre_scaler: 1.,
            k: 2, // Works well in testing
        }
    }
}

impl<const N: usize, const M: usize> MagCalibrator<N, M> {
    /// Create a new calibrator
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new calibrator with a custom number of neighbors to check
    pub fn new_define_k(k: usize) -> Self {
        Self{ k, ..Default::default() }
    }

    /// Calculates mean distance to the `k` nearest neighbors.
    /// A smaller number means the point is "similar" to its neighbors.
    fn mean_distance_from_single(&self, vec: SMatrix<f32, 1, 3>) -> f32 {
        let matrix_view: SMatrixView<f32, N, 3> = self.matrix.fixed_columns::<3>(0);

        // Distane to every other point
        let mut squared_dists: [f32; N] = [0.; N];
        matrix_view.row_iter().enumerate().for_each(|(j, cmp)| {
            let diff = vec - cmp;
            squared_dists[j] = diff.dot(&diff).sqrt(); // ?
        });

        // Sort floats and return mean distance to nearest neighbors
        squared_dists.sort_floats();
        squared_dists
            .iter()
            .take(self.k + 1)
            .rfold(0., |a, &b| a + b)
            / N as f32
    }

    /// Calculates mean squared distance to the `k` nearest neighbors
    /// between all `N` row vectors in the internal buffer.
    /// A smaller number means a point is "similar" to its neighbors
    fn mean_distance_from_all(&self) -> [f32; N] {
        let mut msd: [f32; N] = [0.; N];

        let matrix_view: SMatrixView<f32, N, 3> = self.matrix.fixed_columns::<3>(0);
        matrix_view.row_iter().enumerate().for_each(|(i, row)| {
            msd[i] = self.mean_distance_from_single(row.into());
        });
        msd
    }

    /// Returns index of vector that returns the lowest mean squared distance
    /// Is used when replacing the least useful value in the array
    fn lowest_mean_distance_by_index(&mut self) -> (usize, f32) {
        let msd = self.mean_distance_from_all();

        // Set mean msd now that we are at it
        self.mean_distance = msd.iter().rfold(0., |a, &b| a + b) / N as f32;

        // Obtain index for lowest msd value
        msd.iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal))
            .map(|(index, value)| (index, *value))
            .unwrap()
    }

    /// Evaluates whether the new sample should replace one already in the buffer.
    pub fn evaluate_sample(&mut self, x: [f32; 3]) {
        self.evaluate_sample_vec(Vector3::from(x))
    }

    /// Add a sample if it is deemed more usedful than the least useful sample
    pub fn evaluate_sample_vec(&mut self, x: Vector3<f32>) {
        // Check if buffer is not yet "intialized" with real measurements
        if self.matrix_filled < N {
            self.add_sample_at(self.matrix_filled, x);
            self.matrix_filled += 1;
        }
        // Otherwise check which sample may be best to replace
        else {
            let (low_index, low_msd) = self.lowest_mean_distance_by_index();
            let sample_msd = self.mean_distance_from_single(x.transpose());
            if low_msd < sample_msd {
                self.add_sample_at(low_index, x);
            }
        }
    }

    /// Insert a sample vector into `index` row of buffer matrix
    fn add_sample_at(&mut self, index: usize, sample: Vector3<f32>) {
        self.matrix[(index, 0)] = sample[0] / self.pre_scaler;
        self.matrix[(index, 1)] = sample[1] / self.pre_scaler;
        self.matrix[(index, 2)] = sample[2] / self.pre_scaler;
    }

    /// Get mean distance value between samples in matrix buffer
    pub fn mean_distane(&self) -> f32 {
        self.mean_distance
    }

    /// Set sample pre scaler, prevents ill-conditioning
    pub fn set_pre_scaler(&mut self, scaler: f32) {
        self.pre_scaler = scaler;
    }

    /// Try to calculate calibration offset and scale values. Returns None if
    /// it was not possible to calculate the pseudo inverse, or if some of the
    /// parameters are `NaN`. The tuple contains (offset , scale)
    pub fn perform_calibration(&mut self) -> Option<([f32; 3], [f32; 3])> {
        // Calculate column 4 and 5 of H matrix
        self.matrix.row_iter_mut().for_each(|mut mag| {
            mag[3] = -mag[1] * mag[1];
            mag[4] = -mag[2] * mag[2];
        });

        // Calculate W vector
        let mut w: SMatrix<f32, N, 1> = SMatrix::from_element(0.0);
        self.matrix
            .row_iter()
            .enumerate()
            .for_each(|(i, row)| w[i] = row[0] * row[0]);

        // Perform least squares using pseudo inverse
        let x =
            (self.matrix.transpose() * self.matrix).try_inverse()? * self.matrix.transpose() * w;

        // Calculate offsets and scale factors
        let off = [x[0] / 2., x[1] / (2. * x[3]), x[2] / (2. * x[4])];
        let temp = x[5] + (off[0] * off[0]) + x[3] * (off[1] * off[1]) + x[4] * (off[2] * off[2]);
        let scale = [temp.sqrt(), (temp / x[3]).sqrt(), (temp / x[4]).sqrt()];

        // Check that off and scale vectors contain valid values
        for x in off.iter().chain(scale.iter()) {
            if !x.is_finite() {
                return None;
            }
        }

        // TODO Add option for low-pass filtering this result
        Some((off,scale))
    }
}