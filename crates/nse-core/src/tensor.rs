//! Lightweight dense matrix / vector views shared across crates.
//!
//! For the POC we avoid a heavy BLAS dependency. A [`Matrix`] owns its row-major
//! `Vec<f32>` data; a [`MatrixView`] borrows it. Both are `f32` only — the NSE
//! pipeline keeps activations in `f32` and weights in ternary/PQ forms.

use crate::error::{NseError, NseResult};

/// Owned row-major `f32` matrix of shape `[rows, cols]`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Matrix {
    /// Create a zero matrix of the given shape.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    /// Create from raw row-major data, validating length.
    pub fn from_vec(rows: usize, cols: usize, data: Vec<f32>) -> NseResult<Self> {
        if data.len() != rows * cols {
            return Err(NseError::ShapeMismatch {
                expected: vec![rows * cols],
                got: vec![data.len()],
            });
        }
        Ok(Self { rows, cols, data })
    }

    /// Number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Element access `(row, col)` with bounds checking.
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> NseResult<f32> {
        if row >= self.rows || col >= self.cols {
            return Err(NseError::IndexOutOfBounds {
                index: row * self.cols + col,
                len: self.len(),
            });
        }
        Ok(self.data[row * self.cols + col])
    }

    /// Mutable element access `(row, col)` with bounds checking.
    #[inline]
    pub fn get_mut(&mut self, row: usize, col: usize) -> NseResult<&mut f32> {
        if row >= self.rows || col >= self.cols {
            return Err(NseError::IndexOutOfBounds {
                index: row * self.cols + col,
                len: self.len(),
            });
        }
        Ok(&mut self.data[row * self.cols + col])
    }

    /// A borrowed view over the whole matrix.
    pub fn view(&self) -> MatrixView<'_> {
        MatrixView {
            rows: self.rows,
            cols: self.cols,
            data: &self.data,
        }
    }

    /// A borrowed view over a single row.
    pub fn row(&self, row: usize) -> NseResult<&[f32]> {
        if row >= self.rows {
            return Err(NseError::IndexOutOfBounds {
                index: row,
                len: self.rows,
            });
        }
        Ok(&self.data[row * self.cols..(row + 1) * self.cols])
    }

    /// Transpose into a new owned matrix.
    pub fn transposed(&self) -> Matrix {
        let mut out = Matrix::zeros(self.cols, self.rows);
        for r in 0..self.rows {
            for c in 0..self.cols {
                out.data[c * self.rows + r] = self.data[r * self.cols + c];
            }
        }
        out
    }
}

/// Borrowed row-major `f32` matrix view.
#[derive(Debug, Clone, Copy)]
pub struct MatrixView<'a> {
    pub rows: usize,
    pub cols: usize,
    pub data: &'a [f32],
}

impl<'a> MatrixView<'a> {
    /// View a raw slice as a row-major matrix (unchecked layout).
    pub fn from_slice(rows: usize, cols: usize, data: &'a [f32]) -> NseResult<Self> {
        if data.len() != rows * cols {
            return Err(NseError::ShapeMismatch {
                expected: vec![rows * cols],
                got: vec![data.len()],
            });
        }
        Ok(Self { rows, cols, data })
    }

    /// View over a single row.
    pub fn row(&self, row: usize) -> NseResult<&'a [f32]> {
        if row >= self.rows {
            return Err(NseError::IndexOutOfBounds {
                index: row,
                len: self.rows,
            });
        }
        Ok(&self.data[row * self.cols..(row + 1) * self.cols])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_get_set_roundtrip() {
        let mut m = Matrix::zeros(2, 3);
        *m.get_mut(1, 2).unwrap() = 7.0;
        assert_eq!(m.get(1, 2).unwrap(), 7.0);
        assert!(m.get(2, 0).is_err());
    }

    #[test]
    fn transpose_roundtrip() {
        let mut m = Matrix::zeros(2, 3);
        *m.get_mut(0, 1).unwrap() = 1.0;
        *m.get_mut(1, 2).unwrap() = 2.0;
        let t = m.transposed();
        assert_eq!(t.rows, 3);
        assert_eq!(t.cols, 2);
        assert_eq!(t.get(1, 0).unwrap(), 1.0);
        assert_eq!(t.get(2, 1).unwrap(), 2.0);
    }
}
