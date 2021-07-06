#[derive(PartialEq, Debug)]
pub enum Sense {
    Minimize,
    Maximize,
}

#[derive(PartialEq, Debug)]
pub enum Status {
    Optimal,
    Infeasible,
    Unbounded,
}

// min or max z = c x
// s.t Ax = b
// c <= 0(when maximize)
// c >= 0(when minimize)
// A is row-full rank
pub struct DualSimplexMethod {
    n: usize, // num variables
    m: usize, // num constraints
    sense: Sense,
    tableau: Vec<Vec<f64>>,
    base: Vec<usize>,

    pub objective: f64,
}

#[allow(non_snake_case)]
impl DualSimplexMethod {
    pub fn new(sense: Sense) -> Self {
        DualSimplexMethod { n: 0, m: 0, sense, tableau: vec![vec![]], base: vec![], objective: 0.0 }
    }

    pub fn solve(&mut self, A: Vec<Vec<f64>>, b: Vec<f64>, c: Vec<f64>) -> Status {
        self.n = A[0].len();
        self.m = A.len();
        assert_eq!(b.len(), self.m);
        assert_eq!(c.len(), self.n);

        self.make_tableau(A, b, c);

        self.simplex()
    }

    fn make_tableau(&mut self, A: Vec<Vec<f64>>, b: Vec<f64>, c: Vec<f64>) {
        self.tableau = vec![vec![0.0; self.n + self.m + 1]; self.m + 1];

        // set A
        for i in 0..self.m {
            for j in 0..self.n {
                self.tableau[i][j] = A[i][j];
            }
        }

        // set b
        for i in 0..self.m {
            self.tableau[i][self.n + self.m] = b[i];
        }

        // set c
        for j in 0..self.n {
            self.tableau[self.m][j] = c[j];
            if self.sense == Sense::Minimize {
                self.tableau[self.m][j] *= -1.0;
            }
            assert!(self.tableau[self.m][j] <= 0.0);
        }

        // set artificial variables
        for i in 0..self.m {
            self.tableau[i][self.n + i] = 1.0;
        }

        // set base
        self.base = vec![0; self.m];
        for i in 0..self.m {
            self.base[i] = self.n + i; // artificial variable
        }
    }

    fn simplex(&mut self) -> Status {
        loop {
            let (r, c) = self.bland();

            if c.is_none() {
                break;
            }
            if r.is_none() && !c.is_none() {
                return Status::Unbounded;
            }

            self.pivot(r.unwrap(), c.unwrap());
        }

        self.objective = self.tableau[self.tableau.len() - 1][self.tableau[0].len() - 1];
        if self.sense == Sense::Maximize {
            self.objective *= -1.0;
        }

        Status::Optimal
    }

    fn bland(&self) -> (Option<usize>, Option<usize>) {
        let mut best_r: Option<usize> = None;
        for i in 0..self.tableau.len() - 1 {
            if self.tableau[i][self.tableau[0].len() - 1] < 0.0 {
                best_r = Some(i);
                break;
            }
        }
        // optimal
        if best_r.is_none() {
            return (None, None);
        }

        let mut best_s: Option<usize> = None;
        for s in 0..self.tableau[0].len() - 1 {
            let a = self.tableau[best_r.unwrap()][s];
            if a >= 0.0 {
                continue;
            }
            if best_s.is_none() {
                best_s = Some(s);
            } else {
                let now = self.tableau[self.tableau.len() - 1][s] / a;
                let pre = self.tableau[self.tableau.len() - 1][best_s.unwrap()] / self.tableau[best_r.unwrap()][best_s.unwrap()];
                if now < pre {
                    best_s = Some(s);
                }
            }
        }

        (best_r, best_s)
    }

    fn pivot(&mut self, r: usize, c: usize) {
        for i in 0..self.tableau.len() {
            for j in 0..self.tableau[0].len() {
                if i != r && j != c {
                    self.tableau[i][j] -= self.tableau[r][j] * (self.tableau[i][c] / self.tableau[r][c]);
                }
            }
        }

        for i in 0..self.tableau.len() {
            if i != r {
                self.tableau[i][c] = 0.0;
            }
        }
        for j in 0..self.tableau[0].len() {
            if j != c {
                self.tableau[r][j] /= self.tableau[r][c];
            }
        }

        self.tableau[r][c] = 1.0;
        self.base[r] = c;
    }
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::{DualSimplexMethod, Sense, Status};

    #[test]
    fn test1() {
        let mut solver = DualSimplexMethod::new(Sense::Maximize);

        let A = vec![vec![-2.0, -1.0, 0.0, 1.0, 0.0, 0.0], vec![0.0, 0.0, -3.0, 0.0, 1.0, 0.0], vec![0.0, -2.0, -1.0, 0.0, 0.0, 1.0]];
        let b = vec![-2.0, -3.0, -2.0];
        let c = vec![-4.0, -6.0, -8.0, 0.0, 0.0, 0.0];

        let status = solver.solve(A, b, c);
        assert_eq!(status, Status::Optimal);
        assert_eq!(solver.objective, -14.0);
    }

    #[test]
    fn test2() {
        let mut solver = DualSimplexMethod::new(Sense::Minimize);

        let A = vec![vec![-1.0, 1.0, -1.0, 1.0, 1.0, 0.0, 0.0], vec![-1.0, 2.0, -3.0, 4.0, 0.0, 1.0, 0.0], vec![-3.0, 4.0, -5.0, 6.0, 0.0, 0.0, 1.0]];
        let b = vec![-10.0, -1.0, -3.0];
        let c = vec![2.0, 3.0, 4.0, 5.0, 0.0, 0.0, 0.0];

        let status = solver.solve(A, b, c);
        assert_eq!(status, Status::Optimal);
        assert_eq!(solver.objective, 20.0);
    }

    #[test]
    fn test3() {
        let mut solver = DualSimplexMethod::new(Sense::Maximize);

        let A = vec![vec![-1.0, -1.0, 2.0, 1.0, 0.0, 0.0], vec![-4.0, -2.0, 1.0, 0.0, 1.0, 0.0], vec![1.0, 1.0, -1.0, 0.0, 0.0, 1.0]];
        let b = vec![-3.0, -4.0, 2.0];
        let c = vec![-4.0, -2.0, -1.0, 0.0, 0.0, 0.0];

        let status = solver.solve(A, b, c);
        assert_eq!(status, Status::Optimal);
        assert_eq!(solver.objective, -6.0);
    }
}
