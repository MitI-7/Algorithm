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
// A is row-full rank
pub struct SimplexMethod {
    // num variables
    m: usize,
    // num constraints
    sense: Sense,
    n: usize,
    tableau: Vec<Vec<f64>>,
    base: Vec<usize>,
    eps: f64,

    pub objective: Option<f64>,
}

impl SimplexMethod {
    pub fn new(sense: Sense) -> Self {
        SimplexMethod { n: 0, m: 0, sense, tableau: vec![vec![]], base: vec![], eps: 1e-6, objective: None }
    }

    #[allow(non_snake_case)]
    pub fn solve(&mut self, A: Vec<Vec<f64>>, b: Vec<f64>, c: Vec<f64>) -> Status {
        self.n = A[0].len();
        self.m = A.len();
        assert_eq!(b.len(), self.m);
        assert_eq!(c.len(), self.n);

        self.make_tableau(A, b, c);

        // phase1
        let status = self.phase1();
        if status != Status::Optimal {
            return status;
        }

        // todo: pivot if artificial variables are in base
        for i in 0..self.base.len() {
            assert!(self.base[i] < self.n);
        }

        // remove artificial variables
        for c in 0..self.tableau[0].len() {
            self.tableau[self.m + 1][c] = self.tableau[self.m][c];
        }
        for r in 0..self.tableau.len() {
            self.tableau[r][self.n] = self.tableau[r][self.tableau[r].len() - 1];

            while self.tableau[r].len() > self.n + 1 {
                self.tableau[r].pop();
            }
        }

        self.phase2()
    }

    #[allow(non_snake_case)]
    fn make_tableau(&mut self, A: Vec<Vec<f64>>, b: Vec<f64>, c: Vec<f64>) {
        self.tableau = vec![vec![0.0; self.n + self.m + 1]; self.m + 2];

        // set A and b
        for i in 0..self.m {
            for j in 0..self.n {
                self.tableau[i][j] = A[i][j];
                if b[i] < 0.0 {
                    self.tableau[i][j] *= -1.0;
                }
            }
            self.tableau[i][self.n + self.m] = b[i].abs();
        }

        // set c
        for j in 0..self.n {
            self.tableau[self.m][j] = -c[j];
            if self.sense == Sense::Minimize {
                self.tableau[self.m][j] *= -1.0;
            }
        }

        // set artificial variables
        for i in 0..self.m {
            self.tableau[i][self.n + i] = 1.0;
        }
        for i in 0..self.m {
            self.tableau[self.m + 1][self.n + i] = 1.0;
        }

        self.base = vec![0; self.m];
        // set base variables
        for i in 0..self.m {
            self.base[i] = self.n + i; // artificial variable
        }

        // pivot artificial variables
        for i in 0..self.m {
            for j in 0..self.tableau[0].len() {
                self.tableau[self.m + 1][j] -= self.tableau[i][j];
            }
        }
    }

    fn phase1(&mut self) -> Status {
        let status = self.simplex();
        if status != Status::Optimal {
            return status;
        }

        if self.objective.unwrap().abs() > self.eps {
            return Status::Infeasible;
        }

        Status::Optimal
    }

    fn phase2(&mut self) -> Status {
        self.simplex()
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

        self.objective = Some(self.tableau[self.tableau.len() - 1][self.tableau[0].len() - 1]);
        if self.sense == Sense::Minimize {
            self.objective = Some(-1.0 * self.objective.unwrap());
        }
        Status::Optimal
    }

    fn bland(&self) -> (Option<usize>, Option<usize>) {
        let mut c: Option<usize> = None;
        for j in 0..self.tableau[0].len() - 1 {
            if self.tableau[self.tableau.len() - 1][j] < 0.0 {
                c = Some(j);
                break;
            }
        }
        // optimal
        if c.is_none() {
            return (None, None);
        }

        let mut r: Option<usize> = None;
        for i in 0..self.tableau.len() - 2 {
            if self.tableau[i][c.unwrap()] <= 0.0 {
                continue;
            }
            if r.is_none() {
                r = Some(i);
            } else {
                let now = self.tableau[i][self.tableau[0].len() - 1] / self.tableau[i][c.unwrap()]; // b[i] / a[i][j]
                let best = self.tableau[r.unwrap()][self.tableau[0].len() - 1] / self.tableau[r.unwrap()][c.unwrap()];
                if now < best {
                    r = Some(i);
                }
            }
        }

        (r, c)
    }

    fn pivot(&mut self, r: usize, c: usize) {
        for i in 0..self.tableau.len() {
            for j in 0..self.tableau[0].len() {
                if i != r && j != c {
                    self.tableau[i][j] -= self.tableau[r][j] * self.tableau[i][c] / self.tableau[r][c];
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
    use super::{Sense, SimplexMethod, Status};

    #[test]
    fn test1() {
        let mut solver = SimplexMethod::new(Sense::Maximize);

        let A = vec![vec![1.0, 1.0, 1.0, 0.0, 0.0], vec![1.0, 3.0, 0.0, 1.0, 0.0], vec![2.0, 1.0, 0.0, 0.0, 1.0]];
        let b = vec![6.0, 12.0, 10.0];
        let c = vec![1.0, 2.0, 0.0, 0.0, 0.0];

        let status = solver.solve(A, b, c);
        assert_eq!(status, Status::Optimal);
        assert_eq!(solver.objective.unwrap(), 9.0);
    }

    #[test]
    fn test2() {
        let mut solver = SimplexMethod::new(Sense::Maximize);

        let A = vec![vec![1.0, 1.0, 1.0, 0.0, 0.0], vec![1.0, 3.0, 0.0, 1.0, 0.0], vec![-3.0, -2.0, 0.0, 0.0, 1.0]];
        let b = vec![6.0, 12.0, -6.0];
        let c = vec![1.0, 2.0, 0.0, 0.0, 0.0];

        let status = solver.solve(A, b, c);
        assert_eq!(status, Status::Optimal);
        assert_eq!(solver.objective.unwrap(), 9.0);
    }

    #[test]
    fn test3() {
        let mut solver = SimplexMethod::new(Sense::Minimize);

        let A = vec![vec![-2.0, -1.0, 1.0, 0.0, 0.0], vec![-1.0, -1.0, 0.0, 1.0, 0.0], vec![-1.0, -2.0, 0.0, 0.0, 1.0]];
        let b = vec![-8.0, -6.0, -8.0];
        let c = vec![4.0, 3.0, 0.0, 0.0, 0.0];

        let status = solver.solve(A, b, c);
        assert_eq!(status, Status::Optimal);
        assert_eq!(solver.objective.unwrap(), 20.0);
    }

    #[test]
    fn test4() {
        let mut solver = SimplexMethod::new(Sense::Minimize);

        let A = vec![vec![-1.0, -3.0, 1.0, 0.0], vec![-2.0, -1.0, 0.0, 1.0]];
        let b = vec![-4.0, -3.0];
        let c = vec![4.0, 1.0, 0.0, 0.0];

        let status = solver.solve(A, b, c);
        assert_eq!(status, Status::Optimal);
        assert_eq!(solver.objective.unwrap(), 3.0);
    }

    #[test]
    fn test5() {
        let mut solver = SimplexMethod::new(Sense::Maximize);

        let A = vec![vec![1.0, -2.0, 1.0, 0.0], vec![-1.0, 1.0, 0.0, 1.0]];
        let b = vec![4.0, 2.0];
        let c = vec![2.0, 1.0, 0.0, 0.0];

        let status = solver.solve(A, b, c);
        assert_eq!(status, Status::Unbounded);
    }
}
