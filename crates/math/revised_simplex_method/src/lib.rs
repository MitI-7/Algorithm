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
// b >= 0
// A is row-full rank
#[allow(non_snake_case)]
pub struct RevisedSimplexMethod {
    n: usize,         // num variables
    m: usize,         // num constraints
    A: Vec<Vec<f64>>, // m * n
    b: Vec<f64>,      // m * 1
    c: Vec<f64>,      // 1 * n
    sense: Sense,     // maximize or minimize

    base: Vec<usize>,
    non_base: Vec<usize>,

    pub objective: Option<f64>,
}

#[allow(non_snake_case)]
impl RevisedSimplexMethod {
    pub fn new(sense: Sense) -> Self {
        RevisedSimplexMethod { n: 0, m: 0, A: vec![vec![]], b: vec![], c: vec![], sense, base: vec![], non_base: vec![], objective: None }
    }

    pub fn solve(&mut self, A: Vec<Vec<f64>>, b: Vec<f64>, c: Vec<f64>) -> Status {
        self.n = A[0].len();
        self.m = A.len();
        assert_eq!(b.len(), self.m);
        assert_eq!(c.len(), self.n);

        self.A = A;
        self.b = b;
        self.c = c;

        if self.sense == Sense::Minimize {
            for i in 0..self.n {
                self.c[i] *= -1.0;
            }
        }

        // add slack variables
        for i in 0..self.m {
            for j in 0..self.m {
                if i == j {
                    self.A[i].push(1.0);
                } else {
                    self.A[i].push(0.0);
                }
            }
            self.c.push(0.0);
        }

        // set base variables and non-base variables
        for i in 0..self.m {
            self.base.push(self.n + i);
        }
        for i in 0..self.n {
            self.non_base.push(i);
        }

        self.simplex()
    }

    fn simplex(&mut self) -> Status {
        // inverse of basis matrix
        let mut U = vec![vec![0.0; self.m]; self.m];
        for i in 0..self.m {
            U[i][i] = 1.0;
        }

        loop {
            // U * b
            let mut b_ver = vec![0.0; self.m];
            for i in 0..self.m {
                for j in 0..self.m {
                    b_ver[i] += U[i][j] * self.b[j];
                }
            }

            // select column s
            let s;
            match self.select_s(&U) {
                Some(x) => s = x,
                None => {
                    // optimal
                    break;
                }
            }

            // A[*][s]
            let mut a_s = vec![0.0; self.m];
            for i in 0..a_s.len() {
                for j in 0..self.m {
                    a_s[i] += U[i][j] * self.A[j][self.non_base[s]];
                }
            }

            let r;
            match self.select_r(&a_s, &b_ver) {
                Some(x) => r = x,
                None => {
                    return Status::Unbounded;
                }
            }

            // swap base and non-base
            let tmp = self.non_base[s];
            self.non_base[s] = self.base[r];
            self.base[r] = tmp;

            // U = UT
            let mut new_u = vec![vec![0.0; self.m]; self.m];
            for i in 0..self.m {
                for j in 0..self.m {
                    if i != r {
                        new_u[i][j] += U[i][j] + -a_s[i] / a_s[r] * U[r][j];
                    } else {
                        new_u[i][j] += 1.0 / a_s[r] * U[i][j];
                    }
                }
            }

            U = new_u;
        }

        self.objective = Some(self.calculate_objective(&U));
        Status::Optimal
    }

    fn calculate_objective(&self, U: &Vec<Vec<f64>>) -> f64 {
        let mut obj = 0.0;
        for i in 0..self.m {
            for j in 0..self.m {
                obj += self.c[self.base[j]] * U[j][i] * self.b[i];
            }
        }

        if self.sense == Sense::Minimize {
            obj *= -1.0;
        }
        obj
    }

    // c_ver = c_n - c_b * U * A_n
    fn select_s(&self, U: &Vec<Vec<f64>>) -> Option<usize> {
        // c_b * U
        let mut tmp = vec![0.0; self.m];
        for i in 0..self.m {
            for j in 0..self.m {
                tmp[i] += self.c[self.base[j]] * U[j][i];
            }
        }

        // c_n - tmp * A_n
        let mut c_ver = vec![0.0; self.non_base.len()];
        for i in 0..c_ver.len() {
            for j in 0..self.m {
                c_ver[i] += tmp[j] * self.A[j][self.non_base[i]];
            }
        }
        for i in 0..c_ver.len() {
            c_ver[i] = self.c[self.non_base[i]] - c_ver[i];
        }

        // select s
        for i in 0..c_ver.len() {
            if c_ver[i] > 0.0 {
                return Some(i);
            }
        }

        None
    }

    fn select_r(&self, a_s: &Vec<f64>, b_ver: &Vec<f64>) -> Option<usize> {
        let mut r: Option<usize> = None;
        for i in 0..self.m {
            if a_s[i] <= 0.0 {
                continue;
            }

            if r.is_none() {
                r = Some(i);
            } else {
                let now = b_ver[i] / a_s[i];
                let best = b_ver[r.unwrap()] / a_s[r.unwrap()];
                if now < best {
                    r = Some(i);
                }
            }
        }
        r
    }
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::{RevisedSimplexMethod, Sense, Status};

    #[test]
    fn test1() {
        let mut solver = RevisedSimplexMethod::new(Sense::Maximize);
        let A = vec![vec![2.0, 2.0, -1.0, 1.0, 0.0, 0.0], vec![2.0, 0.0, 4.0, 0.0, 1.0, 0.0], vec![-4.0, 3.0, -1.0, 0.0, 0.0, 1.0]];
        let b = vec![6.0, 4.0, 1.0];
        let c = vec![2.0, 1.0, 1.0, 0.0, 0.0, 0.0];

        let status = solver.solve(A, b, c);
        assert_eq!(status, Status::Optimal);
        assert_eq!(solver.objective.unwrap(), 5.0);
    }

    #[test]
    fn test2() {
        let mut solver = RevisedSimplexMethod::new(Sense::Maximize);

        let A = vec![vec![1.0, 1.0, 1.0, 0.0, 0.0], vec![1.0, 3.0, 0.0, 1.0, 0.0], vec![2.0, 1.0, 0.0, 0.0, 1.0]];
        let b = vec![6.0, 12.0, 10.0];
        let c = vec![1.0, 2.0, 0.0, 0.0, 0.0];

        let status = solver.solve(A, b, c);
        assert_eq!(status, Status::Optimal);
        assert_eq!(solver.objective.unwrap(), 9.0);
    }

    #[test]
    fn test3() {
        let mut solver = RevisedSimplexMethod::new(Sense::Maximize);

        let A = vec![vec![1.0, -2.0, 1.0, 0.0], vec![-1.0, 1.0, 0.0, 1.0]];
        let b = vec![4.0, 2.0];
        let c = vec![2.0, 1.0, 0.0, 0.0];

        let status = solver.solve(A, b, c);
        assert_eq!(status, Status::Unbounded);
    }
}