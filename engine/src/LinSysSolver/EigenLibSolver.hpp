//  Created by Minchen Li on 6/30/18.

#ifndef EigenLibSolver_hpp
#define EigenLibSolver_hpp

#include "LinSysSolver.hpp"

#include <Eigen/Eigen>

#include <vector>
#include <set>

namespace uvgami {

template <typename vectorTypeI, typename vectorTypeS>
class EigenLibSolver : public LinSysSolver<vectorTypeI, vectorTypeS> {
    typedef LinSysSolver<vectorTypeI, vectorTypeS> Base;

  protected:
    bool useDense;
    Eigen::MatrixXd coefMtr_dense;
    Eigen::LDLT<Eigen::MatrixXd> LDLT;
    Eigen::SparseMatrix<double> coefMtr;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> simplicialLDLT;

  public:
    void set_type(int threadAmt, int _mtype, bool is_upper_half = false);

    void set_pattern(const std::vector<std::set<int>> &vNeighbor,
                     const std::set<int> &fixedVert);
    void set_pattern(
        const Eigen::SparseMatrix<double> &mtr); // NOTE: mtr must be SPD

    void update_a(const vectorTypeI &II, const vectorTypeI &JJ,
                  const vectorTypeS &SS);

    void analyze_pattern(void);

    bool factorize(void);

    void solve(Eigen::VectorXd &rhs, Eigen::VectorXd &result);

    double coeffMtr(int rowI, int colI) const;

    void setZero(void);

    virtual void setCoeff(int rowI, int colI, double val);

    virtual void addCoeff(int rowI, int colI, double val);
};

} // namespace uvgami

#endif
