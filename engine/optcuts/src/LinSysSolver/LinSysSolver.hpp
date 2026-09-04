//  Created by Minchen Li on 6/30/18.

#ifndef LinSysSolver_hpp
#define LinSysSolver_hpp

#define DIM 2

#include <Eigen/Eigen>
#include <Eigen/Sparse>

#include <algorithm>
#include <set>
#include <vector>
#include <iostream>

namespace uvgami {

template <typename vectorTypeI, typename vectorTypeS> class LinSysSolver {
  protected:
    int numRows;
    // 0-based CSR of the upper triangle, columns sorted within each row
    Eigen::VectorXi ia, ja;
    Eigen::VectorXd a;
    // destination in a of each assembly triplet, -1 for a lower-triangle entry
    std::vector<int> tripletDest;

    int find_a_index(int rowI, int colI) const {
        const int *rowBegin = ja.data() + ia[rowI];
        const int *rowEnd = ja.data() + ia[rowI + 1];
        const int *found = std::lower_bound(rowBegin, rowEnd, colI);
        if ((found != rowEnd) && (*found == colI)) {
            return static_cast<int>(ia[rowI] + (found - rowBegin));
        }
        return -1;
    }

  public:
    virtual ~LinSysSolver(void) {};

  public:
    virtual void set_type(int threadAmt, int _mtype,
                          bool is_upper_half = false) = 0;

    virtual void set_pattern(const std::vector<std::set<int>> &vNeighbor,
                             const std::set<int> &fixedVert) {
        const int nV = static_cast<int>(vNeighbor.size());
        numRows = nV * DIM;
        tripletDest.clear();

        std::vector<char> isFixed(nV, 0);
        for (const auto fixedVI : fixedVert) {
            isFixed[fixedVI] = 1;
        }

        // count nnz per row first so ia and ja are allocated exactly once
        ia.resize(numRows + 1);
        ia[0] = 0;
        for (int vI = 0; vI < nV; vI++) {
            if (!isFixed[vI]) {
                int nnzBlocks = 1;
                for (const auto &nbVI : vNeighbor[vI]) {
                    if ((nbVI > vI) && !isFixed[nbVI]) {
                        nnzBlocks++;
                    }
                }
                for (int d = 0; d < DIM; d++) {
                    ia[vI * DIM + d + 1] =
                        ia[vI * DIM + d] + nnzBlocks * DIM - d;
                }
            } else {
                for (int d = 0; d < DIM; d++) {
                    ia[vI * DIM + d + 1] = ia[vI * DIM + d] + 1;
                }
            }
        }

        ja.resize(ia[numRows]);
        for (int vI = 0; vI < nV; vI++) {
            if (!isFixed[vI]) {
                for (int d = 0; d < DIM; d++) {
                    int dst = ia[vI * DIM + d];
                    // vNeighbor sets are sorted, so each row's columns come out sorted
                    for (int dj = d; dj < DIM; dj++) {
                        ja[dst++] = vI * DIM + dj;
                    }
                    for (const auto &nbVI : vNeighbor[vI]) {
                        if ((nbVI > vI) && !isFixed[nbVI]) {
                            for (int dj = 0; dj < DIM; dj++) {
                                ja[dst++] = nbVI * DIM + dj;
                            }
                        }
                    }
                    assert(dst == ia[vI * DIM + d + 1]);
                }
            } else {
                for (int d = 0; d < DIM; d++) {
                    ja[ia[vI * DIM + d]] = vI * DIM + d;
                }
            }
        }
        a.resize(ja.size());
    }
    virtual void set_pattern(
        const Eigen::SparseMatrix<double> &mtr) = 0; // NOTE: mtr must be SPD

    virtual void update_a(const vectorTypeI &II, const vectorTypeI &JJ,
                          const vectorTypeS &SS) {
        assert(II.size() == JJ.size());
        assert(II.size() == SS.size());

        // the triplet layout only changes with topology, which goes through set_pattern
        if (tripletDest.size() != static_cast<size_t>(II.size())) {
            tripletDest.resize(II.size());
            for (int tripletI = 0; tripletI < II.size(); tripletI++) {
                int i = II[tripletI], j = JJ[tripletI];
                if (i <= j) {
                    tripletDest[tripletI] = find_a_index(i, j);
                    assert(tripletDest[tripletI] >= 0);
                } else {
                    tripletDest[tripletI] = -1;
                }
            }
        }

        a.setZero(ja.size());
        for (int tripletI = 0; tripletI < II.size(); tripletI++) {
            const int dest = tripletDest[tripletI];
            if (dest >= 0) {
                a[dest] += SS[tripletI];
            }
        }
    }
    virtual void update_a(const Eigen::SparseMatrix<double> &mtr) {
        assert(0 && "please implement in subclass!");
    }

    virtual void analyze_pattern(void) = 0;

    virtual bool factorize(void) = 0;

    virtual void solve(Eigen::VectorXd &rhs, Eigen::VectorXd &result) = 0;

    virtual void multiply(const Eigen::VectorXd &x, Eigen::VectorXd &Ax) {
        assert(x.size() == numRows);

        Ax.setZero(numRows);
        for (int rowI = 0; rowI < numRows; ++rowI) {
            for (int eI = ia[rowI]; eI < ia[rowI + 1]; eI++) {
                const int colI = ja[eI];
                Ax[rowI] += a[eI] * x[colI];
                if (rowI != colI) {
                    Ax[colI] += a[eI] * x[rowI];
                }
            }
        }
    }

  public:
    virtual double coeffMtr(int rowI, int colI) const {
        if (rowI > colI) {
            // return only upper right part for symmetric matrix
            std::swap(rowI, colI);
        }
        const int dest = find_a_index(rowI, colI);
        return (dest >= 0) ? a[dest] : 0.0;
    }
    virtual void getCoeffMtr(Eigen::SparseMatrix<double> &mtr) const {
        mtr.resize(numRows, numRows);
        mtr.setZero();
        mtr.reserve(a.size() * 2 - numRows);
        for (int rowI = 0; rowI < numRows; rowI++) {
            for (int eI = ia[rowI]; eI < ia[rowI + 1]; eI++) {
                const int colI = ja[eI];
                mtr.insert(rowI, colI) = a[eI];
                if (rowI != colI) {
                    mtr.insert(colI, rowI) = a[eI];
                }
            }
        }
    }
    virtual void setCoeff(int rowI, int colI, double val) {
        if (rowI <= colI) {
            const int dest = find_a_index(rowI, colI);
            assert(dest >= 0);
            a[dest] = val;
        }
    }
    virtual void setZero(void) { a.setZero(); }
    virtual void addCoeff(int rowI, int colI, double val) {
        if (rowI <= colI) {
            const int dest = find_a_index(rowI, colI);
            assert(dest >= 0);
            a[dest] += val;
        }
    }

    virtual int getNumRows(void) const { return numRows; }
    virtual int getNumNonzeros(void) const { return a.size(); }
    virtual Eigen::VectorXi &get_ia(void) { return ia; }
    virtual Eigen::VectorXi &get_ja(void) { return ja; }
    virtual Eigen::VectorXd &get_a(void) { return a; }
    virtual const Eigen::VectorXd &get_a(void) const { return a; }
};

} // namespace uvgami

#endif
