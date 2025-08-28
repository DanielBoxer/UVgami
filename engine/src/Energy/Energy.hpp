//  Created by Minchen Li on 8/31/17.

#ifndef Energy_hpp
#define Energy_hpp

#include "TriMesh.hpp"

namespace uvgami {

// an abstract class for energy terms in the objective of an optimization
// problem
class Energy {
  protected:
    const bool needRefactorize;

  public:
    Energy(bool p_needRefactorize);
    virtual ~Energy(void) = default;

  public:
    bool getNeedRefactorize(void) const;

  public:
    virtual void computeEnergyVal(const TriMesh &data, double &energyVal,
                                  bool uniformWeight = false) const;
    virtual void getEnergyValPerElem(const TriMesh &data,
                                     Eigen::VectorXd &energyValPerElem,
                                     bool uniformWeight = false) const = 0;
    virtual void getEnergyValByElemID(const TriMesh &data, int elemI,
                                      double &energyVal,
                                      bool uniformWeight = false) const = 0;

    virtual void computeGradient(const TriMesh &data, Eigen::VectorXd &gradient,
                                 bool uniformWeight = false) const = 0;

    virtual void computeHessian(const TriMesh &data, Eigen::VectorXd *V,
                                Eigen::VectorXi *I = NULL,
                                Eigen::VectorXi *J = NULL,
                                bool uniformWeight = false) const = 0;
    virtual void computeHessian(const TriMesh &data, Eigen::MatrixXd &Hessian,
                                bool uniformWeight = false) const = 0;
    virtual void initStepSize(const TriMesh &data,
                              const Eigen::VectorXd &searchDir,
                              double &stepSize) const = 0;
};

} // namespace uvgami

#endif
