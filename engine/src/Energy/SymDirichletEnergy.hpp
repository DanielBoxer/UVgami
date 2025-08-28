//  Created by Minchen Li on 9/3/17.

#ifndef SymDirichletEnergy_hpp
#define SymDirichletEnergy_hpp

#include "Energy.hpp"

namespace uvgami {

class SymDirichletEnergy : public Energy {
  public:
    SymDirichletEnergy(void);
    virtual void getEnergyValPerElem(const TriMesh &data,
                                     Eigen::VectorXd &energyValPerElem,
                                     bool uniformWeight = false) const;
    virtual void getEnergyValByElemID(const TriMesh &data, int elemI,
                                      double &energyVal,
                                      bool uniformWeight = false) const;

    virtual void computeGradient(const TriMesh &data, Eigen::VectorXd &gradient,
                                 bool uniformWeight = false) const;

    virtual void computeHessian(const TriMesh &data, Eigen::VectorXd *V,
                                Eigen::VectorXi *I = NULL,
                                Eigen::VectorXi *J = NULL,
                                bool uniformWeight = false) const;
    virtual void computeHessian(const TriMesh &data, Eigen::MatrixXd &Hessian,
                                bool uniformWeight = false) const;

    // to prevent element inversion
    virtual void initStepSize(const TriMesh &data,
                              const Eigen::VectorXd &searchDir,
                              double &stepSize) const;
    virtual void computeLocalGradient(const TriMesh &data,
                                      Eigen::MatrixXd &localGradients) const;
    virtual void computeDivGradPerVert(const TriMesh &data,
                                       Eigen::VectorXd &divGradPerVert) const;
};
} // namespace uvgami

#endif
