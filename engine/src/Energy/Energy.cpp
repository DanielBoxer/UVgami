//  Created by Minchen Li on 9/4/17.

#include "Energy.hpp"
#include <igl/avg_edge_length.h>

namespace uvgami {

Energy::Energy(bool p_needRefactorize) : needRefactorize(p_needRefactorize) {}

bool Energy::getNeedRefactorize(void) const { return needRefactorize; }

void Energy::computeEnergyVal(const TriMesh &data, double &energyVal,
                              bool uniformWeight) const {
    Eigen::VectorXd energyValPerElem;
    getEnergyValPerElem(data, energyValPerElem, uniformWeight);
    energyVal = energyValPerElem.sum();
}
} // namespace uvgami
