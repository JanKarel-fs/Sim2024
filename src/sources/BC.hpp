#ifndef BC_HPP
#define BC_HPP

#include <map>
#include "typedefs.hpp"
#include "inlet.hpp"
#include "outlet.hpp"
#include "slipWall.hpp"
#include "noSlipWall.hpp"
#include "symmetry.hpp"
#include "homogeneousNeumann.hpp"

using namespace std;

map<string, bCondition> bcList = {condition("inlet", inlet),
				  condition("outlet", outlet),
				  condition("slipWall", slipWall),
				  condition("noSlipWall", noSlipWall),
				  condition("symmetry", symmetry),
				  condition("homogeneousNeumann", homogeneousNeumann)};


#endif
