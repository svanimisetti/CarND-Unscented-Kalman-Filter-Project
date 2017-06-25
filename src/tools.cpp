#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
	VectorXd rmse(4);
	rmse << 0.0, 0.0, 0.0, 0.0;

    // check the validity of the following inputs:
	//  * the estimation vector size should not be zero
	//  * the estimation vector size should equal ground truth vector size
	if( (estimations.size()==0) ||
	    (estimations.size()!=ground_truth.size()) ) {
	    return rmse;
	}
	
	//accumulate squared residuals
	for(int i=0; i < estimations.size(); ++i){
        VectorXd res(4);
        res = estimations[i]-ground_truth[i];
        res = res.array()*res.array();
        rmse += res;
	}
	
	//calculate the mean
	rmse = rmse/estimations.size();

	//calculate the squared root
	rmse = rmse.array().sqrt();

	//return the result
	return rmse;
	
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
	
	MatrixXd Hj(3,4);
	//recover state parameters
	double px = x_state(0);
	double py = x_state(1);
	double vx = x_state(2);
	double vy = x_state(3);
	
	double pm = sqrt(pow(px,2)+pow(py,2));
    
	if(pm==0) {
        //check division by zero
		Hj << 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, 0.0, 0.0;
    } else {
        //compute the Jacobian matrix
		Hj << px/pm, py/pm, 0.0, 0.0,
             -py/pow(pm,2), px/pow(pm,2), 0.0, 0.0,
              py*(vx*py-vy*px)/pow(pm,3), px*(vy*px-vx*py)/pow(pm,3),
			    px/pm, py/pm;
    }

	return Hj;
	
}
