#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0.0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225,    0.0,
                 0.0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09,    0.0,  0.0,
               0.0, 0.0009,  0.0,
               0.0,    0.0, 0.09;

  //measurement function - laser
  H_laser_ << 1.0, 0.0, 0.0, 0.0,
              0.0, 1.0, 0.0, 0.0;
  
  //measurement function - radar
  Hj_ << 1.0, 0.0, 0.0, 0.0,
         0.0, 1.0, 0.0, 0.0,
         0.0, 0.0, 1.0, 0.0;

}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    
    // Initialize the state ekf_.x_ with the first measurement.
    // Create the covariance matrix.
    // convert radar from polar to cartesian coordinates.
    
    // first measurement
    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1.0, 1.0, 1.0, 1.0;
    ekf_.P_ = MatrixXd(4, 4);
    ekf_.P_ << 1.0, 0.0,    0.0,    0.0,
			         0.0, 1.0,    0.0,    0.0,
			         0.0, 0.0, 1000.0,    0.0,
			         0.0, 0.0,    0.0, 1000.0;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      // convert radar from polar to cartesian coordinates and initialize state.
      double rho = measurement_pack.raw_measurements_[0];
      double psi = measurement_pack.raw_measurements_[1];
      double rhod = measurement_pack.raw_measurements_[2];
      ekf_.x_ << rho*cos(psi), rho*sin(psi), rhod*cos(psi), rhod*sin(psi);
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      // initialize state.
      ekf_.x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0;
    }

    previous_timestamp_ = measurement_pack.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  double noise_ax = 9.0;
  double noise_ay = 9.0;
  
  //compute the time elapsed between the current and previous measurements
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;	//dt - expressed in seconds
	previous_timestamp_ = measurement_pack.timestamp_;

  ekf_.F_ = MatrixXd(4, 4);
  ekf_.F_ << 1.0, 0.0,  dt, 0.0,
             0.0, 1.0, 0.0,  dt,
             0.0, 0.0, 1.0, 0.0,
             0.0, 0.0, 0.0, 1.0;
  
  ekf_.Q_ = MatrixXd(4, 4);
  ekf_.Q_ << pow(dt,4.0)*noise_ax/4.0, 0.0, pow(dt,3.0)*noise_ax/2.0, 0.0,
             0.0, pow(dt,4.0)*noise_ay/4.0, 0.0, pow(dt,3.0)*noise_ay/2.0,
             pow(dt,3.0)*noise_ax/2.0, 0.0, pow(dt,2.0)*noise_ax, 0.0,
             0.0, pow(dt,3.0)*noise_ay/2.0, 0.0, pow(dt,2.0)*noise_ay;

  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  // use the sensor type to perform the update step.
  // update the state and covariance matrices.
  
  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    ekf_.R_ = R_radar_;
    ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
    VectorXd z;
    z = VectorXd(3);
    z << measurement_pack.raw_measurements_[0],
         measurement_pack.raw_measurements_[1],
         measurement_pack.raw_measurements_[2];
    ekf_.UpdateEKF(z);
  } else {
    // Laser updates
    ekf_.R_ = R_laser_;
    ekf_.H_ = H_laser_;
    VectorXd z;
    z = VectorXd(2);
    z << measurement_pack.raw_measurements_[0],
         measurement_pack.raw_measurements_[1];
    ekf_.Update(z);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
