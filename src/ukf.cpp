#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.0;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.5;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  // Complete the initialization. Need to tune above values
  n_x_ = 5;
  n_aug_ = 7;
  lambda_ = 3-n_x_;
  //set vector for weights
  weights_ = VectorXd(2*n_aug_+1);
  weights_.fill(0.0);
  double weight_0 = lambda_/(lambda_+n_aug_);
  weights_(0) = weight_0;
  for (int i=1; i<2*n_aug_+1; i++) {  
    double weight = 0.5/(n_aug_+lambda_);
    weights_(i) = weight;
  }
  Xsig_pred_ = MatrixXd(n_x_, 2*n_aug_+1);
  Xsig_pred_.fill(0.0);

}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  
  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    // first measurement
    cout << "UKF: " << endl;
    x_ = VectorXd(n_x_);
    x_.fill(0.0);
    
    // switch between lidar and radar measurements
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      // convert radar from polar to cartesian coordinates and initialize state.
      double rho = meas_package.raw_measurements_[0];
      double phi = meas_package.raw_measurements_[1];
      double rhod = meas_package.raw_measurements_[2];
      x_ << rho*cos(phi), rho*sin(phi), rhod, phi, 0.0;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      // initialize state.
      x_ << meas_package.raw_measurements_[0],
            meas_package.raw_measurements_[1],
            0.0, 0.0, 0.0;
    }

    P_ = MatrixXd(n_x_, n_x_);
    P_ << 1.0, 0.0, 0.0, 0.0, 0.0,
			    0.0, 1.0, 0.0, 0.0, 0.0,
			    0.0, 0.0, 1.5, 0.0, 0.0,
			    0.0, 0.0, 0.0, 1.0, 0.0,
          0.0, 0.0, 0.0, 0.0, 0.5;

    time_us_ = meas_package.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  // compute the dt (sec) between current and previous measurements
  double dt = (meas_package.timestamp_ - time_us_) / 1000000.0;
	time_us_ = meas_package.timestamp_;

  // do nothing if dt < 1 msec
  if(dt<0.001) return;

  Prediction(dt);

  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    UpdateRadar(meas_package);
  } else {
    // Laser updates
    UpdateLidar(meas_package);
  }

  // print the output
  cout << "x_ = " << x_ << endl;
  cout << "P_ = " << P_ << endl;

}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {

  // estimate the object's location and modify the x_.
  // predict sigma points, the state, and the state covariance matrix.
  
  //create augmented mean vector
  x_aug_ = VectorXd(n_aug_);
  x_aug_.fill(0.0);
  
  //create augmented state covariance
  P_aug_ = MatrixXd(n_aug_, n_aug_);
  P_aug_.fill(0.0);
  
  //create sigma point matrix
  Xsig_aug_ = MatrixXd(n_aug_, 2*n_aug_+1);
  Xsig_aug_.fill(0.0);

  // update augmented state and covariance matrix 
  x_aug_.head(n_x_) = x_;
  P_aug_.topLeftCorner(5,5) = P_;
  P_aug_(5,5)=std_a_*std_a_;
  P_aug_(6,6)=std_yawdd_*std_yawdd_;

  // populate the augmented sigma point matrix
  MatrixXd A_aug = P_aug_.llt().matrixL();
  Xsig_aug_.col(0) = x_aug_;
  for (int i = 0; i< n_aug_; i++) {
    Xsig_aug_.col(i+1)        = x_aug_ + sqrt(lambda_+n_aug_) * A_aug.col(i);
    Xsig_aug_.col(i+1+n_aug_) = x_aug_ - sqrt(lambda_+n_aug_) * A_aug.col(i);
  }

  // use augmented sigma point matrix and pass through prediction
  for (int i=0; i< 2*n_aug_+1; i++) {
    
    //extract values
    double px = Xsig_aug_(0,i);
    double py = Xsig_aug_(1,i);
    double v = Xsig_aug_(2,i);
    double yaw = Xsig_aug_(3,i);
    double yawd = Xsig_aug_(4,i);
    double nu_a = Xsig_aug_(5,i);
    double nu_yawdd = Xsig_aug_(6,i);

    //predicted state values
    double px_p, py_p;

    //avoid division by zero
    if (fabs(yawd) > 0.001) {
      px_p = px + (v/yawd)*(sin(yaw+yawd*delta_t)-sin(yaw));
      py_p = py + (v/yawd)*(cos(yaw)-cos(yaw+yawd*delta_t));
    }
    else {
      px_p = px + v*delta_t*cos(yaw);
      py_p = py + v*delta_t*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    //add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t*cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t*sin(yaw);
    v_p = v_p + nu_a*delta_t;

    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    //write predicted sigma point into right column
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;

  }

  // predict the state mean and state covariance matrix
  x_.fill(0.0);
  for(int i=0; i<2*n_aug_+1; i++) {
    //predict state mean
    x_=x_+weights_(i)*Xsig_pred_.col(i);
  }
  // predict state covariance matrix
  P_.fill(0.0);
  VectorXd dx = VectorXd(n_x_);
  for(int i=0; i<2*n_aug_+1; i++) {
    // state difference
    dx=Xsig_pred_.col(i)-x_;
    //angle normalization
    while (dx(3)> M_PI) dx(3)-=2.*M_PI;
    while (dx(3)<-M_PI) dx(3)+=2.*M_PI;
    P_=P_+weights_(i)*dx*dx.transpose();
  }

}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  
  // use lidar data to update the belief about the object's
  // position. Modify the state vector, x_, and covariance, P_.
  // TODO: NIS calculations

  // predict measurement
  int n_z = 2;
  VectorXd z(n_z);
  z << meas_package.raw_measurements_[0],
       meas_package.raw_measurements_[1];
  MatrixXd H(n_z, n_x_);
  H.fill(0.0);
  H(0,0) = 1.0;
  H(1,1) = 1.0;
  VectorXd z_pred = H*x_;

  VectorXd y = z - z_pred;
  MatrixXd Ht = H.transpose();
  MatrixXd R(n_z,n_z);
  R.fill(0.0);
  R(0,0)=std_laspx_*std_laspx_;
  R(1,1)=std_laspy_*std_laspy_;
  MatrixXd S = H * P_ * Ht + R;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
	MatrixXd K = PHt * Si;

  //new estimate
	x_ = x_ + (K * y);
	long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K * H) * P_;

}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {

  // use radar data to update the belief about the object's
  // position. Modify the state vector, x_, and covariance, P_.
  // TODO: NIS calculations

  int n_z = 3;
  //create measurement vector
  VectorXd z = VectorXd(n_z);
  z << meas_package.raw_measurements_[0],
       meas_package.raw_measurements_[1],
       meas_package.raw_measurements_[2];

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2*n_aug_+1);
  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);
  //noise covariance matrix
  MatrixXd R = MatrixXd(n_z,n_z);
  
  //transform sigma points into measurement space
  for (int i = 0; i < 2*n_aug_+1; i++) {
    // extract values for better readibility
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v  = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);
    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;
    // measurement model
    Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y); //r
    Zsig(1,i) = atan2(p_y,p_x); //phi
    Zsig(2,i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y); //r_dot
  }

  //mean predicted measurement
  z_pred.fill(0.0);
  for (int i=0; i < 2*n_aug_+1; i++) {
      z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  //measurement covariance matrix S
  S.fill(0.0);
  for (int i=0; i<2*n_aug_+1; i++) {  //2n+1 simga points
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix
  R.fill(0.0);
  R << std_radr_*std_radr_, 0.0, 0.0,
       0.0, std_radphi_*std_radphi_, 0.0,
       0.0, 0.0,std_radrd_*std_radrd_;
  S = S + R;

  MatrixXd Tc = MatrixXd(n_x_, n_z);
  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i=0; i<2*n_aug_+1; i++) {
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;
    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //residual
  VectorXd z_diff = z - z_pred;

  //angle normalization
  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();

}
