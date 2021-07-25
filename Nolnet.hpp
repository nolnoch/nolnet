/*
 * Nolnet.hpp
 *
 *  Created on: Jul 16, 2021
 *      Author: braer
 */

#ifndef Nolnet_HPP_
#define Nolnet_HPP_


/* Includes */
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <random>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <eigen3/Eigen/Eigen>


/* Namespaces */
using namespace std;


/* Typedef */
typedef float Scalar;
typedef Eigen::MatrixXf Matrix;
typedef Eigen::VectorXf ColVector;


/* Class */
class Nolnet {

public:
	// ctor
	Nolnet(vector<uint> top, float givenRate);

	// train
	void train(vector<ColVector*> input_data, vector<ColVector*> output_data);

	// utility
	void printLayers();
	void printWeights();
	void printDeltas();
	void printCaches();

private:
	vector<uint> topology;

	vector<ColVector*> neurons;
	vector<ColVector*> caches;
	vector<ColVector*> deltas;
	vector<Matrix*> weights;

	Scalar learnRate;
	Scalar cost;
	uint depth;

	// propagators
	void forwardProp(ColVector& input);
	void backwardProp(ColVector& output);

	// utility
	void errCalc(ColVector& output);
	void updateWeights();
};


#endif /* Nolnet_HPP_ */
