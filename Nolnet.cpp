//============================================================================
// Name        : Nolnet.cpp
// Author      : Wade Burch
// Version     :
// Copyright   : TBD
// Description : Hello World in C++, Ansi-style
//============================================================================

#include "./Nolnet.hpp"

using namespace std;


Nolnet::Nolnet(vector<uint> top, float givenRate)
: topology(top), learnRate(givenRate), depth(0), cost(0) {

	random_device rd;
	mt19937 rgen(rd());

	for (uint i = 0; i < topology.size(); i++) {
		uint t = topology[i];

		// Create Neuron, Cache, and Delta layers
		if (i == topology.size() - 1) {
			neurons.push_back(new ColVector(t));
			deltas.push_back(new ColVector(t));
		} else {
			neurons.push_back(new ColVector(t + 1));
			deltas.push_back(new ColVector(t + 1));
			neurons.back()->coeffRef(t) = 1.0;
		}
		caches.push_back(new ColVector(t));

		// Initialize weights (using He)
		if (i > 0) {
			uint p = neurons[i-1]->size();
			uint c = neurons[i]->size();

			if (i != topology.size() - 1)
				c -= 1;

			normal_distribution<float> dist(0.0, sqrt(2.0/p));
			auto he = [&](){return dist(rgen);};

			Matrix m = Eigen::MatrixXf::NullaryExpr(c, p, he);
			weights.push_back(new Matrix(c, p));
			(*weights.back()) = m;
		}
		this->depth = (t+1 > depth) ? t+1 : depth;
	}

}

/* Currently using ReLU activation */

Scalar activationFunc(Scalar f) {
	return max((Scalar) 0.0, f);
}

Scalar activationFuncDeriv(Scalar f) {
	return ((f>0) ? 1.0 : 0.0);
}

void Nolnet::forwardProp(ColVector& input) {
	// Add input to first nodes (input layer)
	neurons.front()->block(0, 0, topology[0], 1) = input;

	// Propagate forward, store unactivated input as cache, and activate
	for (uint i = 1; i < topology.size(); i++) {
		auto s = topology[i];
		neurons[i]->block(0, 0, s, 1) = ((*weights[i-1]) * (*neurons[i-1]));
		(*caches[i]) = neurons[i]->block(0, 0, s, 1);
		neurons[i]->block(0, 0, s, 1) = neurons[i]->block(0, 0, s, 1).unaryExpr(ptr_fun(activationFunc));
	}

}

void Nolnet::backwardProp(ColVector& output) {
	this->errCalc(output);
	this->updateWeights();
}

void Nolnet::errCalc(ColVector& output) {
	// Diff output layer and expected values, obtain cost
	(*deltas.back()) = output - (*neurons.back());

	// Calculate MSE
	//this->cost = deltas.back()->array().square().sum() / 2.0;
	this->cost = deltas.back()->array().square().sum() / deltas.back()->size();

	// Clear deltas for n-1 (prep for accumulation)
	for (uint i = topology.size() - 2; i > 0; i--) {
		(*deltas[i]) = deltas[i]->Zero(deltas[i]->size());
	}

}

void Nolnet::updateWeights() {
	for (int i = topology.size() - 2; i >= 0; i--) {
		for (uint r = 0; r < weights[i]->rows(); r++) {
			for (uint c = 0; c < weights[i]->cols(); c++) {
				auto dEdO = deltas[i+1]->coeffRef(r);
				auto dOdN = activationFuncDeriv(caches[i+1]->coeffRef(r));
				auto dNdW = neurons[i]->coeffRef(c);

				deltas[i]->coeffRef(c) += dEdO * dOdN * weights[i]->coeffRef(r,c);

				weights[i]->coeffRef(r,c) += learnRate * dEdO * dOdN * dNdW;
			}
		}
	}

}

void Nolnet::train(vector<ColVector*> input_data, vector<ColVector*> output_data) {
	for (uint i = 0; i < input_data.size(); i++) {
		this->forwardProp(*input_data[i]);
		this->backwardProp(*output_data[i]);

		cout << i << "," << this->cost << endl;
	}
}

void Nolnet::printLayers() {
	for (uint d = 0; d < depth; d++) {
		for (uint i = 0; i < neurons.size(); i++) {
			if (neurons[i]->size() > d) {
				cout << setw(12) << neurons[i]->coeffRef(d);
			} else {
				cout << setw(12) << "-";
			}
		}
		cout << endl;
	}
}

void Nolnet::printWeights() {
	for (uint i = 0; i < weights.size(); i++) {
		cout << *weights[i] << "\n" << endl;
	}

}

void Nolnet::printDeltas() {
	for (uint d = 0; d < depth; d++) {
		for (uint i = 0; i < deltas.size(); i++) {
			if (deltas[i]->size() > d) {
				cout << setw(12) << deltas[i]->coeffRef(d);
			} else {
				cout << setw(12) << "-";
			}
		}
		cout << endl;
	}

}

void Nolnet::printCaches() {
	for (uint d = 0; d < depth-1; d++) {
		for (uint i = 0; i < caches.size(); i++) {
			if (caches[i]->size() > d) {
				cout << setw(12) << caches[i]->coeffRef(d);
			} else {
				cout << setw(12) << "-";
			}
		}
		cout << endl;
	}

}
