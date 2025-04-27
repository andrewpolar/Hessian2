//Concept: Andrew Polar and Mike Poluektov
//Developer Andrew Polar

//License
//In case if end user finds the way of making a profit by using this code and earns
//billions of US dollars and meet developer bagging change in the street near McDonalds,
//he or she is not in obligation to buy him a sandwich.

//Symmetricity
//In case developer became rich and famous by publishing this code and meet misfortunate
//end user who went bankrupt by using this code, he is also not in obligation to buy
//end user a sandwich.

//Publications:
//https://www.sciencedirect.com/science/article/abs/pii/S0016003220301149
//https://www.sciencedirect.com/science/article/abs/pii/S0952197620303742
//https://arxiv.org/abs/2305.08194

//Remarks from Andrew Polar.
//This is comparison of two methods Newton-Kaczmarz and LBFGS.
//The executable Hessian2.exe uses library provided by Jorge Nocedal and Naoaki Okazaki.
//I use library as is whithout changes. Library must be compiled prior to executable. 
//Visual studio does not hold build order, so build library first from right-click and Hessian2 next.
//Rebuild menu item usually violates build order, but clean and build supports.

#include <iostream>
#include <thread>
#include <stdio.h>
#include "lbfgs.h"
#include "Helper.h"
#include "Urysohn.h"

//Parameters for dataset and model: 

////Features are 4 * 4 random matrices, targets are determinants.
//const int nTrainingRecords    = 100000;
//const int nValidationRecords  = 20000;
//const int nU1                 = 1;
//const int nU0                 = 50;
//const int nPoints0            = 3;
//const int nPoints1            = 30;
//const int nMatrixSize         = 4;
//const double features_min     = 0.0;
//const double features_max     = 1.0;
//const double termination_rate = 0.980;

//Features are 3 * 3 random matrices, targets are determinants.
const int nTrainingRecords    = 10000;
const int nValidationRecords  = 2000;
const int nU1                 = 1;
const int nU0                 = 10;
const int nPoints0            = 3;
const int nPoints1            = 30;
const int nMatrixSize         = 3;
const double features_min     = 0.0;
const double features_max     = 1.0;
const double termination_rate = 0.980;

///////////// Determinat dataset
std::unique_ptr<std::unique_ptr<double[]>[]> GenerateInput(int nRecords, int nFeatures, double min, double max) {
	auto x = std::make_unique<std::unique_ptr<double[]>[]>(nRecords);
	for (int i = 0; i < nRecords; ++i) {
		x[i] = std::make_unique<double[]>(nFeatures);
		for (int j = 0; j < nFeatures; ++j) {
			x[i][j] = static_cast<double>((rand() % 10000) / 10000.0);
			x[i][j] *= (max - min);
			x[i][j] += min;
		}
	}
	return x;
}

double determinant(const std::vector<std::vector<double>>& matrix) {
	int n = (int)matrix.size();
	if (n == 1) {
		return matrix[0][0];
	}
	if (n == 2) {
		return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
	}
	double det = 0.0;
	for (int col = 0; col < n; ++col) {
		std::vector<std::vector<double>> subMatrix(n - 1, std::vector<double>(n - 1));
		for (int i = 1; i < n; ++i) {
			int subCol = 0;
			for (int j = 0; j < n; ++j) {
				if (j == col) continue;
				subMatrix[i - 1][subCol++] = matrix[i][j];
			}
		}
		det += (col % 2 == 0 ? 1 : -1) * matrix[0][col] * determinant(subMatrix);
	}
	return det;
}

double ComputeDeterminant(std::unique_ptr<double[]>& input, int N) {
	std::vector<std::vector<double>> matrix(N, std::vector<double>(N, 0.0));
	int cnt = 0;
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			matrix[i][j] = input[cnt++];
		}
	}
	return determinant(matrix);
}

std::unique_ptr<double[]> ComputeDeterminantTarget(const std::unique_ptr<std::unique_ptr<double[]>[]>& x, int nMatrixSize, int nRecords) {
	auto target = std::make_unique<double[]>(nRecords);
	int counter = 0;
	while (true) {
		target[counter] = ComputeDeterminant(x[counter], nMatrixSize);
		if (++counter >= nRecords) break;
	}
	return target;
}
///////// End determinant data

void GetGradientAndObjectiveFunction(const std::vector<std::unique_ptr<Urysohn>>& u0, const std::unique_ptr<Urysohn>& u1,
	const std::unique_ptr<std::unique_ptr<double[]>[]>& features,
	const std::unique_ptr<double[]>& targets, int nStart, int nEnd, std::unique_ptr<double[]>& g, double& fx) {
	fx = 0.0;
	int nFeatures = (int)u0[0]->_model.size();
	int nPoints0 = (int)u0[0]->_model[0].size();
	int nPoints1 = (int)u1->_model[0].size();
	int nU0 = (int)u0.size();
	int nCols = nFeatures * nPoints0 * nU0 + nU0 * nPoints1;
	auto intermediate = std::make_unique<double[]>(nU0);
	auto derivatives = std::make_unique<double[]>(nU0);

	std::vector<std::vector<double>> model_derivatives0(nU0, std::vector<double>(nFeatures * 2, 0.0));
	std::vector<std::vector<int>> model_indexes0(nU0, std::vector<int>(nFeatures * 2, 0));

	std::vector<double> model_derivatives1(nU0 * 2, 0.0);
	std::vector<int> model_indexes1(nU0 * 2, 0);

	for (int j = 0; j < nCols; ++j) g[j] = 0.0;
	for (int k = nStart; k < nEnd; ++k) {
		for (int j = 0; j < nU0; ++j) {
			intermediate[j] = u0[j]->GetJacobianElementsAndUrysohn(features[k], model_derivatives0[j], model_indexes0[j]);
		}
		u1->GetUrysohn(intermediate, derivatives);
		double prediction = u1->GetJacobianElementsAndUrysohn(intermediate, model_derivatives1, model_indexes1);
		double residual = targets[k] - prediction;
		fx += residual * residual;

		//we have to update lower level derivatives
		for (int j = 0; j < nU0; ++j) {
			for (size_t s = 0; s < model_derivatives0[j].size(); ++s) {
				model_derivatives0[j][s] *= derivatives[j];
			}
		}

		//we increment urysohn indexes
		int innerUrysohnBlockSize = (int)u0[0]->_model.size() * (int)u0[0]->_model[0].size();
		int currentIncrement = innerUrysohnBlockSize;
		for (int j = 1; j < nU0; ++j) {
			for (size_t s = 0; s < model_indexes0[j].size(); ++s) {
				model_indexes0[j][s] += currentIncrement;
			}
			currentIncrement += innerUrysohnBlockSize;
		}
		for (size_t s = 0; s < model_indexes1.size(); ++s) {
			model_indexes1[s] += currentIncrement;
		}

		//we merge all vectors
		std::vector<double> tmpDerivatives;
		tmpDerivatives.reserve(nCols);
		for (int j = 0; j < nU0; ++j) {
			tmpDerivatives.insert(tmpDerivatives.end(), model_derivatives0[j].begin(), model_derivatives0[j].end());
		}
		tmpDerivatives.insert(tmpDerivatives.end(), model_derivatives1.begin(), model_derivatives1.end());

		std::vector<int> tmpIndexes;
		tmpIndexes.reserve(nCols);
		for (int j = 0; j < nU0; ++j) {
			tmpIndexes.insert(tmpIndexes.end(), model_indexes0[j].begin(), model_indexes0[j].end());
		}
		tmpIndexes.insert(tmpIndexes.end(), model_indexes1.begin(), model_indexes1.end());

		for (size_t j = 0; j < tmpIndexes.size(); ++j) {
			int position = tmpIndexes[j];
			g[position] += -2.0 * residual *  tmpDerivatives[j];
		}
	}
}

double AccuracyAssessment(const std::vector<std::unique_ptr<Urysohn>>& u0, const std::unique_ptr<Urysohn>& u1, 
	const std::unique_ptr<std::unique_ptr<double[]>[]>& features,
	const std::unique_ptr<double[]>& targets, int nRecords, double& pearson) {

	int nU0 = (int)u0.size();
	auto intermediate = std::make_unique<double[]>(nU0);
	auto actual = std::make_unique<double[]>(nRecords);

	double error = 0.0;
	for (int i = 0; i < nRecords; ++i) {
		for (int j = 0; j < nU0; ++j) {
			intermediate[j] = u0[j]->GetUrysohn(features[i]);
		}
		double prediction = u1->GetUrysohn(intermediate);
		error += (targets[i] - prediction) * (targets[i] - prediction);
		actual[i] = prediction;
	}
	pearson = Helper::Pearson(targets, actual, nRecords);
	error /= nRecords;
	error = sqrt(error);
	return error;
}

//this class is needed only to pass objects to callbacks of LBFGS lib
class ObjectsHolder {
public:
	int _nTrainingRecords;
	int _nValidationRecords;
	std::unique_ptr<std::unique_ptr<double[]>[]> _features_training;
	std::unique_ptr<std::unique_ptr<double[]>[]> _features_validation;
	std::unique_ptr<double[]> _targets_training;
	std::unique_ptr<double[]> _targets_validation;
	std::vector<std::unique_ptr<Urysohn>> _u0;
	std::unique_ptr<Urysohn> _u1;
	double _targetMin = 0.0;
	double _targetMax = 0.0;
	double _termination_rate = 0.0;
	int _N;
	ObjectsHolder(int nTrainingRecords, int nValidationRecords, int nMatrixSize, int nU0, int nU1, 
		int nPoints0, int nPoints1, double min, double max, double termination_rate) {
		_nTrainingRecords = nTrainingRecords;
		_nValidationRecords = nValidationRecords;
		_nFeatures = nMatrixSize * nMatrixSize;
		_min = min;
		_max = max;
		_features_training = GenerateInput(_nTrainingRecords, _nFeatures, _min, _max);
		_features_validation = GenerateInput(_nValidationRecords, _nFeatures, _min, _max);
		_targets_training = ComputeDeterminantTarget(_features_training, nMatrixSize, _nTrainingRecords);
		_targets_validation = ComputeDeterminantTarget(_features_validation, nMatrixSize, _nValidationRecords);

		//find limits
		std::vector<double> argmin;
		std::vector<double> argmax;
		Helper::FindMinMax(argmin, argmax, _targetMin, _targetMax, _features_training, _targets_training,
			nTrainingRecords, _nFeatures);

		//instantiate model
		_nU0 = nU0;
		_nU1 = nU1;
		_nPoints0 = nPoints0;
		_nPoints1 = nPoints1;
		_termination_rate = termination_rate;

		_N = _nU0 * _nFeatures * _nPoints0 + _nU0 * _nPoints1;

		for (int i = 0; i < _nU0; ++i) {
			_u0.push_back(std::make_unique<Urysohn>(argmin, argmax, 0.0, 1.0, nPoints0));
		}

		std::vector<double> argmin2;
		std::vector<double> argmax2;
		for (int i = 0; i < nU0; ++i) {
			argmin2.push_back(0.0);
			argmax2.push_back(1.0);
		}

		_u1 = std::make_unique<Urysohn>(argmin2, argmax2, _targetMin, _targetMax, nPoints1);
	}
	void AssignUrysohnsDirect(const lbfgsfloatval_t* x) {
		std::vector<double> u0Model(_nFeatures * _nPoints0);
		std::vector<double> u1Model(_nU0 * _nPoints1);
		int split_size = _nFeatures * _nPoints0;
		int position = 0;
		for (int j = 0; j < _nU0; ++j) {
			for (int s = 0; s < split_size; ++s) {
				u0Model[s] = x[position];
				++position;
			}
			_u0[j]->AssignUrysohnDirect(u0Model);
		}
		for (size_t j = 0; j < u1Model.size(); ++j) {
			u1Model[j] = x[position];
			++position;
		}
		_u1->AssignUrysohnDirect(u1Model);
	}
private:
	int _nU0;
	int _nU1;
	int _nPoints0 = 0;
	int _nPoints1 = 0;
	int _nFeatures = 0;
	double _min = 0.0;
	double _max = 1.0;
};

void RunKaczmarz() {
	int nFeatures = nMatrixSize * nMatrixSize;
	auto features_training = GenerateInput(nTrainingRecords, nFeatures, features_min, features_max);
	auto features_validation = GenerateInput(nValidationRecords, nFeatures, features_min, features_max);
	auto targets_training = ComputeDeterminantTarget(features_training, nMatrixSize, nTrainingRecords);
	auto targets_validation = ComputeDeterminantTarget(features_validation, nMatrixSize, nValidationRecords);

	clock_t start_application = clock();
	clock_t current_time = clock();

	//find limits
	std::vector<double> argmin;
	std::vector<double> argmax;
	double targetMin;
	double targetMax;
	Helper::FindMinMax(argmin, argmax, targetMin, targetMax, features_training, targets_training,
		nTrainingRecords, nFeatures);

	std::vector<std::unique_ptr<Urysohn>> u0(nU0);
	for (int i = 0; i < nU0; ++i) {
		u0[i] = std::make_unique<Urysohn>(argmin, argmax, 0.0, 1.0, nPoints0);
	}

	std::vector<double> argmin2;
	std::vector<double> argmax2;
	for (int i = 0; i < nU0; ++i) {
		argmin2.push_back(0.0);
		argmax2.push_back(1.0);
	}

	auto u1 = std::make_unique<Urysohn>(argmin2, argmax2, targetMin, targetMax, nPoints1);

	//auxiliary data buffers for a quick moving data between methods
	auto intermediate = std::make_unique<double[]>(nU0);
	auto derivatives = std::make_unique<double[]>(nU0);
	
	printf("Newton - Kaczmarz method, features are random %d * %d matrices, targets are determinants\n", nMatrixSize, nMatrixSize);
	for (int epoch = 0; epoch < 256; ++epoch) {
		//training
		for (int i = 0; i < nTrainingRecords; ++i) {
			for (int j = 0; j < nU0; ++j) {
				intermediate[j] = u0[j]->GetUrysohn(features_training[i]);
			}
			double prediction = u1->GetUrysohn(intermediate, derivatives);
			double residual = targets_training[i] - prediction;
			for (int j = 0; j < nU0; ++j) {
				u0[j]->Update(derivatives[j] * residual * 0.5, features_training[i]);
			}
			u1->Update(residual * 0.005, intermediate);
		}

		//validation
		double error, pearson;
		error = AccuracyAssessment(u0, u1, features_validation, targets_validation, nValidationRecords, pearson);
		error /= (targetMax - targetMin);

		current_time = clock();
		printf("Epoch %d, current relative error %f, pearson %f, time %2.3f\n", epoch, error, pearson, (double)(current_time - start_application) / CLOCKS_PER_SEC);

		if (pearson > termination_rate) break;
	}
	printf("\n\n");
}

lbfgsfloatval_t evaluate(void* instance, const lbfgsfloatval_t* x, lbfgsfloatval_t* g, const int n, const lbfgsfloatval_t step) {
	ObjectsHolder* ctx = (ObjectsHolder*)instance;
	ctx->AssignUrysohnsDirect(x);
	double fx;

	const int THREADS = 8;
	int range = ctx->_nTrainingRecords / THREADS;
	int nStart[THREADS];
	int nEnd[THREADS];
	int start = 0;
	int end = range;
	for (int i = 0; i < THREADS; ++i) {
		nStart[i] = start;
		nEnd[i] = end;
		start += range;
		end += range;
	}
	nEnd[THREADS - 1] = ctx->_nTrainingRecords;
	int N = ctx->_N;

	std::vector<std::unique_ptr<double[]>> gBuffers;
	for (int i = 0; i < THREADS; ++i) {
		gBuffers.push_back(std::make_unique<double[]>(N));
	}

	double FX[THREADS];
	for (int i = 0; i < THREADS; ++i) {
		FX[i] = 0.0;
	}

	std::vector<std::thread> threads;
	for (int i = 0; i < THREADS; ++i) {
		threads.push_back(std::thread(GetGradientAndObjectiveFunction, std::ref(ctx->_u0), std::ref(ctx->_u1), std::ref(ctx->_features_training),
			std::ref(ctx->_targets_training), nStart[i], nEnd[i], std::ref(gBuffers[i]), std::ref(FX[i])));
	}

	for (int i = 0; i < (int)threads.size(); ++i) {
		threads[i].join();
	}

	for (int k = 0; k < N; ++k) {
		g[k] = 0.0;
	}
	for (int i = 0; i < THREADS; ++i) {
		for (int j = 0; j < N; ++j) {
			g[j] += gBuffers[i][j];
		}
	}

	fx = 0.0;
	for (int i = 0; i < THREADS; ++i) {
		fx += FX[i];
	}

	return fx;
}

int progress(void* instance,
	const lbfgsfloatval_t* x, const lbfgsfloatval_t* g, const lbfgsfloatval_t fx, const lbfgsfloatval_t xnorm, const lbfgsfloatval_t gnorm,
	const lbfgsfloatval_t step, int n, int k, int ls) {

	ObjectsHolder* ctx = (ObjectsHolder*)instance;
	double pearson = 0.0;
	double error2 = AccuracyAssessment(ctx->_u0, ctx->_u1, ctx->_features_validation, ctx->_targets_validation, ctx->_nValidationRecords, pearson);
	double error = error2 / (ctx->_targetMax - ctx->_targetMin);
	printf("RRMSE for validation = %f, Pearson = %f\n", error, pearson);
	if (pearson > ctx->_termination_rate) return 1;
	return 0;
}

void RunLBFGS() {
	clock_t start_application = clock();
	clock_t current_time = clock();

	int nFeatures = nMatrixSize * nMatrixSize;

	int N = nU0 * nFeatures * nPoints0 + nU0 * nPoints1;
	lbfgsfloatval_t fx;
	lbfgsfloatval_t* x = lbfgs_malloc(N);
	lbfgs_parameter_t param;

	if (x == NULL) {
		printf("ERROR: Failed to allocate a memory block for variables.\n");
		exit(1);
	}

	ObjectsHolder* objectsholder = new ObjectsHolder(nTrainingRecords, nValidationRecords, nMatrixSize, nU0, nU1, nPoints0,
		nPoints1, features_min, features_max, termination_rate);

	for (int i = 0; i < nU0 * nFeatures * nPoints0; ++i) {
		x[i] = rand() % 100 / 1000.0;
	}
	for (int i = nU0 * nFeatures * nPoints0; i < N; ++i) {
		x[i] = (rand() % 100 - 50) / 1000.0;
	}

	lbfgs_parameter_init(&param);
	printf("LBGFS, features are random %d * %d matrices, targets are determinants\n", nMatrixSize, nMatrixSize);

	int ret = lbfgs(N, x, &fx, evaluate, progress, objectsholder, &param);

	lbfgs_free(x);
	delete objectsholder;

	current_time = clock();
	printf("Training time for LBFGS %f\n\n", (double)(current_time - start_application) / CLOCKS_PER_SEC);
}

int main() {
	srand((unsigned int)time(NULL));
	RunKaczmarz();
	RunLBFGS();
}
