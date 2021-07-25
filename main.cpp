// main.cpp

#include "Nolnet.hpp"

typedef vector<ColVector*> data;


void printData(vector<ColVector*> dector) {
	for (uint i = 0; i < dector.size(); i++) {
		cout << *dector[i] << endl;
	}

}

void readCSV(string filename, vector<ColVector*>& data) {

	data.clear();
	ifstream file(filename);
	string line, word;

	getline(file, line, '\n');

	stringstream ss(line);
	vector<Scalar> parsed_vec;

	while (getline(ss, word, ',')) {
		parsed_vec.push_back(Scalar(stof(&word[0])));
	}

	uint cols = parsed_vec.size();
	data.push_back(new ColVector(cols));
	for (uint i = 0; i < cols; i++) {
		data.back()->coeffRef(i) = parsed_vec[i];
	}

	if (file.is_open()) {
		while (getline(file, line, '\n')) {
			stringstream ss(line);
			data.push_back(new ColVector(cols));
			uint i = 0;
			while (getline(ss, word, ',')) {
				data.back()->coeffRef(i++) = Scalar(stof(&word[0]));
			}
		}
	}

}

void genData(string filename) {
	ofstream file1(filename + "-in");
	ofstream file2(filename + "-out");
	for (uint r = 0; r < 500; r++) {
		Scalar x = rand() / Scalar(RAND_MAX);
		Scalar y = rand() / Scalar(RAND_MAX);
		file1 << x << "," << y << endl;
		file2 << 2 * x + 10 + y << endl;
	}
	file1.close();
	file2.close();

}

int main(int argc, char** argv)
{
	float rate = (argc > 1) ? stof(argv[1]) : 0.01;
	//cout << "Learning Rate: " << rate << endl;
	Nolnet n({ 2, 3, 1 }, rate);
	data in_dat, out_dat;

	genData("test");
	readCSV("test-in", in_dat);
	readCSV("test-out", out_dat);

	n.train(in_dat, out_dat);
	return 0;
}
